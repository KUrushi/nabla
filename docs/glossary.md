# 用語集

CLAUDE.md や設計書に出てくる専門用語のうち、大学の学部やジュニアエンジニアの段階では習わないことが多いものを説明する。まず一言で意味を書き、必要ならこのプロジェクトでの使われ方を補足する。

## コンパイラと実行系

**IR（中間表現, Intermediate Representation）**
: ソースコードと機械語の間に置く、プログラムの別の書き方。変換や最適化をしやすい形にしてある。nabla では JAX の jaxpr にならった小さな IR（`aval` / `var` / `eqn` / `graph`）を持つ。

**aval（抽象値, abstract value）**
: 配列の「中身」を持たず、「形状と dtype」だけを持つ値。トレース中は実際の数値が分からないので、aval だけを追って出力の形を決める。

**プリミティブ（primitive）**
: `add` や `reduce-sum` のような、nabla が知っている最小単位の演算。`defprimitive` で宣言し、形状推論（abstract-eval）・StableHLO 出力（emit）・eager 用の CPU 実装（eager）を束ねる。`jvp` / transpose ルール / バッチ化ルールは、その演算を `grad` / `vmap` に対応させるときに、同じプリミティブに追加する。

**contracting dims / batch dims（縮約次元・バッチ次元）**
: StableHLO の `dot_general`（nabla の `dot-general` プリミティブ）が、どの次元をどう扱うかを指定する2種類の次元。contracting dims（縮約次元）は、行列積の「掛けて足し込む」次元（`lhs-contracting` / `rhs-contracting`）で、両側のサイズが一致していなければならない。batch dims（バッチ次元）は、縮約せず両オペランドに共通して残る次元（`lhs-batch` / `rhs-batch`）で、その次元ごとに独立した縮約を行う（バッチ行列積）。どちらにも属さない次元は自由次元（free dims）と呼び、出力にはバッチ次元・lhs の自由次元・rhs の自由次元の順で並ぶ。

**reduction / 初期値（init value）**
: 配列のいくつかの次元を、要素同士を1つずつ組み合わせる二項演算（総和・最大値など）で潰して次元を減らすこと。StableHLO の `stablehlo.reduce`（nabla の `reduce-sum` / `reduce-max` プリミティブ）は、潰す次元ごとに空でないスライスを1つの値にまとめる。init value（初期値）は、その組み合わせを始める前に置いておく値で、演算の単位元（総和なら 0、最大値なら -∞）を使う。潰す次元のサイズが0（スライスが空）のときは、この初期値がそのまま出力になる。

**形状推論（abstract-eval）**
: 入力の `aval`（形状と dtype）とパラメタから、実際の数値を計算せずに出力の `aval` を決めること。プリミティブごとに `defprimitive` の `:abstract-eval` として書く。トレース中は実データが無いので、この計算だけで IR の各 `var` の形状・dtype を決められる。

**トレース（trace）/ トレーサ（tracer）**
: 関数を実際の数値ではなく特別なオブジェクト（トレーサ）で呼び出し、どんな演算が行われたかを記録すること。記録した結果が IR になる。

**jit（Just-In-Time コンパイル）**
: 関数が最初に呼ばれたときにコンパイルし、2回目以降はコンパイル済みのものを使う仕組み。nabla では、トレース → StableHLO 出力 → IREE でコンパイル、の流れになる。

**静的引数（static argument）**
: `jit` に `:static-args` で渡す、トレース時に固定する引数の位置。静的引数はトレース対象の本体へ普通の Lisp の値として渡り（クロージャに閉じ込められる）、`aval`（形状・dtype）を持つ動的引数とは別に、jit キャッシュのキーの一部になる。値が違えば別のコンパイル結果になる（`1` と `1.0` は異なる静的引数として扱う）ので、EQUAL で比較できる値（数値・シンボル・文字列・そのリストなど）にする。JAX の `static_argnums` に相当する。

**インメモリのコンパイルキャッシュ（jit キャッシュ）**
: `jit` が持つ、プロセス内・メモリ上だけのキャッシュ（`src/jit.lisp`）。キーは「関数の同一性（EQ）・aval・静的引数・実行系（ターゲット）」で、同じキーの2回目の呼び出しはトレース・コンパイルをせずコンパイル済みの module をそのまま使う。プロセスをまたいで効く vmfb のディスクキャッシュ（`src/compile-cache.lisp`、issue #10。`BACKEND-COMPILE` の `:AROUND` メソッドとして実装され、実行系がコンパイルした結果そのものをファイルに残す）とは別の層で、両方が独立に効く（jit キャッシュがヒットすればディスクキャッシュまで届かないし、jit キャッシュがミスしてもディスクキャッシュがヒットすれば実際のコンパイラは呼ばれない）。関数を再定義する（`WITH-TRACING` を再評価する、`defjit` を再評価する）と、新しい `TRACEABLE-FUNCTION` オブジェクトになるため、古いキャッシュは（EQ で一致しないので）使われない。古い関数のエントリが消えるとき（`defjit` の再定義、GC による回収）は、読み込んだ module を `BACKEND-UNLOAD` で解放する（GC のときは関数に登録した finalizer が行う。issue #71）。

**defjit**
: `(defjit name-or-(name :static-args positions) (&rest lambda-list) &body body)`。`body` を `with-tracing` でトレース対象にしてから `jit` した通常の関数を `name` に定義するマクロ（`src/jit.lisp`、issue #34）。CL の `defun` と同じ感覚で「関数を定義したら、その名前で呼べる」ようにする糖衣で、内部では毎回新しい `traceable-function` を作って `(setf (fdefinition name) ...)` する。再評価すると古いキャッシュエントリを捨てて module を解放するので、関数を再定義したら次の呼び出しは必ず再コンパイルする。`:static-args` は `jit` と同じ意味。

**リスタート（restart）**
: Common Lisp の条件システムが提供する「コンディションが signal された地点から、あらかじめ用意した別の処理を選んで再開する」仕組み。`error` と違い、呼び出し元（`handler-bind` を書いた側）が `invoke-restart` でどう続けるかを選べる（スタックを一度も巻き戻さずに選べるのが `handler-case` との違い）。nabla の `jit` は、キャッシュミスのコンパイルが `jit-compile-error` を signal したとき2つのリスタートを提供する: `use-eager`（この呼び出しだけ `eval-graph` で eager に評価して返す。何もキャッシュしない）と `recompile`（もう一度コンパイルをやり直す）。

**loc（位置情報）**
: MLIR のテキストで、ある演算がソースのどこに由来するかを添える注釈（`stablehlo.add %a, %b : tensor<4xf32> loc("eqn-3")` の末尾部分）。nabla の `emit-stablehlo` は各 eqn の出力行に `loc("eqn-N")`（N は `graph-eqns` 中の0始まりの位置）を付ける。IREE のコンパイルエラーの診断がこの loc を含んでいれば、`graph-eqn-for-diagnostic` でどの eqn が原因かを逆引きできる。

**診断（diagnostic）**
: コンパイラがエラーや警告を報告するときの1つのメッセージ（ファイル位置・重大度・本文を持つ）。IREE の `iree-compile-error` は複数の診断を持つことがある。

**静的形状（static shape）**
: 配列の形がコンパイルの時点で決まっていること。形が変わるたびに再コンパイルが必要になる代わりに、実装が単純になる。

**MLIR**
: LLVM プロジェクトの一部で、コンパイラの IR を作るための共通の枠組み。「方言（dialect）」という単位で命令セットを定義できる。

**pretty form / generic form（MLIR の省略記法と汎用記法）**
: MLIR のテキスト表現には2つの書き方がある。generic form は
`"stablehlo.add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>`
のように、演算名を文字列にし、属性を `{...}` で持つ、どんな方言の演算にも
使える汎用の書き方。pretty form は `%0 = stablehlo.add %a, %b :
tensor<4xf32>` のように、その方言が独自に定義した、人が読み書きしやすい
省略記法。両方とも同じ演算を表し、IREE のコンパイラはどちらも受け付ける
（`docs/stablehlo-ops.md` の対応表を参照）。nabla の emitter（issue #33）は
pretty form を出力する。

**StableHLO**
: MLIR の方言の1つで、機械学習の計算（行列積、畳み込み、要素ごとの演算など）を表すための命令セット。JAX、PyTorch、TensorFlow の共通の出力形式として使われている。nabla と IREE の境界になる。

**SSA（静的単一代入, Static Single Assignment）**
: 変数への代入を1回だけに制限した書き方。`%0 = ...`, `%1 = ...` のように毎回新しい名前を付ける。依存関係が読み取りやすく、コンパイラが扱いやすい。StableHLO のテキストはこの形で書く。

**IREE**
: StableHLO などの MLIR を受け取り、CPU / CUDA / ROCm / Vulkan / Metal 向けの実行ファイルを作るコンパイラと、それを動かすランタイム。

**vmfb**
: IREE のコンパイラが出力する実行用のファイル形式（VM FlatBuffer）。コンパイルに時間がかかるので、nabla ではディスクにキャッシュする。IREE は既定で「polyglot zip」という形式で vmfb を出す（`--iree-vm-emit-polyglot-zip`）。中身はフラットバッファだが、ファイルの先頭は ZIP の local-file-header シグネチャ（`PK\3\4`、バイト列では `#x50 #x4B #x03 #x04`）で始まる。ランタイムが読み込むときにこの ZIP の皮を剥がすので、フラットバッファそのものの識別子を先頭バイトだと思って比較しないよう注意する。

**HAL（Hardware Abstraction Layer）**
: IREE の中で、CPU や各種 GPU の違いを隠す層。「HAL ドライバ」を切り替えることで実行するデバイスを選ぶ。

**device-array**
: IREE デバイス上の buffer view を包む、JAX の `jax.Array` に相当するクラス（`nabla.iree:device-array`）。実データ（buffer view の foreign pointer）と `aval`（形状と dtype）、そのデバイスへの参照を持つ。生成時に自分のデバイス（`iree_hal_device_t`）を retain するので、呼び出し側がデバイス自身を解放した後でも、生きている device-array から値を読み出せる。

**PJRT**
: XLA（JAX の標準の実行系）を外部から呼ぶための C API。nabla では IREE と並ぶ2つ目の実行系（`nabla/pjrt`）として、`backend` プロトコルの裏に置く。バックエンドごとに「プラグイン」（`.so`）があり、`GetPjrtApi` という1つの関数が `PJRT_Api` 構造体（関数ポインタの表。先頭に API の版 major / minor を持つ）を返す。nabla は `third_party/pjrt.lock` で固定した CPU / CUDA のプラグインを `scripts/fetch-pjrt.sh` で取得する（`docs/pjrt-setup.md`）。PJRT の関数はどれも `*_Args` 構造体（先頭に `struct_size` を持つ）へのポインタを1つ受け取り、失敗すると `PJRT_Error*` を返す（成功なら NULL）。主なオブジェクトは、プラグインの計算資源を持つ `PJRT_Client`、そのデバイス `PJRT_Device`、デバイス上の配列 `PJRT_Buffer`（nabla では `nabla.pjrt:device-array` が包む）、非同期処理の完了を表す `PJRT_Event`（転送の完了などを `PJRT_Event_Await` で待ち、`PJRT_Event_Destroy` で解放する）。

**埋め込み C API（embedding API）**
: IREE がコンパイラ・ランタイムの機能を、別プロセスを起動せずに自分のプロセス内から呼べるように提供している C の関数群。ヘッダは `iree/compiler/embedding_api.h`（コンパイラ）と `iree/runtime/api.h`（ランタイム）。nabla はこれを CFFI で直接 `dlopen` して呼び、`iree-compile` / `iree-run-module` をサブプロセスとして起動しない。

**CFFI**
: Common Lisp から C のライブラリを呼び出すためのライブラリ。

**libffi / cffi-libffi**
: libffi は C の関数呼び出しを実行時に組み立てるライブラリで、構造体を値で渡したり値で返したりする関数（`iree_allocator_t` や `iree_string_view_t` など）を、素の CFFI では扱えない場合に使う。cffi-libffi はこれを CFFI から使うための拡張で、`nabla/iree` のランタイムバインディングが依存する（ビルドには apt の `libffi-dev` が要る）。

**finalizer**
: オブジェクトが GC（ガベージコレクタ）に回収されるときに呼ばれる関数。nabla では GPU 上のメモリを解放するのに使う（`device-array`、`trivial-garbage:finalize` で登録）。finalizer の中で対象オブジェクト自身を参照すると、参照が残るので永遠に回収されなくなる。SBCL は finalizer を別スレッド（finalizer thread）で非同期に実行するため、`(sb-ext:gc :full t)` の直後に確認しても、ほとんどの finalizer はまだ実行されていない。テストでは `(sb-kernel:run-pending-finalizers)` を続けて呼び、保留中の finalizer を同期的に実行させて確認する（`tests/iree/support.lisp` の `gc-and-run-finalizers`）。保守的なスタックルートのせいで、full GC を1回しても少数のオブジェクトが生き残ることがあるので、リークを確かめるテストは0ではなく少量の残留を許容する。

**コードウォーク（code walk）/ コードウォーカ**
: Lisp のコード（リスト）を先頭から順にたどり、特定の形式を別の形式に書き換える処理。nabla では `with-tracing` の中の `if` や `loop` を、トレースできる `cond` / `scan` に書き換えるのに使う。事前に `macroexpand-all` でマクロをすべて展開してからたどる。

**funcallable instance**
: CLOS のオブジェクトでありながら、そのまま `funcall` / `apply` できる（関数としても振る舞う）インスタンス。SBCL では `sb-mop:funcallable-standard-object` をメタクラスに `sb-mop:funcallable-standard-class` を指定して作り、`sb-mop:set-funcallable-instance-function` で実際に呼ばれる関数を差し込む。nabla の `with-tracing` が返す `traceable-function`（`src/trace.lisp`、issue #32）はこれで作る: 呼び出せば eager に実行するふつうの関数として振る舞いつつ、`traceable-function-lambda-list` のようなアクセサでメタデータ（仮引数のリスト）も持てる。

**リフト（lift）**
: トレース中に、実数（Lisp の数値）や生の配列を、その場にあるトレーサ（`tracer`）と同じ dtype・shape の値に持ち上げること。数値は rank 0 の定数トレーサにしてから、必要なら `:broadcast-in-dim` でトレーサの shape まで広げる（`%lift-number`、`src/trace.lisp`）。配列は `array-aval` で aval を決めて、そのまま定数として graph に足す（`%lift-array`）。`(+ x 1)` のようにトレーサと数値・配列が混ざった式を、常にトレーサどうしの演算に揃えるための下ごしらえ。

**最近接偶数丸め（RNE, round to nearest, ties to even）**
: 浮動小数点の丸め方式の1つ。表現できる2つの値のうち近い方に丸め、ちょうど中間（等距離）のときは仮数の最下位ビットが0になる方（偶数）に丸める。IEEE 754 の既定の丸めモードで、nabla では bf16 / f16 と single-float の変換（`src/float16.lisp`）に使う。単純な切り捨てと違い、丸め誤差が特定の方向に偏らない。

**float trap（浮動小数点例外トラップ）**
: CPU が0除算・オーバーフロー・不正な演算（0/0 や sqrt(-1) など）を検出したときに、実行を止めてコンディションを signal する仕組み。SBCL は既定で `:overflow` `:invalid` `:divide-by-zero` の3つのトラップを有効にしているため、`(/ 1.0 0.0)` のような計算はそのままだと `division-by-zero` を signal してしまう。StableHLO / IREE は IEEE 754 どおり無限大・NaN を返す（signal しない）ので、nabla のプリミティブの eager 実装は `sb-int:with-float-traps-masked` でこの3つのトラップをマスクしてから計算し、両者の挙動を揃える（`src/primitives/common.lisp` の `with-ieee-arithmetic`）。マスクは要素ごとではなく、eager 呼び出し全体を1回だけ包む（速度のため）。`nabla/iree` では、これに加えて MXCSR（SSE の浮動小数点制御・ステータスレジスタ）の性質に注意が要る：Linux はスレッド生成（`clone(2)`）時に生成元スレッドの MXCSR をそのままコピーするため、IREE のワーカースレッドや LLVM コード生成を生成・実行する瞬間に呼び出し元スレッドがマスクされていないと、生成された側は未マスクのまま動き続ける（issue #53）。`nabla.ffi-support:with-all-float-traps-masked`（`src/ffi-support/float-traps.lisp`。実験の記録は `docs/float-traps-experiments.md`）は SBCL（x86-64）が制御できる5種類すべて（`:underflow` `:overflow` `:inexact` `:invalid` `:divide-by-zero`。6つ目の `:denormalized-operand` は SBCL では 32bit x86 専用で x86-64 には存在しない）をマスクし、`make-device` / `make-session` / `session-append-module` / `invoke` / `compile-stablehlo` などの生成・呼び出し点を包む。

## 自動微分と変換

**自動微分（automatic differentiation, AD）**
: プログラムとして書かれた関数の微分を、演算ごとの微分ルールを連鎖律でつないで正確に計算する方法。数値微分（差分で近似する）とも、数式処理（式を記号で変形する）とも違う。

**jvp（ヤコビアン・ベクトル積, Jacobian-Vector Product）/ 前進モード**
: 入力をある方向 `v` に少し動かしたとき、出力がどう動くか（`J·v`）を計算する。関数の計算と同時に前から順に求められる。入力が少なく出力が多い関数に向く。

**vjp（ベクトル・ヤコビアン積, Vector-Jacobian Product）/ 逆伝播**
: 出力側の重み `u` から、各入力への影響（`uᵀ·J`）を計算する。深層学習の「逆伝播（バックプロパゲーション）」はこれ。出力がスカラー（損失）で入力が多い関数に向くので、`grad` はこちらを使う。

**linearize（線形化）**
: jvp の計算を「入力の値だけで決まる部分」と「`v` に対して線形な部分」に分けること。線形な部分だけを取り出すと、次の transpose がかけられる。nabla は graph が静的なので、「接線の入力に推移的に依存する eqn か」だけで分ける（`src/ad/linearize.lisp`）。線形な部分が使う主値の中間値を、残差（residuals）と呼ぶ。

**transpose ルール（転置ルール）**
: 線形な演算 `L` に対して、その転置 `Lᵀ` を計算するルール。行列 `A` をかける演算なら、転置は `Aᵀ` をかける演算になる。JAX と nabla は「jvp を線形化して転置すると vjp になる」という性質を使い、演算ごとに書くルールを jvp と transpose の2種類に抑えている。transpose 変換（`src/ad/transpose.lisp`）は、線形な graph を逆順にたどり、各 eqn の出力の余接線からルールで入力の余接線を求め、同じ入力への寄与を足し合わせる。

**symbolic zero（シンボリックなゼロ）**
: 値がゼロと分かっている接線・余接線を、配列も eqn も作らずに表す内部オブジェクト（`src/ad/zero.lisp` の `symbolic-zero`）。jvp / transpose の変換はこれをそのまま伝播させ、ゼロとの加算や、ゼロを使う項の計算を丸ごと省く。graph の出力など、実体が必要になったときだけ `instantiate-zero` が、rank 0 の定数 0 と `broadcast-in-dim` で配列にする。transpose ルールで「まだ値が無い線形入力」を表す `undefined-primal` とは別物。

**grad / value-and-grad（勾配）**
: スカラー（rank 0 の浮動小数点）を返す関数 `f` の、引数についての勾配を返す関数を作る（`nb:grad`、`nb:value-and-grad`。`src/ad/grad.lisp`）。`value-and-grad` は値も一緒に返す。中身は、`f` を引数の `aval` で1回トレースして graph にし、vjp（余接線は rank 0 の `1`）で微分した graph にしたもの。呼び出しが別のトレース（`jit`・`with-tracing` の本体・別の `grad`）の中なら `inline-graph` でそのトレースへ展開し、そうでなければ `eval-graph` で評価する。`(grad f)` は呼ぶたびに新しい関数オブジェクトを作るので、`jit` のキャッシュ（関数の同一性が鍵）が効かず、ループの中で `(jit (grad f))` を作ると毎回コンパイルされる（JAX と同じ）。ループの外で作るか、`defjit` の本体の中で使う。

**stop-gradient（勾配を止める）**
: 値は入力そのままだが、自動微分では定数として扱う演算（`nb:stop-gradient`、プリミティブ `stop-gradient`、JAX の `lax.stop_gradient`）。jvp ルールは常に symbolic zero を返す。StableHLO には恒等の op が無いので、値を変えず最適化の境界になる `stablehlo.optimization_barrier` に出力する。

**balanced eq（等しいときは半分ずつ）**
: `max(x, y)` の微分で `x` と `y` が等しい点では、接線を各側に 0.5 ずつ流す JAX の規約（`jax._src.lax._balanced_eq`）。等しくない点では大きい側（`min` なら小さい側）の接線だけが通る。

**vmap / バッチ化ルール（batching rule）**
: `vmap` は、1つの例を処理する関数を、例の束（バッチ）をまとめて処理する関数に自動で変換する。そのために、各演算に「入力にバッチの軸が増えたら、出力のどこにバッチの軸が来るか」を決めるルールを書く。これがバッチ化ルール。nabla では `defprimitive` の `:batch`（または `def-batch-rule`）で設定し、`(lambda (args batch-dims &key <params>) → (values outs out-dims))` の形をとる。`batch-dims` は各引数のバッチ次元の位置か、バッチされていないことを表す `nil`。出力のバッチ次元の位置 `out-dims` も返す。詳しくは「バッチ次元（vmap）」を見る。

**PyTree**
: リストや構造体を入れ子にした木構造で、葉に配列を持つもの。モデルのパラメータ全体を1つの値として `grad` や `jit` に渡すために使う。`flatten` で葉の列に平たくし、`unflatten` で元の形に戻す。

**PRNG（疑似乱数生成器）/ threefry**
: 決まった計算で乱数のように見える数列を作る仕組み。JAX と nabla は「キー」を明示的に渡し、キーを `split` して子キーを作る方式をとる。同じキーからは必ず同じ乱数が出るので、`jit` や `vmap` と両立する。threefry はその計算に使うアルゴリズムの名前。

**bf16 / f16**
: 16 ビットの浮動小数点数。f16（半精度）は仮数部が多く範囲が狭い。bf16（brain float 16）は f32 と同じ指数部を持ち、範囲が広い代わりに精度が低い。GPU での学習を速くするために使う。

**NaN（Not a Number）/ quiet NaN**
: IEEE 754 で「数として定義できない結果」（`0.0/0.0` や負数の `log` など）を表す特別な浮動小数点値。quiet NaN はそのうち、演算に混ざってもプロセスを落とさず（signal を出さず）そのまま伝播する種類の NaN（対になる signaling NaN は使わない）。nabla では `%quiet-nan`（`src/primitives/common.lisp`）がビットパターンから直接組み立てる。

**NaN propagation（NaN 伝播）**
: 演算の入力のどれかが NaN なら、出力も必ず NaN になるという規則。StableHLO / IREE / JAX の `max` / `min` はこの規則に従うが、Common Lisp の `max` / `min` は引数の順序によって NaN を落としてしまうことがあるため、nabla は `%ieee-max` / `%ieee-min` で明示的に NaN 伝播を実装している。

**指示関数（indicator）**
: 条件を満たす要素で 1、そうでなければ 0 になる配列。`reduce-max` の jvp は、最大値を取る要素の指示関数を主値だけから作り、`reduce-sum(接線 · 指示関数) / reduce-sum(指示関数)` で接線を選ぶ（最大値が重複すれば平均になる。JAX と同じ）。指示関数は主値にしか依存しないので、接線について線形のまま保てる。

## テスト

**property-based testing（PBT, 性質ベーステスト）**
: 具体的な入力と期待値を並べる代わりに、「どんな入力でも成り立つはずの性質」を書き、入力をランダムに大量に生成して確かめるテスト。例: 「どんなリスト `x` でも `(reverse (reverse x))` は `x` に等しい」。

**生成器（generator）**
: PBT で、ランダムな入力を作る関数。どんな範囲の値をどんな分布で作るかを決める。

**縮小（shrinking）**
: PBT で失敗する入力が見つかったとき、同じ失敗を起こす、より小さく単純な入力を自動で探すこと。原因を調べやすくなる。

**中心差分（central difference）**
: 微分を `(f(x + h) - f(x - h)) / 2h` で近似する方法。自動微分の結果が正しいかを確かめるのに使う。丸め誤差を減らすため f64 で計算する。

**内積テスト（dot-product test）**
: jvp と vjp が互いに正しい「転置の関係」にあるかを確かめる方法。任意のベクトル `u`, `v` に対して `<vjp(u), v>` と `<u, jvp(v)>`（`< , >` は内積）が等しくなることを確かめる。元の関数の正しい微分値を知らなくても検査できる。

**rtol / atol（相対誤差と絶対誤差の許容値）**
: 浮動小数点数の比較で使う。`|actual - expected| <= atol + rtol * |expected|` なら一致とみなす。値が大きいときは rtol が、0 に近いときは atol が効く。

**mutation testing（変異テスト）**
: テストの品質を測るテスト。プログラムにわざと小さなバグを入れ、テストがそれを見つけられるか（落ちるか）を確かめる。

**変異体（mutant）**
: mutation testing でわざとバグを入れたプログラム。テストが落ちれば「殺された（killed）」、落ちなければ「生き残った（survived）」と言う。

**等価変異体（equivalent mutant）**
: 変異させても、どんな入力でもプログラムの振る舞いが変わらない変異体。例えば、結果に影響しない `(* x 1)` を `(/ x 1)` に変えたもの。どんなテストでも殺せないので、除外リストに入れる。

**mutation score**
: 殺された変異体の数 ÷（全変異体の数 − 等価変異体の数）。テストがどれだけバグを見つけられるかの目安。

**arid node**
: 変異させても意味のある違いが生まれない、または確かめる価値の低いコードの箇所。ログ出力、エラーメッセージの文字列、型宣言などがこれにあたる。Google の mutation testing ではこれを除外して、ノイズを減らしている。

**テストサイズ（small / medium / large）**
: Google の分類で、テストが使ってよい資源の範囲。small は1プロセス内で完結し、I/O やスレッドを使わない。medium は1台のマシン内で、ファイルやローカルのプロセスを使ってよい。large は複数のマシンや外部の資源を使う。小さいほど速く、安定している。nabla では FiveAM のスイート `:nabla.small` / `:nabla.medium` / `:nabla.large` がこれに対応し、既定のスイート（scripts/run-tests.sh）は small + medium。

**ハーメティック（hermetic）**
: テストが外部の状態（ネットワーク、時刻、他のテスト、実行順序）に依存せず、それだけで完結していること。何度実行しても同じ結果になる。

**フレーキー（flaky）**
: コードを変えていないのに、実行するたびに成功したり失敗したりするテスト。原因の多くは、乱数、時刻、スレッドの実行順序、外部サービスへの依存。

**テストダブル / フェイク（fake）/ モック（mock）**
: テストで本物の代わりに使う部品をまとめてテストダブルと呼ぶ。フェイクは本物と同じように動く軽い実装（例: GPU の代わりの CPU バックエンド）。モックは「どう呼ばれたか」を記録・検証するための部品。Google は、本物 → フェイク → モックの順に優先することを勧めている。

**DAMP（Descriptive And Meaningful Phrases）**
: テストコードでは、重複を減らす（DRY: Don't Repeat Yourself）ことより、1つのテストを読むだけで意味が分かることを優先する、という考え方。

## ソフトウェアエンジニアリングの原則

**ハイラムの法則（Hyrum's Law）**
: 「API の利用者が十分に多ければ、仕様に書いたかどうかに関係なく、観測できるすべての振る舞いに誰かが依存するようになる」という経験則。Google のエンジニア Hyrum Wright にちなむ。公開するものを最小限にする理由になる。

**ビヨンセ・ルール（Beyoncé Rule）**
: 「気に入っていたなら、テストを付けておくべきだった（If you liked it, then you shoulda put a test on it）」。Google の社内ルールで、ビヨンセの曲の歌詞のもじり。インフラなど他のチームの変更で壊れて困る振る舞いは、自分でテストを書いて守る、という意味。

**shift left**
: 開発の流れを左（設計・実装）から右（リリース・運用）に並べたとき、問題をできるだけ左で見つけようという考え方。後で見つかるほど直すコストが大きい。

**非推奨化（deprecation）**
: 古い API をすぐに消さず、「将来なくなる」と告知して移行期間を設けてから消すこと。

**Conventional Commits**
: コミットメッセージを `<type>(<scope>): <説明>` という決まった形式で書く規約。変更の種類が機械的に読み取れるので、変更履歴の自動生成やバージョン番号の決定に使える。

**squash merge**
: PR をマージするとき、PR 内の複数のコミットを1つにまとめてから取り込む方法。`main` の履歴が「1つの PR = 1つのコミット」になり、読みやすく、取り消し（revert）もしやすい。



<!-- フェーズ3 anchor: issue #127 -->

**サブグラフ（subgraph）と高階プリミティブ**
: eqn の `params` の値として持たせる、閉じた `graph`（外側の var を参照せず、定数は自分の定数表に持つ）。`cond` / `while-loop` / `scan` のように、本体の関数を別の graph として持つプリミティブを高階プリミティブと呼ぶ。印字では入れ子に、StableHLO ではリージョンとして出す。`eval-graph` / `inline-graph` / `dce-graph` と jvp / transpose の変換は、サブグラフの中身を書き換えずに素通しする（中身の変換は各プリミティブのルールの仕事）。

**closure conversion（閉包変換）**
: 制御構造の本体が外側のトレーサを閉包で捕まえたとき、その値をサブグラフの追加の入力に持ち上げ、呼び出し側の eqn の入力の末尾に足す変換（`%trace-subgraph`）。同じトレーサは1回だけ持ち上げられ、持ち上げた順に並ぶ。`grad` の `%call-with-fresh-trace` は親を持たないので、外側のトレーサを捕まえると従来どおり `tracing-error` になる。

**複数出力のプリミティブ**
: `defprimitive` に `:multiple-outputs t` を付けたプリミティブ。`abstract-eval` が aval のリスト、`eager` が配列のリストを返し、`emit` は出力の名前と aval をリストで受けて `%8, %9 = ...` の左辺を自分で書く。トレースには、常にトレーサのリストを返す `%trace-eqn*` を使う。



<!-- フェーズ3 anchor: issue #126 -->

**2 の補数での折り返し（整数のオーバーフロー）**
: 整数の演算結果がその dtype の範囲を超えたとき、上位のビットを捨てて範囲内に収める挙動（`:i32` は 2 の補数、`:u32` / `:u64` は法 2^n）。StableHLO の整数演算がこうなので、eager 実装も同じにそろえる（`(+ 2147483647 1)` の `:i32` は -2147483648）。issue #126。



<!-- フェーズ3 anchor: issue #125 -->

**バッチ次元（vmap）**
: `vmap` が、例の束を並べるために配列へ足した軸。`dot_general` の batch dims（縮約せずに両オペランドに共通して残る次元。上の「contracting dims / batch dims」）とは別の概念で、同じ名前で呼ぶので注意する。`vmap` は値ごとに「バッチ次元がどの位置にあるか（無ければ `nil`）」を伝播させながら graph を書き換える。バッチ次元を持たない値だけを入力とする演算は、バッチ化ルールを呼ばずにそのまま残す（不要な複製をしない）。最後に出力ごとに `out-axes` の位置へ動かす。



<!-- フェーズ3 anchor: issue #128 -->

**broadcast_batcher（要素演算のバッチ化）**
: JAX の `broadcast_batcher` に倣った、要素演算のバッチ化ルール†の共通の実装。バッチ軸がすべて同じ位置のときは何も動かさず、違うときは先頭へ `transpose` で揃え、バッチされていない引数は `broadcast-in-dim` でバッチ軸を足してから、元のプリミティブを1つ適用する。



<!-- フェーズ3 anchor: issue #129 -->



<!-- フェーズ3 anchor: issue #130 -->

**cond* と select**
: `select`（`with-tracing` の `if`、`where`）は `:i1` の条件を要素ごとに使い、両方の枝を必ず計算してから選ぶ。計算量は両枝の和で、選ばれなかった枝の NaN / inf は捨てられるが評価はされる。`cond*` は rank 0 の `:i1` の条件で片方の枝だけを実行する高階プリミティブ（StableHLO の `stablehlo.if`）で、選ばれなかった枝は実行時に評価されない。代わりに条件はスカラーに限り、両枝の出力の aval が一致していなければならない。`if` を `cond*` に落とさず `select` のままにしているのは、`if` の条件が要素ごとの配列でありうるため。



<!-- フェーズ3 anchor: issue #131 -->
### while-loop（ホワイルループ）

反復回数がトレース時に決まらないループを表す高階プリミティブ（issue #131）。条件（cond）と本体（body）をサブグラフとして持ち、StableHLO では `stablehlo.while`（cond と body の2つのリージョン）になる。JAX の `lax.while_loop` に相当する。carry（ループで受け渡す値）の aval は本体の前後で一致しなければならない。逆モードの自動微分には対応しない（反復回数が分からないと、各反復の途中の値＝残差を保存できないため）。



<!-- フェーズ3 anchor: issue #132 -->

**scan（制御構造）**
: 配列の先頭の軸に沿って、状態（carry）を持ち回しながら関数を回す高階プリミティブ。JAX の `lax.scan` に相当し、RNN のように「前のステップの出力を次のステップの入力にする」計算を、Lisp のループを展開せずに1つの eqn で表す。eqn の params は JAX と同じく `num-consts`（ループ不変な入力の個数）、`num-carry`、`length`、`reverse`、本体のサブグラフ（入力は consts ++ carry ++ x_t、出力は carry ++ y_t）。StableHLO では `:i32` のカウンタを carry に足した `stablehlo.while` に落とし、x_t は `dynamic_slice`、y_t は `dynamic_update_slice` で読み書きする。



<!-- フェーズ3 anchor: issue #133 -->

**Threefry / rng_bit_generator**

Threefry は、鍵とカウンタから乱数のビット列を作るカウンタベースの乱数生成法（Salmon ら 2011。nabla は32ビット2語の Threefry-2x32、20ラウンド）。状態を持たず、同じ鍵とカウンタからは常に同じビットが出るので、JAX と同じ「明示的なキー渡し」の PRNG の土台になる。`stablehlo.rng_bit_generator`（`algorithm = THREE_FRY`）は、状態 `ui64[2]`（鍵とカウンタ）から新しい状態と乱数ビットを返す StableHLO の op で、nabla では `rng-bit-generator` プリミティブが対応する。




<!-- フェーズ3 anchor: issue #134 -->

### 不動点（fixpoint、while-loop の jvp）

`while-loop` の jvp で、「接線が非ゼロの carry の集合」を求める計算。最初は接線がゼロの carry も、本体を1回通ると他の carry の接線が流れ込んで非ゼロになりうる。そこで、本体を jvp 変換して出力の接線が非ゼロの carry を集合に足す、を集合が変わらなくなるまで繰り返す（集合は増える一方なので有限回で止まる）。JAX の `_while_loop_jvp` と同じ。実装は `src/ad/rules-control.lisp`。



<!-- フェーズ3 anchor: issue #135 -->

### carry の接線の不動点（fixed point）

`scan` を jvp（前向きモード微分）するとき、どの carry が非ゼロの接線を持つかは、ループの本体を通ると変わりうる。たとえば `g' = 0.9 g + h` の `g` は、初期の接線がゼロでも、`h` の接線が非ゼロなら次のステップの `g` の接線は非ゼロになる。そこで「非ゼロの接線を持つ carry の集合」を、本体を jvp してはその結果で集合を広げる、を集合が増えなくなるまで繰り返す。集合は増えるだけで carry の個数が上限なので必ず止まり、止まった集合（不動点）が、jvp した `scan` の carry の接線の組になる。JAX の `_scan_jvp` の `carry_nz` と同じ。→ `src/ad/rules-scan.lisp`



<!-- フェーズ3 anchor: issue #136 -->



<!-- フェーズ3 anchor: issue #137 -->



<!-- フェーズ3 anchor: issue #138 -->

**per-example 勾配（サンプルごとの勾配）**
: バッチ全体の損失の勾配（1つの値）ではなく、バッチの各サンプルについての損失の勾配を、サンプルごとに別々に求めたもの。`(vmap (grad loss) :in-axes ...)` と書く。`grad` は1サンプルの損失を微分し、`vmap` がそれをサンプルの束にまとめて適用する。パラメータはバッチしない（`in-axes` が `nil`）ので、結果の各勾配は `(N ...パラメータの形)` になる。平均すると、バッチ平均損失の `grad` に一致する。差分プライバシー付きの学習（サンプルごとの勾配のクリッピング）などに使う。



<!-- フェーズ3 anchor: issue #139 -->

### partial eval（部分評価）とループ不変な残差

jvp した graph を、主値だけで決まる部分（既知）と接線に依存する部分（未知）に分けること。`scan` の jvp は主値と接線を1つのループで計算するので、そのままでは「接線について線形」な部分だけを取り出して転置できない。そこで `scan` を、主値と各ステップの残差（接線の係数になる中間値。ステップごとに `ys` として積む）を計算する scan と、残差を `xs` で受けて接線だけを回す線形な scan に分ける（JAX の `_scan_partial_eval`）。残差のうち、ループの外の値（consts や本体の定数）だけで決まるもの（ループ不変な残差）は、ステップごとに積まず、`consts` としてそのまま渡す。線形な scan を `reverse` を反転した scan にしたものが、逆モード（BPTT）の本体になる。→ `src/ad/rules-scan-reverse.lisp`



<!-- フェーズ3 anchor: issue #140 -->



<!-- フェーズ3 anchor: issue #141 -->
