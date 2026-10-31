# 用語集

CLAUDE.md や設計書に出てくる専門用語のうち、大学の学部やジュニアエンジニアの段階では習わないことが多いものを説明する。まず一言で意味を書き、必要ならこのプロジェクトでの使われ方を補足する。

## コンパイラと実行系

**IR（中間表現, Intermediate Representation）**
: ソースコードと機械語の間に置く、プログラムの別の書き方。変換や最適化をしやすい形にしてある。nabla では JAX の jaxpr にならった小さな IR（`aval` / `var` / `eqn` / `graph`）を持つ。

**入力の連結（input packing、jit）**
: jit した関数の入力（引数の木の葉）が実行系の入力の上限（`backend-max-inputs`）を超えるときだけ、同じ dtype の葉をホストで 1 本の 1 次元の配列につなげて渡し、コンパイルする関数の先頭で `slice` と `reshape` を使って元の葉に切り分けること（issue #242）。IREE 3.11 の local-task が入力 162 個以上の関数で落ちるのを避ける。入力の個数は dtype の種類の数になる。上限の内側では何もしないので、StableHLO もディスクキャッシュも変わらない。

**aval（抽象値, abstract value）**
: 配列の「中身」を持たず、「形状と dtype」だけを持つ値。トレース中は実際の数値が分からないので、aval だけを追って出力の形を決める。

**プリミティブ（primitive）**
: `add` や `reduce-sum` のような、nabla が知っている最小単位の演算。`defprimitive` で宣言し、形状推論（abstract-eval）・StableHLO 出力（emit）・eager 用の CPU 実装（eager）を束ねる。`jvp` / transpose ルール / バッチ化ルールは、その演算を `grad` / `vmap` に対応させるときに、同じプリミティブに追加する。

**contracting dims / batch dims（縮約次元・バッチ次元）**
: StableHLO の `dot_general`（nabla の `dot-general` プリミティブ）が、どの次元をどう扱うかを指定する2種類の次元。contracting dims（縮約次元）は、行列積の「掛けて足し込む」次元（`lhs-contracting` / `rhs-contracting`）で、両側のサイズが一致していなければならない。batch dims（バッチ次元）は、縮約せず両オペランドに共通して残る次元（`lhs-batch` / `rhs-batch`）で、その次元ごとに独立した縮約を行う（バッチ行列積）。どちらにも属さない次元は自由次元（free dims）と呼び、出力にはバッチ次元・lhs の自由次元・rhs の自由次元の順で並ぶ。

**reduction / 初期値（init value）**
: 配列のいくつかの次元を、要素同士を1つずつ組み合わせる二項演算（総和・最大値など）で潰して次元を減らすこと。StableHLO の `stablehlo.reduce`（nabla の `reduce-sum` / `reduce-max` プリミティブ）は、潰す次元ごとに空でないスライスを1つの値にまとめる。init value（初期値）は、その組み合わせを始める前に置いておく値で、演算の単位元（総和なら 0、最大値なら -∞）を使う。潰す次元のサイズが0（スライスが空）のときは、この初期値がそのまま出力になる。

**gather / scatter-add（ギャザー・スキャッター加算）**
: gather は配列の行を添字の配列で引き集める演算（nabla の `gather` は先頭の軸を引く形だけで、JAX の `jnp.take(x, i, axis=0)` に当たる）。scatter-add はその逆向きで、値を添字の行へ足し込む（同じ添字が何度も出れば足し合わさる）。gather の転置（勾配）が scatter-add になる。embedding の引きと、その勾配に使う。

**FILL_OR_DROP（範囲外の添字の扱い）**
: JAX の gather / scatter の範囲外の添字の扱いの1つ（`jnp.take` の既定）。gather では範囲外の添字の結果を決まった値で埋め（fill）、scatter ではその更新を捨てる（drop）。埋めた結果の勾配が捨てた更新と対応するので、転置として整合する。nabla の `gather` はこの構造で、埋める値は JAX の NaN でなく 0。

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

**channels-last（NHWC / HWIO）**
: 畳み込み（`convolution` プリミティブ、`conv-general-dilated`）の配列の並べ方。入力と出力は `[N, 空間..., C]`（バッチ・空間の各軸・チャネル。2次元なら NHWC）、カーネルは `[空間..., C/G, O]`（2次元なら HWIO）。Flax と同じで、nabla はこの配置に固定している（StableHLO の `dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]`）。G は `feature_group_count`（入力チャネルを G 個のグループに分け、グループごとに畳み込む。G = C なら depthwise）。

**SAME / VALID padding**
: 畳み込みの padding の指定。VALID は padding なし（出力が小さくなる）。SAME は stride 1 なら出力の空間の大きさを入力と同じに、stride があれば `ceil(入力 / stride)` にする。必要な padding の合計は `max((出力 - 1) stride + 実効 kernel - 入力, 0)`（実効 kernel は dilation を含めた `(k - 1) dilation + 1`）で、合計が奇数のときは右（後ろ）に1多く入れる（左が合計の半分の切り捨て）。Flax と JAX の `padtype_to_pads` と同じ。nabla の `conv` は `call` のたびに入力の形から明示の `(low high)` に直して `conv-general-dilated` に渡す。

**batch group（batch_group_count）**
: 畳み込みの `batch_group_count`（B）。`feature_group_count`（入力チャネルを G 個に分ける）の対になるもので、lhs のバッチ軸を連続した B 個の塊に分け、出力チャネルの B 個の塊がそれぞれ1つのバッチの塊だけを読む（出力のバッチは N/B）。ユーザーが書く畳み込みには出てこず、グループ畳み込み・depthwise（G > 1）の rhs の転置（`grad` の rhs 側）が B = G の畳み込みになる（JAX の `_conv_general_dilated_transpose_rhs`）。nabla では `convolution` プリミティブの内部のパラメタ（issue #203）。

**dispatch / binding（IREE）**
: IREE がプログラムを分けた、デバイスで 1 回に起動する計算の単位が dispatch（GPU のカーネル 1 つに当たる）。IREE のコンパイラは要素ごとの演算などを融合（fusion）して 1 つの dispatch にまとめる。binding は 1 つの dispatch が読み書きするバッファの 1 つ 1 つで、ふつうは「その dispatch に入る別々のテンソル（関数の入力や、他の dispatch の結果）+ 結果」の数になる。IREE 3.11 のランタイムは 1 つの dispatch の binding を 32 個までしか受け付けない（`docs/iree-upstream-bugs.md` の 10、issue #264）。

**fsync / F_FULLFSYNC**
: `write` で書いたデータは、まず OS のメモリ（ページキャッシュ）に溜まり、ディスクへは後でまとめて書かれる。プロセスが落ちても OS が書いてくれるが、OS のクラッシュや電源断ではその分が失われる。`fsync(2)` は「このファイルの中身をディスクに届けるまで待つ」システムコール。一時ファイルに書いて rename で置き換えるときは、rename の前にファイルを、後に親ディレクトリ（rename そのものの記録）を fsync しないと、電源断の後に空か壊れたファイルが残りうる（nabla の `with-atomic-output-file`、issue #263）。macOS の `fsync` はデータをドライブに渡すだけで、ドライブ内の書き込みキャッシュまでは書き出さない。そこまで待つのが `fcntl(F_FULLFSYNC)` で、ずっと遅い（nabla では `*full-fsync*` で選ぶ）。

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

**PyTree / 葉（leaf） / treedef**
: リストや構造体を入れ子にした木構造で、葉に配列を持つもの。モデルのパラメータ全体を1つの値として `grad` や `jit` に渡すために使う。`tree-flatten` で葉の列（左から順）と木の形に平たくし、`tree-unflatten` で元の形に戻す。nabla で既定の節（子を持つもの）はリストと `simple-vector`、それに `defpytree` で登録した型だけで、`nil` は空の部分木。それ以外（配列・数値・トレーサなど）はすべて葉になる。treedef は葉を取り除いた「木の形」で、節の種類・子の個数・登録した型の補助データ（層の幅のような静的な値）を持ち、`treedef=` で比べる。`jit` のキャッシュのキーには、引数の葉の `aval` に加えてこの treedef が入る（葉の `aval` が同じでも木の形が違えば別のコンパイルになる）。`grad` の結果は、微分した引数と同じ treedef の木になる。`vmap` の `in-axes` / `out-axes` は、引数・戻り値の木の**接頭辞**（部分木まるごとに整数か `nil` を1つ書けるもの。JAX の `flatten_axes`）で、葉ごとの軸に展開されてから使われる。

**モジュール（module）**
: nabla では、パラメータを持つ構造体（`defmodule` で定義）。パラメータ・サブモジュール（`:param`）、学習しない状態（`:state`、BatchNorm の統計など）、ハイパーパラメータ（`:static`、treedef の補助データ）を別のスロットに持ち、PyTree に自動登録される。Linen / Equinox と同じく「パラメータの PyTree ＋ 純粋関数」で、`(call module x)` で適用し、`init-<name>` が PRNG キーからパラメータを作る。

**lecun_normal / 切断正規分布（truncated normal）**
: Flax の `nn.Dense` の kernel の既定の初期化。分散 1/fan_in（fan_in は入力の幅）の正規分布のうち、平均から標準偏差の2倍より外を取り除いた（[-2, 2] で切った）もの。切ると標準偏差が 0.8796 倍に縮むので、1/0.8796 を前もってかけて、結果の標準偏差を sqrt(1/fan_in) に合わせる。nabla では `nabla:truncated-normal`（CDF の区間の一様乱数に erf の逆関数をかける、JAX と同じ方式）に `init-dense` が標準偏差をかけて作る。

**GroupNorm（グループ正規化）**
: チャンネル（最後の軸）を同じ大きさの群に分け、サンプルごと・群ごとに平均 0・分散 1 に正規化する層（Wu と He、2018）。BatchNorm と違いバッチの統計を使わないので、バッチが小さくても学習時と推論時で同じ計算になり、状態（移動平均）も持たない。群が 1 つなら（バッチ軸以外の全部での）LayerNorm、群の大きさが 1 なら InstanceNorm と同じ。nabla の `group-norm`（issue #260）は Flax の `nn.GroupNorm` に合わせ、分散は LayerNorm と同じ「速い分散」

**LayerNorm の「速い分散」**
: 分散を `E[x^2] - E[x]^2` で求める方式（Flax の `use_fast_variance` の既定）。データを1回なぞれば済むが、平均が標準偏差よりずっと大きいと桁落ちする（f32 で平均 3・標準偏差 2 の入力なら 1 桁ほど失う）。nabla の `layer-norm` は Flax と数値で一致させるため同じ式を使う。。平均が標準偏差よりずっと大きい行では分散が 0 に潰れ、出力が `(x - mean) / sqrt(epsilon)` に発散する（たとえば f32 の行 `[1e4, 1e4+1, 1e4+2, 1e4+3]` は E[x^2] と E[x]^2 が同じ値に丸まって分散が 0 になり、出力は ±1500 程度になる。Flax も同じ結果）

**dense-general**
: 「入力の最後の複数の軸をまとめて縮約し、複数の軸の特徴に置き換える」`dense`（Flax の `nn.DenseGeneral`）。縮約する軸の数は `contract-axes`。multi-head attention の q / k / v は `(in) → (heads head-dim)`、出力の射影は `(heads head-dim) → (out)`。

**multi-head attention**
: Transformer の中心の層。入力から query・key・value を作り（それぞれ「ヘッド」ごとの小さな幅に分ける）、ヘッドごとに `softmax(q k^T / sqrt(幅)) v` を計算して、ヘッドをつなぎ直して出力に射影する。q / k / v と出力の射影は `dense-general`。

**attention のマスク**
: 見てはいけない位置のスコアを softmax の前に最小の有限値（`-inf` ではなく Flax の `jnp.finfo(dtype).min`）に置き換えること。因果マスクは「位置 i は i より後を見ない」下三角のマスク（言語モデルの学習で未来を見ないため）、padding マスクは長さをそろえるために足した詰め物の位置を隠す。全部が隠れた行は全部同じ値になるので、重みは一様になる（NaN にならない）。

**反転 dropout（inverted dropout）**
: 学習時に要素をランダムに落とし、残した要素を 1/(1-rate) 倍して、出力の期待値を入力と同じに保つ方式。推論時は何もしなくてよい（`deterministic` が真なら入力そのもの）。Flax の `nn.Dropout` も同じ。
**オプティマイザ（optimizer）/ 勾配変換（gradient transformation）**
: 勾配から更新量を作る `init`（パラメータ → 状態）と `update`（勾配・状態・パラメータ → 更新量・新しい状態）の組。Optax の `GradientTransformation` と同じで、`sgd` や `adam` だけでなく、勾配のクリップや学習率の掛け算のような小さな変換も同じ型なので、`chain` で順に合成できる（nabla では `nabla.nn:chain`、`defoptimizer`）。状態はパラメータと同じ構造の木（`mu`・`nu` など）とステップ数 `count` を持つ PyTree で、`update` は純粋関数なので、学習ステップ全体を `jit` できる。

**データソース（data source）**
: インデックスでレコードを取れるデータの抽象。長さ（レコードの個数）と「i 番目のレコード」だけを持ち、同じ i には常に同じレコードを返す。Grain の `RandomAccessDataSource` に当たり、サンプラーが決めた添字の列をローダがここで引くので、添字の列さえ再現できればデータの供給を途中から再開できる。nabla では `nabla.data:data-source`（`source-length` / `source-item`）で、メモリ上の配列の木から作る `array-source` がある（issue #279）。

**サンプラー（sampler）/ エポック（epoch）**
: レコードを読む順序を決める部品。エポックはデータセットを 1 周すること。nabla の `index-sampler` は、通し番号の位置 k（0, 1, 2, ...）から「エポック floor(k / n) と、そのエポックで読むレコードの添字」を計算する純粋関数で、シャッフルするときはエポックごとに違う並べ替えになる。順序を表として持たず位置から直接計算するので、途中から再開するには位置だけを覚えておけばよい（Grain の `IndexSampler`、issue #280）。

**SplitMix64**
: 64 ビットの状態に定数を足し、掛け算とビットシフトの xor で攪拌して 64 ビットの数を出す、小さく速い疑似乱数の算法（Java の `SplittableRandom` が使う）。暗号には使えないが、統計的な質は十分で、1 回が数 ns。nabla の `nabla/data` は、シャッフルの鍵と `random-map-transform` のキーを作るのに使う（nabla の PRNG は eager で呼ぶたびに graph を組むので、レコードごとに呼ぶには遅い）。

**Feistel 置換（Feistel network）/ 偶置換**
: 数を上位と下位の半分に分け、「片方に鍵付きの関数をかけた値をもう片方に xor して入れ替える」ラウンドを何回か繰り返す構成。どんな関数を使っても 1 対 1 の写像（置換）になるので、鍵を変えるだけで 0..2^m − 1 の並べ替えを表なしで作れる（暗号の DES が使う）。上位と下位が同じ幅の「均衡」な形は、偶置換（互換を偶数回かけた置換）しか作らないので、`index-sampler` は鍵で決まる互換を最後に 1 回かけて、偶置換と奇置換が半々に出るようにしている（issue #280）。

**cycle-walking**
: 2 のべきの大きさの領域の置換を、それより小さい範囲 [0, n) の置換に縮める方法。置換した値が n 以上なら、範囲に入るまで同じ置換をかけ続ける。置換をたどると必ず元に戻るので、いつか範囲に入り、結果も [0, n) の置換になる。領域を n の 4 倍未満に取れば、平均の繰り返しは 4 回未満で済む。

**変換（transform）: map / filter / random-map**
: ローダがレコードにかける処理。map はレコードを別のレコードに変え、filter は条件に合わないレコードを落とし、random-map はレコードと乱数のキーから新しいレコードを作る（データ拡張用）。どれもワーカーのスレッドから同時に呼ばれるので純粋関数にする。random-map のキーはローダのシードとサンプラーの位置から決まるので、並列に読んでも途中から再開しても同じ拡張になる（Grain の `MapTransform` / `FilterTransform` / `RandomMapTransform`、issue #281、#284）。レコードをバッチにまとめるのは変換ではなくローダの仕事で、エポックの終わりに足りない端数は既定で捨てる（drop-remainder）。

**再開状態（resume state）/ 入力側の位置**
: 学習を中断して、後で同じ続きから再開するために保存する、データローダの最小限の情報。nabla の `iterator-state` は「次に調べるサンプラーの位置」（入力側の位置）をエポックとエポック内の位置に分けたものと、シード・算法の版を持つ。出力したバッチの番号ではなく入力の位置で持つので、filter がレコードを落としても、ワーカーが先読みしていても、再開した続きの列が中断しなかった場合と一致する。モデルとオプティマイザ状態と同じチェックポイントの木に入れて保存する（issue #285）。

**.npy**
: numpy が配列を 1 つ保存するファイル形式。短いヘッダ（要素型・C 順か Fortran 順か・形を書いた Python の dict の文字列）の後に、要素のバイト列がそのまま続く。nabla は `nabla.data:read-npy` / `write-npy` で読み書きし、`npy-source` でデータソースにする（issue #282）。

**先読み（prefetch）/ future**
: 先読みは、呼び出し側が今のバッチを使っている（デバイスへ送り、学習ステップを走らせる）間に、次のバッチを別のスレッドで用意しておくこと。読み込みと計算が重なるので、1 ステップの時間が「読み込み + 計算」から「長いほう」に近づく。future は「後で値が入る箱」で、作った時点で計算をワーカーのスレッドに投げ、`force` で結果を待って受け取る（lparallel の `future` / `force`）。nabla の並列ローダは位置の区間ごとに future を作り、作った順に `force` するので、ワーカーが終わった順に依らずバッチの順序が決まる（`make-data-iterator` の `:num-workers` / `:prefetch`、issue #286）。

**pre-LN / decoder-only**
: Transformer のブロックの書き方。pre-LN は layer-norm を attention や MLP の「前」に置き、`x = x + attn(ln(x))` のように残差の足し算の外側に出さない形（学習が安定しやすい）。decoder-only は因果マスクをかけた自己 attention だけを積んだ形で、位置 i が i より後を見ないので、次のトークンを当てる言語モデルに使う（`examples/transformer.lisp`）。

**Adam / AdamW / バイアス補正**
: 勾配の指数移動平均 `mu`（1次モーメント）と、勾配の二乗の指数移動平均 `nu`（2次モーメント）で更新量 `mu / (√nu + ε)` を作る最適化手法。`mu`・`nu` は0から始めるので最初の数ステップは小さく偏る。その分を `1 - b^t`（`b` は減衰率、`t` はステップ数）で割って直すのがバイアス補正。AdamW は重み減衰（weight decay）を勾配に混ぜず、Adam の更新量に `wd × パラメータ` として足す（decoupled weight decay）。

**batch normalization（BatchNorm）/ 移動平均（running average）**
: 各特徴を、ミニバッチ（とバッチ以外の空間の軸）の平均と分散で正規化してから、学習する scale と bias をかける正規化。推論ではミニバッチの統計の代わりに、学習中に `ra ← momentum × ra + (1 − momentum) × バッチの統計` で更新し続けた指数移動平均（running average。Flax の `batch_stats`）を使う。移動平均は学習しない状態なので、nabla では `defmodule` の `:state` に持ち、`call` が更新後のモジュールを返す（`nabla.nn:batch-norm`、issue #260）。

**has_aux（補助出力）**
: `grad` の対象の関数が、損失と一緒に返す「微分しない値」。JAX の `grad(f, has_aux=True)`。BatchNorm の更新後の状態や、ログに出す精度のような値を、順伝播を2回計算せずに取り出すのに使う（nabla の `grad` / `value-and-grad` の `:has-aux`）。

**学習率スケジュール（learning rate schedule）**
: ステップ数 `count` から学習率を返す関数。一定（constant）、直線で動かす（linear。warmup は 0 から目標値へ上げるこの形）、半周期の cosine で下げる（cosine decay）など。nabla ではステップ数を状態の中の rank 0 の配列に持つので、`jit` したままステップごとに値が変わる。

**大域ノルムによるクリップ（clip by global norm）**
: 全パラメータの勾配をひとつのベクトルとみなしたときのノルム（二乗和の平方根）が上限を超えたら、向きを変えずに全体を上限まで縮める。勾配爆発を抑える。

**safetensors**
: 名前を付けたテンソルの束を保存するファイル形式（Hugging Face）。先頭8バイトにJSONヘッダの長さ、次にヘッダ（名前ごとの dtype・形・データの範囲 `data_offsets`）、残りが生のバイト列。コードを実行しない純粋なデータ形式なので、pickle より安全に重みを配れる。nabla は `nabla/nn` で読み書きし、モデルのチェックポイントに使う（フェーズ4）。

**チェックポイント（checkpoint）**
: 学習の途中経過（モデルのパラメータとオプティマイザ状態）をファイルに保存したもの。nabla は PyTree の葉を `"encoder.layers.0.w"` のようなパス名（リストとベクタは添字、`defmodule` はスロット名）で平坦化して safetensors に書き、復元は「同じ構造のひな形」（init で作ったモデルなど）の葉に名前で対応づけて unflatten する。ひな形と名前・形・dtype が合わなければエラーにする。

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

## 数値の安定性

**absorption（情報落ち）**
: 大きさの違う数を足したとき、小さい方の桁が丸めで消えること。f32 では 1 に 1e-8 を足しても 1 のまま。softmax の正規化項 `1 + s`（s は小さな和）がこれで 1 になるのが Softmax Collapse の原因（`docs/numerics.md`）。

**catastrophic cancellation（桁落ち）**
: ほぼ等しい2つの数を引いたとき、上の桁が打ち消し合い、丸め誤差の残った下の桁だけが結果になること。`p_t − 1`（p_t ≈ 1）や、近い確率どうしの差 `q − p` で起きる。差を作らずに小さな量を直接計算する式（`log1p`、`expm1`、最大値を除いた和）で避ける。

**精度の崖（precision cliff）**
: 1 に足した値が丸めで消え始める大きさの境目を、softmax の余裕（最大のロジットと他の差）で表したもの。ln(1/eps) で、bf16 で約 5.5、f32 で約 16.6、f64 で約 36.7。素朴な softmax はここを越えると値と最大値の位置の勾配が 0 になる。

**Softmax Collapse**
: 1 つのロジットが他より精度の崖以上大きいとき、素朴な log-softmax / 交差エントロピーが、その位置の値と勾配 `p − 1` を 0 に丸め、他の位置の勾配は残すので、勾配の和が 0 でなくなる不具合（Barron ら、2026）。nabla の `log-softmax` / `softmax` は起こさない（`docs/numerics.md`）。

**Softmax Cancellation**
: 蒸留の交差エントロピーの勾配 `q − p` を、別々に丸めた確率の差として計算するために、生徒が教師に近いと桁を失い、等しくても 0 にならない不具合（Barron ら、2026）。nabla の `soft-target-cross-entropy` は起こさない。

**double where（where の両側の微分）**
: `where(c, f(x), y)` で、選ばれない側の f(x) の微分が inf になり（log(0)、1/0 など）、余接線 0 との積が NaN になって勾配を壊すこと。f の入力の前で安全な値に select して避ける。

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

**unit roundoff（u）/ ULP（unit in the last place）**
: ULP は、ある値の付近で隣り合う2つの浮動小数点数の間隔。unit roundoff u は、最も近い値に丸めたときの相対誤差の上限で、仮数部が p ビットなら 2^-p（bf16 は 2^-8、f32 は 2^-24）。総和の丸め誤差は出力ではなく途中の部分和の大きさで決まるので、バックエンド間で総和や内積を比べるときは `accumulation-atol`（2·n·u·Σ|項|）を atol に使う（issue #12）。

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




**サブグラフ（subgraph）と高階プリミティブ**
: eqn の `params` の値として持たせる、閉じた `graph`（外側の var を参照せず、定数は自分の定数表に持つ）。`cond` / `while-loop` / `scan` のように、本体の関数を別の graph として持つプリミティブを高階プリミティブと呼ぶ。印字では入れ子に、StableHLO ではリージョンとして出す。`eval-graph` / `inline-graph` / `dce-graph` と jvp / transpose の変換は、サブグラフの中身を書き換えずに素通しする（中身の変換は各プリミティブのルールの仕事）。

**closure conversion（閉包変換）**
: 制御構造の本体が外側のトレーサを閉包で捕まえたとき、その値をサブグラフの追加の入力に持ち上げ、呼び出し側の eqn の入力の末尾に足す変換（`%trace-subgraph`）。同じトレーサは1回だけ持ち上げられ、持ち上げた順に並ぶ。`grad` の `%call-with-fresh-trace` は親を持たないので、外側のトレーサを捕まえると従来どおり `tracing-error` になる。

**複数出力のプリミティブ**
: `defprimitive` に `:multiple-outputs t` を付けたプリミティブ。`abstract-eval` が aval のリスト、`eager` が配列のリストを返し、`emit` は出力の名前と aval をリストで受けて `%8, %9 = ...` の左辺を自分で書く。トレースには、常にトレーサのリストを返す `%trace-eqn*` を使う。




**2 の補数での折り返し（整数のオーバーフロー）**
: 整数の演算結果がその dtype の範囲を超えたとき、上位のビットを捨てて範囲内に収める挙動（`:i32` は 2 の補数、`:u32` / `:u64` は法 2^n）。StableHLO の整数演算がこうなので、eager 実装も同じにそろえる（`(+ 2147483647 1)` の `:i32` は -2147483648）。issue #126。




**バッチ次元（vmap）**
: `vmap` が、例の束を並べるために配列へ足した軸。`dot_general` の batch dims（縮約せずに両オペランドに共通して残る次元。上の「contracting dims / batch dims」）とは別の概念で、同じ名前で呼ぶので注意する。`vmap` は値ごとに「バッチ次元がどの位置にあるか（無ければ `nil`）」を伝播させながら graph を書き換える。バッチ次元を持たない値だけを入力とする演算は、バッチ化ルールを呼ばずにそのまま残す（不要な複製をしない）。最後に出力ごとに `out-axes` の位置へ動かす。`in-axes` / `out-axes` は PyTree の接頭辞として書ける（パラメータの木全体に `nil`、データの木全体に `0` など。「PyTree / 葉 / treedef」を見る）。




**broadcast_batcher（要素演算のバッチ化）**
: JAX の `broadcast_batcher` に倣った、要素演算のバッチ化ルール†の共通の実装。バッチ軸がすべて同じ位置のときは何も動かさず、違うときは先頭へ `transpose` で揃え、バッチされていない引数は `broadcast-in-dim` でバッチ軸を足してから、元のプリミティブを1つ適用する。







**cond* と select**
: `select`（`with-tracing` の `if`、`where`）は `:i1` の条件を要素ごとに使い、両方の枝を必ず計算してから選ぶ。計算量は両枝の和で、選ばれなかった枝の NaN / inf は捨てられるが評価はされる。`cond*` は rank 0 の `:i1` の条件で片方の枝だけを実行する高階プリミティブ（StableHLO の `stablehlo.if`）で、選ばれなかった枝は実行時に評価されない。代わりに条件はスカラーに限り、両枝の出力の aval が一致していなければならない。`if` を `cond*` に落とさず `select` のままにしているのは、`if` の条件が要素ごとの配列でありうるため。



### while-loop（ホワイルループ）

反復回数がトレース時に決まらないループを表す高階プリミティブ（issue #131）。条件（cond）と本体（body）をサブグラフとして持ち、StableHLO では `stablehlo.while`（cond と body の2つのリージョン）になる。JAX の `lax.while_loop` に相当する。carry（ループで受け渡す値）の aval は本体の前後で一致しなければならない。逆モードの自動微分には対応しない（反復回数が分からないと、各反復の途中の値＝残差を保存できないため）。




**scan（制御構造）**
: 配列の先頭の軸に沿って、状態（carry）を持ち回しながら関数を回す高階プリミティブ。JAX の `lax.scan` に相当し、RNN のように「前のステップの出力を次のステップの入力にする」計算を、Lisp のループを展開せずに1つの eqn で表す。eqn の params は JAX と同じく `num-consts`（ループ不変な入力の個数）、`num-carry`、`length`、`reverse`、本体のサブグラフ（入力は consts ++ carry ++ x_t、出力は carry ++ y_t）。StableHLO では `:i32` のカウンタを carry に足した `stablehlo.while` に落とし、x_t は `dynamic_slice`、y_t は `dynamic_update_slice` で読み書きする。



**carry（キャリー）**
: `while-loop` や `scan` で、反復のたびに次の反復へ持ち回す値。`scan` では本体が `(carry x_t) → (新しい carry, y_t)` を計算し、最後の carry と、各ステップの `y_t` を積んだ `ys` が結果になる。nabla では carry・`xs`・`ys` は PyTree（リスト・ベクタ・`defmodule` の型の入れ子。配列そのものでもよい）で受け渡す。carry の aval は反復の前後で一致しなければならない。

**optimization_barrier（最適化の境界）**
: `stablehlo.optimization_barrier`。値を変えずにそのまま通す op で、コンパイラがこの前後で式を畳んだり移動したりしない境界になる。nabla は `stop-gradient` の出力のほか、IREE 3.11 のコンパイラのバグの回避（定数で初期化された carry を持つ `while` の定数オペランド、整数 → bf16 の `convert` の2段の間）に使う。


**Threefry / rng_bit_generator**

Threefry は、鍵とカウンタから乱数のビット列を作るカウンタベースの乱数生成法（Salmon ら 2011。nabla は32ビット2語の Threefry-2x32、20ラウンド）。状態を持たず、同じ鍵とカウンタからは常に同じビットが出るので、JAX と同じ「明示的なキー渡し」の PRNG の土台になる。`stablehlo.rng_bit_generator`（`algorithm = THREE_FRY`）は、状態 `ui64[2]`（鍵とカウンタ）から新しい状態と乱数ビットを返す StableHLO の op で、nabla では `rng-bit-generator` プリミティブが対応する。





### 不動点（fixpoint、while-loop の jvp）

`while-loop` の jvp で、「接線が非ゼロの carry の集合」を求める計算。最初は接線がゼロの carry も、本体を1回通ると他の carry の接線が流れ込んで非ゼロになりうる。そこで、本体を jvp 変換して出力の接線が非ゼロの carry を集合に足す、を集合が変わらなくなるまで繰り返す（集合は増える一方なので有限回で止まる）。JAX の `_while_loop_jvp` と同じ。実装は `src/ad/rules-control.lisp`。




### carry の接線の不動点（fixed point）

`scan` を jvp（前向きモード微分）するとき、どの carry が非ゼロの接線を持つかは、ループの本体を通ると変わりうる。たとえば `g' = 0.9 g + h` の `g` は、初期の接線がゼロでも、`h` の接線が非ゼロなら次のステップの `g` の接線は非ゼロになる。そこで「非ゼロの接線を持つ carry の集合」を、本体を jvp してはその結果で集合を広げる、を集合が増えなくなるまで繰り返す。集合は増えるだけで carry の個数が上限なので必ず止まり、止まった集合（不動点）が、jvp した `scan` の carry の接線の組になる。JAX の `_scan_jvp` の `carry_nz` と同じ。→ `src/ad/rules-scan.lisp`




**PRNG のキー / split / fold-in**

nabla の PRNG は JAX と同じ「明示的なキー渡し」で、グローバルな乱数状態を持たない。キーは `:u32` の2要素配列で、`prng-key` が整数のシードから作る。`split` は1つのキーから独立な複数のキーを作り、`fold-in` はキーに整数（ループの反復番号など）を混ぜて別のキーを作る。同じキーからは常に同じ乱数が出るので、キーを使い回さずに、使うたびに `split` で分けるのが作法。内部では、キーの2語を `ui64[2]` の状態（鍵 + カウンタ）にして `rng_bit_generator` に渡す。

**bitcast-convert / 仮数部のトリック**

`bitcast-convert` は、値を変えずにビット列だけを別の型として読み替える演算（`stablehlo.bitcast_convert`。`convert` が値を変換するのと違う）。乱数ビットから浮動小数点数を作る「仮数部のトリック」は、32ビットの乱数の上位 23 ビットを仮数部に置き、指数部を 1.0 のものにして `[1, 2)` の f32 として読み替え、1 を引いて `[0, 1)` の一様乱数にする。

**erf の逆関数（正規乱数の作り方）**

正規分布の逆累積分布関数は `√2 · erfinv(2p - 1)`。nabla の `normal` は `(-1, 1)` の一様乱数 u に `erfinv(u) · √2` を計算する（JAX と同じ）。`erfinv` は Giles の多項式近似（`w = -log(1 - u²)` の大きさで多項式を切り替える。単精度用は2つ、`:f64` で使う倍精度用は3つ）で、`exp` / `log` / 四則と比較・選択だけで書ける。




**do ループの scan への展開**（issue #137）: `with-tracing` の中の定型の `do`（カウンタ・carry・不変な変数、終了条件 `(>= i n)` / `(= i n)`）を、反復回数だけ展開せずに1つの `scan`† にする変換。`do` はマクロ展開されると `block` / `tagbody` / `setq` になって形が分からなくなるので、`macroexpand-all` の前の、展開前のフォームに対して行う。`do` の step 式は純粋で、`setq` なしで carry を更新できるので scan の本体にそのまま写せる。定型でない `do` は原因のフォームを示す `unsupported-form`、`dotimes` / `loop` / `do*` は従来どおり展開後の `block` で `unsupported-form`。




**per-example 勾配（サンプルごとの勾配）**
: バッチ全体の損失の勾配（1つの値）ではなく、バッチの各サンプルについての損失の勾配を、サンプルごとに別々に求めたもの。`(vmap (grad loss) :in-axes ...)` と書く。`grad` は1サンプルの損失を微分し、`vmap` がそれをサンプルの束にまとめて適用する。パラメータはバッチしない（`in-axes` が `nil`）ので、結果の各勾配は `(N ...パラメータの形)` になる。平均すると、バッチ平均損失の `grad` に一致する。差分プライバシー付きの学習（サンプルごとの勾配のクリッピング）などに使う。




### partial eval（部分評価）とループ不変な残差

jvp した graph を、主値だけで決まる部分（既知）と接線に依存する部分（未知）に分けること。`scan` の jvp は主値と接線を1つのループで計算するので、そのままでは「接線について線形」な部分だけを取り出して転置できない。そこで `scan` を、主値と各ステップの残差（接線の係数になる中間値。ステップごとに `ys` として積む）を計算する scan と、残差を `xs` で受けて接線だけを回す線形な scan に分ける（JAX の `_scan_partial_eval`）。残差のうち、ループの外の値（consts や本体の定数）だけで決まるもの（ループ不変な残差）は、ステップごとに積まず、`consts` としてそのまま渡す。線形な scan を `reverse` を反転した scan にしたものが、逆モード（BPTT）の本体になる。→ `src/ad/rules-scan-reverse.lisp`




**バッチされる carry の不動点（制御構造の vmap）**
: `while-loop`（と `scan`）の本体を `vmap` するとき、最初はバッチされない carry が、本体でバッチされた値と混ざってバッチされることがある。すると次の反復の入力が変わるので、本体を何度かバッチ化し直して「バッチされる carry の集合」が変わらなくなるまで広げる。この変わらなくなった状態が不動点。jvp で「接線が非ゼロの carry の集合」を不動点まで広げるのと同じ形。



