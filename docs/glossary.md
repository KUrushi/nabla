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
: `jit` が持つ、プロセス内・メモリ上だけのキャッシュ（`src/jit.lisp`）。キーは「関数の同一性（EQ）・aval・静的引数・実行系（ターゲット）」で、同じキーの2回目の呼び出しはトレース・コンパイルをせずコンパイル済みの module をそのまま使う。プロセスをまたいで効く vmfb のディスクキャッシュ（`src/compile-cache.lisp`、issue #10。`BACKEND-COMPILE` の `:AROUND` メソッドとして実装され、実行系がコンパイルした結果そのものをファイルに残す）とは別の層で、両方が独立に効く（jit キャッシュがヒットすればディスクキャッシュまで届かないし、jit キャッシュがミスしてもディスクキャッシュがヒットすれば実際のコンパイラは呼ばれない）。関数を再定義する（`WITH-TRACING` を再評価する、`defjit` を再評価する）と、新しい `TRACEABLE-FUNCTION` オブジェクトになるため、古いキャッシュは（EQ で一致しないので）使われない。

**defjit**
: `(defjit name (&rest lambda-list) &body body)`。`body` を `with-tracing` でトレース対象にしてから `jit` した通常の関数を `name` に定義するマクロ（`src/jit.lisp`、issue #34）。CL の `defun` と同じ感覚で「関数を定義したら、その名前で呼べる」ようにする糖衣で、内部では毎回新しい `traceable-function` を作って `(setf (fdefinition name) ...)` する。再評価すると古いキャッシュエントリを捨てるので、関数を再定義したら次の呼び出しは必ず再コンパイルする。

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
: XLA（JAX の標準の実行系）を外部から呼ぶための C API。nabla では IREE の次の候補として、`backend` プロトコルの裏に置く。

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
: CPU が0除算・オーバーフロー・不正な演算（0/0 や sqrt(-1) など）を検出したときに、実行を止めてコンディションを signal する仕組み。SBCL は既定で `:overflow` `:invalid` `:divide-by-zero` の3つのトラップを有効にしているため、`(/ 1.0 0.0)` のような計算はそのままだと `division-by-zero` を signal してしまう。StableHLO / IREE は IEEE 754 どおり無限大・NaN を返す（signal しない）ので、nabla のプリミティブの eager 実装は `sb-int:with-float-traps-masked` でこの3つのトラップをマスクしてから計算し、両者の挙動を揃える（`src/primitives/common.lisp` の `with-ieee-arithmetic`）。マスクは要素ごとではなく、eager 呼び出し全体を1回だけ包む（速度のため）。`nabla/iree` では、これに加えて MXCSR（SSE の浮動小数点制御・ステータスレジスタ）の性質に注意が要る：Linux はスレッド生成（`clone(2)`）時に生成元スレッドの MXCSR をそのままコピーするため、IREE のワーカースレッドや LLVM コード生成を生成・実行する瞬間に呼び出し元スレッドがマスクされていないと、生成された側は未マスクのまま動き続ける（issue #53）。`nabla.iree::with-all-float-traps-masked`（`src/iree/float-traps.lisp`）は SBCL（x86-64）が制御できる5種類すべて（`:underflow` `:overflow` `:inexact` `:invalid` `:divide-by-zero`。6つ目の `:denormalized-operand` は SBCL では 32bit x86 専用で x86-64 には存在しない）をマスクし、`make-device` / `make-session` / `session-append-module` / `invoke` / `compile-stablehlo` などの生成・呼び出し点を包む。

## 自動微分と変換

**自動微分（automatic differentiation, AD）**
: プログラムとして書かれた関数の微分を、演算ごとの微分ルールを連鎖律でつないで正確に計算する方法。数値微分（差分で近似する）とも、数式処理（式を記号で変形する）とも違う。

**jvp（ヤコビアン・ベクトル積, Jacobian-Vector Product）/ 前進モード**
: 入力をある方向 `v` に少し動かしたとき、出力がどう動くか（`J·v`）を計算する。関数の計算と同時に前から順に求められる。入力が少なく出力が多い関数に向く。

**vjp（ベクトル・ヤコビアン積, Vector-Jacobian Product）/ 逆伝播**
: 出力側の重み `u` から、各入力への影響（`uᵀ·J`）を計算する。深層学習の「逆伝播（バックプロパゲーション）」はこれ。出力がスカラー（損失）で入力が多い関数に向くので、`grad` はこちらを使う。

**linearize（線形化）**
: jvp の計算を「入力の値だけで決まる部分」と「`v` に対して線形な部分」に分けること。線形な部分だけを取り出すと、次の transpose がかけられる。

**transpose ルール（転置ルール）**
: 線形な演算 `L` に対して、その転置 `Lᵀ` を計算するルール。行列 `A` をかける演算なら、転置は `Aᵀ` をかける演算になる。JAX と nabla は「jvp を線形化して転置すると vjp になる」という性質を使い、演算ごとに書くルールを jvp と transpose の2種類に抑えている。

**vmap / バッチ化ルール（batching rule）**
: `vmap` は、1つの例を処理する関数を、例の束（バッチ）をまとめて処理する関数に自動で変換する。そのために、各演算に「入力にバッチの軸が増えたら、出力のどこにバッチの軸が来るか」を決めるルールを書く。これがバッチ化ルール。

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
: Google の分類で、テストが使ってよい資源の範囲。small は1プロセス内で完結し、I/O やスレッドを使わない。medium は1台のマシン内で、ファイルやローカルのプロセスを使ってよい。large は複数のマシンや外部の資源を使う。小さいほど速く、安定している。nabla では FiveAM のスイート `:nabla.small` / `:nabla.medium` / `:nabla.large` がこれに対応し、既定のスイート（scripts/run-tests.sh）は small + medium。`:nabla.isolated-medium`（tests/iree/support.lisp の `define-iree-test/isolated-medium`）は意味論としては medium（1台のマシン内で完結し、既定で実行する）だが、他の medium テストと同じ SBCL プロセスで実行すると in-process の IREE コンパイラが壊れることが分かっている（issue #68）テストを、`:nabla.medium` とは別の SBCL プロセスに隔離するためのスイート。

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
