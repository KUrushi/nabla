# nabla

Common Lisp で書く、[JAX](https://github.com/jax-ml/jax) に相当する深層学習ライブラリ。Lisp の関数をトレース（実行を記録）して自前の中間表現（IR）に変換し、`jit` / `grad` / `vmap` で書き換えてから [StableHLO](https://openxla.org/stablehlo) を出力し、[IREE](https://iree.dev/) で CPU や NVIDIA GPU 上で実行する。

設計と計画の正本は Artifact「Common Lisp × IREE 深層学習ライブラリ 計画」（計画タブ・設計タブ）。開発の規約は [CLAUDE.md](CLAUDE.md)、専門用語は [docs/glossary.md](docs/glossary.md) を見る。

## 現状

フェーズ0（IREE 疎通）は完了した。ただし GPU（`cuda` ターゲット）での local/cuda 数値一致（issue #12）は GPU の無い環境のため未測定で、issue は open のままにしてある。詳しい知見は [docs/phase0-report.md](docs/phase0-report.md) にまとめてある。

フェーズ1（トレースと jit。親 issue #35）も完了した。得た知見は [docs/phase1-report.md](docs/phase1-report.md) にまとめてある。

現在はフェーズ2（grad。逆伝播）に着手している。

ロードマップ（計画タブより）:

| フェーズ | 目標 | 目安 |
| --- | --- | --- |
| 0. IREE 疎通 | Lisp から IREE を動かす | 完了 |
| 1. トレースと jit | Lisp 関数から StableHLO を出す | 完了 |
| 2. grad | 逆伝播 | 1〜1.5ヶ月 |
| 3. vmap と制御構造 | JAX 相当の変換を揃える | 1〜1.5ヶ月 |
| 4. Flax 相当 | モデル記述 | 1〜2ヶ月 |
| 5. Grain 相当 | データ供給 | 2〜3週間 |

## 要件

- Linux x86_64（IREE コンパイラを PyPI ホイールから使う既定モードの制約。他の OS / アーキでは `--compiler=source` を使う）
- SBCL（動作確認は 2.2.9、apt パッケージ）
- apt パッケージ: `sbcl cl-fiveam cl-cffi cl-alexandria cl-trivial-garbage cl-lparallel cl-closer-mop cl-bordeaux-threads cl-ironclad libffi-dev`
- IREE のビルドに使う道具: `clang lld cmake ninja-build python3-pip git`（Python は PyPI ホイールの取得・展開にだけ使い、nabla の実行時依存にはしない）
- Quicklisp は使わない（ネットワーク方針。依存は apt と固定コミットの git clone で揃える）
- ディスク容量: 約 400 MB（`NABLA_IREE_HOME` 配下が約 345 MB）

## セットアップ

```sh
# 1. Lisp の依存を揃える（初回のみ。apt は root で実行、check-it / optima は
#    固定コミットで git clone する。2回目以降はべき等）
scripts/setup-lisp-deps.sh

# 2. IREE をビルドする（既定 --compiler=wheel。コンパイラは同じコミットの
#    PyPI ホイール、ランタイムは常にソースビルド。数十秒〜数分）
scripts/build-iree.sh

# 3. ビルドした IREE で小さな StableHLO を実行して確かめる
scripts/verify-iree.sh
```

主な環境変数（詳細は [docs/iree-build.md](docs/iree-build.md)）:

- `NABLA_IREE_HOME`: IREE のインストール先。既定 `~/.local/share/nabla/iree-3.11.0`
- `NABLA_LISP_DEPS`: apt に無い Lisp 依存（check-it / optima）の置き場。既定 `~/.local/share/nabla/lisp-deps`
- `NABLA_CACHE_DIR`: vmfb ディスクキャッシュの場所。既定 `${XDG_CACHE_HOME:-~/.cache}/nabla/`

## テスト

```sh
# 既定（small + medium。CPU だけで動き、GPU は不要）
scripts/run-tests.sh

# large（GPU での local/cuda 一致など）。GPU が無ければ自動でスキップする
NABLA_TEST_SIZES=large scripts/run-tests.sh
# GPU が要る環境で、スキップの代わりに失敗させたいとき
NABLA_TEST_SIZES=large NABLA_REQUIRE_CUDA=1 scripts/run-tests.sh

# IREE の共有ライブラリが見つからないときにスキップではなく失敗させる（CI 用）
NABLA_REQUIRE_IREE=1 scripts/run-tests.sh

# mutation testing（自前の runner。既定は git diff で変わった行が対象）
tools/mutate/run.sh
tools/mutate/run.sh src/compile-cache.lisp        # ファイルを指定してもよい

# runner 自身のテスト（CI と同じ1行。上の run-tests.sh と違い CL_SOURCE_REGISTRY は
# 再帰的な "//" にする必要がある。tools/mutate/nabla-mutate.asd はリポジトリ直下の
# 非再帰的な "$PWD/:..." では見つからないため）
CL_SOURCE_REGISTRY="$(pwd)//:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:" \
  sbcl --non-interactive \
    --eval '(require :asdf)' \
    --eval '(asdf:load-system "nabla-mutate/tests")' \
    --eval '(uiop:quit (if (nabla.mutate.tests:run-tests) 0 1))'
```

テストサイズの分類（詳しくは `.claude/skills/nabla-testing`）:

| サイズ | 範囲 | 実行タイミング |
| --- | --- | --- |
| small | 1プロセス内。FFI・ファイル・スレッドを使わない | 毎回（既定） |
| medium | 1台のマシン内。IREE の `local`（CPU）、ファイル I/O | 毎回（既定） |
| large | GPU（`cuda`）での実行、JAX フィクスチャの再生成 | 手動または定期実行 |

## 使ってみる（backend プロトコル経由の最小例）

`examples/add.lisp`（このコードブロックと同じ内容）:

```lisp
(require :asdf)
(asdf:load-system "nabla/iree")
(defparameter *stablehlo* "
func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %a, %b : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}")
(let* ((backend (nb:find-backend :iree))                       ; プロセスに1つの IREE backend（CPU）
       (module (nb:backend-load backend (nb:backend-compile backend *stablehlo*)))
       (a (nb:to-device (make-array 4 :element-type 'single-float :initial-contents '(1.0 2.0 3.0 4.0)) backend))
       (b (nb:to-device (make-array 4 :element-type 'single-float :initial-contents '(10.0 20.0 30.0 40.0)) backend))
       (result (nb:backend-invoke backend module "main" a b)))
  (format t "~&~A~%~A~%" (nb:device-array-aval result) (nb:to-host result))
  (nb:backend-unload backend module))
```

実行:

```sh
export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
sbcl --non-interactive --load examples/add.lisp
```

期待される出力（実測済み）:

```
#S(AVAL :SHAPE (4) :DTYPE F32)
#(11.0 22.0 33.0 44.0)
```

2回目以降にこの例を動かすと、`backend-compile` が vmfb のディスクキャッシュ（既定 `~/.cache/nabla/vmfb/`）を引くので、コンパイルにかかる時間はほぼ0になる。device array（`result` や `a` / `b`）は明示的に解放しなくてよい（finalizer が GC 時に解放する）。明示的に解放したいときは `nabla.iree:release-device-array` を呼ぶ。

上記2行の前に `WARNING: redefining IRONCLAD:BLOCK-LENGTH in DEFGENERIC` のような警告が出ることがあるが、これは ironclad が apt と Lisp システムの両方からロードされることによる既知の無害な警告で、この例のバグではない。

`tests/iree/example-test.lisp` がこの例を毎回 `load` して出力を確認しているので、この例が壊れたら既定のテストスイートが落ちる（ビヨンセ・ルール）。

## 使ってみる（2層 MLP の学習）

`examples/mlp.lisp` は、XOR 風の2次元2クラス分類データで 2層 MLP（dense → tanh → dense）を softmax 交差エントロピーで学習する。`(jit (with-tracing ... (value-and-grad loss :argnums '(0 1 2 3))))` をループの外で1回だけ作り、SGD の更新は Lisp 側の配列演算で行う（2ステップ目以降はコンパイルしない）。`argnums` がリストだと勾配のリストは `jit` の出力にできないので、`with-tracing` の本体で `(multiple-value-bind (loss grads) (funcall vg ...) (values-list (cons loss grads)))` と受けて `(値 勾配...)` の多値に直している。学習ステップは `make-mlp-train-step` として関数で呼べる。

```sh
export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
sbcl --non-interactive --load examples/mlp.lisp
```

出力（100ステップの最初と最後の損失）:

```
loss[0] = 0.6404
final loss = 0.0745
```

`tests/iree/mlp-train-test.lisp` が、JAX のフィクスチャ（`tests/fixtures/train/mlp-sgd.lisp`、生成は `generate.py`）との数値一致、損失の減少、コンパイルが1回だけであること、この例の実行を確かめている（issue #88）。

## 使ってみる（jit）

`examples/jit.lisp`（このコードブロックと同じ内容）:

```lisp
(require :asdf)
(asdf:load-system "nabla/iree")
(nb:defjit add2 (a b) (+ a b))
(let ((a (make-array 4 :element-type 'single-float :initial-contents '(1.0 2.0 3.0 4.0)))
      (b (make-array 4 :element-type 'single-float :initial-contents '(10.0 20.0 30.0 40.0))))
  (format t "~&1回目（コンパイルする）: ~A~%" (add2 a b))
  (format t "~&2回目（キャッシュを使う。再コンパイルしない）: ~A~%" (add2 a b)))
```

実行:

```sh
export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
sbcl --non-interactive --load examples/jit.lisp
```

`defjit` は `with-tracing` で本体をトレース対象にしてから `jit` した通常の Lisp 関数を定義する。1回目の呼び出しでトレース・emit・コンパイルし、`add2` の同一性（このマクロ展開1回分）・引数の `aval`・`*default-backend*` が変わらない限り、2回目以降はインメモリのキャッシュを引くだけでコンパイルし直さない（同じ `defjit` フォームを再評価すると、古いキャッシュは捨てて次の呼び出しで作り直す）。`tests/iree/jit-test.lisp` の `example/jit-lisp/prints-expected-sum` がこの例を `load` して出力を確認している。jit の IREE 経由 end-to-end テストは、他の `nabla/iree` の medium テストと同じ既定の small+medium スイート・同じ SBCL プロセスで動く（issue #34）。

## 公開 API

`nabla`（nickname `nb`）が export するシンボルのみ。`nabla.iree` の低水準な C API バインディングはここには載せない（ハイラムの法則に備え、README に載せた名前を事実上の公開約束にしすぎないため）。詳しく知りたければ `src/iree/package.lisp` を見る。

**dtype**（`src/dtype.lisp`）: `dtype`（`:f32` / `:f64` / `:bf16` / `:f16` / `:i1` / `:i32` / `:u32` / `:u64`。`:i1` は issue #37 で追加した真偽値の dtype で、compare の出力・select の条件に使う。`:i32` / `:u32` / `:u64` は issue #126 で追加した整数の dtype）, `dtype-element-type`, `dtype-byte-width`, `array-dtype`, `dtype-mismatch`, `dtype-mismatch-element-type`, `dtype-mismatch-dtype`

**aval**（`src/aval.lisp`）: `aval`, `make-aval`, `aval-p`, `aval-shape`, `aval-dtype`, `aval-rank`, `aval-size`, `aval-byte-length`, `array-aval`

**backend プロトコル**（`src/backend.lisp`、issue #9）: `backend`, `make-backend`, `find-backend`, `backend-target`, `backend-fingerprint`, `backend-compile`, `backend-load`, `backend-unload`, `backend-invoke`, `to-device`, `to-host`, `device-array-aval`, `backend-error`, `backend-not-available`, `backend-not-available-kind`, `unsupported-dtype`, `unsupported-dtype-dtype`（issue #37。`to-device` にその実行系がデバイス上の表現を持たない dtype を渡すと signal する。IREE backend は issue #72 以降、すべての `dtype` を扱える）

**vmfb ディスクキャッシュ**（`src/compile-cache.lisp`、issue #10）: `*compile-cache-directory*`

**IR と defprimitive**（`src/primitive.lisp`、`src/ir.lisp`、issue #29）: `defprimitive`, `primitive-name`, `var-aval`, `eqn-prim`, `eqn-params`, `eqn-invars`, `eqn-outvars`, `graph-invars`, `graph-eqns`, `graph-outvars`, `graph-constants`, `unknown-primitive`, `unknown-primitive-name`, `primitive-error`

**graph の印字**（`src/ir-print.lisp`、issue #29）: `print-graph`（graph を jaxpr 風のテキストに変換する。読み込み側の `read-graph` はテキスト形式をフェーズ1では公開契約にしないため export しない）

**graph の eager 評価**（`src/eval.lisp`、issue #39）: `eval-graph`（graph を各プリミティブの `:eager` 実装で CPU 上で評価し、出力を多値で返す。`check-graph` は呼ばない）, `graph-input-mismatch`（渡した配列の個数・aval が graph の invars と合わないときに signal する）, `primitive-not-evaluable`, `primitive-not-evaluable-name`（`:eager` を持たないプリミティブに当たったときに signal する）

**トレーサ**（`src/trace.lisp`、`src/walk.lisp`、issue #32）: `with-tracing`（Lisp のコードをコードウォークしてトレース対象にするマクロ。CL の標準関数呼び出しを内部の演算に書き換え、`setq` など対応していない特殊形式は `unsupported-form` にする。`multiple-value-bind` / `multiple-value-list` / `nth-value` / `multiple-value-call` / `multiple-value-prog1` は、トレーサ以外の値の多値も含めてふつうの Lisp の多値として使える（issue #115）。ただし `(multiple-value-call #'+ ...)` のように `+` などの書き換え対象の CL 関数を直接（`#'+` または `'+` で）渡す形は `unsupported-form`）, `trace-to-graph`（`with-tracing` が返す関数を実際の `aval` でトレースし `graph` にする）, `traceable-function`（`with-tracing` が返す関数のクラス）, `unsupported-form`, `unsupported-form-form`, `unsupported-form-path`, `tracing-error`。Lisp の数値はトレース対象の演算に自動でリフト（`%lift-number`）されるが、rank 0 の値どうし・rank 0 と rank ≥1 のブロードキャストは issue #32 の後続 PR（t2）が対応する。shape の不一致はプリミティブの `primitive-error` になる。

**if を select に、配列レベルの公開 API**（`src/trace-ops.lisp`、`src/array-api.lisp`、issue #32）: `with-tracing` の本体の `if`（および `when`/`unless`/`cond`/`and`/`or` のように `if` にマクロ展開されるもの）は、条件がトレーサ・配列で dtype が `:i1`（比較の結果）なら、THEN・ELSE を両方評価してから `select` の演算に書き換える。条件が `:i1` でないトレーサ・配列なら `tracing-error`、それ以外（ふつうの Lisp の値）ならふつうの `if` のまま。`and`/`or` はトレーサの条件に対して `(if x x else)` に展開され、`:i1` の値がそのまま分岐に来るので `tracing-error`（明示的な比較や `where` を使う）。`when`/`unless` は省略された ELSE が `NIL` になるので、条件がトレーサ・配列だと同じ理由で `tracing-error` になる（`nil` は数値としてリフトできない）。

配列レベルの公開 API は9個の総称関数（`array` と `tracer` の両方のメソッドを持つ）: `dot`（最後の軸と最初の軸を縮約する、バッチ無しの `dot-general`。rank 0 は `tracing-error`。配列とトレーサを混ぜて呼ぶと、配列をトレーサと同じ dtype の定数としてリフトする）, `reshape`, `transpose`（`perm` 省略時は軸を逆順にする）, `broadcast-in-dim`, `reduce-sum` / `reduce-max`（`axes` 省略時は全軸を潰す。`axes` を明示的に空リストで渡すと reduce しない＝X をそのまま返す。トレース時は eqn も足さない。省略とは区別される）, `convert`, `where`（`pred`・`a`・`b`。`pred` が真の要素は `a`、偽の要素は `b`。`pred` が eager な bit 配列でも `a`・`b` の少なくとも一方がトレーサなら `pred` を定数としてリフトしてトレースする。`pred` が rank 0 で `a`・`b` が rank 1 以上なら `pred` を分岐の shape にブロードキャストする（`if` の条件も同じ）。`pred` が Lisp のブール値 `t`／`nil` なら、ふつうの `if` と同じく片方の分岐を静的に選び `select` をトレースしない（結果の shape・dtype は同じ真偽の rank 0 の `pred` と同じ）。それ以外のふつうの Lisp の値は `tracing-error`）, `stop-gradient`（issue #80。値は引数そのままで、自動微分では定数として扱われる＝接線がゼロ。任意の dtype を通し、配列には中身の等しい新しい配列を返す。StableHLO では `stablehlo.optimization_barrier` になる）。数値・rank 0 の値は、もう一方の分岐・オペランドの shape・dtype に合わせて自動でブロードキャストされる（それ以外の shape の不一致は `primitive-error`）。`where`（および `if`/`select` への書き換え）の両方の分岐が数値だと dtype を決められず `tracing-error` になる。

**自動微分のコンディション**（`src/ad/rules.lisp`、issue #77）: `autodiff-error`（自動微分の変換が続けられないときの親。ルールの無いプリミティブは子の `no-jvp-rule` / `no-transpose-rule` で、`no-jvp-rule-name` / `no-transpose-rule-name` がプリミティブ名を返す）。jvp / transpose ルールの宣言（`def-jvp-rule` など）と symbolic zero は内部 API で、公開していない。

**grad / value-and-grad**（`src/ad/grad.lisp`、issue #86）: `grad`（`f &key argnums`。`with-tracing` が作った関数、`#'name`（`defjit` が定義した関数）、`jit` した関数を `f` に渡せる（`jit` は `:static-args` の無いものだけ。中の `traceable-function` だけを使い、その `:backend` は無視される。backend 上で動かすには外側を `(jit (grad f) :backend b)` にする）。スカラー出力についての `f` の勾配を返す関数を作る）, `value-and-grad`（同じ引数。多値の `(値 勾配)` を返す）, `grad-requires-scalar-output`, `grad-requires-scalar-output-aval`（`autodiff-error` の子。`f` の出力が rank 0 の浮動小数点ちょうど1つでないときに signal する。`-aval` は問題の出力の aval）。`argnums` は微分する引数の位置（0始まりの整数、または整数のリスト。既定は 0）で、整数なら勾配1つ、リストならリスト（`argnums` の順。JAX と同じ）を返す。範囲外・重複・整数でない `argnums` と、微分する引数が浮動小数点でないとき（`:i1` など）は `autodiff-error`。勾配の shape・dtype はその引数と同じ。戻り値は `traceable-function` なので、配列・実数を渡して直接呼べる（eager。実数は rank 0 の配列として扱い、`double-float` は `:f64`、それ以外は `:f32`）ほか、`(jit (grad f))`、`with-tracing` / `defjit` の本体の中、別の `grad` の対象（`(grad (grad f))` の高階微分）にもできる。トレース中の呼び出しは、微分した graph を呼び出し元のトレースへ展開する。`argnums` がリストのときの勾配のリストは `jit` の出力にできない（`jit` が返せるのは配列かトレーサだけ）ので、`(with-tracing (...) (values-list (funcall g ...)))` で包む（`value-and-grad` の `(値 勾配)` は `(multiple-value-bind (v g) (funcall vg ...) ...)` で受けられる）。既知の制限: `f` が外側のトレースのトレーサを閉包で捕まえていると `tracing-error` になる（外側の値は `f` の引数として渡す）。`(grad f)` は呼ぶたびに新しい関数オブジェクトを作り、`jit` のキャッシュは関数の同一性が鍵なので、**ループの中で `(jit (grad f))` を作ると毎回コンパイルされる**（JAX と同じ）。ループの外で `(jit (grad f))` を1回だけ作るか、`defjit` の本体の中で `grad` を使うこと。

**StableHLO テキスト emitter**（`src/stablehlo.lisp`、issue #33）: `emit-stablehlo`（graph を、無名の module の中に1つの `func.func`（既定名 `main`）を持つ StableHLO テキストに変換する。`backend-compile` にそのまま渡せる。各 eqn の出力行には `loc("eqn-N")` が付く）, `primitive-not-emittable`, `primitive-not-emittable-name`（`:emit` を持たないプリミティブに当たったときに signal する）

**jit**（`src/jit.lisp`、issue #34）: `jit`（`with-tracing` / `defjit` が作った `traceable-function` を、呼ぶたびに必要なら1回だけコンパイルしてから実行する関数にする。`:static-args` で0始まりの引数位置を静的引数に指定でき、`:backend` で使う backend（`nil` なら `*default-backend*`）を指定できる。呼び出し時の動的引数は、CL の配列（`array-aval` で aval を推論する）か device array のどちらでもよい。bf16 / f16 は生の `(unsigned-byte 16)` 配列のままでは dtype を推論できないため、先に `to-device` で device array にしてから渡すこと。戻り値は graph の出力の個数だけ `to-host` した多値になる——v1 は常に host 配列を返す。IREE の `local` backend は f64 の `exp` / `log` / `tanh` を libm の呼び出しとして残すので、これらを含む関数のコンパイルにだけ PATH 上の `ld.lld` が要る（`docs/stablehlo-ops.md`）), `jit-error`（`jit` / jit した関数の呼び出しが誤った使い方を検出したときに signal する）, `*default-backend*`（`jit` に `:backend` を渡さなかったときに使う既定の backend。`nil`・backend の KIND（キーワード）・backend インスタンスのいずれか。`nabla/iree` をロードするとまだ未設定のときに限りこの変数を自分の KIND に設定する）。jit キャッシュは関数の同一性（EQ）・aval・静的引数の値・backend（フィンガープリント込み）をキーにするインメモリのキャッシュで、vmfb のディスクキャッシュ（`*compile-cache-directory*`）とは別の層（`docs/glossary.md` の「インメモリのコンパイルキャッシュ」参照）。

`defjit`（`name-and-options (&rest lambda-list) &body body`。マクロ）は `body` を `with-tracing` でトレース対象にしてから `jit` した、通常の関数として呼べるものを `name` に定義する。`name-and-options` は `name` か `(name :static-args positions)` で、`positions` は評価されて `jit` の `:static-args` と同じ意味になる（例: `(nb:defjit (f :static-args '(1)) (a axis) ...)`。ドキュメント文字列は未対応）。同じ `defjit` フォームを再評価するたびに新しい `traceable-function` を作り直し、古いキャッシュエントリはその場で捨てて、読み込んだモジュールを `backend-unload` する（Lisp の関数を再定義したときの直感どおり、古いキャッシュを使い続けない）。`jit` に渡した関数が GC で回収されたときも、そのキャッシュのモジュールは finalizer で `backend-unload` される。キャッシュのロックは関数ごとで、トレース・コンパイルの間は持たないので、別々の関数の jit は複数のスレッドから並行してコンパイルでき、同じ関数・同じキーへの同時の呼び出しは1回だけコンパイルして結果を共有する。jit した関数を別の関数のトレース中にトレーサを引数にして呼ぶと、コンパイルせずに呼び出し元の graph に展開する（JAX の jit の入れ子と同じ）。`jit-compile-error`（`jit-error` のサブタイプ。`backend-compile` / `backend-load` が `backend-error` を signal したときに、失敗した `graph` と、可能なら原因の eqn（`jit-compile-error-eqn` / `-eqn-index`。見つからなければ NIL）を添えて signal する）, `jit-compile-error-condition`（元の `backend-error`）, `jit-compile-error-graph`, `jit-compile-error-eqn`, `jit-compile-error-eqn-index`。`jit-compile-error` を待ち受けるキャッシュミスの経路には2つのリスタートがある: `use-eager`（この呼び出しだけ `eval-graph` で eager 実行し、何もキャッシュしない）, `recompile`（もう一度コンパイルをやり直す。再帰ではなくループでやり直すので、ハンドラが何度選んでもスタックは深くならない）。`use-eager` はコンパイルに失敗した graph をそのまま評価し、本体をトレースし直さない。使い方は `(handler-bind ((nb:jit-compile-error (lambda (c) (invoke-restart 'nb:use-eager)))) (funcall jitted ...))` のように `invoke-restart` で選ぶ。




**整数 dtype**（`src/dtype.lisp`、`src/primitives/*.lisp`、issue #126）: `:i32`（Lisp では `(signed-byte 32)`、StableHLO では `i32`）、`:u32`（`(unsigned-byte 32)`、`ui32`）、`:u64`（`(unsigned-byte 64)`、`ui64`。THREE_FRY の状態 `ui64[2]` のために足した）。`add` `sub` `mul` `max` `min` `neg` `compare` `select` `convert` `broadcast-in-dim` `reshape` `transpose` `reduce-sum` `reduce-max` は整数を受け付け、`div` `exp` `log` `tanh` `dot-general` は整数を `primitive-error`（トレース時）で拒否する。整数どうしの演算は dtype が一致していなければならず（暗黙の型昇格はしない）、eager の算術は StableHLO と同じく 2 の補数（符号なしは法 2^n）で折り返す。`convert` は整数 ⇔ 浮動小数点 ⇔ `:i1` を変換できる（浮動小数点 → 整数は 0 方向へ丸め、NaN は 0、範囲外（±無限大を含む）は整数の端に飽和。これはどのバックエンドでも同じになるよう、出力する StableHLO が clamp と select で飽和と NaN → 0 を明示する。整数 → `:bf16` は f32 を経由する2段の convert。→ `:i1` は 0 以外が真）。`with-tracing` の中の整数リテラルは相手の整数 dtype にリフトされる（浮動小数点のリテラルや範囲外の整数は `tracing-error`。浮動小数点のトレーサと組むリテラルは従来どおり `:f32`）。整数の接線は変換の中では常に symbolic zero で、整数の入力に対する `grad` は `autodiff-error`、整数を経由する計算の勾配はゼロ。IREE と PJRT（CPU）の `to-device` / `to-host` は整数に対応する。新しい export はない。




#### vmap（issue #125）

`(vmap f &key (in-axes 0) (out-axes 0))` は、`f`（`with-tracing` / `defjit` / `jit` が作った関数。`jit` は静的引数なしのもの）を、引数のバッチ軸についてまとめて適用する関数を返す。結果は「バッチ軸で切り出した各要素に `f` を適用して、出力の `out-axes` の位置に積み直したもの」と一致する。`in-axes` は引数ごとの軸（0始まり、負なら末尾から）か `nil`（その引数はバッチせず `f` にそのまま渡す）で、整数か `nil` を1つ渡すと全引数に共通、リストなら引数ごと。`out-axes` は出力ごとの軸か `nil`（出力がバッチに依存しないときだけ。依存しない出力に整数を渡すと複製する）。戻り値はトレースできる関数なので、配列を渡して直接呼ぶ（eager）ほか、`(jit (vmap f))`、`with-tracing` の中、`grad` の対象、別の `vmap` の対象（`(vmap (vmap f))`）として使える。バッチされていない入力だけの演算は、バッチ化ルールを呼ばずにそのまま残る。既知の制限は `grad` と同じ（`f` が外側のトレーサを閉包で捕まえると `tracing-error`）。`f` はリストを返せない（`(with-tracing … (values-list …))` で包む）。`(vmap f)` は呼ぶたびに新しい関数オブジェクトを作るので、ループの中で `(jit (vmap f))` を作ると毎回コンパイルされる（jit キャッシュのキーが関数の同一性のため。ループの外で1回だけ作る）。`cond*` / `while-loop` / `scan` を含む `f` も `vmap` できる（下の「制御構造のバッチ化」）。

コンディションは `vmap-error`（親。`in-axes` / `out-axes` の個数・型・範囲の不正、軸長の不一致、バッチされた引数が無い、など）と、その子の `no-batch-rule`（バッチ軸を持つ値がバッチ化ルールの無いプリミティブに渡った。`no-batch-rule-name` がプリミティブ名）。バッチ化ルールは `def-batch-rule`（内部。`src/ad/rules-batch-*.lisp`）で書く。全プリミティブがバッチ化ルールを持つ（要素演算は #128、形状・縮約・`dot-general` は #129、制御構造は #140）。




#### 要素演算のバッチ化ルール（issue #128）

`add sub mul div neg exp log tanh max min compare select convert stop-gradient` は共通のバッチ化ルール1つ（`src/ad/rules-batch-elementwise.lisp`）を持つ。バッチされた引数のバッチ軸がすべて同じ位置ならそのまま元の演算を適用し（`transpose` も `broadcast-in-dim` も足さない）、位置が違うときは先頭へ `transpose` で揃える。バッチされていない引数には `broadcast-in-dim` でその位置に軸を足す（要素演算は形が全引数で一致する前提で、暗黙のスカラー拡張は無い）。`select` は条件だけがバッチされる場合も扱う。




形状演算・縮約・`dot-general` のバッチ化ルール（`src/ad/rules-batch-shape.lisp`）: `broadcast-in-dim` / `reshape` / `transpose`（バッチ軸を先頭に置く）、`reduce-sum` / `reduce-max`（縮約軸をずらし、出力のバッチ軸は自然な位置）、`dot-general`（片側だけバッチなら自由次元、両側なら新しい batch 次元の先頭。出力のバッチ軸は自然な位置）。公開 API の追加は無い（`vmap` を通して使う）。




**cond\*（条件分岐）**（`src/cond.lisp`、`src/primitives/cond.lisp`、issue #130）: `(cond* pred then-fn else-fn &rest operands)`。`pred` が真なら `(then-fn operands...)`、偽なら `(else-fn operands...)` を評価する高階プリミティブ `:cond` で、eager では選ばれた枝のサブグラフだけを評価する（CL の `cond` と衝突するので名前は `cond*`）。`then-fn` / `else-fn` は `with-tracing` で作った関数で、普通のトレース対象の関数と同じく1つの値か多値を返し、`cond*` も同じ個数の多値を返す。両枝の出力の aval が一致しなければトレース時に `cond-error`（`tracing-error` の子。`pred` がトレーサなのに rank 0 の `:i1` でない場合、枝が `traceable-function` でない場合も）。枝が閉包で捕まえた外側のトレーサも使える（closure conversion。両枝は同一の入力シグネチャ「operands と両枝の捕捉値の和集合」を持ち、片方が使わない捕捉値の位置には使われない入力が置かれる）。operand はトレーサ・実数・dtype を推論できる配列（`:f32` / `:f64` / `:i1` / `:i32` / `:u32` / `:u64`。bf16 / f16 の生の配列はトレーサで渡す。違えば `cond-error`）。枝の引数の個数が operand の個数と違うときも `cond-error`。`pred` が `t` / `nil` / rank 0 の bit 配列なら、選ばれた枝をそのまま呼ぶ（eqn は作らない）。StableHLO は `stablehlo.if`（IREE でコンパイル・実行できることを medium テストで確認）。jvp / transpose / grad のルールは #134（`src/ad/rules-control.lisp`）、バッチ化ルールは下の「制御構造のバッチ化」。**`with-tracing` の `if` は `cond*` に落とさず、これまでどおり `select` のままにする**: `if` の条件は要素ごとの `:i1` 配列でありうるので、`select`（要素ごとの意味）を保つ必要がある。スカラー条件で片枝だけを評価したい（重い計算や、範囲外の値の `log` など片方の枝でしか意味を持たない計算を避けたい）ときに、`cond*` を明示的に呼ぶ。



**`while-loop`**（`src/while-loop.lisp`、issue #131）: `(while-loop cond-fn body-fn init)` は、`cond-fn` が真の間 `body-fn` を繰り返して最後の carry のリストを返す（JAX の `lax.while_loop`）。`init` は配列（トレース中はトレーサでもよい）の空でないリストで、`cond-fn` / `body-fn` は carry のリストを1つ受け取る関数（`with-tracing` で作る）。`cond-fn` は rank 0 の `:i1` を返し、`body-fn` は `init` と同じ aval（個数・shape・dtype）のリストを返す。配列だけで `with-tracing` の外から呼べば eager（Lisp のループ）、`with-tracing` / `jit` の中では `:while-loop` の eqn になり StableHLO の `stablehlo.while` で出る。`cond-fn` / `body-fn` は外側のトレーサを閉包で捕まえてよく、捕まえた値は loop 不変の追加のオペランドになる。エラーは `while-loop-error` の子: `while-loop-argument-error`（`init` がリストでない・空・要素が配列でない、関数でない、`body-fn` がリストを返さない）、`while-loop-carry-mismatch`（`body-fn` の出力の aval が `init` と違う。トレース時に検出し、0回で終わるループでも出る。`while-loop-carry-mismatch-expected` / `-actual`）、`while-loop-condition-error`（`cond-fn` の結果が rank 0 の `:i1` でない）。前進モードの jvp には対応する（`src/ad/rules-control.lisp`、issue #134。接線を持つ carry の接線を carry に足した `while-loop` にする。最初は接線がゼロの carry も本体を通ると非ゼロになりうるので、JAX と同じく不動点まで広げる）。逆モードの `grad` は対応しない（反復回数が分からず残差を保存できない）。`grad` が通ると、プリミティブ名 `:while-loop` を含む `autodiff-error` になる。




#### scan（issue #132）

`(scan f init xs &key length reverse)` は、`xs`（配列のリスト）の先頭の軸に沿って `f` を回し、`(values 最終の carry のリスト ys のリスト)` を返す（JAX の `lax.scan` 相当）。`f` は `with-tracing` で作った2引数の関数 `(carry-list x-list)` で、`(values 新しい carry のリスト y のリスト)` を返す。`ys` は各ステップの `y` を先頭の軸に積んだ配列のリスト。carry は `init` と個数・shape・dtype が同じでなければならない（違うと `scan-carry-mismatch`）。`xs` が空のときは `length` が必須で、そうでなければ `xs` の先頭の軸の長さと一致しなければならない（`scan-length-error`）。長さ 0 の scan は `init` をそのまま返し、`ys` は先頭の軸が 0 の空の配列になる。`reverse` が真なら添字 `length-1` から 0 へ辿る（`ys[t]` にはそのときも添字 `t` のステップの `y` が入る）。`f` が閉包で捕まえた外側の値はループ不変な入力（consts）になる。eager でも `with-tracing` / `jit` の中でも使える。引数や `f` の戻り値の形が不正なときは `scan-error`（親）。順方向・jvp（#135、下記）・`grad`（#139、下記）に対応し、`vmap`（#140、下の「制御構造のバッチ化」）にも対応する。StableHLO では `:i32` のカウンタを持つ `stablehlo.while` に落ちる。




#### rng-bit-generator プリミティブ（issue #133、内部）

`:rng-bit-generator`（複数出力）は `stablehlo.rng_bit_generator`（THREE_FRY）に対応し、状態 `ui64[2]` から `(新しい状態, 乱数ビット)` を作る。params は出力の `:shape` と `:dtype`（`:u32` / `:u64`）。eager 実装は IREE の lowering を写した Threefry-2x32 で、IREE（local）とも PJRT（XLA CPU）ともビット単位で一致する（`docs/stablehlo-ops.md`）。状態もビットも整数なので微分しない。公開の PRNG API（`prng-key` / `split` / `uniform` など）は #136（下の「PRNG」）で、このプリミティブは内部（`nb::rng-bit-generator`）。状態の先頭にバッチ次元を付けられる（#136）。





**制御構造の jvp**（`src/ad/rules-control.lisp`、issue #134）: `cond` の jvp ・transpose ルールと `while-loop` の jvp ルール（前進モードのみ）。`cond*` は `grad` / `jvp` を通せる: jvp は主値の `:cond`（残差を枝の出力として出す）と、接線について線形な `:cond` の2つの eqn にし、`grad` は線形な `:cond` の各枝を転置する（JAX の `_cond_partial_eval` / `_cond_transpose` の写し）。公開 API の追加は無い（`grad` の逆モードは `while-loop` を通すと `autodiff-error`）。




**scan の jvp ルール**（`src/ad/rules-scan.lisp`、issue #135。内部のみで export は無い）: `:scan` の jvp ルールは JAX の `_scan_jvp` と同じ形で、本体を `jvp-graph` した「主値と接線を一緒に回す1つの `scan`」を作る（ループを2回回さない）。並びは consts ++ 接線のある consts の接線、carry ++ 接線のある carry の接線、xs ++ 接線のある xs の接線（出力は 最終 carry ++ その接線 ++ ys ++ ys の接線）。symbolic zero の接線は入力にも出力にもならない。carry の接線の有無は本体を通ると変わりうる（初期の接線がゼロでも、本体で非ゼロの接線を受ければ次のステップで非ゼロ）ので、「非ゼロの接線を持つ carry の集合」を増えなくなるまで広げる（不動点）。逆モードは次の段落。




#### PRNG（issue #136）

JAX の `jax.random` と同じ、明示的なキー渡しの PRNG。キーは `:u32` の `(2)` の配列で、乱数が要る関数にはキーを引数として渡し、同じキーからは必ず同じ値が出る。キーは「使う（`uniform` / `normal`）か、`split` する」のどちらか一方にだけ使い、元のキーで両方を引かない。

| 関数 | 役割 |
| --- | --- |
| `(prng-key seed)` | 整数のシード（64ビットに収まる整数）からキー `[上位32ビット 下位32ビット]` を作る |
| `(split key &optional (n 2))` | 独立な `n` 個のキー（shape `(n 2)`、`:u32`）を作る |
| `(fold-in key data)` | キーに整数 `data`（0 以上 2^32 未満、または `:u32` / `:i32` の rank 0 のトレーサ）を混ぜた新しいキー |
| `(uniform key shape &key dtype minval maxval)` | `[minval, maxval)` の一様乱数（`dtype` は `:f32`（既定）/ `:f64`、範囲の既定は 0 と 1） |
| `(normal key shape &key dtype)` | 標準正規分布の乱数（`dtype` は `:f32`（既定）/ `:f64`） |
| `prng-error` | 不正な引数のコンディション |

eager でも `jit` / `grad` / `vmap` の中でも使える。`vmap` でキーをバッチすると各要素はそのキーで単独に呼んだ結果とビット単位で一致する（`(vmap (with-tracing (k) (uniform k '(3))))` を `(split key 8)` に適用する、など）。

決めたこと:

- **キー → 状態**: キー `[k0 k1]` を `rng-bit-generator` の状態 `ui64[2] = [k0 | k1 << 32, カウンタ]` にする（`bitcast-convert` で2語を1語にまとめる）。`uniform` / `normal` / `split` はカウンタ 0 から引き、`fold-in` はカウンタ `2^32 + data` の2語を新しいキーにする（引く量が 2^32 要素未満なら、`fold-in` の出力が `uniform` / `split` の列と重なることはない）。
- **JAX とビット単位では一致しない**: JAX の既定は `threefry_2x32` を直接呼ぶ実装で `rng_bit_generator` を使わないため、同じシードでも値が違う。nabla は `stablehlo.rng_bit_generator`（THREE_FRY）を使い、IREE・PJRT・eager が互いにビット単位で一致する（`docs/stablehlo-ops.md`）。分布としては同じ（統計検定で確かめている）。
- **uniform**: 乱数ビットの仮数部だけを取り出して `[1, 2)` の浮動小数点数にし、1 を引いて範囲に伸ばす（JAX と同じ。`:f32` は 23 ビット、`:f64` は 52 ビットの粒度）。丸めのため `maxval` にちょうど等しい値が出うる。
- **normal**: JAX と同じく、`(-1, 1)` の一様乱数に erf の逆関数をかけて √2 倍する。Box–Muller は `sin` / `cos` のプリミティブが無いため採らなかった。erf の逆関数は Giles の単精度多項式近似（JAX の f32 と同じ係数。相対誤差 約 1e-7）で、`:f64` でも同じ近似を使うので精度は f32 並みで、極端な裾は過小評価される（`u = ±(1 - 2^-53)` で約 ±7.32、真の分位点は約 8.2）。
- **バッチ化**: バッチ次元を持つ状態 `ui64[..., 2]` を `rng-bit-generator` が受け付け、各行は単独に呼んだ結果とビット単位で一致する。StableHLO の `rng_bit_generator` は `ui64[2]` しか受けないので、`stablehlo.while` で1行ずつ `dynamic_slice` → `rng_bit_generator` → `dynamic_update_slice` を回して出力する（issue #164。出力する StableHLO の大きさは行数に依らない）。**コンパイルコスト**（IREE local、バッチされた rng の eqn 1つ）: 32 行 0.79 秒、64 行 0.80 秒、256 行 0.85 秒（行ごとに展開していた以前は 4.0 秒 / 6.9 秒 / 42.8 秒）。**実行時間**は IREE ではループ1回ごとの起動と、ビットのバッファ全体のコピー（`scan` の ys と同じ。#159）のぶん以前より遅い（256 キー × `uniform` 1024 要素で 1回 約 57 ms、展開していたときは約 3 ms）。PJRT（XLA CPU）では遅くならない。
- 新しいビット演算プリミティブ（内部）: `:shift-right-logical`、`:bitwise-or`（整数専用の2入力の要素演算）と、ビット列を再解釈する `:bitcast-convert`（`:f32 :f64 :i32 :u32 :u64`。幅が違うときは StableHLO と同じく末尾の次元が増減し、並びはリトルエンディアン）。どれも整数・ビット列の演算なので微分しない。





#### with-tracing の do ループ（issue #137）

`with-tracing` の本体にある定型の `do` は、反復回数の分だけ展開されるのではなく、1つの `scan` になる（反復回数を大きくしても eqn は増えない）。`setq` が使えないので、carry は `do` 変数の step 式で更新する。

```lisp
(nb:with-tracing (h0 w)
  (do ((i 0 (1+ i))                    ; カウンタ。step は (1+ i) / (+ i 1) / (+ 1 i)
       (h h0 (tanh (+ (* h s) 0.1)))   ; carry。step は純粋な式（全 step が古い値を見る）
       (s w))                          ; step が無い変数は不変
      ((>= i n) h)))                   ; 終了条件は (>= i n) か (= i n)。結果形式は最終値で評価
```

- カウンタの初期値と上限 `n` はトレース時に決まる整数（閉包で捕まえた Lisp の整数など）。上限は `do` 変数を参照できない。整数でないと `scan-error`。反復回数は `(>= i n)` なら `max(0, n - 初期値)`、`(= i n)` なら `n - 初期値`（`n` が初期値より小さいと Lisp では止まらないので `scan-length-error`）
- カウンタは step の中では `:i32` のスカラーのトレーサ、結果形式の中では最終値の Lisp の整数。浮動小数点の carry と組むときは `(nb:convert i :f32)` のように明示的に変換する（暗黙の型昇格はしない）。carry の dtype・shape は init と step で一致しなければならない（`scan-carry-mismatch`）
- 上限 `n` は CL の `do` のように反復ごとではなく、ループに入る前に1回だけ評価する（init 式のあとに評価する）
- 整数リテラルの carry（例 `(k 0 (+ k 1))`）は Lisp の整数ではなく、f32 の rank 0 の配列として戻る（scan が Lisp の実数を f32 のスカラーにするため。整数の carry にしたいときは `:i32` の配列を init に渡す）
- 本体のフォーム（`do` の body）は書けない（宣言だけ可）。この形に合わない `do`（終了条件が無い、step が別の形、本体にフォームがある、など）は、原因の `do` フォームを持つ `unsupported-form`（`path` は `nil`）
- `do*` / `dotimes` / `loop` は展開しない。従来どおり、展開後の `block` で `unsupported-form` になる。`dotimes` は carry を `setq` でしか渡せず、`loop` の `for ... = ... then` は更新と終了判定の順序が `do` と違って同じ意味に写せないため。ユーザー定義のマクロが `do` に展開されるものも、展開前のフォームだけを見るので対象外
- `quote` とバッククォートの中の `do` は書き換えない。`flet` / `labels` / `macrolet` の局所関数の定義（名前が `do` でも）も見ないが、`do` という名前の局所関数の呼び出しが `do` の構文に読める形だと `do` として扱ってしまう（制限）
- 展開は `src/loop-scan.lisp`（`%expand-do-loops`。`with-tracing` が `macroexpand-all` の前に呼ぶ）。`%walk` / `*rewrite-table*` は変えない
- 新しい export はない




#### per-example 勾配（issue #138）

`(vmap (grad loss) :in-axes ...)` で、サンプルごとの勾配を一度に求める。`examples/mlp.lisp` の `make-per-example-grad` は、2層 MLP の1サンプルの損失（`make-mlp-example-loss`。x は `(D)`、y は `(C)` の one-hot）を `grad` し、パラメータはバッチせず（`in-axes` が `nil`）x と y だけを軸 0 でバッチする（`:in-axes '(nil nil nil nil 0 0)`）。各勾配の形は `(N ...パラメータの形)`で、JAX の `jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))` と f32 の許容誤差で一致する（フィクスチャ `tests/fixtures/per-example/`。生成は `generate.py`、jax 0.10.2・x64 無効）。サンプルごとの勾配の平均は、バッチ全体の損失（平均）の `grad` と一致する。`grad` が返す勾配のリストは `vmap` の出力にできないので、`(with-tracing … (values-list …))` で多値に直してから `vmap` に渡す。

合成は次のどれも eager と IREE の `local` で動く: `(jit (vmap (grad f)))`、`(vmap (grad f))`、`(grad (… (vmap f) …))`（`grad` の中の `vmap`。x や y は閉包でなく引数として渡す）、`(vmap (vmap f))`。`vmap` が `jit` した関数をトレース中に呼ぶ場合は `grad` と同じく中の関数だけを使い、その `:backend` は見ない（外側の `jit` の backend で動く）。




**scan の逆モード（partial eval と transpose）**（`src/ad/partial-eval.lisp`、`src/ad/rules-scan-reverse.lisp`、issue #139。内部のみで export は無い）: `linearize-graph` は、jvp した graph を接線への依存で主値側と線形側に分ける前に、プリミティブごとの partial eval ルール（`set-partial-eval-rule`）で eqn を置き換える。`:scan` のルールは JAX の `_scan_partial_eval` と同じで、jvp した1つの scan を「主値と各ステップの残差（`ys` として積む）を計算する scan」と「残差を `xs`、ループ不変な残差を `consts` として受ける、接線について線形な scan」に分ける。未知（接線に依存する）carry の集合は不動点まで広げる。ループ不変な残差（consts と本体の定数だけで決まる値。閉包で捕まえたホストの配列は本体の定数なので不変）は積まず、scan の外で（必要なら計算して）渡す。`:scan` の transpose ルールは JAX の `_scan_transpose` と同じで、線形な scan を `reverse` を反転した scan にし（carry の余接線は carry、`xs` の余接線は `ys`、consts の余接線は carry に足し込む和）、`ys` の余接線が symbolic zero なら `xs` に積まず本体の中でゼロを作る。carry が線形入力に依存しない scan（主値と接線が混ざった scan）の transpose は `autodiff-error`。これで `grad` は scan を通る。





**制御構造のバッチ化**（`src/ad/rules-batch-control.lisp`、issue #140。JAX の `_cond_batching_rule` / `_while_loop_batching_rule` に倣う）: 本体のサブグラフを再帰的に `vmap` する（`%vmap-subgraph`）。`cond*`: 条件がバッチされなければ、両枝を同じ入力のバッチ軸でバッチ化した `:cond` のままにし、どちらかの枝でバッチされる出力は両枝でバッチして先頭に揃える。条件がバッチされると、片方の枝だけを評価する性質は失われ、両枝を評価して `select` で選ぶ。`while-loop`: 「バッチされる carry の集合」を、本体の出力でバッチされる carry（最初はバッチされない carry が本体でバッチされる場合）を足しながら不動点まで広げ、バッチされる carry は軸 0 に揃える。loop 不変の（閉包で捕まえた）値は元のバッチ軸のまま素通しする。条件がバッチされると（全 carry がバッチされる）、どれかの要素の条件が真の間回し（`convert` + `reduce-max` + `compare`）、条件が偽になった要素の carry は `select` で据え置く。`scan`: 本体を同じ不動点でバッチ化する。consts は元のバッチ軸のまま、バッチされる carry は軸 0、`xs` のバッチ軸は走査の軸（先頭）とぶつからないよう 1 に動かし、`ys` のバッチ軸も 1 に出る（`out-axes` で動かす）。




**RNN を scan で学習する例**（`examples/rnn.lisp`、issue #141）: Elman RNN（`h' = tanh(h W_h + x_t W_x + b)`、最後の隠れ状態から線形層、損失は平均二乗誤差）を `scan` で書き、`(jit (with-tracing ... (value-and-grad loss :argnums '(0 1 2 3 4))))` をループの外で1回だけ作って SGD で学習する。バッチは `vmap` ではなく `dot` の行方向で持つ（系列は `(T B D)`、T=8 B=4 D=4 H=8 O=2）。`tests/iree/rnn-train-test.lisp` が、JAX（`jax.lax.scan`）のフィクスチャ（`tests/fixtures/rnn/rnn-sgd.lisp`、生成は `generate.py`）との1ステップ目の損失・勾配と30ステップの損失の軌跡の一致、損失の減少、コンパイルが1回だけであることを確かめる。



`nabla.iree` パッケージからは、上の総称関数の IREE 向けメソッドに加えて次を使う:

- `iree-backend`（`(nb:make-backend :iree :target :local | :cuda :cuda-arch "sm_80")`）
- `compile-stablehlo` / `compile-flags`
- コンディション階層: `iree-error`（`backend-error` のサブクラス）

## プロジェクト構成

```
src/                 core（nabla パッケージ）: package, dtype, aval, primitive, ir, backend, compile-cache
src/iree/            nabla.iree パッケージ: IREE の埋め込み C API バインディングと backend 実装
tests/               nabla/tests のテスト（support-test, dtype-test, aval-test, backend-test, ...）
tests/support/       nabla/test-support: FiveAM のスイート、check-it の生成器、比較関数、フェイク backend
tests/iree/          nabla/iree/tests のテスト
tests/fixtures/stablehlo/  手書きの StableHLO フィクスチャ
tests/regressions/   check-it が見つけた失敗例の回帰テスト
tools/mutate/         自前の mutation testing runner（nabla-mutate）
scripts/             setup-lisp-deps.sh, build-iree.sh, verify-iree.sh, run-tests.sh
docs/                glossary.md, iree-build.md, stablehlo-ops.md, pjrt-setup.md, float-traps-experiments.md, phase0-report.md 〜 phase3-report.md
examples/            add.lisp, jit.lisp, mlp.lisp, rnn.lisp（この README の使用例）
third_party/         iree.lock（固定した IREE のコミットとホイールの sha256）
.claude/skills/nabla-testing/  テスト戦略の詳しい手順
```

ASDF システムは `nabla`（コア、nickname `nb`）、`nabla/test-support`、`nabla/tests`、`nabla/ffi-support`、`nabla/ffi-support/tests`、`nabla/iree`、`nabla/iree/tests`、`nabla/pjrt`、`nabla/pjrt/tests`（PJRT プラグインのロード、クライアント・デバイス・device-array、StableHLO のコンパイル・ロード・実行。`(find-backend :pjrt)` と `to-device` / `to-host` / `backend-compile` / `backend-invoke`、`(jit f :backend :pjrt)`。docs/pjrt-setup.md）の9つに加え、mutation testing 用の `nabla-mutate`（`tools/mutate/`）がある。

## 開発の進め方

- 規約は [CLAUDE.md](CLAUDE.md) を読む（構成、コマンド、設計上の約束、開発の原則、テスト戦略、Git の運用）
- テストの書き方・実行の仕方は `.claude/skills/nabla-testing` を読む
- コミットメッセージと PR タイトルは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従う
- PR は stacked PR（下位ブランチに積む）で、squash merge をデフォルトにする
- ライセンスは [Apache License 2.0](LICENSE)
