# nabla

Common Lisp で書く、[JAX](https://github.com/jax-ml/jax) に相当する深層学習ライブラリ。Lisp の関数をトレース（実行を記録）して自前の中間表現（IR）に変換し、`jit` / `grad` / `vmap` で書き換えてから [StableHLO](https://openxla.org/stablehlo) を出力し、[IREE](https://iree.dev/) で CPU や NVIDIA GPU 上で実行する。

設計と計画の正本は Artifact「Common Lisp × IREE 深層学習ライブラリ 計画」（計画タブ・設計タブ）。開発の規約は [CLAUDE.md](CLAUDE.md)、専門用語は [docs/glossary.md](docs/glossary.md) を見る。

## 現状

フェーズ0〜4は完了した。現在はフェーズ5（Grain 相当。データ供給）に着手する。モデルを書いて学習し、保存するところまでを、下の「チュートリアル」の節で通して試せる。

- フェーズ0（IREE 疎通）: GPU（`cuda` ターゲット）での local/cuda 数値一致（issue #12）は、2026-10-10 に Colab の Tesla T4 で確かめた（`scripts/colab-gpu-check.sh`、結果は [docs/iree-build.md](docs/iree-build.md)）。知見は [docs/phase0-report.md](docs/phase0-report.md)
- フェーズ1（トレースと jit。親 issue #35）: 知見は [docs/phase1-report.md](docs/phase1-report.md)
- フェーズ2（grad。2層 MLP の学習、PJRT バックエンドと IREE の比較。親 issue #90）: 知見は [docs/phase2-report.md](docs/phase2-report.md)
- フェーズ3（vmap・制御構造 `cond*` / `while-loop` / `scan`・PRNG。親 issue #124）: 知見は [docs/phase3-report.md](docs/phase3-report.md)
- フェーズ4（Flax 相当。PyTree と `defmodule`・レイヤー・オプティマイザ・チェックポイント・小さな Transformer の学習。親 issue #194）: 知見は [docs/phase4-report.md](docs/phase4-report.md)
- フェーズ5（Grain 相当。データ供給）: 着手する（`nabla/data` はまだ無い）

ロードマップ（計画タブより）:

| フェーズ | 目標 | 状態 |
| --- | --- | --- |
| 0. IREE 疎通 | Lisp から IREE を動かす | 完了 |
| 1. トレースと jit | Lisp 関数から StableHLO を出す | 完了 |
| 2. grad | 逆伝播 | 完了 |
| 3. vmap と制御構造 | JAX 相当の変換を揃える | 完了 |
| 4. Flax 相当 | モデル記述 | 完了 |
| 5. Grain 相当 | データ供給 | 着手 |

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
- `NABLA_CACHE_DIR`: vmfb ディスクキャッシュの場所（キャッシュはその下の `vmfb/` に置く）。既定 `${XDG_CACHE_HOME:-~/.cache}/nabla/`。相対パスはプロセスの作業ディレクトリを基準に絶対パスへ解決する（`*compile-cache-directory*` に束縛した相対パスも同じ）

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

## チュートリアル（モデル定義から学習・保存まで）

`defmodule` でモデルを書き、PRNG キーで初期化し、`jit` した学習ステップ（`value-and-grad` + オプティマイザ）を手書きのループで回し、モデルとオプティマイザの状態を1つのチェックポイントに保存して、復元して続きを学習するまでを、XOR 風の2クラス分類（2層 MLP）で説明する。コードは全部 [`examples/tutorial.lisp`](examples/tutorial.lisp) にあり、下のコードブロックはそのファイルの該当部分をそのまま貼ったもの。`tests/nn/tutorial-test.lisp` が (1) README のブロックと例のファイルが一字一句同じであること、(2) 例が動いて損失が下がること、(3) 保存・復元した続きが止めずに続けた学習とビットまで同じであること、を毎回確かめるので、コードが古くなると既定のテストスイートが落ちる。README を直すときは、先に `examples/tutorial.lisp` を直し、`[[tutorial:名前]]` で挟んだ部分を貼り直す。

以降のコードブロックは、次のパッケージ（`nabla/nn` と `nabla/iree` をロードした後）で書いている。`nabla.nn` のシンボル（`defmodule` `call` `chain` `adam` `save-checkpoint` など）はそのまま、コアの関数は `nabla:` を付けて呼ぶ（`nabla:jit` `nabla:value-and-grad` `nabla:with-tracing` など）。

```lisp
(require :asdf)
(asdf:load-system "nabla/nn")
(asdf:load-system "nabla/iree")
(defpackage #:nabla-example-tutorial
  (:use #:cl #:nabla.nn))
(in-package #:nabla-example-tutorial)
```

```sh
export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
sbcl --non-interactive --load examples/tutorial.lisp
```

出力の例（30 ステップ学習 → 保存 → 復元 → 30 ステップ続ける。数値は検査していない。テストが確かめるのは損失が下がることと、復元した続きが一致すること）:

```
loss[0] = 0.7050
loss[29] = 0.4449
復元して続けた loss[59] = 0.1164
```

### 1. モデルを `defmodule` で書く

モデルは「パラメータの PyTree ＋ 純粋関数」。スロット（`hidden` `out`）が子のモジュールで、`:init` が PRNG キーから初期値を作り、`:call` が順伝播になる。キーは `split-keys` で独立な部分キーに分けて、子に1つずつ渡す（同じキーからは常に同じ初期値になる）。`dense` や `layer-norm` などの層は `nabla.nn` にある。

<!-- tutorial:model -->
```lisp
;; 2層 MLP の分類器。dense は nabla.nn の層、(key in width classes) の key は PRNG キー。
(defmodule mlp (hidden out)
  (:init (key in width classes)
   (destructuring-bind (hidden-key out-key) (split-keys key 2)
     (make-mlp :hidden (init-dense hidden-key in width)
               :out (init-dense out-key width classes))))
  (:call (self x)
   (call (mlp-out self) (tanh (call (mlp-hidden self) x)))))
```

`(init-mlp (nabla:prng-key 0) 2 16 2)` で入力 2・隠れ 16・クラス 2 のモデルができる。`call` は総称関数で、`(call model x)` が順伝播。`:call` の本体は `with-tracing` の本体として書かれるので、`jit` / `grad` / `vmap` の中でそのまま呼べる。

### 2. データ

乱数の配列は `nabla:uniform` / `nabla:normal` に PRNG キーを渡して作る。ステップ番号を `nabla:fold-in` でキーに混ぜると、ステップごとに別の、でも再現できるバッチになる。

<!-- tutorial:data -->
```lisp
;; XOR 風の2クラス分類。x は [-1, 1) の2次元の点で、x0 * x1 > 0 ならクラス 1、そうでなければクラス 0。
;; y はクラスの one-hot（f32 の (batch 2)）。ステップ番号 step から決まるので、何度呼んでも同じ。
(defun make-batch (step &key (batch-size 32))
  (let* ((key (nabla:fold-in (nabla:prng-key 100) step))
         (x (nabla:uniform key (list batch-size 2) :minval -1 :maxval 1))
         (y (make-array (list batch-size 2) :element-type 'single-float :initial-element 0.0)))
    (dotimes (i batch-size)
      (setf (aref y i (if (> (* (aref x i 0) (aref x i 1)) 0) 1 0)) 1.0))
    (values x y)))
```

### 3. `jit` した学習ステップ

`value-and-grad` は、第1引数がモデル（木）でも、勾配を同じ構造の木で返す。オプティマイザは `chain` で部品（ここでは大域ノルムのクリップと Adam）を組み合わせる。`optimizer-update` は更新量と新しい状態を返し、`apply-updates` がモデルに足す。学習ステップ全体が純粋関数なので、ひとまとめに `jit` できる。`jit` した関数はループの外で1回だけ作る（2回目以降の呼び出しはコンパイルしない）。

<!-- tutorial:train-step -->
```lisp
;; 損失は交差エントロピー。サンプル数は one-hot の総和 (reduce-sum y) で割る。
;; value-and-grad の引数 model が木（defmodule）でも、grad は同じ構造の木で返る。
;; 学習ステップ (model opt-state x y) → (values 損失 新しい model 新しい opt-state) は jit する。
;; jit はループの外で1回だけ作る（2回目以降の呼び出しはコンパイルしない）。
(defun make-optimizer ()
  (chain (clip-by-global-norm 1.0) (adam 0.05)))

(defun make-train-step (optimizer)
  (let ((value-and-grad
          (nabla:value-and-grad
           (nabla:with-tracing (model x y)
             (- (/ (nabla:reduce-sum (* y (log-softmax (call model x))))
                   (nabla:reduce-sum y)))))))
    (nabla:jit (nabla:with-tracing (model opt-state x y)
                 (multiple-value-bind (loss grads) (funcall value-and-grad model x y)
                   (multiple-value-bind (updates new-state)
                       (optimizer-update optimizer grads opt-state model)
                     (values loss (apply-updates model updates) new-state)))))))
```

オプティマイザの状態（Adam の `mu` / `nu` / `count`）もモデルと同じ PyTree で、`(optimizer-init optimizer model)` で作る。ステップ数 `count` は配列なので、ステップごとに再コンパイルされない。

### 4. 手書きのループ

ループは普通の Lisp で書く。ステップが返す新しいモデルと状態を、次のステップに渡す。

<!-- tutorial:loop -->
```lisp
;; 手書きのループ。FROM から TO の手前までのステップを回し、損失のリストと最後の model / opt-state を返す。
(defun run-steps (train-step model opt-state from to)
  (let ((losses '()))
    (loop for step from from below to
          do (multiple-value-bind (x y) (make-batch step)
               (multiple-value-bind (loss new-model new-state) (funcall train-step model opt-state x y)
                 (push (aref loss) losses)
                 (setf model new-model opt-state new-state))))
    (values (nreverse losses) model opt-state)))
```

### 5. チェックポイントの保存と復元

モデルとオプティマイザの状態は、1つのリストにまとめれば1ファイル（safetensors）に入る。パス名は `0.hidden.kernel`（リストの添字とスロット名を `.` でつないだもの）のようになる。復元には同じ構造の「ひな形」を渡す。値は使わず、名前・形・dtype の照合と、`tree-unflatten` の型情報にだけ使う。名前の過不足や形・dtype の違いは `checkpoint-error` になる。

<!-- tutorial:checkpoint -->
```lisp
;; model と opt-state を1つのリストにして1ファイル（safetensors）に保存する。
;; 復元には同じ構造の「ひな形」が要る（葉の名前・形・dtype を照合する）。
(defun save-training (path model opt-state)
  (save-checkpoint path (list model opt-state)))

(defun load-training (path template-model template-opt-state)
  (destructuring-bind (model opt-state) (load-checkpoint path (list template-model template-opt-state))
    (values model opt-state)))
```

復元したモデルと状態から `run-steps` を続けると、止めずに続けた学習と（ビットまで）同じ損失の列になる。

### 6. （おまけ）サンプルごとの勾配

`vmap` と `grad` を重ねると、バッチの各サンプルの勾配を一度に求められる。`in-axes` が `nil` の引数（モデル）はバッチせず、`0` の引数（`x` と `y`）だけを軸 0 でバッチする。

<!-- tutorial:per-example -->
```lisp
;; vmap でサンプルごとの勾配。パラメータはバッチせず (in-axes が nil)、x と y だけを軸 0 でバッチする。
;; 戻り値は model と同じ構造の木で、各葉に先頭のバッチ軸がつく。
(defun per-example-grads (model x y)
  (funcall (nabla:vmap (nabla:grad (nabla:with-tracing (model x y)
                                      (- (nabla:reduce-sum (* y (log-softmax (call model x)))))))
                       :in-axes '(nil 0 0))
           model x y))
```

次に読むもの: 「公開 API」の defmodule・基本レイヤー・オプティマイザ・チェックポイントの各節と、全部を使った Transformer の例。

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

**配列の不変性**（issue #166）: nabla は、渡された配列を書き換えない。一方で eager の結果は、入力・閉包で捕まえた配列・graph の定数と `eq` な配列でありうる（`eval-graph`、`with-tracing` で作った関数を配列で直接呼ぶとき、`cond*` / `while-loop` / `scan`、`axes` を空リストで渡した `reduce-sum` / `reduce-max`、jit の `use-eager` リスタート）。JAX と同じく、入力も結果も不変（immutable）として扱うこと。書き換えると graph の定数や jit のキャッシュが壊れうる。

**rsqrt / sqrt**（issue #196）: `rsqrt`（export。`(rsqrt x)` = 1/sqrt(x) を要素ごとに返す関数で、配列・トレーサ・数値を取る。`with-tracing` の中でも同じ関数として書ける）と、CL の `sqrt` を `with-tracing` が書き換えたもの。どちらも bf16 / f16 / f32 / f64 に対応し、負の入力は NaN、`rsqrt(0)` は +無限大。数値（リテラル）の引数は CL の演算にそのまま委ね、負の値や 0 では CL の挙動（複素数・除算エラー）になる（`log` と同じ制限）。

**value-aval**（issue #204）: `(value-aval x)` は `x` の `aval`（`aval-shape` / `aval-dtype` で形と dtype が読める）を返す関数。`with-tracing` の本体（`defmodule` の `:call` など）が入力の形を知るために使う。受け付けるのはトレーサ・dtype が決まる配列・device array で、文字列・数値・リストは `tracing-error`、dtype が決まらない配列（`(unsigned-byte 16)`・`simple-vector`）は `dtype-mismatch`。

配列レベルの公開 API は11個の総称関数（`array` と `tracer` の両方のメソッドを持つ。`conv-general-dilated` は下の「畳み込み」）: `dot`（最後の軸と最初の軸を縮約する、バッチ無しの `dot-general`。rank 0 は `tracing-error`。配列とトレーサを混ぜて呼ぶと、配列をトレーサと同じ dtype の定数としてリフトする）, `dot-general`（`(dot-general a b :lhs-contracting .. :rhs-contracting .. :lhs-batch .. :rhs-batch ..)`。縮約軸とバッチ軸（先頭でなくてもよい）を明示する StableHLO の dot_general。結果の軸はバッチ軸、a の残り、b の残りの順。attention のスコアのように複数のバッチ軸があるときに使う。issue #208）, `reshape`, `transpose`（`perm` 省略時は軸を逆順にする）, `broadcast-in-dim`, `reduce-sum` / `reduce-max`（`axes` 省略時は全軸を潰す。`axes` を明示的に空リストで渡すと reduce しない＝X をそのまま返す。トレース時は eqn も足さない。省略とは区別される）, `convert`, `where`（`pred`・`a`・`b`。`pred` が真の要素は `a`、偽の要素は `b`。`pred` が eager な bit 配列でも `a`・`b` の少なくとも一方がトレーサなら `pred` を定数としてリフトしてトレースする。`pred` が rank 0 で `a`・`b` が rank 1 以上なら `pred` を分岐の shape にブロードキャストする（`if` の条件も同じ）。`pred` が Lisp のブール値 `t`／`nil` なら、ふつうの `if` と同じく片方の分岐を静的に選び `select` をトレースしない（結果の shape・dtype は同じ真偽の rank 0 の `pred` と同じ）。それ以外のふつうの Lisp の値は `tracing-error`）, `stop-gradient`（issue #80。値は引数そのままで、自動微分では定数として扱われる＝接線がゼロ。任意の dtype を通し、配列には中身の等しい新しい配列を返す。StableHLO では `stablehlo.optimization_barrier` になる）。数値・rank 0 の値は、もう一方の分岐・オペランドの shape・dtype に合わせて自動でブロードキャストされる（それ以外の shape の不一致は `primitive-error`）。`where`（および `if`/`select` への書き換え）の両方の分岐が数値だと dtype を決められず `tracing-error` になる。

**畳み込み**（`src/primitives/conv.lisp`、`src/array-api.lisp`、issue #197）: `conv-general-dilated`（総称関数。JAX の `conv_general_dilated` 相当。`(conv-general-dilated lhs rhs &key strides padding lhs-dilation rhs-dilation feature-group-count)`）。配置は Flax と同じ channels-last に固定: lhs `[N, 空間..., C]`（NHWC）、rhs `[空間..., C/G, O]`（HWIO）、出力 `[N, 空間..., O]`。空間次元は 1 つ以上で任意（rank 3 以上）。`strides` / `lhs-dilation` / `rhs-dilation` は空間次元ごとの正の整数のリスト（既定は 1）、`padding` は空間次元ごとの `(low high)`（明示的な値。負も可。既定は `(0 0)`。`SAME` / `VALID` のような名前は無い）、`feature-group-count` は G（既定 1）。float dtype の同じ dtype の2入力だけ。整数 dtype、サイズ 0 の次元を持つ入力（N=0 も。JAX は受け付けるが v1 では拒否）、出力の空間サイズが 0 以下になる形、不正なパラメタは `primitive-error`。配列とトレーサを混ぜると配列を定数としてリフトする。プリミティブ名は `:convolution`（StableHLO の `stablehlo.convolution`）。`grad` / `vmap` / `jvp` を通せる（issue #203。`src/ad/rules-conv.lisp`、`src/ad/rules-batch-conv.lisp`。JAX の `_conv_general_dilated_transpose_lhs` / `_rhs` / `_conv_general_dilated_batch_rule` を channels-last 固定の配置に写したもの）。jvp は lhs と rhs の双線形（`conv(dl, r) + conv(l, dr)`）。lhs の転置はカーネルを空間軸で反転（内部のプリミティブ `:rev`、`stablehlo.reverse`）し入出力チャネルを入れ替えた畳み込み、rhs の転置は lhs のバッチ軸とチャネル軸を入れ替えて ct をカーネルとみなした畳み込みで、畳み込みの前後に明示的な `transpose` / `reshape` が入る。`feature-group-count` G > 1（depthwise を含む）の rhs の転置は、内部のパラメタ `:batch-group-count`（B。既定 1。lhs のバッチ軸を連続した B 個の塊に分け、出力チャネルの B 個の塊がそれぞれ1つの塊を読む。N と O が B で割り切れ、G と同時に 1 より大きくできない。`conv-general-dilated` のキーワードには出さない）が G の畳み込みになる。バッチ化ルールはバッチ軸を feature group / batch group / 出力チャネルに畳んで1回の畳み込みにする。2階微分（`grad` の `grad`）も通る。bf16 / f16 の畳み込みは、IREE 3.11 のコンパイラのバグ（f32 に直した畳み込みで片方の入力の dtype が元に戻る。`docs/iree-upstream-bugs.md` の 4、5）を避けるため、f32 に直した2入力を `optimization_barrier` に通してから畳み込む（`jit(grad)` の lhs の勾配と O/G = 1 の depthwise の bf16 / f16 が IREE で動く）。両方の入力が線形な転置は `autodiff-error`。内部のプリミティブ `:rev`（`dims` の軸を反転する。jvp / transpose / バッチ化ルールを持つ）を新しく足した。

**自動微分のコンディション**（`src/ad/rules.lisp`、issue #77）: `autodiff-error`（自動微分の変換が続けられないときの親。ルールの無いプリミティブは子の `no-jvp-rule` / `no-transpose-rule` で、`no-jvp-rule-name` / `no-transpose-rule-name` がプリミティブ名を返す）。jvp / transpose ルールの宣言（`def-jvp-rule` など）と symbolic zero は内部 API で、公開していない。

**grad / value-and-grad**（`src/ad/grad.lisp`、issue #86）: `grad`（`f &key argnums`。`with-tracing` が作った関数、`#'name`（`defjit` が定義した関数）、`jit` した関数を `f` に渡せる（`jit` は `:static-args` の無いものだけ。中の `traceable-function` だけを使い、その `:backend` は無視される。backend 上で動かすには外側を `(jit (grad f) :backend b)` にする）。スカラー出力についての `f` の勾配を返す関数を作る）, `value-and-grad`（同じ引数。多値の `(値 勾配)` を返す）, `grad-requires-scalar-output`, `grad-requires-scalar-output-aval`（`autodiff-error` の子。`f` の出力が rank 0 の浮動小数点ちょうど1つでないときに signal する。`-aval` は問題の出力の aval）。`argnums` は微分する引数の位置（0始まりの整数、または整数のリスト。既定は 0）で、整数なら勾配1つ、リストならリスト（`argnums` の順。JAX と同じ）を返す。引数は PyTree でもよく（#199）、勾配はその引数と同じ treedef の木になる（`(grad (lambda (model) …))` でモデル全体の勾配が取れる）。木の中の葉は配列・実数・トレーサで、微分する引数の葉はすべて浮動小数点でなければならない。`f` の出力は rank 0 の浮動小数点を木で包まずに1つ返す（`has-aux` 相当は未対応。葉1つでもリストで包むと `grad-requires-scalar-output`）。範囲外・重複・整数でない `argnums` と、微分する引数が浮動小数点でないとき（`:i1` など）は `autodiff-error`。勾配の shape・dtype はその引数と同じ。戻り値は `traceable-function` なので、配列・実数を渡して直接呼べる（eager。実数は rank 0 の配列として扱い、`double-float` は `:f64`、それ以外は `:f32`）ほか、`(jit (grad f))`、`with-tracing` / `defjit` の本体の中、別の `grad` の対象（`(grad (grad f))` の高階微分）にもできる。トレース中の呼び出しは、微分した graph を呼び出し元のトレースへ展開する。`argnums` がリストのときの勾配のリストも `jit` の出力にできる（`jit` の戻り値は PyTree。#199）。`value-and-grad` の `(値 勾配)` は `(multiple-value-bind (v g) (funcall vg ...) ...)` で受けられる。既知の制限: `f` が外側のトレースのトレーサを閉包で捕まえていると `tracing-error` になる（外側の値は `f` の引数として渡す）。`(grad f)` は呼ぶたびに新しい関数オブジェクトを作り、`jit` のキャッシュは関数の同一性が鍵なので、**ループの中で `(jit (grad f))` を作ると毎回コンパイルされる**（JAX と同じ）。ループの外で `(jit (grad f))` を1回だけ作るか、`defjit` の本体の中で `grad` を使うこと。

**StableHLO テキスト emitter**（`src/stablehlo.lisp`、issue #33）: `emit-stablehlo`（graph を、無名の module の中に1つの `func.func`（既定名 `main`）を持つ StableHLO テキストに変換する。`backend-compile` にそのまま渡せる。各 eqn の出力行には `loc("eqn-N")` が付く）, `primitive-not-emittable`, `primitive-not-emittable-name`（`:emit` を持たないプリミティブに当たったときに signal する）

**jit**（`src/jit.lisp`、issue #34）: `jit`（`with-tracing` / `defjit` が作った `traceable-function` を、呼ぶたびに必要なら1回だけコンパイルしてから実行する関数にする。`:static-args` で0始まりの引数位置を静的引数に指定でき、`:backend` で使う backend（`nil` なら `*default-backend*`）を指定できる。呼び出し時の動的引数は PyTree（#199。リスト・`simple-vector`・`defpytree` で登録した型を入れ子にしたもの）でもよく、その葉は CL の配列（`array-aval` で aval を推論する）・device array・Lisp の実数（rank 0 の配列として扱う。`double-float` は `:f64`、それ以外は `:f32`）のどれでもよい（要素型 `t` の `simple-vector` は配列ではなく PyTree の節なので、葉に分かれる。Lisp の整数の葉は `:f32` の rank 0 になり、`3d0` を `:f32` の配列と並べると dtype の不一致になる。配列・device array・実数以外の葉は `jit-error`）。**jit した関数の入力（引数の木の葉）が 162 個以上だと、IREE 3.11 の local-task device は落ちる**ので、`nabla.iree` は呼ぶ前に `iree-too-many-inputs` を signal する（回避策は `(make-backend :iree :driver :local-sync)`。Adam の状態を含む大きなモデルは葉が多いので、パラメータをまとめるか local-sync を使う。`docs/iree-upstream-bugs.md` の 6）。bf16 / f16 は生の `(unsigned-byte 16)` 配列のままでは dtype を推論できないため、先に `to-device` で device array にしてから渡すこと。戻り値は元の関数の戻り値と同じ形になる（葉は常に host 配列。値を複数返す関数は多値、木を1つ返す関数は同じ形の木。木を返す関数と `(values …)` で返す関数は区別される）。IREE の `local` backend は f64 の `exp` / `log` / `tanh` を libm の呼び出しとして残すので、これらを含む関数のコンパイルにだけ PATH 上の `ld.lld` が要る（`docs/stablehlo-ops.md`）), `jit-error`（`jit` / jit した関数の呼び出しが誤った使い方を検出したときに signal する）, `*default-backend*`（`jit` に `:backend` を渡さなかったときに使う既定の backend。`nil`・backend の KIND（キーワード）・backend インスタンスのいずれか。`nabla/iree` をロードするとまだ未設定のときに限りこの変数を自分の KIND に設定する）。jit キャッシュは関数の同一性（EQ）・動的引数の葉の aval と動的引数の treedef・静的引数の値・backend（フィンガープリント込み）をキーにする（aval が同じでも木の形が違えば別のコンパイル。同じ treedef・aval の呼び出しは再コンパイルしない）インメモリのキャッシュで、vmfb のディスクキャッシュ（`*compile-cache-directory*`）とは別の層（`docs/glossary.md` の「インメモリのコンパイルキャッシュ」参照）。

`defjit`（`name-and-options (&rest lambda-list) &body body`。マクロ）は `body` を `with-tracing` でトレース対象にしてから `jit` した、通常の関数として呼べるものを `name` に定義する。`name-and-options` は `name` か `(name :static-args positions)` で、`positions` は評価されて `jit` の `:static-args` と同じ意味になる（例: `(nb:defjit (f :static-args '(1)) (a axis) ...)`。ドキュメント文字列は未対応）。同じ `defjit` フォームを再評価するたびに新しい `traceable-function` を作り直し、古いキャッシュエントリはその場で捨てて、読み込んだモジュールを `backend-unload` する（Lisp の関数を再定義したときの直感どおり、古いキャッシュを使い続けない）。`jit` に渡した関数が GC で回収されたときも、そのキャッシュのモジュールは finalizer で `backend-unload` される。キャッシュのロックは関数ごとで、トレース・コンパイルの間は持たないので、別々の関数の jit は複数のスレッドから並行してコンパイルでき、同じ関数・同じキーへの同時の呼び出しは1回だけコンパイルして結果を共有する。jit した関数を別の関数のトレース中にトレーサを引数にして呼ぶと、コンパイルせずに呼び出し元の graph に展開する（JAX の jit の入れ子と同じ）。`jit-compile-error`（`jit-error` のサブタイプ。`backend-compile` / `backend-load` が `backend-error` を signal したときに、失敗した `graph` と、可能なら原因の eqn（`jit-compile-error-eqn` / `-eqn-index`。見つからなければ NIL）を添えて signal する）, `jit-compile-error-condition`（元の `backend-error`）, `jit-compile-error-graph`, `jit-compile-error-eqn`, `jit-compile-error-eqn-index`。`jit-compile-error` を待ち受けるキャッシュミスの経路には2つのリスタートがある: `use-eager`（この呼び出しだけ `eval-graph` で eager 実行し、何もキャッシュしない）, `recompile`（もう一度コンパイルをやり直す。再帰ではなくループでやり直すので、ハンドラが何度選んでもスタックは深くならない）。`use-eager` はコンパイルに失敗した graph をそのまま評価し、本体をトレースし直さない。使い方は `(handler-bind ((nb:jit-compile-error (lambda (c) (invoke-restart 'nb:use-eager)))) (funcall jitted ...))` のように `invoke-restart` で選ぶ。




**整数 dtype**（`src/dtype.lisp`、`src/primitives/*.lisp`、issue #126）: `:i32`（Lisp では `(signed-byte 32)`、StableHLO では `i32`）、`:u32`（`(unsigned-byte 32)`、`ui32`）、`:u64`（`(unsigned-byte 64)`、`ui64`。THREE_FRY の状態 `ui64[2]` のために足した）。`add` `sub` `mul` `max` `min` `neg` `compare` `select` `convert` `broadcast-in-dim` `reshape` `transpose` `reduce-sum` `reduce-max` は整数を受け付け、`div` `exp` `log` `tanh` `sqrt` `rsqrt` `dot-general` は整数を `primitive-error`（トレース時）で拒否する。整数どうしの演算は dtype が一致していなければならず（暗黙の型昇格はしない）、eager の算術は StableHLO と同じく 2 の補数（符号なしは法 2^n）で折り返す。`convert` は整数 ⇔ 浮動小数点 ⇔ `:i1` を変換できる（浮動小数点 → 整数は 0 方向へ丸め、NaN は 0、範囲外（±無限大を含む）は整数の端に飽和。これはどのバックエンドでも同じになるよう、出力する StableHLO が clamp と select で飽和と NaN → 0 を明示する。整数 → `:bf16` は f32 を経由する2段の convert。→ `:i1` は 0 以外が真）。`with-tracing` の中の整数リテラルは相手の整数 dtype にリフトされる（浮動小数点のリテラルや範囲外の整数は `tracing-error`。浮動小数点のトレーサと組むリテラルは従来どおり `:f32`）。整数の接線は変換の中では常に symbolic zero で、整数の入力に対する `grad` は `autodiff-error`、整数を経由する計算の勾配はゼロ。IREE と PJRT（CPU）の `to-device` / `to-host` は整数に対応する。新しい export はない。




#### vmap（issue #125）

`(vmap f &key (in-axes 0) (out-axes 0))` は、`f`（`with-tracing` / `defjit` / `jit` が作った関数。`jit` は静的引数なしのもの）を、引数のバッチ軸についてまとめて適用する関数を返す。結果は「バッチ軸で切り出した各要素に `f` を適用して、出力の `out-axes` の位置に積み直したもの」と一致する。`in-axes` は引数ごとの軸（0始まり、負なら末尾から）か `nil`（その部分木はバッチせず `f` にそのまま渡す）で、整数か `nil` を1つ渡すと全引数の全葉に共通、リストなら引数ごと。引数と戻り値は PyTree（#200）で、`in-axes` / `out-axes` は木の**接頭辞**として書ける（JAX の `flatten_axes` と同じ。規則は下の箇条書き）。例: `(vmap (grad loss) :in-axes '(nil 0 0))` は、パラメータの木（`defmodule` のモデルなど）を共有して x と y だけを軸 0 でバッチし、サンプルごとの勾配をモデルと同じ木（各葉の先頭にバッチ軸）で返す。

接頭辞の規則:

- 整数・`nil` は部分木まるごとに効く。
- リスト・`simple-vector`・`defpytree` / `defmodule` で登録した型の節は、種類と子の個数（登録した型は補助データが `equal` なことも）が木と同じで、子をそれぞれ接頭辞として指定する。合わなければ `vmap-error`（呼び出し時に検査。葉が整数・`nil` 以外、整数・`nil`・リスト以外の `in-axes`、`in-axes` のリストの個数が引数の個数と違う場合は `vmap` を呼んだ時点）。
- `vmap` は `in-axes` / `out-axes` を複写して持つので、後から書き換えても影響しない。
- `out-axes` は `f` の戻り値（多値は「値のリスト」という1つの木）の接頭辞で、出力がバッチに依存しないときだけ `nil` にでき（依存しない出力に整数を渡すと複製する）、`f` が `(list a b)` を返すなら `((軸a 軸b))`、多値 `(values a b)` なら `(軸a 軸b)` と書く。

戻り値はトレースできる関数なので、配列を渡して直接呼ぶ（eager）ほか、`(jit (vmap f))`、`with-tracing` の中、`grad` の対象、別の `vmap` の対象（`(vmap (vmap f))`）として使える。バッチされていない入力だけの演算は、バッチ化ルールを呼ばずにそのまま残る。既知の制限は `grad` と同じ（`f` が外側のトレーサを閉包で捕まえると `tracing-error`）。`(vmap f)` は呼ぶたびに新しい関数オブジェクトを作るので、ループの中で `(jit (vmap f))` を作ると毎回コンパイルされる（jit キャッシュのキーが関数の同一性のため。ループの外で1回だけ作る）。`cond*` / `while-loop` / `scan` を含む `f` も `vmap` できる（下の「制御構造のバッチ化」）。

コンディションは `vmap-error`（親。`in-axes` / `out-axes` の個数・型・範囲の不正、軸長の不一致、バッチされた引数が無い、など）と、その子の `no-batch-rule`（バッチ軸を持つ値がバッチ化ルールの無いプリミティブに渡った。`no-batch-rule-name` がプリミティブ名）。バッチ化ルールは `def-batch-rule`（内部。`src/ad/rules-batch-*.lisp`）で書く。全プリミティブがバッチ化ルールを持つ（要素演算は #128、形状・縮約・`dot-general` は #129、制御構造は #140）。




#### 要素演算のバッチ化ルール（issue #128）

`add sub mul div neg exp log tanh sqrt rsqrt max min compare select convert stop-gradient` は共通のバッチ化ルール1つ（`src/ad/rules-batch-elementwise.lisp`）を持つ。バッチされた引数のバッチ軸がすべて同じ位置ならそのまま元の演算を適用し（`transpose` も `broadcast-in-dim` も足さない）、位置が違うときは先頭へ `transpose` で揃える。バッチされていない引数には `broadcast-in-dim` でその位置に軸を足す（要素演算は形が全引数で一致する前提で、暗黙のスカラー拡張は無い）。`select` は条件だけがバッチされる場合も扱う。




形状演算・縮約・`dot-general` のバッチ化ルール（`src/ad/rules-batch-shape.lisp`）: `broadcast-in-dim` / `reshape` / `transpose`（バッチ軸を先頭に置く）、`reduce-sum` / `reduce-max`（縮約軸をずらし、出力のバッチ軸は自然な位置）、`dot-general`（片側だけバッチなら自由次元、両側なら新しい batch 次元の先頭。出力のバッチ軸は自然な位置）。公開 API の追加は無い（`vmap` を通して使う）。




**cond\*（条件分岐）**（`src/cond.lisp`、`src/primitives/cond.lisp`、issue #130）: `(cond* pred then-fn else-fn &rest operands)`。`pred` が真なら `(then-fn operands...)`、偽なら `(else-fn operands...)` を評価する高階プリミティブ `:cond` で、eager では選ばれた枝のサブグラフだけを評価する（CL の `cond` と衝突するので名前は `cond*`）。`then-fn` / `else-fn` は `with-tracing` で作った関数で、普通のトレース対象の関数と同じく1つの値か多値を返し、`cond*` も同じ個数の多値を返す。operands の各引数と枝の戻り値は PyTree でよい（issue #201。内部では葉に flatten して `:cond` の eqn にし、結果を枝の戻り値の木に戻す。リストは既定の節なのでこれまでの呼び出しはそのまま動く）。両枝の戻り値の木の形（treedef）か葉の aval が一致しなければトレース時に `cond-error`（`tracing-error` の子。`pred` がトレーサなのに rank 0 の `:i1` でない場合、枝が `traceable-function` でない場合も）。枝が閉包で捕まえた外側のトレーサも使える（closure conversion。両枝は同一の入力シグネチャ「operands と両枝の捕捉値の和集合」を持ち、片方が使わない捕捉値の位置には使われない入力が置かれる）。operand はトレーサ・実数・dtype を推論できる配列（`:f32` / `:f64` / `:i1` / `:i32` / `:u32` / `:u64`。bf16 / f16 の生の配列はトレーサで渡す。違えば `cond-error`）。枝の引数の個数が operand の個数と違うときも `cond-error`。`pred` が `t` / `nil` / rank 0 の bit 配列なら、選ばれた枝をそのまま呼ぶ（eqn は作らない）。StableHLO は `stablehlo.if`（IREE でコンパイル・実行できることを medium テストで確認）。jvp / transpose / grad のルールは #134（`src/ad/rules-control.lisp`）、バッチ化ルールは下の「制御構造のバッチ化」。**`with-tracing` の `if` は `cond*` に落とさず、これまでどおり `select` のままにする**: `if` の条件は要素ごとの `:i1` 配列でありうるので、`select`（要素ごとの意味）を保つ必要がある。スカラー条件で片枝だけを評価したい（重い計算や、範囲外の値の `log` など片方の枝でしか意味を持たない計算を避けたい）ときに、`cond*` を明示的に呼ぶ。



**`while-loop`**（`src/while-loop.lisp`、issue #131）: `(while-loop cond-fn body-fn init)` は、`cond-fn` が真の間 `body-fn` を繰り返して最後の carry を返す（JAX の `lax.while_loop`）。`init` は PyTree（リスト・ベクタ・`defpytree` / `defmodule` の型の入れ子、配列そのものでもよい。issue #201）で、葉は配列（トレース中はトレーサでもよい）を1つ以上持つ。`cond-fn` / `body-fn` は carry の木を1つ受け取る関数（`with-tracing` で作る）。`cond-fn` は rank 0 の `:i1` を返し、`body-fn` は `init` と同じ木の形（`treedef=`）・同じ葉の aval（shape・dtype）の木を返す。結果は `init` と同じ木の形。配列だけで `with-tracing` の外から呼べば eager（Lisp のループ）、`with-tracing` / `jit` の中では `:while-loop` の eqn になり StableHLO の `stablehlo.while` で出る。`cond-fn` / `body-fn` は外側のトレーサを閉包で捕まえてよく、捕まえた値は loop 不変の追加のオペランドになる。エラーは `while-loop-error` の子: `while-loop-argument-error`（`init` / `body-fn` の戻り値が PyTree として不正、`init` に葉が無い・葉が配列でない、関数でない）、`while-loop-carry-mismatch`（`body-fn` の戻り値の木の形か葉の aval が `init` と違う。木の形が違うときもメッセージに両方の treedef が出る。トレース時に検出し、0回で終わるループでも出る。`while-loop-carry-mismatch-expected` / `-actual`）、`while-loop-condition-error`（`cond-fn` の結果が rank 0 の `:i1` でない）。前進モードの jvp には対応する（`src/ad/rules-control.lisp`、issue #134。接線を持つ carry の接線を carry に足した `while-loop` にする。最初は接線がゼロの carry も本体を通ると非ゼロになりうるので、JAX と同じく不動点まで広げる）。逆モードの `grad` は対応しない（反復回数が分からず残差を保存できない）。`grad` が通ると、プリミティブ名 `:while-loop` を含む `autodiff-error` になる。




#### scan（issue #132）

`(scan f init xs &key length reverse)` は、`xs` の先頭の軸に沿って `f` を回し、`(values 最終の carry ys)` を返す（JAX の `lax.scan` 相当）。`init` / `xs` / `ys` は PyTree（リスト・ベクタ・`defpytree` / `defmodule` の型の入れ子。issue #201）で、`xs` の葉はどれも先頭の軸で走査する。`f` は `with-tracing` で作った2引数の関数 `(carry x)` で、`x` は `xs` の各葉の先頭の軸を外した同じ形の木、`(values 新しい carry y)` を返す。`ys` は `y` と同じ木の形で、各葉を先頭の軸に積んだもの。carry は `init` と木の形（`treedef=`）も葉の shape・dtype も同じでなければならない（違うと `scan-carry-mismatch`）。内部ではこれまでどおり葉のフラットなリストで扱う。`xs` が空のときは `length` が必須で、そうでなければ `xs` の先頭の軸の長さと一致しなければならない（`scan-length-error`）。長さ 0 の scan は `init` と同じ値の carry を返し、`ys` は先頭の軸が 0 の空の配列になる。eager の結果は `init`・`xs`・`f` が閉包で捕まえた配列と `eq` でありうる（上の「配列の不変性」）。`reverse` が真なら添字 `length-1` から 0 へ辿る（`ys[t]` にはそのときも添字 `t` のステップの `y` が入る）。`f` が閉包で捕まえた外側の値はループ不変な入力（consts）になる。eager でも `with-tracing` / `jit` の中でも使える。引数や `f` の戻り値が PyTree として不正なとき（`init` / `xs` に生の `(unsigned-byte 16)` の bf16 / f16 配列を渡したときも。トレーサで渡す）は `scan-error`（親）。順方向・jvp（#135、下記）・`grad`（#139、下記）に対応し、`vmap`（#140、下の「制御構造のバッチ化」）にも対応する。StableHLO では `:i32` のカウンタを持つ `stablehlo.while` に落ちる。




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
| `(truncated-normal key shape &key dtype lower upper)` | `[lower, upper]`（既定 `[-2, 2]`。どちらも `[-5, 5]` の実数。それより外では f32 の `erf(upper/√2)` が 1 に丸まり erf の逆関数が無限大になるため）に切った標準正規分布の乱数（issue #204。JAX の `truncated_normal` と同じ方式。Flax の `lecun_normal` に要る） |
| `prng-error` | 不正な引数のコンディション |

eager でも `jit` / `grad` / `vmap` の中でも使える。`vmap` でキーをバッチすると各要素はそのキーで単独に呼んだ結果とビット単位で一致する（`(vmap (with-tracing (k) (uniform k '(3))))` を `(split key 8)` に適用する、など）。

決めたこと:

- **キー → 状態**: キー `[k0 k1]` を `rng-bit-generator` の状態 `ui64[2] = [k0 | k1 << 32, カウンタ]` にする（`bitcast-convert` で2語を1語にまとめる）。`uniform` / `normal` / `split` はカウンタ 0 から引き、`fold-in` はカウンタ `2^32 + data` の2語を新しいキーにする（引く量が 2^32 要素未満なら、`fold-in` の出力が `uniform` / `split` の列と重なることはない）。
- **JAX とビット単位では一致しない**: JAX の既定は `threefry_2x32` を直接呼ぶ実装で `rng_bit_generator` を使わないため、同じシードでも値が違う。nabla は `stablehlo.rng_bit_generator`（THREE_FRY）を使い、IREE・PJRT・eager が互いにビット単位で一致する（`docs/stablehlo-ops.md`）。分布としては同じ（統計検定で確かめている）。
- **uniform**: 乱数ビットの仮数部だけを取り出して `[1, 2)` の浮動小数点数にし、1 を引いて範囲に伸ばす（JAX と同じ。`:f32` は 23 ビット、`:f64` は 52 ビットの粒度）。丸めのため `maxval` にちょうど等しい値が出うる。
- **normal**: JAX と同じく、`(-1, 1)` の一様乱数に erf の逆関数をかけて √2 倍する。Box–Muller は `sin` / `cos` のプリミティブが無いため採らなかった。erf の逆関数は Giles の多項式近似で、`:f32` は単精度用（JAX の f32 と同じ係数）、`:f64` は倍精度用（XLA の f64 と同じ係数）。相対誤差の実測は f32 で最大 2.0e-7、f64 で最大 4.9e-16（多倍長の参照値と比較）。値の絶対値の上限は f32 で約 5.42、f64 で約 8.29。
- **バッチ化**: バッチ次元を持つ状態 `ui64[..., 2]` を `rng-bit-generator` が受け付け、各行は単独に呼んだ結果とビット単位で一致する。StableHLO の `rng_bit_generator` は `ui64[2]` しか受けないので、`stablehlo.while` の1回で 16 行ずつ `dynamic_slice` → 16 回の `rng_bit_generator` → `dynamic_update_slice` を回して出力する（issue #164 / #178。16 行以下なら while を出さずに展開する。行数が 16 を超えれば出力する StableHLO の大きさは行数に依らない）。**コンパイルコスト**（IREE local、バッチされた rng の eqn 1つ）: 32 行・256 行とも約 1.0 秒（1行ずつ回していたときは約 0.5 秒、全行を展開していた以前は 32 行 4.0 秒、256 行 42.8 秒）。**実行時間**: ビットのバッファは `scan` の ys と同じく `optimization_barrier` を通して in-place に書く（#159）ので、バッファ全体のコピーは無い。残る費用はループ1回（16 行）ごとの起動で、IREE では行数 / 16 に比例する（4コアの機械で `vmap` した `uniform` の jit 1回が、256 キー × 4 要素で約 4 ms、1024 要素で約 7 ms、64 キー × 16384 要素で約 12 ms。1行ずつ回していたときは約 46 / 52 / 21 ms、全行を展開していたときは約 1 / 3 / 8 ms）。PJRT（XLA CPU）では遅くならない。
- 新しいビット演算プリミティブ（内部）: `:shift-right-logical`、`:bitwise-or`（整数専用の2入力の要素演算）と、ビット列を再解釈する `:bitcast-convert`（`:f32 :f64 :i32 :u32 :u64`。幅が違うときは StableHLO と同じく末尾の次元が増減し、並びはリトルエンディアン）。どれも整数・ビット列の演算なので微分しない。





#### PyTree（issue #195）

リスト・`simple-vector`・`defpytree` で登録した型を入れ子にした木（PyTree†）の操作。フェーズ4で `jit` / `grad` / `vmap` / 制御構造の引数と戻り値を PyTree に一般化する土台（`jit` / `grad` / `value-and-grad` は #199、`vmap` は #200 で対応済み、制御構造は #201）。

| 関数 | 役割 |
| --- | --- |
| `(tree-flatten tree)` | 葉を左から順に並べたリストと treedef の2値 |
| `(tree-unflatten treedef leaves)` | treedef の形の木を葉のリストから作る（葉の個数が違えば `pytree-error`） |
| `(tree-leaves tree)` / `(tree-structure tree)` | `tree-flatten` の1つ目 / 2つ目の値 |
| `(tree-map function tree &rest more-trees)` | 葉ごとに `function` を呼んだ同じ形の木。複数の木は同じ位置の葉を引数にし、形が違えば `pytree-structure-mismatch` |
| `(defpytree type :flatten f :unflatten g)` | クラス名 `type` を節として登録する。`f` は `(object) → (values 子のリスト 補助データ)`、`g` は `(aux children) → object` |
| `treedef` / `treedef-num-leaves` / `treedef=` | 木の形（不透明な構造体）、その葉の個数、形の比較 |
| `pytree-error` / `pytree-structure-mismatch` | 不正な木・引数のコンディション / その子で、形の不一致 |

決めたこと:

- **節はリストと `simple-vector`（要素型 `t`）だけ**。型特化した配列（`(simple-array single-float (3))` などの1次元の配列も）・文字列・数値・トレーサは葉。nabla の配列は要素型 `t` を使わないので、配列の葉とベクタの節が紛れない
- **`nil` は空の部分木**（葉にならない）。真リストでないコンス（`(1 . 2)`）は `pytree-error`
- **登録の無い型はすべて葉**（JAX と同じ）。plist / alist / ハッシュ表や構造体を節にするには `defpytree` で明示的に登録する。登録はクラス名で引き、サブクラスには継承しない。リスト・ベクタなど組み込みの型は登録できない
- **補助データは EQUAL で比べられる値**（数値・文字・シンボル・文字列・パス名と、それらのコンス）に限る（treedef に入り、後で `jit` のキャッシュのキーになるため）。違えば `tree-flatten` が `pytree-error`
- `tree-unflatten` はリストとベクタの節を新しく作り、葉はそのまま（`eq` で）置く

#### defmodule（`nabla/nn`、issue #202）

Linen / Equinox 風の「モジュール = パラメータの PyTree ＋ 純粋関数」を書く。`(defmodule name (slot...) clause...)`:

```lisp
(nabla.nn:defmodule dense ((w :param) (b :param) (act :static))
  (:init (key in out act)
   (destructuring-bind (kw kb) (nabla.nn:split-keys key 2)
     (make-dense :w (nabla:normal kw (list in out)) :b (nabla:uniform kb (list out)) :act act)))
  (:call (self x) ...))
(init-dense (nabla:prng-key 0) 3 4 :relu)    ; => #S(DENSE ...)
(nabla.nn:call layer x)
```

| 名前 | 役割 |
| --- | --- |
| `defmodule` | 構造体 `name`・コンストラクタ `make-name`（スロットのキーワード引数、すべて必須）・アクセサ `name-slot` を作り、PyTree に登録する。`(:init (key . ll) body)` があれば関数 `init-name`、`(:call (self . ll) body)` があれば `call` のメソッド |
| `call` | モジュールを引数に適用する総称関数 |
| `split-keys` `(key n)` | PRNG キーから独立な `n` 個のキーのリスト（eager 専用）。サブモジュールの init に分けて渡す |
| `map-state` `(function tree)` | `:state` スロットの葉すべてに `function` をかけた同じ形の木。`(map-state #'nabla:stop-gradient model)` で状態を `grad` から外す |
| `module-slots` `(name-or-instance)` | 宣言順の `(スロット . 種類)`。チェックポイントのパス名（#206）用 |

- スロットの種類: `:param`（省略時。学習する葉・サブモジュール・そのリスト）、`:state`（学習しない状態。葉だが `map-state` で扱う）、`:static`（層の幅などのハイパーパラメータ。葉でなく treedef の補助データ。`equal` で比べられる値に限る）。葉は `:param` / `:state` の宣言順
- `:call` の本体は `with-tracing` の本体として書く（コードウォークされる。引数リストは必須引数のシンボルだけ、`setq` 禁止、`self` は閉包で捕まえる）ので、`+` `*` や `nb:dot` が配列にもトレーサにも使え、`grad` / `vmap` / `jit` の中で呼べる。`(:init ...)` と `(:call ...)` の節はそれぞれ1つまで。述語 `name-p` と複写関数は作らない。`:static` の値は `equal` で比べられるものに限る（関数は不可。活性化関数はシンボルで持つ。`1e-5` と `1d-5` は別の treedef）。`defmodule` を再評価するときはフォーム全体を評価し直す（構造体だけ再定義すると古い `call` メソッドが残る）
- 呼び出しの名前は `call`。`apply` は CL のシンボルで、`call` は CL に無い（`call-method` だけ）ので衝突しない
- `init` はキーを `split-keys` で分けて渡すので、同じキーからは常に同じパラメータ。継承はサポートしない

#### 基本レイヤー（`nabla/nn`、issue #204）

`dense` / `layer-norm` / `embedding` / `dropout` / `sequential` / `conv` を `defmodule` で作ったもの。初期化と計算は Flax Linen の既定に合わせ、スロット名（チェックポイントのパス名になる）も Flax のパラメータ名と同じ。init は `init-<name>` で、第1引数が PRNG キー。

| モジュール | スロット | init | 呼び出し |
| --- | --- | --- | --- |
| `dense` | `kernel (in out)`, `bias (out)`, `use-bias`（`:static`） | `(init-dense key in out &key dtype use-bias)`。kernel は `lecun_normal`（切断正規分布、分散 1/in）、bias は 0。`use-bias` 既定は真（Flax の `use_bias`） | `(call m x)` = `x kernel + bias`（x は rank 1 以上、最後の軸が in）。bias が `nil` なら足さない |
| `layer-norm` | `scale (f)`, `bias (f)`, `epsilon` / `use-scale` / `use-bias` / `reduction-axes` / `feature-axes`（`:static`） | `(init-layer-norm key features &key epsilon dtype use-scale use-bias reduction-axes feature-axes)`。scale 1、bias 0、epsilon 既定 `1f-6`（Flax と同じ）。`use-scale` / `use-bias` 既定は真、`reduction-axes` / `feature-axes` 既定は `-1`（Flax の `use_scale` `use_bias` `reduction_axes` `feature_axes` と同じ。整数かそのリスト）。`features` は `feature-axes` が 1 つなら整数、複数ならその軸の大きさのリスト | `(call m x)`。`reduction-axes` の軸で正規化（既定は最後の軸）。分散は Flax と同じ `E[x^2] - E[x]^2`（0 で切る） |
| `embedding` | `embedding (vocab features)` | `(init-embedding key vocab features &key dtype)`。`N(0, 1/features)` | `(call m ids)`（整数の配列）。`[ids... features]` を返す。`(embedding-attend m query)`（Flax の `Embed.attend`）は `query [... features]` と埋め込みの転置の内積で `[... vocab]` の logits を返す（出力の層と埋め込みの重みの共有。勾配は引きと attend の和が同じ葉に入る） |
| `dropout` | `rate`（`:static`） | `(init-dropout key rate)`（rate は 0 以上 1 以下） | `(call m x key deterministic)`。反転 dropout（保持した要素を `1/(1-rate)` 倍）。`deterministic` が真なら `x` そのもの |
| `sequential` | `layers`（モジュールのリスト） | `(init-sequential key init-functions)`。各関数に `split-keys` で分けたキーを渡して層を作る | `(call m x)` = 各層を順に適用 |
| `activation` | `fn`（`:static`。`:relu` `:gelu` `:tanh` `:softmax`） | `(init-activation key fn)` | `(call m x)`。`sequential` の層として活性化関数を並べる |
| `conv` | `kernel (空間... in/G out)`（HWIO）, `bias (out)`, `strides` / `padding` / `kernel-dilation` / `feature-group-count` / `use-bias`（`:static`） | `(init-conv key in out kernel-size &key strides padding kernel-dilation feature-group-count dtype use-bias)`。`kernel-size` は空間の次元ごとのリスト（`(3)` で1次元、`(3 3)` で2次元、`(3 3 3)` で3次元。Flax の `nn.Conv` と同じ N 次元）、`strides` と `kernel-dilation` は整数か次元ごとのリスト（既定 1）、`padding` はキーワードの `:same`（既定）/ `:valid` か次元ごとの `((low high) ...)`（Flax の文字列 `"SAME"` は受け付けない）、`feature-group-count` 既定 1、`use-bias` 既定は真。kernel は `lecun_normal`（fan_in = kernel の面積 × in/G）、bias は 0 | `(call m x)`。x は `[N 空間... C]`（2次元なら NHWC）かバッチ軸の無い `[空間... C]`。出力は `conv-general-dilated` + bias |

活性化関数は `nabla.nn` の普通の関数（配列にもトレーサにも使え、`with-tracing` の中でも書ける）: `relu`, `gelu`（Flax の既定の tanh 近似）, `(softmax x &key axis)`, `(log-softmax x &key axis)`（`axis` 既定 -1、負の値は後ろから数える）。コアのプリミティブにはせず、`max` / `exp` / `tanh` / `reduce-*` の合成で書いてある。

決めたこと:

- **embedding は gather ではなく one-hot と `dot`**: gather の勾配は scatter-add で、プリミティブ・StableHLO 出力・jvp / transpose / バッチ化ルールが要り、この issue の範囲を超える。one-hot + `dot` は既存の演算だけで、勾配は one-hot の転置との `dot` になる。代償は計算量が ids 数 × 語彙数 × features に比例すること（小さな語彙では問題にならない）。語彙が大きいモデルでは gather が要る。範囲外の id（負・語彙数以上）は 0 の行になる（Flax の `jnp.take` は NaN）
- **embedding-attend（issue #241）**: Flax の `Embed.attend` はモジュールのメソッドだが、nabla の `call` は引きの 1 つなので `embedding-attend` という関数にした。`query` の最後の軸が `features` でなければ層名入りの error。重みの共有は、同じ `embedding` モジュールを `(call m ids)` と `(embedding-attend m h)` の両方に渡す（`defmodule` のスロットに 1 つだけ持つ）。`examples/transformer.lisp` は出力の dense を埋め込みと共有していない（変えていない）
- **dropout**: モジュールは状態を持たないので、キーと `deterministic` は呼び出しの引数。`deterministic` は Lisp の真偽値で、`jit` では静的引数（`:static-args`）にするか閉包で渡す。`rate = 1` は全部 0（入力に inf / NaN があっても 0。Flax の `zeros_like` と同じ）。`deterministic` が偽でキーが `nil` なら error。`sequential` は1引数の層だけを並べる（Flax の `nn.Sequential` と同じ。dropout は並べられない）
- 初期値は Flax と分布が同じで、値はビット単位では一致しない（nabla の PRNG は JAX と違う）。Flax との比較は同じパラメータを与えて順伝播と勾配で行う
- **layer-norm の引数（issue #241）**: Flax の `LayerNorm` と同じ。`reduction-axes` は平均と分散を取る軸、`feature-axes` は `scale` / `bias` を持つ軸で、どちらも負の値は後ろから数え、重複は取り除いて昇順にそろえる（Flax の `_canonicalize_axes` と同じ。`call` のとき入力の rank で決める）。`scale` / `bias` の形は入力の `feature-axes` の大きさ（軸の昇順）で、形が合わなければ error。例: 入力 `[2 3 4]` で `:reduction-axes '(1 2) :feature-axes '(1 2)` なら `(init-layer-norm key '(3 4) ...)` で scale の形は `(3 4)`。`use-scale` / `use-bias` が偽のときは `use_bias` と同じ規則（対応するスロットが `nil` で、パス名からも消える）。`reduction-axes` と `feature-axes` は `:static` なので、`-1` と `(2)`（rank 3 の入力の最後の軸）のように同じ軸でも書き方が違えば別の treedef になる（`jit` の再コンパイルやチェックポイントのひな形の比較で区別される）。`reduction-axes` が空のもの（Flax では `x.mean(())` で退化する）は受け付けない。`axis_name` / `axis_index_groups`（pmap の軸）、`use_fast_variance=False`、`force_float32_reductions`、`mask`、`dtype` / `param_dtype` の分離は持たない（nabla に pmap が無い・速い分散が Flax の既定・bf16 / f16 の layer-norm が未対応のため）
- **conv の SAME は Flax（JAX）と同じ**: 出力の長さは `ceil(入力 / stride)`、padding の合計は `max((出力 - 1) stride + 実効 kernel - 入力, 0)`（実効 kernel = `(k - 1) dilation + 1`）で、左が合計の半分（切り捨て）、右が残り。偶数の入力や stride 2 では左右非対称（例: 入力 6、kernel 3、stride 2 は `(0 1)`）。入力の形に依存するので呼ぶたびに計算する。`vmap` の中ではバッチ軸の無い `[H W C]` の画像として書ける。`make-conv` で直接作ったモジュールの strides・padding・dilation も呼ぶたびに `init-conv` と同じに正規化する（`:strides 2` や `:padding :same` が使える）。入力の dilation（転置畳み込み）、CIRCULAR / CAUSAL padding などは持たない（下の「Flax との差」の表。CAUSAL は明示の padding で書ける）
- **`use_bias`（dense / dense-general / conv / multi-head-attention。issue #241）**: Flax と同じ引数名（`init-*` の `:use-bias`、既定は真）。`:static` のスロット `use-bias` を持ち、偽のとき `bias` スロットは `nil`（空の部分木）になる。`nil` は葉でないので、チェックポイントのパス名は変わらず（`kernel` だけになる）、Flax の「bias のパラメータが無い」と一致する。`call` は bias が `nil` なら足さない。`use-bias` と `bias` が食い違う（真なのに `nil`、偽なのにある）モジュールは `call` が層名入りの error にする。`make-*` はスロットがすべて必須（`defmodule` の規則）なので、#241 で増えたスロットは手で作るモジュールでも明示する（**破壊的変更**）: `make-dense` / `make-dense-general` / `make-conv` は `:use-bias`、`make-layer-norm` は `:use-scale` `:use-bias` `:reduction-axes` `:feature-axes`、`make-multi-head-attention` は `:broadcast-dropout`（以前のコードは `:use-bias t`、`:use-scale t :use-bias t :reduction-axes '(-1) :feature-axes '(-1)`、`:broadcast-dropout nil`（以前の挙動）などを足す）。`defmodule` の `:static` スロットに既定値を許せば、この種の変更が壊れなくなる（#261）。`multi-head-attention` は `:use-bias` を query / key / value / out の 4 つの `dense-general` すべてに渡す（Flax の `use_bias` も 4 つの `DenseGeneral` すべてに効く）。`multi-head-attention` 自身に `use-bias` のスロットは持たない（各射影が持つ）

#### Flax との差（`nabla/nn`、issue #241）

レイヤーのうち Flax と引数や挙動が違うものを、1 項目ずつ「実装した」か「やらない」かで決めた表。実装したものは上の各レイヤーの表と説明にある。やらないものは、要るようになった時点で別の Issue にする。

| Flax の機能 | 結論 | 理由・書き方 |
| --- | --- | --- |
| `use_bias`（`Dense` `DenseGeneral` `Conv` `MultiHeadDotProductAttention`） | 実装 | `init-*` の `:use-bias`。偽なら `bias` スロットが `nil` でパス名から消える |
| `LayerNorm` の `use_scale` `use_bias` `reduction_axes` `feature_axes` | 実装 | 同名の `:use-scale` `:use-bias` `:reduction-axes` `:feature-axes`。既定も Flax と同じ |
| `MultiHeadDotProductAttention` の `broadcast_dropout` | 実装 | `:broadcast-dropout`。既定は Flax と同じ真（**挙動変更**）。値は JAX とビット一致しないので統計的な性質で確かめる |
| `Embed.attend` | 実装 | `embedding-attend`（関数。`call` は引きの 1 つなので、メソッドにしなかった） |
| `Embed` の範囲外 id が NaN | 別の作業（#240） | 今は 0 の行。引きを gather に替える #240 で扱う |
| `Conv` の `input_dilation`、`ConvTranspose` | やらない | コアの `conv-general-dilated` は `lhs-dilation` を持つが、nn の層にするには padding の仕様（Flax / JAX の SAME は膨らませる前の入力の形で決め、`conv_transpose` は別の式を使う）を写してフィクスチャで確かめる別の作業になる。使い道はほぼ転置畳み込みなので、デコーダや U-Net を書く時点でまとめて足す |
| `Conv` の `padding='CIRCULAR'`（と `'REFLECT'`） | やらない | Flax は入力を `jnp.pad(mode='wrap')` で端から複写して VALID にする。nabla には slice / concatenate / pad のプリミティブが無く（畳み込みの 0 埋めだけ）、足すのは別の作業。周期的な境界の畳み込みは当面の対象外 |
| `Conv` の `padding='CAUSAL'` | やらない（明示の padding で書ける） | 1 次元で左だけ `(k-1) × dilation` を埋めるだけなので、`:padding '(((k-1)d 0))` と書けば同じ（`conv/explicit-left-padding-makes-a-causal-convolution` が因果性を確かめる）。層に別名を持たせない |
| `Conv` の文字列 padding `'SAME'` `'VALID'` | やらない | キーワード `:same` `:valid` で同じ。`:static` の値を `equal` で比べるため文字列を受け付けない（#207 の決定） |
| `Conv` の整数 padding | やらない | 明示の `((p p) ...)` で書ける |
| `BatchNorm` `GroupNorm`（`InstanceNorm`） | やらない（#260） | `BatchNorm` は `:state` の移動平均を更新して返す `call` の規約が要り、`map-state` の最初の実使用になる設計判断なので #260 で扱う。`GroupNorm` は軸の分割（`num_groups`）が要る。`LayerNorm` の `reduction-axes` / `feature-axes` で `InstanceNorm` 相当は書ける |
| `LayerNorm` の `axis_name` `axis_index_groups`（pmap の軸）、`use_fast_variance=False`、`force_float32_reductions`、`mask`、`dtype` / `param_dtype` | やらない | nabla に pmap が無い。速い分散が Flax の既定。layer-norm は f32 / f64 のみ（bf16 / f16 は未対応）。`mask` 付きの統計は要望が出たら |
| `Dropout` の `broadcast_dims` | やらない（attention の内部では使っている） | `dropout` レイヤー自身には出していない。`multi-head-attention` の `broadcast-dropout` が内部で同じ仕組み（`%dropout-apply` の `broadcast-dims`）を使う |

#### オプティマイザ（`nabla/nn`、issue #205）

Optax 相当。オプティマイザは `init`（パラメータの木 → 状態の木）と `update`（更新量の木・状態・パラメータ → 更新量の木と新しい状態）の組で、状態はモデルと同じ構造の PyTree、`update` は純粋関数。学習ステップ全体を `jit` できる。

```lisp
(let* ((opt (nabla.nn:chain (nabla.nn:clip-by-global-norm 1.0)
                            (nabla.nn:adam (nabla.nn:warmup-cosine-decay-schedule 0.0 1e-3 100 1000))))
       (state (nabla.nn:optimizer-init opt model)))
  (multiple-value-bind (updates state) (nabla.nn:optimizer-update opt grads state model)
    (values (nabla.nn:apply-updates model updates) state)))
```

| 名前 | 役割 |
| --- | --- |
| `optimizer` | init と update の組の型。`defoptimizer` か下の関数で作る |
| `optimizer-init` `(optimizer params)` | 状態を作る。`params` はモデルでもよい |
| `optimizer-update` `(optimizer updates state &optional params)` | `(values 更新量 新しい状態)`。`params` は `adamw` のように値を使う変換だけが読む。木の構造が違えば `pytree-structure-mismatch` |
| `apply-updates` `(params updates)` | 各葉に更新量を足した同じ構造の木 |
| `defoptimizer` `(name (hyper...) (:init (params) ...) (:update (updates state params) ...))` | `name` というオプティマイザを返す関数を定義する。本体は `with-tracing` の本体（`defmodule` の `:call` と同じ） |
| `sgd` `(lr &key momentum nesterov)` | `lr` は実数かスケジュール。`momentum` 付きは先に `momentum-trace` を通す |
| `adam` `(lr &key b1 b2 eps eps-root)` / `adamw` `(lr &key ... weight-decay)` | Optax の既定値（0.9, 0.999, 1e-8, 0, adamw の decay は 1e-4）。adamw は decoupled weight decay（mask は未対応）で、`update` に `params` が要る |
| `chain` `(&rest optimizers)` | 左から順に適用する。結合的で、引数なしは恒等。状態は各状態のリスト |
| `clip-by-global-norm` `(max-norm)` | 全葉をまとめたノルムが `max-norm` を超えるとき向きを変えずに縮める |
| `scale` `(factor)` / `scale-by-adam` / `scale-by-schedule` `(schedule)` / `momentum-trace` `(decay &key nesterov)` / `add-decayed-weights` `(wd)` | 上を組み立てる部品（Optax の同名の変換。`trace` は CL の `trace` と衝突するので `momentum-trace`） |
| `constant-schedule` `(v)` / `linear-schedule` `(init end steps &key transition-begin)` / `cosine-decay-schedule` `(init steps &key alpha)` / `warmup-cosine-decay-schedule` `(init peak warmup decay &key end-value)` | 学習率スケジュール。`count`（rank 0 の `:i32`）を受け取って rank 0 の `:f32` を返す関数 |

- **ステップ数は状態の中の rank 0 の `:i32` の配列**（`count`）で、Lisp の整数ではない。jit の入力として渡るので、ステップごとに再コンパイルされず、スケジュールの値も毎ステップ変わる。`update` ごとに 1 増える（`:i32` の最大値で止まる）
- 状態は `defmodule` の構造体（`adam-state` の `count` `mu` `nu`、`momentum-state` の `trace`、`schedule-state` の `count`）で、スロット名がチェックポイントのパス名になる。状態を持たない変換の状態は `nil`。葉の並びは Optax の `tree_leaves` と同じ
- 数値は Optax と f32 の許容誤差で一致する（`tests/fixtures/optimizer/`）。cos と pow のプリミティブが無いので、cosine は sin の Taylor 展開（打ち切り誤差は 1e-11 未満で、f32 の丸め程度の差が残る）、adam のバイアス補正 `1 - b^t` は `-expm1` の Taylor 展開と `1 - exp(-t log b)` で計算する
- **チェックポイントとの関係**: `chain` の状態はリストなので、パス名は位置で決まる（`1.0.mu.l1.w` の `0` は chain の何番目か）。chain の途中に変換を足すと古いチェックポイントは読めなくなる（`checkpoint-error`）。状態のスロットはすべて `:state` で、`(save-checkpoint path (list model opt-state))` で1ファイルに入る。`optimizer-update` / `apply-updates` に渡す葉は、jit の外ではホストの配列（`device-array` は eager では扱えない。`to-host` で戻す）
- **`chain` は `check-it:chain` と名前が衝突する**。check-it も使うパッケージ（テストなど）では `(:shadowing-import-from #:nabla.nn #:chain)` を書く
- 対象は f32 / f64 のパラメータ。スカラー（学習率・バイアス補正）は葉の dtype に合わせて `convert` する。bf16 / f16 は未対応
- IREE 3.11 の local-task は jit の入力が 162 個以上だと落ちる（`iree-too-many-inputs`）。adam の学習ステップの入力は パラメータ + `mu` + `nu` + `count` + バッチ なので、パラメータの葉が 50 個を超えるモデルでは足りなくなる

#### dense-general と attention のマスク（`nabla/nn`、issue #208）

Flax の `nn.DenseGeneral` と `make_causal_mask` / `make_attention_mask` / `combine_masks` に合わせたもの。スロット名は Flax のパラメータ名と同じ（`kernel` / `bias`）。

| モジュール | スロット | init | 呼び出し |
| --- | --- | --- | --- |
| `dense-general` | `kernel (in... features...)`, `bias (features...)`, `contract-axes` / `use-bias`（`:static`。`contract-axes` は縮約する入力の軸の数） | `(init-dense-general key in-shape features &key dtype use-bias)`（形は整数かそのリスト。`contract-axes` は `in-shape` の長さ。`use-bias` 既定は真）。kernel は平らな形の `lecun_normal` を reshape したもの、bias は 0 | `(call m x)`。x の最後の `contract-axes` 個の軸を縮約して特徴の軸に置き換える。q / k / v は `(in) → (heads head-dim)`、out は `(heads head-dim) → (out)` |

- `contract-axes` は kernel と bias の rank の差から推論せず、`:static` のスロットに持つ（`init-dense-general` が設定する。`make-dense-general` では明示する）。`:static` は葉でないので、チェックポイントのパス名は Flax と同じ `kernel` / `bias` のまま。bias の形が kernel の後ろの軸と合わなければ層名入りの error
- マスクの部品（bit の配列を返す。1 が「見える」）: `(make-causal-mask length)` は形 `(1 length length)` の因果マスク（クエリ位置 q がキー位置 k 以上で 1。先頭の 1 がヘッドとバッチ軸に広がる）、`(make-attention-mask query-valid key-valid)` は `[... q]` と `[... k]` の有効フラグから padding のマスク `[... 1 q k]`、`(combine-masks &rest masks)` は論理積（ランクが違えば右に揃えてブロードキャスト、`nil` は無視）。multi-head attention の `mask` に渡す

#### multi-head attention（`nabla/nn`、issue #208）

Flax の `nn.MultiHeadDotProductAttention` に合わせたもの。スロット名は Flax のパラメータ名と同じなので、チェックポイントのパス名が Flax の `flatten_dict` と一致する（`query.kernel` / `query.bias` / `key.*` / `value.*` / `out.*`）。`query` `key` `value` `out` は上の `dense-general`。

| モジュール | スロット | init | 呼び出し |
| --- | --- | --- | --- |
| `multi-head-attention` | `query` `key` `value` `out`（`dense-general`）, `dropout-rate` / `broadcast-dropout`（`:static`） | `(init-multi-head-attention key num-heads in-features &key qkv-features kv-features out-features dtype dropout-rate use-bias broadcast-dropout)`。`qkv-features` / `kv-features` / `out-features` の既定は `in-features`（Flax と同じ）。`head-dim = qkv-features / num-heads` | `(call m inputs-q inputs-kv mask key deterministic)` → `[... 長さq out-features]` |

- 入力は `[... 長さ 幅]`（先頭のバッチ軸は何個でもよい。Flax と同じ）。自己 attention は `inputs-q` と `inputs-kv` に同じものを渡す。`mask` は `nil` か bit（`:i1`）の配列で、`[... ヘッド 長さq 長さkv]` に右から揃えてブロードキャストできる形。1 が「見える」。配列でない値（`t` やリスト）は層名入りの error
- 計算は Flax と同じ順: `softmax((q / sqrt(head-dim)) k^T)` に `v` をかけ、`out` で射影する。マスクで隠す位置には、その dtype（f32 / f64 / bf16 / f16）の最小の有限値（Flax の `jnp.finfo(dtype).min`。`-inf` ではない）を `select` で入れ、最大値を引く `softmax` で正規化する。全部のキーが隠れた行は一様な重みになる（Flax と同じ。NaN にならない）
- attention の dropout は重みにかける（`dropout` と同じ規則: `deterministic` は Lisp の真偽値で `jit` では静的引数、偽で `dropout-rate > 0` ならキーが要る）。**`broadcast-dropout`（issue #241。既定は Flax と同じ真）**: 真のとき、マスクをバッチ軸（何個でも）とヘッドで共有する。Flax は重み `[バッチ... ヘッド q k]` に形 `[1 ... 1 q k]` のマスクを引いてブロードキャストする（`dot_product_attention_weights` の `dropout_shape`。`nn.Dropout` の `broadcast_dims` と同じ意味で、共有する軸の大きさを 1 にして引く）ので、nabla も同じ形で引く。偽なら重みの要素ごとに独立（Flax の `broadcast_dropout=False`）。**挙動変更**: #241 より前は偽だけだった（`make-multi-head-attention` も `:broadcast-dropout` が必須になった）。`dropout-rate` が 0 のモデル、`deterministic` が真の呼び出し、Flax のフィクスチャ、`examples/transformer.lisp`（dropout なし）には影響しない。dropout を使うモデルは、以前の挙動に戻すなら `:broadcast-dropout nil` を渡す。nabla の PRNG は JAX とビット単位で一致せず Flax と値を比べられないので、保持率・平均の保存・共有される軸では同じマスク・共有されない軸では独立、という統計的な性質をテストで確かめる（`tests/nn/attention-test.lisp`）
- **バッチ次元は「先頭の任意個」**（`vmap` で足す前提にしない）。学習のコードはバッチ付きの入力を直接渡すのが普通で、Flax のフィクスチャとそのまま比べられる。中身は `dot-general` のバッチ軸（先頭のバッチ軸とヘッド）で書いてあるので、先頭の軸が 0 個でも 2 個でも同じコードで、`vmap` で外側に軸を足すこともできる
- `key` と `value` は同じ入力 `inputs-kv` から作る（Flax の `inputs_k` と `inputs_v` を別々に渡す形は持たない。交差 attention には足りる）。`normalize_qk` などのオプションは、必要になった時点で足す

#### 小さな Transformer を学習する例（`examples/transformer.lisp`、issue #209）

フェーズ4の完了条件。decoder-only の Transformer を `defmodule` で書き（埋め込み + 学習する位置埋め込み、pre-LN のブロック 2 つ（因果マスクの multi-head attention と gelu の MLP、どちらも残差）、最後の layer-norm と出力の dense。出力は埋め込みと共有しない）、系列のコピー（`x1..x4 SEP x1..x4` の `SEP` 以降の 4 位置を当てる）を `jit` した学習ステップ（`value-and-grad` + `chain (clip-by-global-norm 1) (adam 0.01)`）の手書きのループで学習する。語彙 12・長さ 8・幅 16・2 ヘッド・MLP の幅 32、パラメータの葉 38 個。

```sh
sbcl --non-interactive --load examples/transformer.lisp   # 100 ステップ。損失 2.8 → 0.05 くらい
```

- パラメータ名（チェックポイントのパス名）は Flax の `flatten_dict` と同じ（`embed.embedding`, `pos`, `blocks.0.attn.query.kernel` ...。Flax の `blocks_0` は `blocks.0`）。`tests/fixtures/transformer/generate.py` が同じモデルを Flax + Optax で書き、同じ初期値を `init.safetensors` で nabla に渡し（nabla の PRNG は JAX とビット単位で一致しないため、初期値は乱数で合わせない）、同じ系列で学習した 20 ステップの損失と 1・5 ステップ後のパラメータを記録する。IREE の local で、損失とパラメータ（`attn.key.bias` 以外）の差は 4e-6 以下。`attn.key.bias` の勾配は数学的に 0 で、Adam が浮動小数点の雑音を学習率の大きさの更新に増幅するため値は一致せず、テストは差の上限だけを見る（理由はテストの docstring）
- jit する学習ステップの入力は モデル 38 + `mu` 38 + `nu` 38 + `count` 1 + バッチ 2 = 117 個で、IREE の local-task の上限（161。`docs/iree-upstream-bugs.md` の 6）に収まる。大きなモデルでは `local-sync` を使うか、葉をまとめる
- 学習の途中のモデルとオプティマイザの状態は 1 つの木 `(list model opt-state)` として 1 つの safetensors に保存でき、復元して続けた学習は止めずに続けた学習とビットまで同じになる（同じコンパイル済みの関数を同じ入力で決定的に走らせるため）
- 測った時間の目安（IREE local、このマシン相当の CPU。`tests/nn/transformer-test.lisp` は実行のたびに出力に書くだけで、値は検査しない）: 1 回目の呼び出し（トレース + コンパイル + 実行）約 7〜9 秒、2 回目以降は約 3〜4 ms/ステップ（バッチ 8、ホスト側のバッチ作りを含む）
- テスト: `tests/nn/transformer-test.lisp`（small: モデル全体の因果性の PBT、パス名、データの部品、入力の葉の数。medium: Flax + Optax との一致、損失の減少と決定性、チェックポイントからの再開、スクリプトが動くこと）

#### safetensors の読み書き（`nabla/nn`、issue #198）

`nabla/nn`（パッケージ `nabla.nn`）の部品の1つ。チェックポイントの形式 safetensors を、平坦な「名前 → 配列」の対応として読み書きする（PyTree をパス名で平坦化して保存・復元するのは次の「チェックポイント」）。`(asdf:load-system "nabla/nn")` で使う。

| 関数 | 役割 |
| --- | --- |
| `(write-safetensors path entries &key metadata)` | `entries` を `path` に書く。一時ファイルへの書き込み + rename で、失敗しても既存のファイルは壊れない |
| `(read-safetensors path)` | `(values entries metadata)` |
| `(encode-safetensors entries &key metadata)` / `(decode-safetensors octets &key path)` | ファイルを介さないバイト列版 |
| `safetensors-error` / `-path` / `-reason` / `-detail` | 壊れた入力・不正な引数のコンディション。`reason` は `:truncated` `:header-length` `:invalid-header` `:unknown-dtype` `:size-mismatch` `:bad-offsets` `:invalid-bool` `:invalid-entries`（意味は docstring） |

- `entries` は `(名前 配列 [dtype])` のリスト（読むと `(名前 配列 dtype)`、ファイル内のデータの順）。metadata は `(キー . 値)`（文字列）のリスト
- dtype の対応: `:f32` `:f64` `:bf16` `:f16` `:i32` `:u32` `:u64` と `:i1`（`BOOL`、1要素1バイト）が `F32` `F64` `BF16` `F16` `I32` `U32` `U64` `BOOL`。bf16 / f16 は `(unsigned-byte 16)` のビット列のまま読み書きするので、書くときは dtype を明示する（省略は `dtype-mismatch`）。`I64` `U8` など他の dtype を含むファイルは `:unknown-dtype`
- 書くときテンソルは要素の大きい順、同じなら名前順に並べる（入力の順序に依らず同じバイト列。Python の safetensors が書くバイト列とは限らず、保証するのは読み書きの意味が同じこと）。ヘッダは 8 バイト境界まで空白で埋める。ビット列はそのままコピーする（NaN のペイロードも保つ）。ホストはリトルエンディアン前提
- ヘッダの JSON は自前の最小の読み書き（`src/nn/json.lisp`）。`cl-yason` に頼らず依存を増やさない。小数・指数の数値は読まない（safetensors のヘッダには現れない）

#### チェックポイント（`nabla/nn`、issue #206）

PyTree の葉をパス名で平坦化して safetensors に保存し、「同じ構造のひな形」に unflatten して復元する。

| 関数 | 役割 |
| --- | --- |
| `(save-checkpoint path tree &key metadata dtypes)` | `tree` の葉を `path` に保存する（`write-safetensors` 経由なのでアトミック）。葉は配列か `device-array`（`to-host` でホストに戻して書く） |
| `(load-checkpoint path template &key dtypes)` | `(values 木 メタデータ)`。`template` の葉と名前で対応づけて `tree-unflatten` する。葉は配列・`device-array`・`aval` のどれでもよく、形と dtype だけを使う。復元した葉はホストの新しい配列 |
| `(tree-paths tree)` | 葉のパス名のリスト（`tree-leaves` と同じ順） |
| `checkpoint-error` / `-reason` / `-name` / `-detail` | `reason` は `:empty-segment` `:inconsistent-tree` `:bad-leaf` `:duplicate-path` `:unknown-path` `:reserved-metadata` `:unsupported-version` `:missing-entries` `:unexpected-entries` `:shape-mismatch` `:dtype-mismatch`（意味は docstring。`name` は原因の葉） |

- パス名は `"."` でつなぐ（`"enc.layers.0.w"`）。リストとベクタの子は添字、`defmodule` の子はスロット名（`:static` は出ない）。Lisp の既定で大文字になるスロット名は小文字にし、`|Dense_0|` のように小文字を含む名前はそのままなので、Flax の `flatten_dict(..., sep=".")` の `"params.Dense_0.kernel"` と同じ名前にできる。`defmodule` 以外で `defpytree` した型の子には名前が無いので、`flatten` が返す子の添字を使い、入れ子は再帰する（`"opt.0.0.w"`、`"opt.2"`）。スロット名が空のシンボル（`||`）は `:empty-segment`。根が葉なら `""`。名前が衝突する木（スロット名に `.` を含むなど）は `:duplicate-path`
- モデルとオプティマイザ状態は1つの木にまとめれば同じファイルに入る: `(defmodule train-state ((params) (opt)))` → `"params.w"`、`"opt.0"`。別々に保存してもよい（API に接頭辞は持たない）
- `(unsigned-byte 16)` の葉は bf16 か f16 か決められない。保存では `:dtypes`（`(パス名 . :bf16)` の連想リスト）で言うか、`device-array` なら aval から決まる（言わなければ `dtype-mismatch`）。復元でも、ひな形の葉が `(unsigned-byte 16)` の配列なら `:dtypes` で言う必要がある（言わなければ `:dtype-mismatch`。bf16 のモデルを f16 のファイルから生のビットで復元する事故を防ぐ。`device-array` と `aval` のひな形は自分の dtype を持つ）。それ以外の葉は dtype が一致しなければ `:dtype-mismatch`
- 照合は厳密: ファイルとひな形の名前の集合が違えば（足りなくても余っても）エラー。形の違いも `:shape-mismatch`。壊れたファイルは `safetensors-error`
- メタデータはそのまま往復する（データローダの再開状態を文字列にして入れられる）。`"nabla."` で始まるキーは予約で、版（`nabla.checkpoint_version` = `"1"`）を書く。版の無い（Python が書いた）ファイルも読め、知らない版は `:unsupported-version`
- Python との相互運用は `tests/fixtures/checkpoint/`（`generate.py` が Flax の2層 MLP を `flatten_dict` と同じ規則で書き、nabla が復元して numpy と同じ順伝播の結果になることを確かめる）

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
src/nn/              nabla.nn パッケージ: safetensors の読み書き、defmodule、オプティマイザ、チェックポイント、レイヤー、multi-head attention
tests/               nabla/tests のテスト（support-test, dtype-test, aval-test, backend-test, ...）
tests/support/       nabla/test-support: FiveAM のスイート、check-it の生成器、比較関数、フェイク backend
tests/iree/          nabla/iree/tests のテスト
tests/nn/            nabla/nn/tests のテスト
tests/fixtures/stablehlo/  手書きの StableHLO フィクスチャ
tests/regressions/   check-it が見つけた失敗例の回帰テスト
tools/mutate/         自前の mutation testing runner（nabla-mutate）
scripts/             setup-lisp-deps.sh, build-iree.sh, verify-iree.sh, run-tests.sh
docs/                glossary.md, iree-build.md, stablehlo-ops.md, pjrt-setup.md, float-traps-experiments.md, phase0-report.md 〜 phase4-report.md
examples/            add.lisp, jit.lisp, mlp.lisp, rnn.lisp, transformer.lisp, tutorial.lisp（この README の使用例）
third_party/         iree.lock（固定した IREE のコミットとホイールの sha256）
.claude/skills/nabla-testing/  テスト戦略の詳しい手順
```

ASDF システムは `nabla`（コア、nickname `nb`）、`nabla/test-support`、`nabla/tests`、`nabla/ffi-support`、`nabla/ffi-support/tests`、`nabla/iree`、`nabla/iree/tests`、`nabla/pjrt`、`nabla/pjrt/tests`（PJRT プラグインのロード、クライアント・デバイス・device-array、StableHLO のコンパイル・ロード・実行。`(find-backend :pjrt)` と `to-device` / `to-host` / `backend-compile` / `backend-invoke`、`(jit f :backend :pjrt)`。`to-device` / `to-host` と jit の入出力は IREE と同じ全 dtype に対応し、`:i1` は PJRT の `PRED`（ホストでは1要素1バイト）に写す（issue #166）。docs/pjrt-setup.md）、`nabla/nn`（safetensors の読み書き #198、defmodule #202、PyTree のチェックポイント #206）、`nabla/nn/tests` の11個に加え、mutation testing 用の `nabla-mutate`（`tools/mutate/`）がある。

## 開発の進め方

- 規約は [CLAUDE.md](CLAUDE.md) を読む（構成、コマンド、設計上の約束、開発の原則、テスト戦略、Git の運用）
- テストの書き方・実行の仕方は `.claude/skills/nabla-testing` を読む
- コミットメッセージと PR タイトルは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従う
- PR は stacked PR（下位ブランチに積む）で、squash merge をデフォルトにする
- ライセンスは [Apache License 2.0](LICENSE)
