# nabla

Common Lisp で書く、[JAX](https://github.com/jax-ml/jax) に相当する深層学習ライブラリ。Lisp の関数をトレース（実行を記録）して自前の中間表現（IR）に変換し、`jit` / `grad` / `vmap` で書き換えてから [StableHLO](https://openxla.org/stablehlo) を出力し、[IREE](https://iree.dev/) で CPU や NVIDIA GPU 上で実行する。

設計と計画の正本は Artifact「Common Lisp × IREE 深層学習ライブラリ 計画」（計画タブ・設計タブ）。開発の規約は [CLAUDE.md](CLAUDE.md)、専門用語は [docs/glossary.md](docs/glossary.md) を見る。

## 現状

フェーズ0（IREE 疎通）は完了した。ただし GPU（`cuda` ターゲット）での local/cuda 数値一致（issue #12）は GPU の無い環境のため未測定で、issue は open のままにしてある。詳しい知見は [docs/phase0-report.md](docs/phase0-report.md) にまとめてある。

現在はフェーズ1（トレースと jit。親 issue #35）に着手している。

ロードマップ（計画タブより）:

| フェーズ | 目標 | 目安 |
| --- | --- | --- |
| 0. IREE 疎通 | Lisp から IREE を動かす | 完了 |
| 1. トレースと jit | Lisp 関数から StableHLO を出す | 1〜1.5ヶ月 |
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

## 公開 API（フェーズ0時点）

`nabla`（nickname `nb`）が export するシンボルのみ。`nabla.iree` の低水準な C API バインディングはここには載せない（ハイラムの法則に備え、README に載せた名前を事実上の公開約束にしすぎないため）。詳しく知りたければ `src/iree/package.lisp` を見る。

**dtype**（`src/dtype.lisp`）: `dtype`（`:f32` / `:f64` / `:bf16` / `:f16` / `:i1`。`:i1` は issue #37 で追加した真偽値の dtype で、compare の出力・select の条件に使う）, `dtype-element-type`, `dtype-byte-width`, `array-dtype`, `dtype-mismatch`, `dtype-mismatch-element-type`, `dtype-mismatch-dtype`

**aval**（`src/aval.lisp`）: `aval`, `make-aval`, `aval-p`, `aval-shape`, `aval-dtype`, `aval-rank`, `aval-size`, `aval-byte-length`, `array-aval`

**backend プロトコル**（`src/backend.lisp`、issue #9）: `backend`, `make-backend`, `find-backend`, `backend-target`, `backend-fingerprint`, `backend-compile`, `backend-load`, `backend-unload`, `backend-invoke`, `to-device`, `to-host`, `device-array-aval`, `backend-error`, `backend-not-available`, `backend-not-available-kind`, `unsupported-dtype`, `unsupported-dtype-dtype`（issue #37。`to-device` にその実行系がデバイス上の表現を持たない dtype——フェーズ1では `:i1` と `:f64`——を渡すと signal する）

**vmfb ディスクキャッシュ**（`src/compile-cache.lisp`、issue #10）: `*compile-cache-directory*`

**IR と defprimitive**（`src/primitive.lisp`、`src/ir.lisp`、issue #29）: `defprimitive`, `primitive-name`, `var-aval`, `eqn-prim`, `eqn-params`, `eqn-invars`, `eqn-outvars`, `graph-invars`, `graph-eqns`, `graph-outvars`, `graph-constants`, `unknown-primitive`, `unknown-primitive-name`, `primitive-error`

**graph の印字**（`src/ir-print.lisp`、issue #29）: `print-graph`（graph を jaxpr 風のテキストに変換する。読み込み側の `read-graph` はテキスト形式をフェーズ1では公開契約にしないため export しない）

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
docs/                glossary.md, iree-build.md, phase0-report.md
examples/            add.lisp（この README の使用例）
third_party/         iree.lock（固定した IREE のコミットとホイールの sha256）
.claude/skills/nabla-testing/  テスト戦略の詳しい手順
```

ASDF システムは `nabla`（コア、nickname `nb`）、`nabla/test-support`、`nabla/tests`、`nabla/iree`、`nabla/iree/tests` の5つに加え、mutation testing 用の `nabla-mutate`（`tools/mutate/`）がある。

## 開発の進め方

- 規約は [CLAUDE.md](CLAUDE.md) を読む（構成、コマンド、設計上の約束、開発の原則、テスト戦略、Git の運用）
- テストの書き方・実行の仕方は `.claude/skills/nabla-testing` を読む
- コミットメッセージと PR タイトルは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従う
- PR は stacked PR（下位ブランチに積む）で、squash merge をデフォルトにする
- ライセンスは [Apache License 2.0](LICENSE)
