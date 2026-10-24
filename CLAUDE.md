# CLAUDE.md

nabla は Common Lisp で書く、JAX に相当する深層学習ライブラリ。Lisp の関数をトレース（実行を記録）して自前の中間表現（IR）に変換し、`jit` / `grad` / `vmap` で書き換えてから StableHLO を出力し、IREE で CPU や NVIDIA GPU 上で実行する。

- 計画と設計の正本は Artifact「Common Lisp × IREE 深層学習ライブラリ 計画」（計画タブ・設計タブ）。設計に迷ったらまずそこを確認する。このファイルと食い違うときは Artifact を優先し、このファイルを直す
- 現在はフェーズ5（Grain 相当。データ供給。`nabla/data` はまだ無く、最初の作業で新設する）に着手する。フェーズ4（Flax 相当。PyTree と `defmodule`・レイヤー・オプティマイザ・safetensors のチェックポイント・小さな Transformer の学習、親 issue #194）は 2026-10-19 に完了。知見は docs/phase4-report.md（積み残し: expm1 / log1p / sin / cos #223 は済み、`stablehlo.if` の素通しで落ちる #226 は済み、`while-loop` の実数 #227 は済み、本体に制御構造がある scan の ys のコピー #233 は調査済み（上流の修正待ち。改善の候補の評価と x86 ネイティブで確かめる手順は docs/iree-upstream-bugs.md の 7）、emitter の mutation の生き残り #237 は済み、core の内部シンボルへの依存 #239（`tree-children` で解消）、gather #240、Flax との差の残り #241（済み: use_bias・layer-norm の引数・broadcast_dropout・attend を実装し、残りは README の「Flax との差」に理由を書いて「やらない」）、jit の入力が 161 個を超えるモデル #242、IREE のバグ（4〜8）の上流報告 #243 は下書き済み（`docs/iree-upstream-bugs.md`。上流への起票はリポジトリのオーナー）、チェックポイントの大きなファイルとデータローダの状態 #244（測定済み、docs/checkpoint-memory.md）、check-it の datum #245）。フェーズ3（vmap・制御構造 `cond*` / `while-loop` / `scan`・PRNG、親 issue #124）は 2026-10-03 に完了（GPU での数値一致 #12 は 2026-10-10、CUDA のベンチは 2026-10-16 に Colab の T4 で測った。docs/phase2-report.md §4.4）。知見は docs/phase3-report.md（積み残し: scan の ys が IREE で遅い #159 は済み、バッチされた rng の emit の while 化 #164 と K 行ずつの処理 #178 は済み、IREE 3.11 の while のバグの上流報告と回避策の撤去 #165、小さな積み残し #166、while-loop の逆モードは対象外）。フェーズ2（grad、2層 MLP の学習、PJRT バックエンドと IREE の比較、親 issue #90）は 2026-10-02 に完了（CUDA のベンチは 2026-10-16 に追記）。知見は docs/phase2-report.md。フェーズ1（トレースと jit、親 issue #35）は 2026-09-27 に完了。知見は docs/phase1-report.md。フェーズ0（IREE 疎通、#2）は 2026-09-26 に完了（GPU での数値一致 #12 は 2026-10-10 に確認）。知見は docs/phase0-report.md。ロードマップは フェーズ0 IREE 疎通 → 1 トレースと jit → 2 grad → 3 vmap と制御構造 → 4 Flax 相当 → 5 Grain 相当
- 専門用語の説明は [docs/glossary.md](docs/glossary.md) にある。本文で † が付いた語は用語集に解説がある。新しい専門用語を使い始めたら用語集にも追加する

## 構成

- 処理系は SBCL のみ。C ライブラリの呼び出しは CFFI、GC との連携は trivial-garbage、並列処理は lparallel を使う
- ASDF システムは `nabla`（コア、パッケージのニックネームは `nb`）、`nabla/ffi-support`（IREE と PJRT が共有する、C ライブラリ呼び出しの保護。シグナルハンドラと浮動小数点トラップ。`src/ffi-support/`、core にも IREE にも依存しない。issue #79）、`nabla/iree`、`nabla/pjrt`（XLA の PJRT プラグイン（CPU。CUDA はベンチの MLP が Colab の T4 で動くことだけを確かめた。テストは CPU のみ）を dlopen して使う、IREE と並ぶ `backend` の実装。core は PJRT の名前を知らない）、`nabla/nn`（モデル記述の層。safetensors の読み書き、defmodule、レイヤー、オプティマイザ、PyTree のチェックポイント `src/nn/`。ヘッダの JSON は自前の最小の読み書きで、`cl-yason` は使わない。issue #198）の5つ。フェーズ5で `nabla/data` を足す。テストは各システムに対応する `<system>/tests` に置く（テストの共通部品は `nabla/test-support` に置き、そこに依存する）。公開 API の一覧は README.md の「公開 API」を正とする
- Quicklisp は使えない（ネットワーク方針）。Lisp の依存は apt パッケージと、固定コミットで git clone したもの（check-it など）を `scripts/setup-lisp-deps.sh` で揃える。システムのロードは ASDF の `CL_SOURCE_REGISTRY` で行い、`ql:quickload` は使わない
- IREE は固定したコミットで使う（詳細は `docs/iree-build.md`）。ランタイムの共有ライブラリは常にそのコミットからソースビルドする。コンパイラ (`libIREECompiler.so`) は既定では同じコミットからビルドされた PyPI ホイール（`third_party/iree.lock` に記録）を使う。フルソースビルドは `scripts/build-iree.sh --compiler=source` で選べるが、このマシン相当のスペックでは実用的な時間で終わらないことを確認している。コンパイラは埋め込み C API を dlopen して呼び、`iree-compile` をサブプロセスで起動しない（ビルドスクリプト内の動作確認を除く）
- IREE ランタイムの C API（`iree_allocator_t` / `iree_string_view_t` / `iree_hal_buffer_params_t` / `iree_timeout_t` など）は構造体を値で渡し、値で返す関数もある。素の CFFI はこれに対応しないため `nabla/iree` は `cffi-libffi`（apt の `cl-cffi` に同梱）を使う。`cffi-libffi` は libffi-dev をビルド時に必要とするので `scripts/setup-lisp-deps.sh` の APT_PACKAGES に `libffi-dev` を含めてある。C 側のヘルパーは書かない（`cffi:defcfun` / `cffi:defcstruct` をそのまま使える）
- Python はライブラリの実行時依存にしない。JAX は、テストで比べる期待値（フィクスチャ）の生成にだけ使う

## コマンド

```sh
# 依存の準備（初回のみ。apt は root で実行、check-it / optima は git clone）
scripts/setup-lisp-deps.sh

# テスト（既定は small + medium。CPU だけで動き、GPU は不要）
scripts/run-tests.sh                      # NABLA_TEST_SIZES=small,medium が既定
NABLA_TEST_SIZES=large scripts/run-tests.sh
# GPU で local/cuda の一致を確かめる（issue #12）。GPU が無ければ
# skip-unless-cuda がスキップする。NABLA_REQUIRE_CUDA=1 を立てると
# （GPU が要る環境で）スキップの代わりに失敗させる
NABLA_TEST_SIZES=large NABLA_REQUIRE_CUDA=1 scripts/run-tests.sh
# nabla/iree の medium テストは IREE の共有ライブラリ（NABLA_IREE_HOME 配下）が
# 無いと自動でスキップされる。CI では NABLA_REQUIRE_IREE=1 を立てて、その
# スキップを失敗にする
NABLA_IREE_HOME=~/.local/share/nabla/iree-3.11.0 NABLA_REQUIRE_IREE=1 scripts/run-tests.sh
# nabla/pjrt の medium テストは PJRT プラグイン（NABLA_PJRT_HOME 配下）が無いと
# 自動でスキップされる。CI では NABLA_REQUIRE_PJRT=1 を立てて失敗にする

# Colab の GPU で local / cuda の一致を確かめる（issue #12）。手元のマシンで実行する
# （このクラウド環境からは Colab に接続できない）。詳細は docs/iree-build.md
scripts/colab-gpu-check.sh --gpu T4
scripts/colab-gpu-check.sh --bench-only   # CUDA のベンチ（bench-backends.sh --cuda）だけを Colab で測る。--bench でテストの後に測る
scripts/colab-gpu-check.sh --profile      # jit した関数の呼び出し1回の内訳（to-device / invoke / to-host）を測る（issue #190、scripts/profile-call-overhead.lisp）。--with-tests でテストも

# PJRT プラグイン（CPU。--cuda で CUDA も）を third_party/pjrt.lock の wheel から
# 取得・sha256 検査・展開する。インストール先は NABLA_PJRT_HOME（既定
# ~/.local/share/nabla/pjrt-0.0.1）。詳細は docs/pjrt-setup.md
scripts/fetch-pjrt.sh

# vmfb ディスクキャッシュ（既定 ${XDG_CACHE_HOME:-~/.cache}/nabla/vmfb/。
# NABLA_CACHE_DIR で場所を変える。消してよい。テストで検査した挙動を
# 変えたときは、変更した行に mutation testing もかける）
tools/mutate/run.sh src/compile-cache.lisp

# IREE のビルド（third_party/iree.lock で固定したコミットから）。
# コンパイラは既定で PyPI ホイールを使い、ランタイムは常にソースビルドする
scripts/build-iree.sh                     # --compiler=wheel（既定）+ ソースランタイム（CPU）
scripts/build-iree.sh --compiler=source   # コンパイラもフルソースビルド（CI 向け、手元では非現実的な時間がかかる）
scripts/build-iree.sh --cuda              # CUDA も有効化
scripts/build-iree.sh --configure-only    # cmake configure までで止める
# 環境変数（詳細は docs/iree-build.md）: NABLA_IREE_HOME（インストール先。既定
#   ~/.local/share/nabla/iree-3.11.0）、NABLA_IREE_SRC / NABLA_IREE_BUILD /
#   NABLA_IREE_WHEEL_DIR（既定はいずれも ${XDG_CACHE_HOME:-~/.cache}/nabla/ 以下）、
#   NABLA_IREE_COMPILER（wheel|source、--compiler と同じ）、NABLA_IREE_CUDA、NABLA_IREE_JOBS
# wheel モード（既定）は x86_64 Linux 専用。他のプラットフォームでは --compiler=source を使う

# ビルドした iree-compile / iree-run-module で matmul フィクスチャを実行して確かめる
NABLA_IREE_HOME=~/.local/share/nabla/iree-3.11.0 scripts/verify-iree.sh
scripts/verify-iree.sh --cuda             # llvm-cpu に加えて CUDA でも確かめる

# IREE 3.11 のバグ（docs/iree-upstream-bugs.md の 1〜8）の最小再現（docs/iree-repros/）が今も
# 再現するかを単体の iree-compile で確かめる。IREE を上げたときに回避策を外せるかの
# 確認に使う（手順は docs/iree-build.md、上流への報告の下書きは docs/iree-upstream-bugs.md）
NABLA_IREE_HOME=~/.local/share/nabla/iree-3.11.0 scripts/check-iree-repros.sh --runs 20

# mutation testing（既定は main から HEAD までの git diff で変わった行が対象）
tools/mutate/run.sh
tools/mutate/run.sh src/core/foo.lisp:10-40           # ファイル・行範囲を指定する
tools/mutate/run.sh --system nabla --base main --trials 20 --timeout 300

# IREE (local) と PJRT (XLA CPU) で 2層 MLP のコンパイル時間と学習ステップ時間を測る
# （issue #89。結果と測定条件は docs/phase2-report.md。--cuda で CUDA も、GPU が無ければ「未測定」）
scripts/bench-backends.sh --configs small --steps 50 --reps 1   # 小さな設定。引数なしで small,medium,large

# save-checkpoint / load-checkpoint のピークメモリ（最大 RSS）を測る（issue #244。結果と測り方は docs/checkpoint-memory.md）
scripts/measure-checkpoint-memory.sh                              # NABLA_CKPT_SIZES="64 256" NABLA_CKPT_HEAPS="8GB" で絞る

# 本体に cond* がある scan の呼び出し時間を、外側の長さ L と ys の大きさを変えて測る（issue #233。IREE が要る。
# 改善の候補の emit は --variants でテンプレート scripts/scan-ys-candidates/*.mlir.in を渡して比べる）
scripts/bench-scan-ys.sh --lengths 16,64 --widths 4096,65536

# README の使用例（backend プロトコル経由で StableHLO を実行する）
export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
sbcl --non-interactive --load examples/add.lisp
```

mutation testing の詳しいオプションは [`tools/mutate/README.md`](tools/mutate/README.md)、
考え方は `.claude/skills/nabla-testing` スキルの `references/mutation.md` を見る。

## CI

`.github/workflows/ci.yml` が PR（stacked PR のため base ブランチは問わない）と
`main` への push で動く。実行順: apt で SBCL と IREE のビルド道具（clang / lld /
cmake / ninja / python3-pip）を入れる → `scripts/setup-lisp-deps.sh`（apt の
Lisp パッケージ + check-it / optima の git clone。`$NABLA_LISP_DEPS` を
`hashFiles('scripts/setup-lisp-deps.sh')` でキャッシュ）→ IREE
（`third_party/iree.lock` と `scripts/build-iree.sh` のハッシュをキーに
`$NABLA_IREE_HOME` をキャッシュし、当たればビルドをスキップ、外れれば
`scripts/build-iree.sh --compiler=wheel` を実行）→ `scripts/verify-iree.sh`
→ PJRT CPU プラグイン（`third_party/pjrt.lock` と `scripts/fetch-pjrt.sh` の
ハッシュをキーに `$NABLA_PJRT_HOME` をキャッシュし、外れれば
`scripts/fetch-pjrt.sh`）→ `NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_PJRT=1
scripts/run-tests.sh`（IREE / PJRT 未検出によるスキップを失敗にする）→ `nabla-mutate` 自身のテスト（`tools/mutate/README.md` のコマンド）。
GPU を使う large テストは CI では動かさない。

`.github/workflows/pr-title.yml` が PR タイトルを Conventional Commits の
形式かどうか確かめる（squash merge で PR タイトルがそのまま `main` の
コミットメッセージになるため）。

実行結果は `gh api repos/KUrushi/nabla/actions/runs?branch=<branch>` で見る
（`gh pr view` などの GraphQL 系コマンドはこのプロジェクトの認証では使えない）。

実測（PR #22、ubuntu-24.04）: IREE キャッシュが無いとき（cold）はジョブ全体で
約 4分43秒（うち `scripts/build-iree.sh` が約3分4秒）、キャッシュが当たったとき
（warm）は約44秒（`scripts/build-iree.sh` はスキップされる）。この数字は環境
によって変わるので、目安として扱う。

## 設計上の約束（コードを読んでも分かりにくいもの）

- StableHLO† は出力先であって、内部表現ではない。`grad` / `vmap` は自前 IR（`aval`† / `var` / `eqn` / `graph`）を別の IR に書き換える変換として書く
- 演算（プリミティブ）は `defprimitive` で宣言する。形状推論（出力の形と型を計算する関数）、StableHLO 出力、eager 用の CPU 実装の3つは同じ変更の中で書く。jvp† / transpose ルール†とバッチ化ルール†は、その演算を `grad` / `vmap` に対応させるときに必須になる
- 自動微分は JAX と同じ「jvp + transpose」方式にする（`jax._src.interpreters.ad` を参考にし、演算ごとのルールは `jax._src.lax` から写す）
- v1 は静的形状（配列の形がコンパイル時に決まっている）だけを扱う。jit キャッシュのキーは「関数の同一性 + 動的引数の葉の `aval` + 動的引数の treedef（PyTree の形。`treedef-spec` を EQUAL で比べる。#199）+ 静的引数 + コンパイルターゲット（`sm_XX` などの GPU 世代を含む）」。戻り値の treedef はキーに入れず、キャッシュのエントリに持つ（同じ入力の形からは同じ形の戻り値ができるため）
- トレースは `with-tracing` によるコードウォーク†方式。トレースされるコードでは `setq` を禁止し、対応していない形式はコンディション（Lisp の例外）で報告する。フェーズ1の作業単位は #35 の子 issue を参照。対応する特殊形式・書き換える CL 関数の一覧は `src/walk.lisp`（issue #32）の `%walk` と `*rewrite-table*` を正とする（CL の `sin` / `cos` もここで書き換わる。CL に無い `expm1` / `log1p` / `rsqrt` は nabla の関数で、同じ表で書き換える）。`do` を `scan` に展開する対応表（対応する形）は `src/loop-scan.lisp`（issue #137）にある
- `defmodule`（`src/nn/module.lisp`、issue #202）: スロットは `:param`（省略時）/ `:state`（学習しない状態）/ `:static`（treedef の補助データ。EQUAL で比べられる値）で、子の葉は `:param` / `:state` の宣言順。構造体は定義時に `defpytree` へ登録する（スロットは定義時（ロード時）に表 `*defined-modules*` に持ち、`module-slots` で引ける。#206 のパス名用）。呼び出しは総称関数 `call`（`:call` の本体は `with-tracing` の本体に展開され、grad / vmap の中で呼べる。引数リストは必須引数のみ）、init は `init-<name>`（キーを `split-keys` で分ける）、状態を grad から外すには `map-state`。継承は持たない
- 基本レイヤー（`src/nn/layers.lisp`、issue #204）は Flax Linen の既定に合わせる（スロット名 kernel / bias / scale / embedding もチェックポイントのパス名なので同じ）。`dense` の kernel は `lecun_normal`（`nb:truncated-normal` を 1/0.8796 sqrt(in) 倍）、`layer-norm` は epsilon 1e-6 と「速い分散」（#241 で `use-scale` / `use-bias` / `reduction-axes` / `feature-axes` を足した。axes は call のとき rank で正の値・重複なし・昇順にする。scale / bias の形は入力の feature-axes の大きさ）、`embedding` は gather でなく one-hot と `dot`（勾配は dot の転置。範囲外の id は 0 の行）、`embedding-attend`（#241。Flax の `Embed.attend`。query と埋め込みの転置の `dot` の普通の関数。重みの共有用）、`dropout` は反転 dropout で `(call m x key deterministic)`（`deterministic` は Lisp の真偽値、jit では静的引数）。**`use_bias`（#241。dense / dense-general / conv / multi-head-attention）**は `:static` のスロット `use-bias`（既定 t）で、nil のとき `bias` スロットは nil（空の部分木。葉でないのでパス名は変わらず kernel だけ）、`call` は bias が nil なら足さない（`%add-bias`。食い違いは `%check-bias-slot` が層名入りの error）。`make-*` はスロットが必須なので `:use-bias` も書く。multi-head-attention は `:use-bias` を 4 つの `dense-general` に渡すだけで自身のスロットは持たない。活性化関数 `relu` / `gelu` / `softmax` / `log-softmax` は nabla.nn の関数で、コアのプリミティブではない。`defmodule` の `:call` の本体では `loop` が使えないので、ループが要る部品は普通の関数に出す。ロジックは `defmodule` の外の `defun` に置く（mutation testing の runner は defmodule の中身を変異させられない）。トレースされる部品は `(funcall (load-time-value (nabla:with-tracing (x ...) body)) x ...)` の形で書く。`with-tracing` の本体は `progn` で包まれるので、先頭の文字列は docstring ではなく普通の式になる（docstring は `defun` の側に書く）
- オプティマイザ（`src/nn/optimizer.lisp`、issue #205）は Optax と同じ「`init` と `update` の組」（`optimizer` 構造体。`defoptimizer` で定義、`chain` で合成）。`update` は純粋関数で、**ステップ数 `count` は状態の中の rank 0 の `:i32` の配列**（Lisp の整数にすると jit がステップごとに再コンパイルする）。状態は `defmodule` の構造体（`adam-state` など。スロットはすべて `:state` で、スロット名がチェックポイントのパス名になる）で、状態を持たない変換の状態は `nil`、`chain` の状態は各状態のリスト。内蔵のオプティマイザは `defun` + `%traced`（算術だけ with-tracing）で書く（mutation testing が `defoptimizer` のフォームを変異させられないため）。cosine decay は `0.5 (1 + cos(pi t))`、adam のバイアス補正 `1 - b^t` は `-expm1(t log b)` で計算する（Optax の `1 - b^t` を pow なしで書いた同値な式。Optax は `decay**count` を直接書く。pow のプリミティブは無く、`exp(t log b)` で足りる。`1 - exp(...)` だけだと `b2 = 0.999`・`t = 1` で f32 の桁落ちが大きい）。スカラーは葉の dtype（`nabla:value-aval`）に `convert` する（f32 / f64 のみ。bf16 は eager の配列だけでは dtype を区別できない）。adam の jit は入力が パラメータ + mu + nu + count + バッチ になるので、IREE 3.11 の 162 入力の上限（`iree-too-many-inputs`）に注意
- `dense-general` とマスクの部品（`src/nn/dense-general.lisp`、`src/nn/attention-mask.lisp`、issue #208）は Flax の `nn.DenseGeneral` と `make_causal_mask` などに合わせる。`dense-general` のスロットは `kernel (in... features...)` / `bias (features...)` と `:static` の `contract-axes`（縮約する入力の軸の数。`init-dense-general` が設定し、rank の差からは推論しない。葉でないのでチェックポイントのパス名は Flax と同じ kernel / bias）。計算は kernel と入力を平らにした `dot`。init は平らな形の `lecun_normal` を reshape する。マスクは bit 配列で、`make-causal-mask` は長さを取って `(1 n n)` を返し（Flax は入力の配列を取る）、`combine-masks` は右に揃えたブロードキャストの論理積。`nabla:dot-general`（バッチ軸・縮約軸を明示する公開 API。eager でも形状推論で `primitive-error`）はこの issue で足した
- multi-head attention（`src/nn/attention.lisp`、issue #208）は Flax の `nn.MultiHeadDotProductAttention` に合わせる。スロット名は Flax のパラメータ名（`query` / `key` / `value` / `out` が `dense-general` で、それぞれ `kernel (in heads head-dim)` / `bias (heads head-dim)`、`out` は `kernel (heads head-dim out)`。チェックポイントのパス名が Flax の `flatten_dict` と一致する）。入力は `[... 長さ 幅]` で先頭のバッチ軸は任意個（`vmap` 前提にしない）。呼び出しは `(call m inputs-q inputs-kv mask key deterministic)`（必須引数のみ。`mask` は `nil` か `:i1` の配列で `[... ヘッド q k]` に右から揃えてブロードキャスト。配列でない値は層名入りの error）。スコアは `q / sqrt(head-dim)` と k の `nabla:dot-general`（バッチ軸 = 先頭の軸とヘッド）、隠す位置は dtype（f32 / f64 / bf16 / f16）の最小の有限値を `if`（select）で入れて `softmax`、dropout は重みに `%dropout-apply`（`:static` の `broadcast-dropout`、既定 t は Flax と同じ: マスクを形 `[1 ... 1 q k]` で引いてバッチ軸とヘッドで共有する（`%dropout-apply` の `broadcast-dims` に最後の 2 軸以外を渡す）。nil なら要素ごとに独立。#241 より前は nil だけだった。マスクは JAX とビット一致しないので、統計的な性質（保持率・平均の保存・共有される軸で同じマスク）で確かめる）。Flax との比較で `key.bias` の勾配だけは値でなく「絶対値が 1e-5 未満」で比べる（softmax はスコアの定数ずらしに不変で、key の bias は全スコアに同じずれを足すだけなので、勾配は数学的に 0 で、実測は丸め誤差）。ほかの葉は f32 の既定の許容誤差
- `conv`（`src/nn/conv.lisp`、issue #207）は Flax の `nn.Conv`。スロットは `kernel [空間..., C/G, O]`（HWIO）/ `bias [O]` と `:static` の `strides` / `padding` / `kernel-dilation` / `feature-group-count` / `use-bias`、入力は `[N, 空間..., C]`（バッチ軸の無い入力も可。Flax と同じ）。kernel の init は lecun_normal で fan_in = kernel の面積 × C/G（`%lecun-normal-of-shape`）。padding は `:same` / `:valid`（Flax の文字列は受け付けない。EQUAL で比べられるキーワード）/ 明示の `((low high) ...)`。SAME は入力の形に依存するので `call` のたびに JAX の `padtype_to_pads` と同じ式（出力 ceil(n/s)、合計 max((out-1)s+実効kernel-n, 0)、low = 合計/2 切り捨て）で計算する（`%same-padding`）。N 次元（`init-conv` の kernel-size の長さ。1 次元以上）で、`call` は kernel の rank から空間の次元数を決める。strides・padding・kernel-dilation は `call` のたびに正規化する（`make-conv` で直接作ったモジュールでも `:strides 2` が使える）。`use-bias` は `:static`（nil なら bias は nil。#241）。入力の dilation・CIRCULAR / CAUSAL padding は持たない
- PyTree† として既定で扱うのは、リスト・ベクタ・`defmodule` で定義した構造体だけ。plist / alist / ハッシュ表は明示的に登録する（`defpytree`、`src/pytree.lisp`、issue #195）。ベクタの節は `simple-vector`（要素型 `t`）だけで、型特化した配列・文字列は葉。`nil` は空の部分木。登録の無い型は葉。treedef は EQUAL で比べられる spec を持ち（登録した型の補助データも EQUAL で比べられる値に限る）、jit のキャッシュのキーに入れている（#199）。`jit` / `grad` の引数と戻り値は PyTree で、木の直下の子と補助データを内部表現を知らずに取る公開関数は `tree-children`（節の判定は `tree-flatten` と共通の `%node-parts` で、nabla/nn のチェックポイントのパス名が使う。nn は core の内部シンボル `nabla::` を参照しない。#239）。木を受けて返す関数を「葉だけを受けて返す関数」に変える部品は `src/pytree.lisp` の `%flatten-arguments` / `%flatten-function` / `%unflatten-results`（戻り値の多値は「値のリスト」という1つの木として flatten する。vmap・制御構造もこれを使う）。`vmap` の `in-axes` / `out-axes` は、引数のリスト・戻り値のリスト（多値は値のリスト）の木の接頭辞で、`%vmap-broadcast-axes`（`src/vmap.lisp`）が treedef の spec を歩いて葉ごとの軸に展開する（JAX の `flatten_axes`。整数・`nil` は部分木まるごと、節は種類・子の個数・補助データが一致しなければ `vmap-error`）。軸の木では `nil` は葉（空の部分木ではない）なので `tree-flatten` では歩かず、登録した型は登録された flatten で子を取る。`grad` の `f` の出力は、木で包まない rank 0 の浮動小数点1つのまま
- bf16 / f16† は `(unsigned-byte 16)` の配列で持ち、`aval` の dtype タグで区別する
- IREE の C API は版によって関数名が変わる。関数名は記憶や設計書から書かず、固定コミットのヘッダ（`iree/runtime/api.h`、`iree/compiler/embedding_api.h`）から写す
- デバイス上のバッファは `device-array` で包む。`device-array` は生成時に自分のデバイス（`iree_hal_device_t`）を retain し、解放時に buffer view → device の順で release する（IREE の heap buffer が確保元 allocator の統計ブロックへの生ポインタを持ち、その allocator を device が所有しているため。device を先に解放すると use-after-free になる）。finalizer† はポインタだけを捕まえる（オブジェクト本体を捕まえると、いつまでも GC に回収されない）。この解放は `trivial-garbage:finalize` で自動化されており（`device-array` 生成時に登録）、明示的な `release-device-array` は `tg:cancel-finalization` で finalizer を先に取り消してから自分で解放するので、二重解放にはならない。SBCL は finalizer を別スレッド（finalizer thread）で非同期に実行するため、テストで確認するときは `gc-and-run-finalizers`（`tests/iree/support.lisp`）のように GC の後で明示的に保留中の finalizer を実行させる。ホストから見えないバッファ（CUDA の device-local）の to-host は、device ごとに1つ持つ作業用バッファ（ホストから見える。`%read-via-staging`、`src/iree/device-array.lisp`）へのコピーと map_read で読む（`iree_hal_device_transfer_d2h` は呼ぶたびに作業用バッファを確保・解放し、T4 で1回約 1.35 ms かかっていた。issue #190）。作業用バッファは `release-device` が device より先に解放する
- 自動微分のルールは `src/primitives/*` ではなく `src/ad/rules-*.lisp` に `def-jvp-rule` / `def-transpose-rule` で書く（プリミティブ定義との衝突を避ける。vmap のバッチ化ルールも同じ配置）。jvp ルールは接線を、その被演算子について線形なプリミティブにしか流さない（接線同士の積や、接線への exp / log などは禁止。transpose が各ルールの局所的な転置で済むため）
- ゼロの接線・余接線は配列ではなく symbolic zero† で表す。`:i1` など非 float の接線は変換の中では常に symbolic zero で、出力では全 false の配列に実体化する（`jvp-graph` の既定は float の invar だけが非ゼロ。非 float に接線を渡すと `autodiff-error`）。`jvp-graph` の中では DCE（不要な eqn の削除）しない（主値と接線が混ざった出力の使い道が分からなくなるため）。DCE は `linearize` が接線部分にかける
- `grad` は f を aval で新しいトレースにトレースし、`inline-graph` で外側へ展開する（`src/ad/grad.lisp`）。f が外側のトレーサを閉包で捕まえると `tracing-error` になる（既知の制限）
- convolution の `grad` / `vmap`（`src/ad/rules-conv.lisp`、`src/ad/rules-batch-conv.lisp`、issue #203）は JAX の `_conv_general_dilated_transpose_lhs` / `_rhs` / `_batch_rule` を、channels-last 固定の配置に写したもの（JAX が dimension_numbers で入れ替える軸は、畳み込みの前後の明示的な `transpose` / `reshape` で入れ替える）。lhs の転置はカーネルの空間軸を反転する内部プリミティブ `:rev`（`stablehlo.reverse`）を使う。グループ畳み込みの rhs の転置は `:batch-group-count`（内部のパラメタ。既定 1 で `%conv-params` が明示する。`make-eqn` がキーの完全一致を求めるため。`conv-general-dilated` のキーワードには出さない。G と同時に 1 より大きくならない）が G の畳み込みになる。bf16 / f16 の `convolution` の emit は、f32 に直した2入力を `stablehlo.optimization_barrier` に通してから畳み込む（IREE 3.11 が convert を畳み込みに折り畳んで片方の operand の dtype を取り違えるため。lhs の勾配と O/G = 1 の depthwise で落ちる。`docs/iree-upstream-bugs.md` の 4、5。「今も落ちること」を `convolution/iree-bf16-convert-conv-bugs-still-reproduce` が守る）。テストの期待値は `reference-conv`（窓を数え上げる別実装。`batch-groups` も持つ）と JAX のフィクスチャ `tests/fixtures/conv/conv-grad.lisp`（`generate-grad.py` で再生成）
- バッチ化ルールも `src/ad/rules-batch-*.lisp` に `def-batch-rule`（`src/vmap.lisp`）で書く。形は `(name (args batch-dims &rest param-lambda-list) body)` で、body は `(values outs out-dims)` を返し、**outs と out-dims は単一出力のプリミティブでも常にリスト**。`batch-dims` は各引数のバッチ軸（整数）か `nil`（全部 `nil` の eqn は変換側が短絡する）。ルールは新しい外側のトレースの中で呼ばれる。`primitive` の `batch` スロットは構造体の最後で、`defprimitive` の最後のキー `:batch`。`defprimitive` に `#'fn` を書くと mutation testing が変異させられない（本体を `lambda` で書く）
- 複数出力のプリミティブは `defprimitive` の `:multiple-outputs t` で示す（出力の個数では切り替えない）。`abstract-eval` は aval のリスト、`eager` は配列のリスト、`emit` は `(in-names in-avals out-names out-avals &key ...)` で `%8, %9 = ...` の左辺を自分で書く。トレースは `%trace-eqn*`（常にトレーサのリスト。単一出力用の `%trace-eqn` に複数出力を渡すと `tracing-error`）。複数出力の jvp ルールは `(rule primals tangents &rest params)` → `(values primal-outs tangent-outs)`（主値の eqn もルールが足す。先に足すと高階プリミティブが2回走るため）。`jvp-graph` の本体は `%jvp-graph-core` に一本化してある
- 高階プリミティブ（`cond` / `while-loop` / `scan`）は、eqn の params に閉じた `graph`（サブグラフ）を持つ。本体は `traceable-function` を `%trace-subgraph` でトレースし、外側のトレーサを閉包で捕まえたら closure conversion で新しい invar に持ち上げる（EQ でメモ化、明示的な invar の後ろに最初に使った順で足し、呼び出し側が eqn の invars の末尾に足す）。`%call-with-fresh-trace`（`grad` / `vmap`）は `parent = nil` のままで、`f` が外側のトレーサを捕まえると `tracing-error`。StableHLO のリージョンは `%stablehlo-region-lines`（`arg-names` ありは `stablehlo.if` の枝用、無しは `^bb0` 付きで `while` 用）で出し、SSA 名には `%s<k>_` の接頭辞を付ける。サブグラフの中身の変換は各プリミティブのルールの仕事（`eval-graph` / `inline-graph` / `dce-graph` と jvp / transpose は素通し）
- 制御構造の API の形（carry・xs・ys・operands は **PyTree**。issue #201。内部では `tree-flatten` して今までどおりフラットな葉のリストで扱い、結果を `tree-unflatten` で木に戻す。プリミティブの eqn の形は変えない。リストは既定の節なのでリストの呼び出しはそのまま動く。本体が返す carry の treedef が init と違えば、各構文の不一致のコンディション（`while-loop-carry-mismatch` / `scan-carry-mismatch` / `cond-error`）で報告する。`cond*` の operands は引数ごとの木、両枝の戻り値の treedef は一致が必要。`scan` の `ys` の木は本体の y の形）: `(cond* pred then-fn else-fn &rest operands)`（プリミティブ名 `:cond`。`with-tracing` の `if` は要素ごとの条件がありうるので `select` のまま。eqn の invars は `pred ++ operands の葉 ++ captures`（captures は両枝の捕捉値の和集合）、両枝のサブグラフの invars は同一の `operands の葉 ++ captures` で、出力の aval も一致する）、`(while-loop cond-fn body-fn init)` → carry の木（葉の Lisp の実数は `scan` と同じ規則で rank 0 の配列に持ち上げる。double-float は `:f64`、整数・有理数・single は `:f32` で、整数も `:f32` になる。整数のカウンタは `:i32` の配列で渡す。issue #227。params `:cond :body :n-carries`。捕捉値は素通しで返す追加の carry で、本体の出力で入力そのままであることを EQ で検査する）、`(scan f init xs &key length reverse)` → `(values carry ys)`（params は JAX と同じ `:num-consts :num-carry :length :reverse :body`、本体の入力は consts ++ carry ++ x_t、出力は carry ++ y_t。ホストの配列を閉包で捕まえたものは本体のサブグラフの定数）。`while-loop` の逆モード（`grad`）は JAX と同じく対応しない（`autodiff-error`）
- `linearize` には、プリミティブごとの partial eval フック `*partial-eval-rules*`（`set-partial-eval-rule`、`src/ad/partial-eval.lisp`）がある。`:scan` はこれで主値と残差を計算する scan と接線について線形な scan に分け、`cond` は jvp ルールの中で分ける。接線が非ゼロ（vmap ではバッチされる）carry の集合は、本体を変換しては広げる不動点計算で求める
- **IREE 3.11 の `stablehlo.while` は、cond を駆動する carry が `stablehlo.constant` で初期化されていると（ほかに carry が2つ以上、1つは rank 1 以上）Stream の AffinityAnalysis で非決定的に segfault する**。while / scan を emit するコードは、定数のオペランドを必ず `stablehlo.optimization_barrier` に通す（`%while-barrier-lines`。長さ 1 の scan だけは barrier があると型不一致で落ちるので付けない）。比較由来の `:i1` の carry を持つ while の結果を jit の戻り値にすると別のバグで落ちる（回避策なし。`docs/stablehlo-ops.md`）。どちらも「今も落ちること」を子プロセスの medium テストで守っている（直れば失敗するので、回避策ごと消す）。float → int の `convert` は飽和と NaN → 0 を StableHLO 側で明示し、整数 → bf16 は f32 経由で間に `optimization_barrier` を置く
- **IREE 3.11 の local-task は、ys を barrier に通した scan の ys を負荷のもとでまれに（1〜2%）0 の行のまま返す**。本体が ys を `optimization_barrier` に通し、その結果へ本体の中の制御構造（while / if）をはさんで書くのが原因なので、本体の中に制御構造（入れ子の scan・while-loop・`cond*`）がある scan には ys の barrier を付けない（代償は ys の丸ごとのコピー。長さの2乗。大きさは `scripts/bench-scan-ys.sh` で測れる。ys と制御構造の結果を1つの barrier に通す候補はコピーを消せるが競合の安全性が未検証（#233）なので既定は変えない）。`docs/iree-upstream-bugs.md` の 7、`docs/iree-repros/nested-scan-ys-barrier-race.mlir`、「今も起きること」は `scripts/check-iree-repros.sh`（race-bug。確率的で負荷が要るので既定のスイートには入れない）。テスト側は local-sync や worker 数 1 に逃げない（利用者が踏むため）
- **IREE 3.11 の local-task device は、入力が 162 個以上の関数を呼ぶと `iree_task_call_initialize` で NULL を参照して落ちる**（jit した関数の入力は引数の PyTree の葉の個数。#199、`docs/iree-upstream-bugs.md` の 6、最小の再現は `docs/iree-repros/local-task-162-inputs.mlir`）。`nabla.iree:invoke` は呼ぶ前に `iree-too-many-inputs` を signal する（`+local-task-max-inputs+` = 161。core は IREE の名前を知らないので、検査は `src/iree/execute.lisp` にある）。回避策は `(make-backend :iree :driver :local-sync)`。別の制限として、入力 170 個の要素ごとの加算の連鎖はロード時に `binding count 51 > 32` で失敗する。「今も落ちること」は `tests/iree/input-limit-test.lisp` が子プロセスで守る
- **IREE 3.11 は、`stablehlo.if` の枝が入力をそのまま `stablehlo.return` する出力（素通し）を持つと、もう片方の枝を選んだ実行時に `ref is null` で落ちることがある**（local-task・local-sync の両方。出力の並びによる。#226、`docs/iree-upstream-bugs.md` の 8、最小の再現は `docs/iree-repros/if-passthrough-operands.mlir`）。素通しを枝の中で `optimization_barrier` に通しても直らない。`cond` の emit（`%cond-emit`）は、素通しの出力を `stablehlo.if` の結果から外して if の外の `stablehlo.select(pred, then 側, else 側)` にする（素通しの枝は 0 の定数を返す。全部が素通しなら `stablehlo.if` 自体を出さない）。eqn の形・eager・jvp・vmap は変えない。scan の ys の barrier を外す判定（本体のテキストに `stablehlo.if` があるか）は、select に書き換えた後も `stablehlo.if` が残る限り変わらない。全出力が素通しで `stablehlo.if` が消えた `cond*` だけを本体に持つ scan は「制御構造なし」と判定されて ys の barrier が付くが、制御構造が無くなっているので #220 の競合の条件を満たさず安全「今も落ちること」は `tests/iree/cond-passthrough-test.lisp` が子プロセスで守る（直れば失敗するので、回避策ごと消す）。`scripts/check-iree-repros.sh` の fail-bug / fail-control の行でも確かめられる
- 配列は不変（immutable）として扱う。nabla は渡された配列を書き換えないが、eager の結果は入力・閉包で捕まえた配列・graph の定数と記憶域を共有しうる（`eval-graph`、`cond*` / `while-loop` / `scan`、空の axes の `reduce-*` など）。これはコピーで守らない（新しい配列を返すと約束しない。README「配列の不変性」、issue #166）。`%scan-fresh-row` は x_t の行バッファの使い回しを防ぐ正しさのためのコピーで、方針とは別。`stop-gradient` は #80 から新しい配列を返す
- 小さな Transformer の例（`examples/transformer.lisp`、issue #209、フェーズ4の受け入れテスト）: `defmodule` の `transformer`（`embed` / `pos`（生の配列の :param）/ `blocks`（`decoder-block` のリスト）/ `norm` / `head`）。出力の dense は埋め込みと共有しない（パス名を Flax と1対1にするため）。Flax とのフィクスチャ（`tests/fixtures/transformer/`）は初期値を safetensors（`init.safetensors`、Flax のパス名の `blocks_0` は `blocks.0`）で渡す。Flax との比較で `attn.key.bias` だけは値でなく差の上限で見る（勾配が数学的に 0 で Adam が雑音を増幅する。理由は `tests/nn/transformer-test.lisp`）。jit する学習ステップの入力は 117 個で local-task の上限 161 に収まる。`*features*` に `:nabla-no-demo` を足して load すると、末尾の学習（デモ）が走らない（テストが使う）
- PyTree のチェックポイント（`src/nn/checkpoint.lisp`、issue #206）: 葉を `"."` でつないだパス名（リスト・ベクタは添字、`defmodule` はスロット名で、大文字になる名前だけ小文字にし `|Dense_0|` は保つ。`defmodule` 以外の `defpytree` 型の子は `flatten` が返す子の添字で、入れ子は再帰する）で safetensors に保存し、ひな形（配列・`device-array`・`aval`）に名前で対応づけて `tree-unflatten` する。名前の過不足・形・dtype の違いは `checkpoint-error`。`(unsigned-byte 16)` の葉の bf16 / f16 は、保存でも復元のひな形でも `:dtypes`（パス名 → dtype）か `device-array` / `aval` の dtype で決める（言わなければ `dtype-mismatch`）。モデルとオプティマイザ状態は1つの木（`defmodule`）にまとめて1ファイルに入れる。メタデータの `"nabla."` 接頭辞のキーは予約（版）。データパイプラインの状態（エポック・位置・PRNG キー）も同じ木に入れ、**Lisp の整数は葉にできない（`:bad-leaf`）ので rank 0 の `:i32` の配列と `:u32` の `(2)` のキーで持つ**。ピークメモリは load がファイルの約 2 倍、save が約 3 倍で、ヒープ 1 GiB（測定環境 SBCL 2.6.7 / macOS arm64 の既定）のとき save は 256 MiB が通り 320 MiB で枯渇、load は 448 MiB が通り 512 MiB で枯渇（遅延読み込み・世代管理は作らない判断。`docs/checkpoint-memory.md`、測定は `scripts/measure-checkpoint-memory.sh`）
- PRNG（`src/prng.lisp`）: キーは `:u32` の `(2)`（`prng-key` は `[上位32ビット 下位32ビット]`）で、`rng-bit-generator` の状態 `ui64[2] = [k0 | k1<<32, カウンタ]` に写す（`uniform` / `normal` / `split` はカウンタ 0 から、`fold-in` は 2^32 + data から）。ビットは常に1次元で作って reshape する。JAX とはビット単位で一致しない。バッチされた `rng-bit-generator`（状態 `ui64[..., 2]`）の emit は `stablehlo.while` の1回で K 行（`*rng-rows-per-iteration*`、16）ずつ回す（#164 / #178。行数が K を超えれば StableHLO の大きさは行数に依らず、K 以下なら while を出さずに展開する。行数が K の倍数でないと最後の1回は添字のクランプで前の回と重なるので、本体は書き換えない元の状態の carry から読む。ビットのバッファは scan の ys と同じく in-place に書く。IREE ではループ1回ごとの起動の費用が行数 / K に比例して残る）
- PJRT の `device-array` も finalizer でバッファを解放する。`PJRT_Client` は `client-state`（ポインタとカウンタだけの構造体）が持ち、所有者（`pjrt-client`）が GC または明示解放で消えた後で生きているバッファが0になった時点で、最後に解放した側が `PJRT_Client_Destroy` を1回だけ呼ぶ（バッファより先にクライアントを壊さない。Lisp の mutex を finalizer が取らず、CAS と atomic カウンタだけで済ます。`src/pjrt/client.lisp`）
- 実行系は `backend` プロトコル（`src/backend.lisp`）の裏に置く。core は IREE の名前を知らない（medium テストで検査）。総称関数は `backend-` 接頭辞（CL の `compile` / `load` と衝突させない）
- vmfb のディスクキャッシュ（`src/compile-cache.lisp`、issue #10）は `BACKEND-COMPILE` に `:AROUND` メソッドを足す形で実装し、`BACKEND` そのものは変えない。キーは `BACKEND-FINGERPRINT`（ターゲット・GPU 世代・解決済みのコンパイルフラグ・IREE のリビジョンを含む文字列のリスト）と `TEXT` を、それぞれ「UTF-8 バイト長:」の ASCII 接頭辞つきで連結した SHA-256（ironclad）の16進文字列。ファイルは `<hex>.module` = マジック `"NBLMOD01"`（8バイト）+ payload の SHA-256（32バイト、生）+ payload（vmfb）で、一時ファイルへの書き込み + `uiop:rename-file-overwriting-target` でアトミックに作る。場所は `nabla:*compile-cache-directory*`（既定 `${XDG_CACHE_HOME:-~/.cache}/nabla/vmfb/`、`NABLA_CACHE_DIR` で変更、`NIL` で無効化。相対パスは `uiop:getcwd` を基準に絶対パスへ解決する。SBCL の `rename-file` が相対パスの移動先を移動元を基準に解決するため、issue #180）
- LLVM を呼びうる FFI エントリポイント（IREE コンパイラ。PJRT は CPU プラグインで調べた限りシグナル処分を変えないので世界を止めた登録は要らないが、PJRT_Client_Create も多重防御として同じマクロで包む。docs/phase2-report.md §3.2）は、必ず `with-lisp-signal-handlers-preserved`（`nabla/ffi-support`、`src/ffi-support/signals.lisp`）で本体を包む。LLVM は初回の呼び出し中にプロセス全体のシグナルハンドラを sigaction で登録し直し、SBCL が GC の stop-the-world に使う SIGUSR2 を上書きする。放置すると、以後どこかのスレッドが GC を始めた瞬間に "no SP known for thread" で SBCL が確実に落ちる（issue #5）
- 上記の LLVM のシグナルハンドラ登録そのものは、`ensure-compiler-loaded` が他の全 Lisp スレッドを SBCL の GC と同じ仕組み（`%call-with-world-stopped`、`src/ffi-support/signals.lisp`）で止めた、制御された1点で済ませる（IREE 固有の登録手順 `%register-llvm-signal-handlers` は `src/iree/signals.lisp`）。世界が止まっている間はどのスレッドもシグナルを受け取れないので、登録の瞬間に別スレッドが GC を始める競合の隙間が無くなる（詳しい根拠は両方の signals.lisp 冒頭のコメント。浮動小数点トラップのマスク `with-all-float-traps-masked` も `nabla/ffi-support`）

## 開発の原則

Google の *Software Engineering at Google* と Engineering Practices の考え方をこのプロジェクトに当てはめたもの。

- **ソフトウェアエンジニアリングは「時間をかけて積み重ねたプログラミング」**。今動くことより、数年後も安全に変更できることを優先する。迷ったら「半年後に別の人がこのコードを直せるか」で判断する
- **ハイラムの法則†に備える**。利用者は、公開したものすべてにいずれ依存する。パッケージから `export` するシンボルは必要最小限にし、内部の関数は `export` しない。エラーメッセージの文言や出力の順序のような偶然の性質も、公開 API の一部とみなされうる
- **ビヨンセ・ルール†**: 壊されて困る振る舞いには、必ず自動テストを書く。テストのない振る舞いは、他の変更で壊れても文句を言えない
- **変更は小さく**。1つの PR は1つの目的だけを持ち、目安として差分 200 行程度に収める。リファクタリングと振る舞いの変更は別の PR に分ける。テストは、それが確かめるコードと同じ PR に入れる
- **コードはレビューで読まれるために書く**。レビューでは (1) 正しいか、(2) 読んで理解できるか、(3) 設計がプロジェクトの方針に合っているか、を見る。「動くけれど読めない」コードは直す
- **スタイルのルールは理由があるものだけ**。ツールで自動的にそろえられるものはツールに任せ、人が覚えるルールを増やさない
- **ドキュメントはコードと一緒に更新する**。`export` するシンボルには docstring を書く。このファイルや用語集が古くなったら、気づいた変更の中で直す
- **問題は早く見つけるほど安い（shift left）**。本番やユーザーの手元で見つかる前に、テスト・レビュー・型宣言で見つける
- **非推奨化（deprecation）は段階的に**。公開 API を消すときは、まず非推奨の警告を出し、移行先を docstring に書き、利用箇所がなくなってから消す

## テスト戦略

詳しい手順は `nabla-testing` スキル（`.claude/skills/nabla-testing/`）にある。テストを書く・直す・実行するとき、および nabla のコードを変更するときは、このスキルを使う。

- 自動テストは property-based testing†（FiveAM + check-it）と mutation testing†（自前の runner `tools/mutate/`）の2本柱。例ベースのテストは、JAX との数値一致フィクスチャと、PBT が見つけた失敗例の回帰テストに限る
- テストは実装より先に書き、失敗することを確認してから実装する
- テストは small / medium / large のテストサイズ†に分ける。既定のスイートは small + medium で、GPU なしで通るようにする
- 浮動小数点の比較は、テキストの一致ではなく許容誤差つきの数値の一致で行う

### 作業の終え方

変更を終える前に、既定のテストスイート（small + medium）を実行して通ることを確認する。プリミティブや変換のルールを変えたときは、変更した行に mutation testing もかける。テストを実行できなかったときは、そのことを報告に書く。

## Git の運用

- **コミットメッセージと PR タイトルは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従う**。形式は `<type>(<scope>): <説明>`
  - type: `feat`（機能追加）、`fix`（バグ修正）、`docs`、`test`、`refactor`（振る舞いを変えない変更）、`perf`、`build`、`ci`、`chore`
  - scope: `core`、`iree`、`pjrt`、`nn`、`data`、`mutate` など、変更したシステムやツールの名前。複数にまたがるときは省略してよい
  - 説明は英語の命令形で、先頭は小文字、末尾にピリオドを付けない。例: `feat(core): add broadcast primitive`
  - 公開 API を壊す変更は、type の後ろに `!` を付け（例: `feat(nn)!: rename dense to linear`）、本文の末尾に `BREAKING CHANGE: <内容>` を書く
- **PR は squash merge をデフォルトにする**。PR の全コミットが1つにまとめられ、PR タイトルがそのまま `main` のコミットメッセージになる。そのため PR タイトルは必ず Conventional Commits の形式にし、PR の中身が変わったらタイトルも直す。PR 内の途中のコミットは形式が崩れていてもよい
- コミットメッセージと PR の説明には、署名の行を一切付けない（`Co-Authored-By:` トレーラー、`Claude-Session:` のセッション URL、「Generated with Claude Code」のフッターを含む）
