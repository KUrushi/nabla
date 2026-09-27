# CLAUDE.md

nabla は Common Lisp で書く、JAX に相当する深層学習ライブラリ。Lisp の関数をトレース（実行を記録）して自前の中間表現（IR）に変換し、`jit` / `grad` / `vmap` で書き換えてから StableHLO を出力し、IREE で CPU や NVIDIA GPU 上で実行する。

- 計画と設計の正本は Artifact「Common Lisp × IREE 深層学習ライブラリ 計画」（計画タブ・設計タブ）。設計に迷ったらまずそこを確認する。このファイルと食い違うときは Artifact を優先し、このファイルを直す
- 現在はフェーズ0（IREE 疎通）。ロードマップは フェーズ0 IREE 疎通 → 1 トレースと jit → 2 grad → 3 vmap と制御構造 → 4 Flax 相当 → 5 Grain 相当
- 専門用語の説明は [docs/glossary.md](docs/glossary.md) にある。本文で † が付いた語は用語集に解説がある。新しい専門用語を使い始めたら用語集にも追加する

## 構成

- 処理系は SBCL のみ。C ライブラリの呼び出しは CFFI、GC との連携は trivial-garbage、並列処理は lparallel を使う
- ASDF システムは `nabla`（コア、パッケージのニックネームは `nb`）、`nabla/iree`、`nabla/pjrt`、`nabla/nn`、`nabla/data` の5つ。テストは各システムに対応する `<system>/tests` に置く（テストの共通部品は `nabla/test-support` に置き、そこに依存する）。`nabla` コアの公開 API は `src/dtype.lisp`（dtype タグと Lisp 要素型の対応表）と `src/aval.lisp`（`aval`: 形状と dtype の組）から始まる（issue #7）
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
# nabla/iree の medium テストは IREE の共有ライブラリ（NABLA_IREE_HOME 配下）が
# 無いと自動でスキップされる。CI では NABLA_REQUIRE_IREE=1 を立てて、その
# スキップを失敗にする
NABLA_IREE_HOME=~/.local/share/nabla/iree-3.11.0 NABLA_REQUIRE_IREE=1 scripts/run-tests.sh

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

# mutation testing（既定は main から HEAD までの git diff で変わった行が対象）
tools/mutate/run.sh
tools/mutate/run.sh src/core/foo.lisp:10-40           # ファイル・行範囲を指定する
tools/mutate/run.sh --system nabla --base main --trials 20 --timeout 300
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
→ `NABLA_REQUIRE_IREE=1 scripts/run-tests.sh`（IREE 未検出によるスキップを
失敗にする）→ `nabla-mutate` 自身のテスト（`tools/mutate/README.md` のコマンド）。
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
- v1 は静的形状（配列の形がコンパイル時に決まっている）だけを扱う。jit キャッシュのキーは「関数の同一性 + 引数の `aval` + 静的引数 + コンパイルターゲット（`sm_XX` などの GPU 世代を含む）」
- トレースは `with-tracing` によるコードウォーク†方式。トレースされるコードでは `setq` を禁止し、対応していない形式はコンディション（Lisp の例外）で報告する
- PyTree† として既定で扱うのは、リスト・ベクタ・`defmodule` で定義した構造体だけ。plist / alist / ハッシュ表は明示的に登録する
- bf16 / f16† は `(unsigned-byte 16)` の配列で持ち、`aval` の dtype タグで区別する
- IREE の C API は版によって関数名が変わる。関数名は記憶や設計書から書かず、固定コミットのヘッダ（`iree/runtime/api.h`、`iree/compiler/embedding_api.h`）から写す
- デバイス上のバッファは `device-array` で包む。`device-array` は生成時に自分のデバイス（`iree_hal_device_t`）を retain し、解放時に buffer view → device の順で release する（IREE の heap buffer が確保元 allocator の統計ブロックへの生ポインタを持ち、その allocator を device が所有しているため。device を先に解放すると use-after-free になる）。finalizer† はポインタだけを捕まえる（オブジェクト本体を捕まえると、いつまでも GC に回収されない）。この解放は `trivial-garbage:finalize` で自動化されており（`device-array` 生成時に登録）、明示的な `release-device-array` は `tg:cancel-finalization` で finalizer を先に取り消してから自分で解放するので、二重解放にはならない。SBCL は finalizer を別スレッド（finalizer thread）で非同期に実行するため、テストで確認するときは `gc-and-run-finalizers`（`tests/iree/support.lisp`）のように GC の後で明示的に保留中の finalizer を実行させる
- 実行系は `backend` プロトコル（`src/backend.lisp`）の裏に置く。core は IREE の名前を知らない（medium テストで検査）。総称関数は `backend-` 接頭辞（CL の `compile` / `load` と衝突させない）
- LLVM を呼びうる FFI エントリポイント（IREE コンパイラ、将来の PJRT）は、必ず `with-lisp-signal-handlers-preserved`（`src/iree/signals.lisp`）で本体を包む。LLVM は初回の呼び出し中にプロセス全体のシグナルハンドラを sigaction で登録し直し、SBCL が GC の stop-the-world に使う SIGUSR2 を上書きする。放置すると、以後どこかのスレッドが GC を始めた瞬間に "no SP known for thread" で SBCL が確実に落ちる（issue #5）
- 上記の LLVM のシグナルハンドラ登録そのものは、`ensure-compiler-loaded` が他の全 Lisp スレッドを SBCL の GC と同じ仕組み（`%call-with-world-stopped`、`src/iree/signals.lisp`）で止めた、制御された1点で済ませる。世界が止まっている間はどのスレッドもシグナルを受け取れないので、登録の瞬間に別スレッドが GC を始める競合の隙間が無くなる（詳しい根拠と残る課題は signals.lisp 冒頭のコメント）

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
