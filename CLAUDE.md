# CLAUDE.md

nabla は Common Lisp で書く JAX 相当の深層学習ライブラリ。Lisp の関数をトレースして自前 IR に落とし、`jit` / `grad` / `vmap` で変換してから StableHLO を出力し、IREE で CPU / NVIDIA GPU 上で実行する。計画と設計の正本は Artifact「Common Lisp × IREE 深層学習ライブラリ 計画」（計画タブ・設計タブ）にある。設計判断に迷ったらまずそこを確認し、このファイルと食い違う場合は Artifact を優先して、このファイルを直す。

現在はフェーズ0（IREE 疎通）。ロードマップは フェーズ0 IREE 疎通 → 1 トレースと jit → 2 grad → 3 vmap と制御構造 → 4 Flax 相当 → 5 Grain 相当 の順。

## 構成

- 処理系は SBCL のみ。FFI は CFFI、GC 連携は trivial-garbage、並列は lparallel
- ASDF システム: `nabla`（コア、package nickname `nb`）、`nabla/iree`、`nabla/pjrt`、`nabla/nn`、`nabla/data`。テストは各システムに対応する `<system>/tests`
- IREE は固定コミットからソースビルドする（`libIREECompiler.so` とランタイム共有ライブラリ）。コンパイラは埋め込み C API を dlopen して呼び、`iree-compile` のサブプロセスは使わない
- Python はライブラリの実行時依存にしない。JAX は期待値フィクスチャの生成にだけ使う

## コマンド

```sh
# テスト（既定: CPU のみ、GPU 不要）
sbcl --non-interactive --eval '(ql:quickload "nabla/tests")' --eval '(asdf:test-system "nabla")'
```

ビルドスクリプト、GPU テスト、mutation test のコマンドを追加したら、ここに追記する。

## 設計上の約束（コードから読み取りにくいもの）

- StableHLO は出力先であって内部表現ではない。grad / vmap は自前 IR（`aval` / `var` / `eqn` / `graph`）上の IR→IR 変換として書く
- プリミティブは `defprimitive` で宣言し、形状推論・StableHLO 出力・eager 用 CPU 実装を同じ変更で書く。jvp / transpose ルールとバッチ化ルールは grad / vmap 対応時に必須
- 自動微分は jvp + transpose の JAX 方式（`jax._src.interpreters.ad`、ルールは `jax._src.lax` を写す）
- v1 は静的形状のみ。jit キャッシュのキーは 関数の同一性 + 引数の `aval` + 静的引数 + コンパイルターゲット（`sm_XX` などのアーキを含む）
- トレースは `with-tracing` のコードウォーク方式。`setq` はトレース対象で禁止、対応外の形式はコンディションで報告する
- PyTree の既定はリスト・ベクタ・`defmodule` 構造体のみ。plist / alist / ハッシュ表は明示登録
- bf16 / f16 は `(unsigned-byte 16)` 配列と `aval` の dtype タグで表す
- IREE の C API 名は版で変わる。関数名は記憶や設計書ではなく、固定コミットのヘッダ（`iree/runtime/api.h`、`iree/compiler/embedding_api.h`）から写す
- デバイスバッファは `device-array` で包み、finalizer はポインタだけを捕捉する（オブジェクト本体を捕捉すると回収されない）

## テスト戦略

自動テストは property-based testing（PBT）と mutation testing の2本立てにする。例ベースのテストは JAX との数値一致フィクスチャと、PBT で見つかった回帰例に限る。

### Property-based testing

- フレームワークは FiveAM、生成器は check-it（`(is (check-it gen #'prop))`）。回帰例は check-it の `regression-file` で `tests/regressions/` に保存してコミットする
- 実装より先に性質を書き、失敗することを確認してから実装する
- 数値比較はテキスト一致ではなく許容誤差つきの数値一致。既定の許容誤差は f32 で `rtol 1e-5` / `atol 1e-6`、bf16 / f16 で `rtol 1e-2`。これを緩めるときは理由をテストに書く
- 生成器の方針: rank 0〜4、各次元 1〜8 の小さな形状、dtype も生成する。定義域のある演算（`log`, `sqrt`, 除算）は定義域内の値を生成する。シードは失敗時に出力して再現できるようにする

主に守らせる性質:

| 対象 | 性質 |
| --- | --- |
| プリミティブ | `abstract-eval` の `aval` = 実際の出力の形状と dtype。eager 実装 = jit（IREE `local`）の結果 |
| grad | 中心差分（f64）と一致。`<vjp(u), v> = <u, jvp(v)>`（内積テスト） |
| transpose ルール | 線形性と `<T(u), v> = <u, L(v)>` |
| vmap | `vmap(f)(xs)` = 各要素に `f` を適用して積み上げた結果。`in-axes` を変えても一致 |
| 変換の合成 | `jit(grad f)` = `grad f`、`jit(vmap f)` = `vmap f`（eager） |
| StableHLO 出力 | 生成したテキストが IREE でコンパイルできる |
| PyTree / safetensors | `unflatten(flatten(x))` = `x`、保存→読み込みの往復で一致 |
| PRNG | 同じキーから同じ値、`split` した子キーの値は重ならない |
| データ層 | 同じシードで同じ順序、ワーカー数に依存しない、`iterator-state` から再開しても続きが一致 |
| jit キャッシュ | 同じキーでヒット、`aval` やターゲットが違えばミス |

GPU（`cuda`）で実行するテストは別スイートに分け、既定のテストは CPU だけで通るようにする。

### Mutation testing

- Common Lisp には実用的な既存ツールがないため、リポジトリ内に自前の mutation runner を持つ（`tools/mutate/`）。ソースを reader で読み、変異させた定義を image にロードしてテストスイートを走らせ、生き残った変異体を報告する
- 変異演算子: 算術演算子の入れ替え（`+`↔`-`、`*`↔`/`）、比較の境界（`<`↔`<=`）、定数の置換（`0`, `1`, `-x`）、条件の反転、`if` の分岐入れ替え、`progn` 内の式の削除、ルール関数の返り値を `0` にする
- 対象はコア（IR、変換、プリミティブのルール、StableHLO 出力）と nn / data の純粋関数。CFFI バインディングは対象外
- 変異体が生き残ったら、まず殺す性質を足す。等価変異体だと判断したら理由をつけて除外リストに記録する
- PR では変更したファイルだけを対象にする。mutation score の目標は 80% 以上

### 作業の終え方

変更を終える前に、既定のテストスイートを実行して通ることを確認する。プリミティブや変換ルールを変えたときは、変更したファイルに mutation test もかける。

## Git commits and pull requests

- Do not add any attribution lines to commit messages or pull request descriptions.
  This includes `Co-Authored-By:` trailers, `Claude-Session:` session URLs,
  and "Generated with Claude Code" footers.
