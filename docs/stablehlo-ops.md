# StableHLO op 対応表（issue #30）

フェーズ1で emitter（issue #33）が出す StableHLO op と、IREE が実際に受け付ける
綴りの対応表。プリミティブ実装（issue #31 / #39）と emitter は、ここに書いた
綴りをそのままコピーする。表の全行に `tests/fixtures/stablehlo/ops/<name>.mlir`
というフィクスチャがあり、`tests/iree/ops-test.lisp` の medium テストが
`find-backend :iree` の `backend-compile` で非空の vmfb になることを確かめる。

コンパイルが通ることだけをここでは確認する。数値の正しさはプリミティブ実装
（子3、issue #31 / #39）の仕事。

## 検証環境

- IREE: `third_party/iree.lock` のコミット `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`（v3.11.0）
- コンパイルフラグ: `src/iree/compiler.lisp` の `(compile-flags :local)` と同じ
  （`--iree-input-type=stablehlo --iree-hal-target-device=local
  --iree-hal-local-target-device-backends=llvm-cpu
  --iree-llvmcpu-target-cpu=host`、embedded linker があれば追加）
- 確認方法: `nabla.iree::compile-stablehlo` に各フィクスチャを渡し、例外なく
  vmfb（先頭4バイトが ZIP local-file-header `PK\3\4`）が返ることを確認した
  （本 PR の `tests/iree/ops-test.lisp` と同じ経路）

## 観察: `loc(...)` と verifier 診断

各 eqn の末尾に ` loc("eqn-N")` を付けてコンパイルすると、verifier エラーの
診断に `<unknown>:0: error: loc("eqn-7"): ...` のように `eqn-N` がそのまま
現れる（パースエラー自体は file:line:col になる）。emitter（issue #33）が
`iree-compile-error` の diagnostics から失敗した eqn を逆引きするのに使える。

## 対応表

| op | フィクスチャの綴り（form） | nabla プリミティブ名 | f32 | bf16 | 代替・備考 |
| --- | --- | --- | --- | --- | --- |
| add | `%0 = stablehlo.add %a, %b : tensor<4x8xf32>`（pretty） | `add` | ○ | ○ | |
| subtract | `%0 = stablehlo.subtract %a, %b : tensor<4x8xf32>`（pretty） | `sub` | ○ | ○ | |
| multiply | `%0 = stablehlo.multiply %a, %b : tensor<4x8xf32>`（pretty） | `mul` | ○ | ○ | |
| divide | `%0 = stablehlo.divide %a, %b : tensor<4x8xf32>`（pretty） | `div` | ○ | ○ | |
| maximum | `%0 = stablehlo.maximum %a, %b : tensor<4x8xf32>`（pretty） | `max` | ○ | ○ | |
| minimum | `%0 = stablehlo.minimum %a, %b : tensor<4x8xf32>`（pretty） | `min` | ○ | ○ | |
| negate | `%0 = stablehlo.negate %a : tensor<4xf32>`（pretty、単項） | `neg` | ○ | ○ | |
| exponential | `%0 = stablehlo.exponential %a : tensor<4xf32>`（pretty、単項） | `exp` | ○ | ○ | |
| log | `%0 = stablehlo.log %a : tensor<4xf32>`（pretty、単項） | `log` | ○ | ○ | |
| tanh | `%0 = stablehlo.tanh %a : tensor<4xf32>`（pretty、単項） | `tanh` | ○ | ○ | |
| compare | `%0 = stablehlo.compare LT, %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>`（pretty。方向は LT LE GT GE EQ NE、`, FLOAT` の compare_type 付きも可。generic form `"stablehlo.compare"(%a, %b) {comparison_direction = #stablehlo<comparison_direction LT>}` も通る） | `compare` | ○ | ○ | 出力 dtype は nabla の `:i1`（issue #37）。IREE の `iree-run-module` は `4xi1=1,0,...` 形式の入力を受け付けないため、`:i1` は関数の内部値としてのみ使う（to-device は `:i1` を拒否する。issue #37） |
| select | `%1 = stablehlo.select %pred, %a, %b : tensor<4xi1>, tensor<4xf32>`（pretty） | `select` | ○ | ○ | |
| convert | `%0 = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xbf16>`（pretty） | `convert` | ○ | ○ | |
| constant | `%c = stablehlo.constant dense<[1.0, 2.5]> : tensor<2xf32>`（rank 0 は `dense<3.0> : tensor<f32>`）。bf16/f16 は16進ビット列: `dense<[0x3F80, 0x4020]> : tensor<2xbf16>`（実行結果も正しい: 1, 2.5）。max の初期値のような単一値も同じ書き方: `dense<0xFC00> : tensor<f16>`、`dense<0xFF800000> : tensor<f32>` | プリミティブではなく `graph-constants` | ○ | ○ | |
| broadcast_in_dim | `%0 = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>`（pretty。rank 0 元は `dims = []`） | `broadcast-in-dim` | ○ | ○ | |
| reshape | `%0 = stablehlo.reshape %a : (tensor<2x3xf32>) -> tensor<3x2xf32>`（pretty。`-> tensor<f32>` も可） | `reshape` | ○ | ○ | |
| transpose | `%0 = stablehlo.transpose %a, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>`（pretty） | `transpose` | ○ | ○ | |
| dot_general | `%0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>`（pretty。バッチ付きは `batching_dims = [0] x [0], contracting_dims = [2] x [1]`、`precision = [DEFAULT, DEFAULT]` 付きも可）。既存 `tests/fixtures/stablehlo/matmul.mlir` の generic form も通る | `dot-general` | ○ | ○ | |
| reduce（add） | `%0 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>`（pretty。全軸 `dimensions = [0, 1]` → `tensor<f32>` も可）。既存 `tests/fixtures/stablehlo/reduce_sum.mlir` の generic form も通る | `reduce-sum` | ○ | ○ | init は `stablehlo.constant dense<0.0>`（f32/f64）または `dense<0x0000>`（bf16。ビット列そのまま） |
| reduce（max） | reduce（add）と同じ pretty form で `applies stablehlo.maximum`。init は `-inf` を16進で: f32 `0xFF800000`、bf16 `0xFF80`、f16 `0xFC00`、f64 `0xFFF0000000000000`（`dense<-inf>` は書かない） | `reduce-max` | ○ | ○ | |

f16 / f64 は上記すべての op で advisor が確認済み（確認済み、フィクスチャは
未収録）。ただし f64 は `to-device` が未対応（`unsupported-dtype`、issue
#37）のままなので、実行系連携のフィクスチャには使わない。

## IREE 未対応・要注意の op（代替・備考）

| op | 状況 | 代替 |
| --- | --- | --- |
| `stablehlo.custom_call` | `failed to legalize operation` でコンパイル不可（IREE の入力パイプラインが明示的に illegal にしている） | 使わない |
| `stablehlo.rng_bit_generator` | `%s, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)` の形でコンパイルは通る（計画時の懸念と異なる） | フェーズ3で確認する。`ui64` / `ui32` の device 表現（to-device）が前提になるため、フェーズ1では扱わない |

## その他の確認事項

- 関数の引数・返り値に rank 0（`tensor<f32>`）と `tensor<4xi1>` を使える
- 多値返し `func.return %0, %1 : tensor<4xf32>, tensor<4xf32>` は可
- `module { ... }` で包んでも包まなくても可（`backend-invoke` の
  `"module.main"` 前提は無名モジュールで満たされる）
- `iree-run-module` は `4xi1=1,0,1,0` 形式の i1 入力を受け付けない
  （"binary hex element count mismatch"）。そのため `:i1` は関数の内部値
  としてのみ使い、`to-device` は `:i1` を拒否する（issue #37 側の対応）

## フィクスチャとテスト

- `tests/fixtures/stablehlo/ops/<op>.mlir`（f32）と `<op>_bf16.mlir`（bf16）。
  `reduce` は `reduce_add.mlir` / `reduce_max.mlir` というファイル名にした
  （`reduce-sum` / `reduce-max` という2つのプリミティブに対応するため）
- `tests/iree/ops-test.lisp`:
  - 全38フィクスチャ（19 op × f32/bf16）を `backend-compile` して非空の
    vmfb になることを確かめる
  - フィクスチャの個数と `tests/fixtures/stablehlo/ops/` のファイル数が
    一致することを確かめる整合テスト
  - `add` / `dot_general` / `reduce`（add）の3つは、`backend-load` /
    `backend-invoke` まで通して実際に実行し、期待値と数値が一致することを
    確かめる

フィクスチャ40個（既存 add/matmul/reduce_sum の f32/bf16 各3個 + 本 PR の
38個）のコンパイルがスイートに加わる。1フィクスチャあたり約350ms
（vmfb ディスクキャッシュのヒット時はほぼ0ms）で、既定スイートに約15秒
（コールド時）が加わる見込み。CI の warm 実行時間の目安（約44秒、
CLAUDE.md 参照）に対する影響として記録しておく。
