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
  （`--iree-input-type=stablehlo --iree-input-demote-f64-to-f32=false
  --iree-hal-target-device=local
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
| add | `%0 = stablehlo.add %a, %b : tensor<4x8xf32>`（pretty†） | `add` | ○ | ○ | |
| subtract | `%0 = stablehlo.subtract %a, %b : tensor<4x8xf32>`（pretty） | `sub` | ○ | ○ | |
| multiply | `%0 = stablehlo.multiply %a, %b : tensor<4x8xf32>`（pretty） | `mul` | ○ | ○ | |
| divide | `%0 = stablehlo.divide %a, %b : tensor<4x8xf32>`（pretty） | `div` | ○ | ○ | |
| maximum | `%0 = stablehlo.maximum %a, %b : tensor<4x8xf32>`（pretty） | `max` | ○ | ○ | |
| minimum | `%0 = stablehlo.minimum %a, %b : tensor<4x8xf32>`（pretty） | `min` | ○ | ○ | |
| negate | `%0 = stablehlo.negate %a : tensor<4xf32>`（pretty、単項） | `neg` | ○ | ○ | |
| exponential | `%0 = stablehlo.exponential %a : tensor<4xf32>`（pretty、単項） | `exp` | ○ | ○ | |
| log | `%0 = stablehlo.log %a : tensor<4xf32>`（pretty、単項） | `log` | ○ | ○ | |
| tanh | `%0 = stablehlo.tanh %a : tensor<4xf32>`（pretty、単項） | `tanh` | ○ | ○ | |
| compare | `%0 = stablehlo.compare LT, %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>`（pretty。方向は LT LE GT GE EQ NE、`, FLOAT` の compare_type 付きも可。generic form† `"stablehlo.compare"(%a, %b) {comparison_direction = #stablehlo<comparison_direction LT>}` も通る） | `compare` | ○ | ○ | 出力 dtype は nabla の `:i1`（issue #37）。`:i1` は IREE 側では1要素1バイトの `IREE_HAL_ELEMENT_TYPE_BOOL_8` になり、`to-device` / `to-host` が BIT 配列との間で詰め直す（issue #72）。関数の引数・返り値にも使える |
| select | `%1 = stablehlo.select %pred, %a, %b : tensor<4xi1>, tensor<4xf32>`（pretty） | `select` | ○ | ○ | |
| convert | `%0 = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xbf16>`（pretty） | `convert` | ○ | ○ | |
| constant | `%c = stablehlo.constant dense<[1.0, 2.5]> : tensor<2xf32>`（rank 0 は `dense<3.0> : tensor<f32>`）。bf16/f16 は16進ビット列: `dense<[0x3F80, 0x4020]> : tensor<2xbf16>`（実行結果も正しい: 1, 2.5）。max の初期値のような単一値も同じ書き方: `dense<0xFC00> : tensor<f16>`、`dense<0xFF800000> : tensor<f32>` | プリミティブではなく `graph-constants` | ○ | ○ | |
| broadcast_in_dim | `%0 = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>`（pretty。rank 0 元は `dims = []`） | `broadcast-in-dim` | ○ | ○ | |
| reshape | `%0 = stablehlo.reshape %a : (tensor<2x3xf32>) -> tensor<3x2xf32>`（pretty。`-> tensor<f32>` も可） | `reshape` | ○ | ○ | |
| transpose | `%0 = stablehlo.transpose %a, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>`（pretty） | `transpose` | ○ | ○ | |
| dot_general | f32/f64: `%0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>`（pretty。バッチ付きは `batching_dims = [0] x [0], contracting_dims = [2] x [1]`、`precision = [DEFAULT, DEFAULT]` 付きも可）。既存 `tests/fixtures/stablehlo/matmul.mlir` の generic form も通る。bf16/f16: IREE（llvm-cpu）は縮約を入力の dtype のまま累積し、K が大きいと eager（single-float 累積）と許容誤差を超えてずれる（issue #54）ので、f32 の結果型を持つ `stablehlo.dot_general` を出してから `stablehlo.convert` で戻す2行にする: `%acc_0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xbf16>, tensor<3x2xbf16>) -> tensor<2x2xf32>` に続けて `%0 = stablehlo.convert %acc_0 : (tensor<2x2xf32>) -> tensor<2x2xbf16>`（JAX の `preferred_element_type=f32` と同じ考え方）。K=0（contracting 次元のサイズが 0）では IREE 3.11.0 のコンパイラが AnnotateDispatches の整数 0 除算で落ちる（issue #62）ので、dot_general を出さず `%0 = stablehlo.constant dense<0.0> : <出力型>` を1行出す（全 float dtype。bf16/f16 の f32 累積も経由しない。数学的にも空和 = 0 で正しい） | `dot-general` | ○ | ○ | |
| reduce（add） | f32/f64: `%0 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>`（pretty。全軸 `dimensions = [0, 1]` → `tensor<f32>` も可）。既存 `tests/fixtures/stablehlo/reduce_sum.mlir` の generic form も通る。bf16/f16: IREE（llvm-cpu）は reduce（add）を入力の dtype のまま累積し、軸長が大きいと eager（single-float 累積）と許容誤差を超えてずれる（issue #63、dot_general の issue #54 と同じ原因）ので、入力を f32 に `stablehlo.convert` → f32 の init で reduce → 結果を元の dtype に `stablehlo.convert` して戻す4行にする: `%in32 = stablehlo.convert %a : (tensor<4x8xbf16>) -> tensor<4x8xf32>` → `%init = stablehlo.constant dense<0.0> : tensor<f32>` → `%acc = stablehlo.reduce(%in32 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>` → `%0 = stablehlo.convert %acc : (tensor<4xf32>) -> tensor<4xbf16>`（dot_general の `preferred_element_type=f32` と同じ考え方） | `reduce-sum` | ○ | ○ | init は `stablehlo.constant dense<0.0>`。bf16/f16 も f32 累積の init として同じ `tensor<f32>` の `0.0` を使う（`dense<0x0000>` は使わなくなった） |
| reduce（max） | reduce（add）と同じ pretty form で `applies stablehlo.maximum`。init は `-inf` を16進で: f32 `0xFF800000`、bf16 `0xFF80`、f16 `0xFC00`、f64 `0xFFF0000000000000`（`dense<-inf>` は書かない） | `reduce-max` | ○ | ○ | |
| optimization_barrier | `%0 = stablehlo.optimization_barrier %a : tensor<4xf32>`（pretty、1オペランド） | `stop-gradient` | ○ | ○ | StableHLO には恒等の op が無い（`stablehlo.convert` を同じ dtype でかけると定数畳み込みで消えうる）ので、値を変えず最適化の境界になる optimization_barrier を使う。IREE 3.11.0 がコンパイル・実行できることを確認済み。任意の dtype（`:i1` も）を通す。issue #80 |

f16 / f64 は上記すべての op で advisor が確認済み（フィクスチャは
未収録）。f64 は issue #72 で `to-device` / `to-host` が対応し、`jit` で
実行できる。IREE は既定で f64 を f32 に落とす（`demoteF64ToF32 = true`、
`compiler/src/iree/compiler/Pipelines/Options.h`）ので、`compile-flags` は
`--iree-input-demote-f64-to-f32=false` を付ける。llvm-cpu では
exponential / log / tanh の f64 版が多項式近似されず（`MathTransformPass.cpp`
の近似・f32 展開は f32 以下の型だけが対象）libm の `exp` / `log` / `tanh`
の呼び出しとして残る。既定の embedded ELF（`-nostdlib -static`）ではこれが
リンクできずにコンパイルが失敗する（`iree-lld: error: undefined symbol:
tanh`）。そこで `backend-compile` は、StableHLO のテキストに f64 の
この3つがあるとき（`nabla.iree::%needs-libm-p`）だけ
`--iree-llvmcpu-link-embedded=false` を付け、ランタイムが dlopen で読み込む
system library を作る。読み込み時にプロセスの libm に解決されるので、f64 の
この3つも `jit` できる。代わりにそのモジュールのコンパイルには `ld.lld` が
要る（IREE は system linker を `-nostdlib -static -shared` で呼び、これを
共有ライブラリとして扱えるのは `ld.lld` だけ。`ld.bfd` / `ld.gold` では
失敗する。固定コミットの `iree-lld` は `-flavor` を要求するので使えない）。
PATH 上の `ld.lld` が使われ、`IREE_LLVM_SYSTEM_LINKER_PATH` で変えられる。
それ以外のモジュールは従来どおり固定コミットの `iree-lld` で embedded ELF
にリンクする。

### 制御構造（issue #130）

| op | 形 | nabla プリミティブ名 | 備考 |
| --- | --- | --- | --- |
| if | `%o = "stablehlo.if"(%pred) ({ ...; stablehlo.return %r : T }, { ...; stablehlo.return %r : T }) : (tensor<i1>) -> (T)`（generic 形。リージョンは2つで、外側の SSA 値を直接参照する） | `cond`（公開名 `cond*`） | IREE の `llvm-cpu` でコンパイル・実行できる（`tests/iree/cond-test.lisp`）。複数出力は `%a, %b = ...`、入れ子も可。`stablehlo.case` は添え字が `i32` なので、`i1` の pred を変換せずに使える `if` を選んだ |

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
- `iree-run-module` は `4xi1=1,0,1,0`（`4xi1=1 0 1 0` も）形式の i1 入力を
  受け付けない（`string_util.c:506: binary hex element count mismatch`）。
  これは iree-run-module のテキスト入力の解析だけの制限で、ランタイムの
  C API で `IREE_HAL_ELEMENT_TYPE_BOOL_8` の buffer view を渡せば i1 の
  引数として使える（issue #72 の `to-device`）。iree-run-module では
  `4xi8=1,0,1,0` と書けば通り、i1 の返り値は `4xi8=1 0 1 0` と表示される
  （issue #72 で確認）

## フィクスチャとテスト

- `tests/fixtures/stablehlo/ops/<op>.mlir`（f32）と `<op>_bf16.mlir`（bf16）。
  `reduce` は `reduce_add.mlir` / `reduce_max.mlir` というファイル名にした
  （`reduce-sum` / `reduce-max` という2つのプリミティブに対応するため）
- `tests/iree/ops-test.lisp`:
  - 全42フィクスチャ（19 op + constant + optimization_barrier の21行 × f32/bf16）を `backend-compile` して非空の
    vmfb になることを確かめる
  - フィクスチャの個数と `tests/fixtures/stablehlo/ops/` のファイル数が
    一致することを確かめる整合テスト
  - `add` / `dot_general` / `reduce`（add）の3つは、`backend-load` /
    `backend-invoke` まで通して実際に実行し、期待値と数値が一致することを
    確かめる

フィクスチャ48個（既存 add/matmul/reduce_sum の f32/bf16 各3個 = 6個 + 本 PR
の42個。表の21行（19 op + constant + optimization_barrier）× f32/bf16。
`ls tests/fixtures/stablehlo/*.mlir tests/fixtures/stablehlo/ops/*.mlir | wc -l`
で数えられる）のコンパイルがスイートに加わる。1フィクスチャあたり約350ms
（vmfb ディスクキャッシュのヒット時はほぼ0ms）で、既定スイートに約15秒
（コールド時）が加わる見込み。CI の warm 実行時間の目安（約44秒、
CLAUDE.md 参照）に対する影響として記録しておく。
