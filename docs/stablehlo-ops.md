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
| dot_general | f32/f64: `%0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>`（pretty。バッチ付きは `batching_dims = [0] x [0], contracting_dims = [2] x [1]`、`precision = [DEFAULT, DEFAULT]` 付きも可）。既存 `tests/fixtures/stablehlo/matmul.mlir` の generic form も通る。bf16/f16: IREE（llvm-cpu）は縮約を入力の dtype のまま累積し、K が大きいと eager（single-float 累積）と許容誤差を超えてずれる（issue #54）ので、f32 の結果型を持つ `stablehlo.dot_general` を出してから `stablehlo.convert` で戻す2行にする: `%acc_0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xbf16>, tensor<3x2xbf16>) -> tensor<2x2xf32>` に続けて `%0 = stablehlo.convert %acc_0 : (tensor<2x2xf32>) -> tensor<2x2xbf16>`（JAX の `preferred_element_type=f32` と同じ考え方）。K=0（contracting 次元のサイズが 0）では IREE 3.11.0 のコンパイラが AnnotateDispatches の整数 0 除算で落ちる（issue #62。最小の再現は `docs/iree-repros/dot-general-k0.mlir`、上流への報告は #73 と `docs/iree-upstream-bugs.md`）ので、dot_general を出さず `%0 = stablehlo.constant dense<0.0> : <出力型>` を1行出す（全 float dtype。bf16/f16 の f32 累積も経由しない。数学的にも空和 = 0 で正しい） | `dot-general` | ○ | ○ | |
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

## 制御構造の op（対応表の外。フィクスチャは持たない）

`while-loop`（issue #131）は `stablehlo.while`（generic form）で出す。cond と body はブロック引数を持つリージョンで、cond は `stablehlo.return %c : tensor<i1>`、body は全オペランドと同じ型の値を `stablehlo.return` する。

```
%r0, %r1 = "stablehlo.while"(%a, %b) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>): ...
  stablehlo.return %c : tensor<i1>
}, {
  ^bb0(%x: tensor<f32>, %y: tensor<f32>): ...
  stablehlo.return %x1, %y1 : tensor<f32>, tensor<f32>
}) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
```

オペランドは carry と、cond / body が閉包で捕まえた値（body は素通しで返す）の全部。IREE でコンパイル・実行できることは `tests/iree/while-loop-test.lisp`（medium）で確かめる。

既知の制限（IREE 3.11 のコンパイラのバグ。issue #131 のレビューで確認）: 本体の中で比較から作った値（`:i1` のフラグ、またはそれを `i32` に変換・`select` した値）を carry にした while の結果が関数の戻り値になると、コンパイラが LLVM の `out of memory` / メモリフォルトでプロセスごと落ちる（`:i1` を `i32` として通す・`optimization_barrier` を挟む、のどれでも直らない）。戻り値にしない場合（フラグはループの継続判定にだけ使う）は動く。eager と PJRT は影響を受けない。nabla 側では防げないので、このような while の結果は jit の戻り値にしない（最小の再現は `docs/iree-repros/while-i1-carry-returned.mlir`、上流への報告の下書きは `docs/iree-upstream-bugs.md`）。`tests/iree/while-loop-test.lisp` の子プロセスのテストが、このバグが IREE に残っていることを守る（直れば失敗するので、この注意書きごと消す）。

もう1つの既知のバグ（IREE 3.11 のコンパイラ。issue #134 の jvp の CI で見つかった。上の `:i1` の carry のバグとは別）: `stablehlo.while` で、cond を駆動する carry（カウンタ）の初期値が `stablehlo.constant` で、ほかに carry が 2 つ以上あり、そのうち少なくとも 1 つが rank 1 以上のとき、コンパイラが非決定的に SIGSEGV / SIGBUS で落ちる（単体の `iree-compile` で 10 回中 6 回、定数の carry を 2 つにした最小の形では 10 回中 10 回）。単純な while の jvp（接線の carry が増える）がこの形になる。バックトレース（`ScheduleAllocationPass` の AffinityAnalysis）:

```
mlir::iree_compiler::Explorer::getTraversalAction(mlir::Operation*)
mlir::iree_compiler::Explorer::walkTransitiveUses(mlir::Value, ...)
mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS::updateValue(mlir::Value, mlir::iree_compiler::DFX::Solver&)
mlir::iree_compiler::DFX::Solver::updateElement(...)
```

nabla を使わない最小の再現（`docs/iree-repros/while-constant-carry.mlir`。carry 3 つで 50 回中 49 回落ちる）と上流への報告の下書きは `docs/iree-upstream-bugs.md`、回避策を外す条件は `docs/iree-build.md` の「IREE を上げたときに回避策を外せるか確かめる手順」にある。

回避策（nabla 側で入れてある）: while のオペランドのうち graph の定数（`stablehlo.constant` で出す値）のものを、while の前の `stablehlo.optimization_barrier` に通す（`src/while-loop.lisp` の `%while-barrier-lines`。値は変わらない）。実際の jvp の graph と最小の形で 10 回中 0 回に減ることを確かめた（引数のオペランドは通さない）。同じ値の定数は CSE で1つになるので、同じ定数で初期化した while が2つあると barrier どうしもまったく同じ形になってまとめられ、2つのループが同じ SSA 値から始まりうる。そこで barrier には、モジュールの中で一意な整数の `stablehlo.constant`（`%stablehlo-unique-id`。`%wbar_<out>_u_c`）も一緒に通す（結果は使わない。scan の `%scan-ys-init-lines` と同じやり方。issue #179。`tests/while-loop-test.lisp` の `while-loop/emits-a-distinct-salt-for-each-constant-barrier` が守る）。`tests/iree/while-loop-test.lisp`（medium）の、定数の carry を持つ生の StableHLO が今も落ちることを子プロセスで確かめるテストが、このバグが IREE に残っていることを守り（直れば失敗するので、回避策・このテスト・この注意書きを消す）、barrier 付きの StableHLO が 10 回続けてコンパイルできることを別のテストが確かめる。

### 制御構造（issue #130）

| op | 形 | nabla プリミティブ名 | 備考 |
| --- | --- | --- | --- |
| if | `%o = "stablehlo.if"(%pred) ({ ...; stablehlo.return %r : T }, { ...; stablehlo.return %r : T }) : (tensor<i1>) -> (T)`（generic 形。リージョンは2つで、外側の SSA 値を直接参照する） | `cond`（公開名 `cond*`） | IREE の `llvm-cpu` でコンパイル・実行できる（`tests/iree/cond-test.lisp`）。複数出力は `%a, %b = ...`、入れ子も可。`stablehlo.case` は添え字が `i32` なので、`i1` の pred を変換せずに使える `if` を選んだ |

## IREE 未対応・要注意の op（代替・備考）

| op | 状況 | 代替 |
| --- | --- | --- |
| `stablehlo.custom_call` | `failed to legalize operation` でコンパイル不可（IREE の入力パイプラインが明示的に illegal にしている） | 使わない |
| `stablehlo.rng_bit_generator` | 対応済み（issue #133、`rng-bit-generator` プリミティブ）。`%s2, %b = stablehlo.rng_bit_generator %s, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<…xui32>)` の形で、rank 0 と rank 4、奇数の次元、`ui64` の出力でもコンパイル・実行できる。**IREE（local）と PJRT（XLA CPU）は同じ状態から同じ結果（新しい状態とビット）をビット単位で返し、eager 実装とも一致する**（rank 0〜4・23通りの形状・`ui32` / `ui64`・64ビット全域の状態で確認。`tests/iree/rng-test.lisp` と `tests/pjrt/rng-test.lisp`） | 使える。状態 `ui64[2]` は `[0]` = 鍵（下位32ビット = key0、上位32ビット = key1）、`[1]` = カウンタ。新しい状態は `[0]` を保ち `[1]` を生成した64ビット単位の個数だけ進める。アルゴリズムの写し元と配置は `src/primitives/rng.lisp` の冒頭。`ui32` は Threefry-2x32 の2出力を別の要素にし、最初の偶数の次元（無ければ最大の次元）を半分にして並べる（要素数が偶数なら count は要素数の半分）。公開の PRNG API は issue #136（`prng-key` / `split` / `uniform` など。README の「PRNG」） |

## PRNG が使う op（issue #136。IREE（local）と PJRT（XLA CPU）の両方で実測）

| op | 状況 |
| --- | --- |
| `stablehlo.bitcast_convert` | 同じ幅（`ui32` ⇄ `f32` / `i32`、`ui64` ⇄ `f64`）に加え、幅が違う `tensor<2xui32>` → `tensor<ui64>`（末尾の次元が消える）と、その逆（`tensor<…xui64>` → `tensor<…x2xui32>`）もコンパイル・実行できる。並びはリトルエンディアン（先頭の要素が下位32ビット）。eager 実装（`bitcast-convert`）と一致 |
| `stablehlo.shift_right_logical` / `stablehlo.or`（`ui32` / `ui64`） | 使える。量がビット幅以上のとき 0（eager も同じ） |
| バッチ次元つきの `rng_bit_generator`（`stablehlo.while` の本体で `dynamic_slice` → K 回の `rng_bit_generator` → `dynamic_update_slice`。issue #164 / #178） | `ui64[2]` の状態を while の1回で K 行（16）ずつ取り出して呼ぶ形が、IREE・PJRT とも eager とビット単位で一致する。カウンタとビットのバッファの 0 初期値は、scan の ys と同じく `%scan-ys-init-lines` で作る（制御構造の節の while のバグの回避策と、ループごとに別のバッファにするための一意な整数。K 行以下のときは while を出さずに `slice` / `concatenate` で展開する）。行数が K の倍数でないとき、最後の1回の開始位置は `dynamic_slice` / `dynamic_update_slice` の添字のクランプで行数 - K になり、前の回と重なる。本体は書き換えない元の状態（while に同じ値を2つ目の carry として渡す）から読むので、重なった行は同じ値で上書きされる（IREE・PJRT とも行数 1・K-1・K+1・2K+3 で eager と一致）。本体ではビットのバッファを `optimization_barrier` に通してから `dynamic_update_slice` に渡し、in-place に書く（#159 と同じ回避。状態の carry は小さいので通さない）。**コンパイルコスト**（IREE local、eqn 1つ）: K = 16 で 32 行・256 行とも約 1.0 秒（MLIR 約 12.5 KB で、16 行を超えれば行数に依らない）。1行ずつ回す形（#164）は約 0.5 秒（MLIR 約 3 KB）、全行の展開は 32 行 4.0 秒、256 行 42.8 秒（MLIR 179 KB）だった。**実行時間**は IREE ではループ1回ごとの起動の費用（4コアの機械で1回約 0.16 ms）× ceil(行数 / K) が残る（`vmap` した `uniform` の jit、256 キー × 4 / 1024 要素が 4.4 / 7.2 ms、64 キー × 16384 要素が 12.4 ms。1行ずつ回す形では 46 / 52 / 21 ms、全行の展開では 0.8 / 2.8 / 8.0 ms。バッファを in-place に書く前は 1024 要素で 99 ms、16384 要素で 72 ms だった）。PJRT（XLA CPU）は 4 ms で変わらない |

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

## scan が使う op（issue #132）

`scan` プリミティブ（`src/scan.lisp`）は、次の op を `:emit` の中だけで使う（IR のプリミティブではない）。IREE（`iree-3.11.0`）が受け付けることは `tests/iree/scan-test.lisp` で確かめている。

| op | 綴り（form） | 用途 |
| --- | --- | --- |
| while | `%n, %c, %y = "stablehlo.while"(%i0, %init, %ybuf) ({ ^bb0(...): ... stablehlo.return %p : tensor<i1> }, { ^bb0(...): ... stablehlo.return ... })` | ループ本体。carry の先頭に `tensor<i32>` のカウンタ、末尾に ys のバッファを足す。consts は外側の SSA 名をリージョンの中から直接参照する |
| compare | `%p = stablehlo.compare LT, %i, %len : (tensor<i32>, tensor<i32>) -> tensor<i1>` | cond: カウンタ < length |
| dynamic_slice | `%s = stablehlo.dynamic_slice %xs, %idx, %z, sizes = [1, 3] : (tensor<4x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x3xf32>` | x_t を読む（先頭の軸だけ動的、他の添字は 0） |
| reshape | `%x = stablehlo.reshape %s : (tensor<1x3xf32>) -> tensor<3xf32>` | 先頭の軸 1 を落とす / y_t に足す |
| dynamic_update_slice | `%w = stablehlo.dynamic_update_slice %ybuf, %y1, %idx, %z : (tensor<4x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>` | y_t を ys のバッファに書く |
| subtract / add | `%idx = stablehlo.subtract %last, %i : tensor<i32>` | reverse の添字 `length-1-i` / カウンタの増分 |
| optimization_barrier | `%k = stablehlo.optimization_barrier %ybuf : tensor<4x3xf32>` | while の定数オペランドを通す（下の IREE のバグの回避）/ 本体で ys のバッファを通す（下の性能の節） |

IREE 3.11 のコンパイラバグの回避: cond を決める carry が `stablehlo.constant` で初期化された `stablehlo.while`（他に carry が2つ以上、うち1つは rank 1 以上）は、Stream の AffinityAnalysis（ScheduleAllocationPass）が非決定的に segfault する。scan の while はこの形なので、カウンタと ys の0初期値は `stablehlo.optimization_barrier` を通してから while に渡す。ただし長さ 1 の scan は IREE が while を `scf.for` に変換し、barrier があると `stream.resource` の型の不一致でコンパイルに失敗するので、barrier を付けない（長さ 1 ではクラッシュしない）。バグの詳細は制御構造の節（PR #157）を参照。

性能（issue #159）: ys のバッファは、本体の先頭で `stablehlo.optimization_barrier` に通してから `dynamic_update_slice` に渡す（`%p_bk<j> = stablehlo.optimization_barrier %p_by<j>`）。

- 原因: IREE 3.11 の `iree-stream-conversion`（ConvertToStreamPass）は、ループ（`scf.for` / `scf.while`）の carry であるブロック引数の配置先（affinity）を AffinityAnalysis から引けない。そのため本体の中で carry を使うたびに、転送元が不明の `stream.async.transfer ... -> to(@__device_0)` を挟み、この転送は同じデバイスへのものでも消されず（`ElideAsyncCopiesPass` が消すのは `clone` だけ）、毎ステップ carry 全体のコピー（`stream.cmd.copy`）と新しいバッファの確保になる。ys のバッファ（`length × |y_t|`）も carry なので、ys を持つ scan は長さに対して2乗の時間がかかっていた。`--compile-to=stream` の出力で、本体に ys 全体の `stream.cmd.copy` があることで確かめられる。`--iree-stream-affinity-solver-max-iterations` を増やしても変わらない
- 回避: `util.optimization_barrier`（`stablehlo.optimization_barrier` から変換される）の結果は配置先が引けるので、転送が挟まらず、`dynamic_update_slice` が ys のバッファを in-place に書く（本体のコピーは y_t の1行ぶんだけになる）。carry 自体（`|carry|` ぶんのコピー）は本体の計算と同じ程度なので、そのままにしている。結果の値は変わらない
- in-place にすると、それまで毎ステップのコピーで隠れていた IREE 3.11 の別の問題が出るので、ys の初期値は次の形にする（`%scan-ys-init-lines`）:
  - **ys ごとに別のバッファにする**。同じ型の 0 の `stablehlo.constant` は CSE で1つになり、別々の `optimization_barrier` も1つの値に通るので、2つの ys が同じバッファに書き込んで同じ値になる（jvp / vjp の scan で再現）。スカラーの 0 をまとめて1つの多出力の `optimization_barrier` に通し（結果はそれぞれ別の値）、それぞれを `broadcast_in_dim` で広げる。同じモジュールに同じ入力の scan が2つ（順方向と逆方向など）あると、barrier のオペランドが同じになって CSE で1つにまとめられ、2つの scan が同じ ys のバッファやカウンタを書き換える（#167 の rank 0 / rank 2 のテストと重ねて見つかった。カウンタを共有すると、先の while が進めた値から次の while が始まり、逆方向の scan が1回も回らない。どの組み合わせで壊れるかは定数の値などで変わり、予測できない）。そこでカウンタの 0・ys のスカラーの 0・モジュールの中で一意な整数の `stablehlo.constant`（`%stablehlo-unique-id`）を1つの多出力の barrier に通し、ループごとに別の値にする
  - **広げたバッファをそれぞれ `optimization_barrier` に通してから while に渡す**。通さないと初期値が1つの確保（subview の詰め合わせ）にまとめられ、ループの中で使っているのに while の前で `stream.resource.dealloca` される（ys が3つ以上だと実行時に `ref is null; while invoking native function hal.buffer.subspan` で落ち、2つでは偶然動く use-after-free）
  - 長さ 1 の scan は1回しか書かないので、従来どおり初期値は constant のまま、本体の barrier も付けない
- 実測（IREE local、f32、幅 1024、carry は `tanh(h)+h`、jit の呼び出し3回の最短。ホストへの転送を含む。値は実行ごとに揺れる）:

| 長さ | ys なし | ys あり（回避前） | ys あり（回避後） |
| --- | --- | --- | --- |
| 1000 | 20〜40 ms | 956 ms | 20 ms |
| 4000 | 48〜116 ms | 16488 ms | 64〜256 ms |

  守るテストは、出力の形を検査する small テスト `tests/scan-test.lisp` の `scan/emits-ys-buffers-through-barriers`（本体で ys ごとに barrier を通して dynamic_update_slice に渡すこと、初期値が ys ごとに別の broadcast_in_dim + barrier の組であること、長さ 1 ではどちらも無いこと）、実行時間を測る large テスト `tests/iree/scan-test.lisp` の `scan/iree-ys-write-cost-is-linear-in-length`（長さ 1000 で ys ありが ys なしの 5 倍 + 100 ms 以内。回避前は 748 ms 対 20 ms で落ちる。時間は負荷で揺れるので既定のスイートには入れない）と `scan/iree-same-typed-ys-do-not-share-a-buffer`

長さ 0 の scan は、`dynamic_slice` の切り出し幅 1 が長さ 0 の軸を超えて不正になるので `while` を出さない。carry は入力と同じ型の `stablehlo.reshape` で素通しにし、ys は `stablehlo.constant dense<> : tensor<0x...>` にする（この形を IREE が受け付けることを確認済み）。

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
