# IREE 上流に報告するバグ（issue #73、#165、#199、#203、#220）

nabla が固定している IREE 3.11.0 で見つけたバグ7件（1〜3・5 はコンパイラ、4・6・7 はランタイム）の、nabla を使わない最小の再現と、iree-org/iree にそのまま貼れる issue の下書き。**上流への報告はユーザーが行う**（このリポジトリの作業では iree-org/iree に何も投稿しない）。報告したら、各節の「上流 issue」の行にリンクを書き、nabla の issue（#73 / #165）にも同じリンクを記録する。

IREE を上げたときに、これらのバグが直ったかを確かめて回避策を外す手順は [`docs/iree-build.md`](iree-build.md) の「IREE を上げたときに回避策を外せるか確かめる手順」にある。

## 共通の環境

| 項目 | 値 |
| --- | --- |
| IREE | 3.11.0（`third_party/iree.lock`。`iree-compile --version` は `IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`、`LLVM version 23.0.0git`、Optimized build） |
| コンパイラの入手元 | PyPI の `iree-base-compiler==3.11.0`（manylinux、x86_64。`scripts/build-iree.sh --compiler=wheel` が `$NABLA_IREE_HOME/bin/` に置いたもの） |
| ターゲット | `--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu` |
| OS / CPU | Linux x86_64（Ubuntu 24.04 相当、glibc。AVX-512 のある CPU） |
| 確認日 | 2026-10-04 |

コマンドはどれも次の形（`<file>` は `docs/iree-repros/` のファイル）。入力の形式は自動判定に任せる（`--iree-input-type` は付けない）。

```sh
$NABLA_IREE_HOME/bin/iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  docs/iree-repros/<file> -o /dev/null
```

まとめて確かめるには `scripts/check-iree-repros.sh`（各ファイルを `--runs` 回、既定 10 回コンパイルし、落ちた回数を出す）。

| # | 再現ファイル | 落ちる pass | 症状 | 再現率（単体の iree-compile） | 回避策の形 | nabla の issue | 上流 issue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `dot-general-k0.mlir` | `AnnotateDispatchesPass`（Flow） | SIGFPE（整数 0 除算、終了コード 136） | 決定的（24/24） | （ファイル無し。dot_general を出さずゼロ定数にする） | #62、#73 | 未報告（ユーザーが報告する）: `<link>` |
| 2 | `while-constant-carry.mlir` | `ScheduleAllocationPass`（Stream の AffinityAnalysis） | SIGSEGV（終了コード 139） | 非決定的。49/50（30/30 と 19/20） | `while-constant-carry-barrier.mlir`: 0/20 | #165 | 未報告（ユーザーが報告する）: `<link>` |
| 3 | `while-i1-carry-returned.mlir` | `ConvertToStreamPass`（`ScfWhileOpConversion`） | SIGSEGV（139）または `LLVM ERROR: out of memory` で SIGABRT（134） | 落ちること自体は決定的（33/33）、どちらのシグナルかは非決定的 | `while-i1-carry-not-returned.mlir`: 0/20（回避策ではなく、戻り値にしない対照） | #131、#165 | 未報告（ユーザーが報告する）: `<link>` |

## 1. K=0 の `stablehlo.dot_general` で AnnotateDispatches が整数 0 除算（#73）

- **状態**: 未報告（ユーザーが報告する）。上流 issue: `<link>`
- **再現ファイル**: [`docs/iree-repros/dot-general-k0.mlir`](iree-repros/dot-general-k0.mlir)
- **期待**: コンパイルが通る（K=0 の行列積は空和なので結果は全 0。少なくともクラッシュせずにエラーで止まる）
- **実際**: `uninitialized values` の警告を2回出した後、`AnnotateDispatchesPass` の中で SIGFPE（x86 の整数 0 除算 #DE）。毎回落ちる
- **原因の見立て**: `compiler/src/iree/compiler/Dialect/Flow/Transforms/AnnotateDispatches.cpp` の `saturatingMul` が `assert(lhs > 0 && rhs > 0)` のうえで `lhs > kMaxCost / rhs` を計算する。リリースビルドでは assert が消えるので、ゼロサイズの次元（rhs = 0）で `kMaxCost / 0` になる。バックトレースの `summarizeDispatchRegion` の lambda はこの関数をインライン展開したものと読める
- **上流の既知の情報（2026-10-04 時点）**: 同じ関数の assert を「非負を許し、0 なら 0 を返す」に緩める PR [iree-org/iree#24947](https://github.com/iree-org/iree/pull/24947)（ONNX の空集合 reduction `2x0x4` で見つかった assert 失敗の修正。Open、未マージ）がある。ゼロ除算の issue としての報告は見つからなかった。報告するときはこの PR に触れ、「リリースビルドでは assert ではなく SIGFPE になる」「dot_general（matmul）の K=0 でも起きる」ことを足す
- **nabla 側の回避策**: `src/primitives/dot.lisp` の `%dot-zero-contracting-p` / `%dot-zero-constant-line` / `%dot-emit-lines` が、K=0 のとき dot_general を出さずに `stablehlo.constant dense<0.0>` を出す（PR #65）。in-process のコンパイラでこの #DE を踏んだ後はコンパイラを使用不可（poisoned）にする（`src/iree/compiler.lisp`、PR #69）

バックトレース（抜粋）:

```
Stack dump:
 #4 ... llvm::function_ref<void (mlir::Operation*)>::callback_fn<mlir::iree_compiler::IREE::Flow::summarizeDispatchRegion[abi:cxx11](mlir::Region&)::$_0>(long, mlir::Operation*) AnnotateDispatches.cpp:0:0
 #5 ... mlir::iree_compiler::IREE::Flow::summarizeDispatchRegion[abi:cxx11](mlir::Region&) AnnotateDispatches.cpp:0:0
 #6 ... mlir::iree_compiler::IREE::Flow::AnnotateDispatchesPass::runOnOperation() AnnotateDispatches.cpp:0:0
 ...
#10 ... mlir::iree_compiler::ConstEval::(anonymous namespace)::JitGlobalsPass::runOnOperation() JitGlobals.cpp:0:0
...
Floating point exception
```

### 下書き（英語）

**Title**: `[Flow] AnnotateDispatches crashes with SIGFPE (integer divide-by-zero) on stablehlo.dot_general with a zero-size contracting dimension`

````markdown
### What happened?

Compiling a `stablehlo.dot_general` whose contracting dimension has size 0 (K = 0)
crashes `iree-compile` with SIGFPE (integer divide-by-zero) inside
`AnnotateDispatchesPass` / `summarizeDispatchRegion`. The crash is deterministic.

The likely cause is `saturatingMul` in
`compiler/src/iree/compiler/Dialect/Flow/Transforms/AnnotateDispatches.cpp`:
it asserts `lhs > 0 && rhs > 0` and then computes `kMaxCost / rhs`. In release
builds the assert is compiled out, so a zero-extent dimension divides by zero.
#24947 relaxes that assert for the empty-reduction case; this report shows that
release builds hit a hard SIGFPE (not an assert) and that a plain K = 0 matmul is
enough to trigger it.

Expected: the module compiles (an empty contraction is all zeros), or at least
fails with a diagnostic instead of killing the process.

### Steps to reproduce your issue

`k0.mlir`:

```mlir
func.func @main(%a: tensor<2x0xf32>, %b: tensor<0x3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x0xf32>, tensor<0x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
```

```
iree-compile --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu k0.mlir -o /dev/null
```

Output (after two "reads uninitialized values" warnings on the `linalg.matmul`):

```
Stack dump:
 #4 ... callback_fn<mlir::iree_compiler::IREE::Flow::summarizeDispatchRegion[abi:cxx11](mlir::Region&)::$_0>(long, mlir::Operation*) AnnotateDispatches.cpp:0:0
 #5 ... mlir::iree_compiler::IREE::Flow::summarizeDispatchRegion[abi:cxx11](mlir::Region&) AnnotateDispatches.cpp:0:0
 #6 ... mlir::iree_compiler::IREE::Flow::AnnotateDispatchesPass::runOnOperation() AnnotateDispatches.cpp:0:0
 #7 ... mlir::detail::OpToOpPassAdaptor::run(...)
#10 ... mlir::iree_compiler::ConstEval::(anonymous namespace)::JitGlobalsPass::runOnOperation() JitGlobals.cpp:0:0
#13 ... ireeCompilerInvocationPipeline
Floating point exception (exit code 136)
```

24 out of 24 runs crashed.

### What component(s) does this issue relate to?

Compiler

### Version information

`iree-base-compiler==3.11.0` from PyPI
(`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`,
LLVM 23.0.0git, optimized build), Linux x86_64.

### Additional context

Related: #24947 (same function, assert on zero-extent dims). When the compiler is
used in-process through the embedding C API, the SIGFPE takes down the host
process; we work around it by not emitting `dot_general` when K = 0.
````

## 2. 定数で初期化した carry がループを駆動する `stablehlo.while` で AffinityAnalysis が segfault（#165 の 1）

- **状態**: 未報告（ユーザーが報告する）。上流 issue: `<link>`
- **再現ファイル**: [`docs/iree-repros/while-constant-carry.mlir`](iree-repros/while-constant-carry.mlir)。対照（回避策を入れた形）: [`while-constant-carry-barrier.mlir`](iree-repros/while-constant-carry-barrier.mlir)（定数を `stablehlo.optimization_barrier` に通しただけで、ほかは同じ）
- **期待**: コンパイルが通る
- **実際**: `ScheduleAllocationPass` の Stream AffinityAnalysis（`ValueConsumerAffinityPVS::updateValue` → `Explorer::walkTransitiveUses` の再帰）で SIGSEGV。非決定的で、50 回中 49 回落ちた（30/30 と 19/20）。対照は 20/20 通る
- **最小化で分かった条件**（いずれも 10 回ずつ）:
  - carry は (f32 のカウンタ = `stablehlo.constant`、`tensor<3xf32>` の引数、f32 の引数) の3つ。**carry を2つ**（カウンタと rank 1 の carry）に減らすと 0/10 で落ちない。3つ目を、ループの上限を素通しするだけの carry（本体で更新しない）にした形も 0/10
  - カウンタを **i32** にすると 0/10（ループ回数が定数で決まるので別の経路になると見られる）。nabla の `scan` は i32 のカウンタだが、ys のバッファも定数で初期化するので、回避策を同じく入れてある
  - ループの上限は cond の中の定数（再現ファイル）でも、4つ目の carry（引数）にして (定数, rank 1, 引数, 上限) の4つにしても落ちる（後者は 9/10）
  - その4つの形で3つ目の carry も同じ定数にすると 10/10（`tests/iree/while-loop-test.lisp` のガードテストの形に近い）
  - `stablehlo.optimization_barrier` を定数とループの間に挟むと 0/20
- **nabla 側の回避策**: `src/while-loop.lisp` の `%while-barrier-lines`（while のオペランドのうち定数のものを barrier に通す）と、`src/scan.lisp` の `%scan-ys-init-lines`（scan とバッチされた rng の while のカウンタ・ys の 0 初期値を、モジュールの中で一意な整数と一緒に1つの barrier に通す。長さ 1 の scan は除く）

バックトレース（抜粋）:

```
 #4 ... mlir::iree_compiler::Explorer::walkTransitiveUses(mlir::Value, std::function<mlir::WalkResult (mlir::OpOperand&)>, mlir::iree_compiler::TraversalBehavior) Explorer.cpp:0:0
 #5 ... mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS::updateValue(mlir::Value, mlir::iree_compiler::DFX::Solver&) Affinity.cpp:0:0
 #6 ... mlir::iree_compiler::DFX::Solver::updateElement(mlir::iree_compiler::DFX::AbstractElement&) Solver.cpp:0:0
 #7 ... DFX::Solver::getOrCreateElementFor<mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS>(...)
 #8 ... mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS::updateFromUse(...)
 （#4〜#8 の組がさらに3回入れ子になる）
#35 ... mlir::iree_compiler::IREE::Stream::AffinityAnalysis::run() Affinity.cpp:0:0
#36 ... mlir::iree_compiler::IREE::Stream::(anonymous namespace)::ScheduleAllocationPass::runOnOperation() ScheduleAllocation.cpp:0:0
Segmentation fault (exit code 139)
```

### 下書き（英語）

**Title**: `[Stream] Nondeterministic segfault in AffinityAnalysis (ValueConsumerAffinityPVS / walkTransitiveUses) for stablehlo.while whose loop counter is initialized by a constant`

````markdown
### What happened?

`iree-compile` segfaults in `ScheduleAllocationPass` -> `AffinityAnalysis::run()`
-> `ValueConsumerAffinityPVS::updateValue` -> `Explorer::walkTransitiveUses` for a
small `stablehlo.while`. The crash is nondeterministic but very frequent:
49 out of 50 runs of the same command crashed.

Conditions found while reducing (10 runs each):

- the carry that drives the condition (an f32 counter) is initialized by
  `stablehlo.constant`;
- there are at least two other carries that the body updates, one of which
  has rank >= 1 (with only the counter and one rank-1 carry: 0/10 crashes; with
  a third carry that is only passed through unchanged: 0/10);
- with an `i32` counter instead of f32 it did not crash (0/10);
- putting the constant through `stablehlo.optimization_barrier` before the loop
  makes it compile reliably (0/20 crashes). We use that as a workaround.

Expected: the module compiles deterministically.

### Steps to reproduce your issue

`while_constant_carry.mlir`:

```mlir
func.func @main(%x: tensor<3xf32>, %y: tensor<f32>) -> tensor<3xf32> {
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %r:3 = "stablehlo.while"(%c0, %x, %y) ({
    ^bb0(%i: tensor<f32>, %v: tensor<3xf32>, %s: tensor<f32>):
      %n = stablehlo.constant dense<4.0> : tensor<f32>
      %p = stablehlo.compare LT, %i, %n : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
  }, {
    ^bb0(%i: tensor<f32>, %v: tensor<3xf32>, %s: tensor<f32>):
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %i1 = stablehlo.add %i, %one : tensor<f32>
      %v1 = stablehlo.add %v, %v : tensor<3xf32>
      %s1 = stablehlo.add %s, %one : tensor<f32>
      stablehlo.return %i1, %v1, %s1 : tensor<f32>, tensor<3xf32>, tensor<f32>
  }) : (tensor<f32>, tensor<3xf32>, tensor<f32>) -> (tensor<f32>, tensor<3xf32>, tensor<f32>)
  return %r#1 : tensor<3xf32>
}
```

```
for i in $(seq 10); do
  iree-compile --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    while_constant_carry.mlir -o /dev/null > /dev/null 2>&1; echo $?
done
```

prints `139` (SIGSEGV) almost every time. Backtrace (abridged):

```
 #4 mlir::iree_compiler::Explorer::walkTransitiveUses(mlir::Value, std::function<mlir::WalkResult (mlir::OpOperand&)>, mlir::iree_compiler::TraversalBehavior) Explorer.cpp
 #5 mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS::updateValue(mlir::Value, mlir::iree_compiler::DFX::Solver&) Affinity.cpp
 #6 mlir::iree_compiler::DFX::Solver::updateElement(mlir::iree_compiler::DFX::AbstractElement&) Solver.cpp
 #7 mlir::iree_compiler::DFX::Solver::getOrCreateElementFor<mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS>(...)
 #8 mlir::iree_compiler::IREE::Stream::ValueConsumerAffinityPVS::updateFromUse(...)
    ... (#4-#8 repeat three more times) ...
#35 mlir::iree_compiler::IREE::Stream::AffinityAnalysis::run() Affinity.cpp
#36 mlir::iree_compiler::IREE::Stream::(anonymous namespace)::ScheduleAllocationPass::runOnOperation() ScheduleAllocation.cpp
```

Workaround that compiles 20/20: replace the first two lines of the body with

```mlir
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %b0 = stablehlo.optimization_barrier %c0 : tensor<f32>
  %r:3 = "stablehlo.while"(%b0, %x, %y) ({
```

### What component(s) does this issue relate to?

Compiler

### Version information

`iree-base-compiler==3.11.0` from PyPI
(`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`,
LLVM 23.0.0git, optimized build), Linux x86_64.

### Additional context

The nondeterminism suggests iteration over pointer-keyed containers or a
use-after-free / dangling reference in the DFX solver state.
````

## 3. 比較から作った `i1` の while carry を関数から返すと ConvertToStream が落ちる（#165 の 2）

- **状態**: 未報告（ユーザーが報告する）。上流 issue: `<link>`
- **再現ファイル**: [`docs/iree-repros/while-i1-carry-returned.mlir`](iree-repros/while-i1-carry-returned.mlir)。対照: [`while-i1-carry-not-returned.mlir`](iree-repros/while-i1-carry-not-returned.mlir)（同じ while で、`i1` の carry ではなく f32 の carry を返す）
- **期待**: コンパイルが通る
- **実際**: `ConvertToStreamPass` の `ScfWhileOpConversion` → `ConversionPatternRewriterImpl::applySignatureConversion` の中で、SIGSEGV（`memcpy` の中）か `LLVM ERROR: out of memory` / `Allocation failed` の SIGABRT。33 回中 33 回落ちる（シグナルは 139 と 134 が混ざる。`%r#0` と `%r#1` の両方を返す元の形でも 11/11）。対照は 20/20 通る
- **見立て**: `i1` のテンソルは `i8` のストレージに変わる（型変換）ので、`scf.while` の領域のシグネチャ変換で、変換前後の型の個数・対応がずれて範囲外を読んでいると見られる。nabla の記録（docs/stablehlo-ops.md）では、`i1` を `i32` に通す・barrier を挟む・`select` で作るのどれでも直らなかった
- **nabla 側の回避策**: 無い。`src/while-loop.lisp` の冒頭と `while-loop` の docstring、`docs/stablehlo-ops.md` に制限として書き、jit の戻り値にしないよう案内している

バックトレース（SIGSEGV の場合、抜粋）:

```
 #4 ... __memcpy_avx512_unaligned_erms ./string/../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:265:0
 #5 ... mlir::detail::ConversionPatternRewriterImpl::applySignatureConversion(mlir::Block*, mlir::TypeConverter const*, mlir::TypeConverter::SignatureConversion&) DialectConversion.cpp
 #6 ... mlir::detail::ConversionPatternRewriterImpl::convertRegionTypes(mlir::Region*, mlir::TypeConverter const&, mlir::TypeConverter::SignatureConversion*) DialectConversion.cpp
 #7 ... mlir::iree_compiler::(anonymous namespace)::ScfWhileOpConversion::matchAndRewrite(mlir::scf::WhileOp, ...)
 ...
#18 ... mlir::iree_compiler::IREE::Stream::(anonymous namespace)::ConvertToStreamPass::runOnOperation() ConvertToStream.cpp
```

SIGABRT の場合は `LLVM ERROR: out of memory` / `Allocation failed` を出して `raise` で止まる。

### 下書き（英語）

**Title**: `[Stream] ConvertToStream crashes (segfault / "LLVM ERROR: out of memory") in ScfWhileOpConversion when a stablehlo.while i1 carry is returned from the function`

````markdown
### What happened?

A `stablehlo.while` with an `i1` carry that is produced by `stablehlo.compare`
crashes `iree-compile` in `ConvertToStreamPass` when that carry is returned from
the function. Every run crashes (33/33); the failure mode alternates between
SIGSEGV inside `memcpy` called from
`ConversionPatternRewriterImpl::applySignatureConversion` and an abort with
`LLVM ERROR: out of memory` / `Allocation failed`.

Returning only the f32 carry of the same loop compiles fine (20/20), so the
problem seems to be the signature conversion of the `scf.while` regions when an
`i1` tensor result escapes (i1 -> i8 storage type conversion).

Expected: the module compiles.

### Steps to reproduce your issue

`while_i1_carry.mlir`:

```mlir
func.func @main(%x: tensor<f32>) -> tensor<i1> {
  %five = stablehlo.constant dense<5.0> : tensor<f32>
  %f0 = stablehlo.compare LT, %x, %five : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %r:2 = "stablehlo.while"(%x, %f0) ({
    ^bb0(%a: tensor<f32>, %b: tensor<i1>):
      stablehlo.return %b : tensor<i1>
  }, {
    ^bb0(%a: tensor<f32>, %b: tensor<i1>):
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %n = stablehlo.add %a, %one : tensor<f32>
      %f = stablehlo.compare LT, %n, %five : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %n, %f : tensor<f32>, tensor<i1>
  }) : (tensor<f32>, tensor<i1>) -> (tensor<f32>, tensor<i1>)
  return %r#1 : tensor<i1>
}
```

```
iree-compile --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu while_i1_carry.mlir -o /dev/null
```

Backtrace (SIGSEGV case, abridged):

```
 #4 __memcpy_avx512_unaligned_erms
 #5 mlir::detail::ConversionPatternRewriterImpl::applySignatureConversion(mlir::Block*, mlir::TypeConverter const*, mlir::TypeConverter::SignatureConversion&) DialectConversion.cpp
 #6 mlir::detail::ConversionPatternRewriterImpl::convertRegionTypes(mlir::Region*, mlir::TypeConverter const&, mlir::TypeConverter::SignatureConversion*) DialectConversion.cpp
 #7 mlir::iree_compiler::(anonymous namespace)::ScfWhileOpConversion::matchAndRewrite(mlir::scf::WhileOp, ...)
 #8 mlir::OpConversionPattern<mlir::scf::WhileOp>::matchAndRewrite(...)
    ...
#18 mlir::iree_compiler::IREE::Stream::(anonymous namespace)::ConvertToStreamPass::runOnOperation() ConvertToStream.cpp
```

The abort case prints `LLVM ERROR: out of memory` / `Allocation failed`.

Changing the function to return `%r#0 : tensor<f32>` instead compiles every time.
We also tried converting the flag to `i32` inside the loop, wrapping it in
`stablehlo.optimization_barrier`, and producing it via `stablehlo.select`;
none of those avoided the crash.

### What component(s) does this issue relate to?

Compiler

### Version information

`iree-base-compiler==3.11.0` from PyPI
(`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`,
LLVM 23.0.0git, optimized build), Linux x86_64.
````

## 上流の検索（2026-10-04）

`AnnotateDispatches` `saturatingMul` / `summarizeDispatchRegion` / `ValueConsumerAffinityPVS` / `walkTransitiveUses` / `ScfWhileOpConversion` / `applySignatureConversion` などで github.com の iree-org/iree を検索した結果:

- バグ1: [iree-org/iree#24947](https://github.com/iree-org/iree/pull/24947)（PR、Open）が同じ `saturatingMul` の assert を緩める。issue としての報告は見つからなかった
- バグ2・3: 該当する issue は見つからなかった（近いが別物: [#22974](https://github.com/iree-org/iree/issues/22974) は ConvertToStream の list 型の変換での segfault）

## 4. f32 に convert した畳み込みで片方の operand が元の dtype に戻る（#203）

- **状態**: 未報告（ユーザーが報告する）。上流 issue: `<link>`
- **再現ファイル**: [`docs/iree-repros/conv-convert-operand-type.mlir`](iree-repros/conv-convert-operand-type.mlir)（対照: `conv-convert-operand-type-barrier.mlir`）
- **期待**: bf16 の2入力を `stablehlo.convert` で f32 にして f32 で畳み込み、結果を bf16 に戻す形がコンパイルできる
- **実際**: `expects lhs and rhs to have compatible element type. Got: 'bf16' and 'f32'`（クラッシュではなく通常のコンパイルエラー）。カーネルが `stablehlo.transpose`（入出力チャネルの入れ替え）を経ているときに出る。`reverse` + convert + convolution は通る。convert を transpose の前に置いても、片方の入力だけの `optimization_barrier` でも直らない。f16 も同じ
- **原因の見立て**: convert を畳み込みの operand に畳み込む（折り畳む）最適化が、片方の operand だけ元の dtype に戻すように見える（未調査）
- **nabla 側の回避策**: `src/primitives/conv.lisp` の `%conv-emit-lines` が、f32 に convert した2入力をまとめて `stablehlo.optimization_barrier`（多値）に通してから畳み込む。再現ファイルの `-barrier` が通ることで確かめた。`tests/iree/conv-grad-test.lisp` の `convolution/iree-bf16-convert-conv-bugs-still-reproduce` が「今も落ちる」ことを守っている（直れば失敗するので、回避策ごと消す）

## 5. O/G = 1 の depthwise 畳み込みを bf16 / f16 から f32 に convert すると tensor.collapse_shape で落ちる（#203）

- **状態**: 未報告（ユーザーが報告する）。上流 issue: `<link>`
- **再現ファイル**: [`docs/iree-repros/conv-depthwise-collapse-shape.mlir`](iree-repros/conv-depthwise-collapse-shape.mlir)（対照: `conv-depthwise-collapse-shape-barrier.mlir`）
- **期待**: コンパイルが通る（f32 のままなら通る）
- **実際**: `'tensor.collapse_shape' op expected collapsed type to be 'tensor<3x2x3xbf16>', but got 'tensor<3x2x3xf32>'`（rhs `[3, 2, 1, 3]`、`feature_group_count = 3`）。前方の畳み込みの時点で落ちる（nabla issue #197 で入った形）
- **原因の見立て**: #4 と同じ（convert の折り畳み後に dtype が食い違う）
- **nabla 側の回避策**: #4 と同じ `optimization_barrier`。同じテストが守っている

なお、この2件は SIGSEGV ではなく通常のコンパイルエラーなので、`scripts/check-iree-repros.sh`（クラッシュの数を数える）の対象には入れていない。確かめるには上の `iree-compile` のコマンドを使う。

## 6. local-task device で入力が 162 個以上の関数を呼ぶと NULL を参照して落ちる（#199）

コンパイルは通り、**実行時**に落ちる（このファイルの他の3件はコンパイラが落ちる）。`main` の入力（テンソル）の個数だけで決まり、入力の形・演算には依らない。

| 項目 | 内容 |
| --- | --- |
| 再現ファイル | `docs/iree-repros/local-task-162-inputs.mlir`（入力 162 個、各 `tensor<3xf32>` に `tanh` して 162 個を返す）。対照は `local-task-161-inputs.mlir`（161 個） |
| 再現のしかた | `iree-compile` した vmfb を `iree-run-module --device=local-task --function=main --input=3xf32=1,2,3 …`（入力の個数ぶん）で実行する。`scripts/check-iree-repros.sh` が自動でやる |
| 症状 | SIGSEGV（終了コード 139）。161 個までは通り、162 個で決定的に落ちる |
| 落ちる場所 | `iree_task_call_initialize` ← `iree_hal_task_queue_issue_cmd_allocate`（local-task ドライバの runtime） |
| 回避策 | `--device=local-sync` は 170 個・309 個の入力でも通る。nabla では `(make-backend :iree :driver :local-sync)` で選べる |
| nabla の対応 | `nabla.iree:invoke` が local-task device で入力 162 個以上を呼ぶ前に `iree-too-many-inputs`（`iree-error` の子）を signal する（`+local-task-max-inputs+` = 161） |

jit した関数の入力の個数は、引数の PyTree の葉の個数に等しい（#199）。Adam の学習ステップ（パラメータ + m + v + バッチ）のような大きなモデルは、この上限に当たりうる。

2つ目の観察（別の制限）: 入力 170 個で全部を足していく要素ごとの加算の連鎖（`sum170.mlir` 相当）は、モジュールのロード時に `hal.command_buffer.dispatch ... OUT_OF_RANGE; binding count 51 > 32` で失敗する（1つの dispatch に融合されるオペランドが多すぎる。160 個でも `41 > 32`）。こちらはロードの失敗（`iree-status-error`）で、プロセスは落ちない。

### 下書き（英語）

````markdown
### What happened?

With the local-task HAL driver, invoking a function with 162 or more tensor inputs
crashes the process with a NULL dereference in `iree_task_call_initialize`
(called from `iree_hal_task_queue_issue_cmd_allocate`). 161 inputs work, 162 inputs
crash every time. `--device=local-sync` runs the same module with 170 and 309 inputs.

### Steps to reproduce your issue

`func.func @main(%a0: tensor<3xf32>, ..., %a161: tensor<3xf32>) -> (tensor<3xf32>, ...)`
returning `stablehlo.tanh %aN` for every input (see local-task-162-inputs.mlir).

```sh
iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  local-task-162-inputs.mlir -o m.vmfb
iree-run-module --device=local-task --module=m.vmfb --function=main \
  $(for i in $(seq 162); do printf -- '--input=3xf32=1,2,3 '; done)   # exit 139
```

### What component(s) does this issue relate to?

Runtime

### Version information

`iree-base-compiler==3.11.0` from PyPI (`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`), runtime built from the same commit, Linux x86_64.
````

## 7. local-task device で、入れ子の scan の外側の ys が負荷のもとでまれに 0 の行を返す（#220）

コンパイルは通り、**実行時**にまれに結果が壊れる（クラッシュはしない）。CPU に負荷があるときだけ、1〜2% の実行で起きる。

| 項目 | 内容 |
| --- | --- |
| 再現ファイル | `docs/iree-repros/nested-scan-ys-barrier-race.mlir`（外側の `stablehlo.while` の本体が ys `tensor<4x3xf32>` を `optimization_barrier` に通し、内側の `stablehlo.while` を回してから、その結果へ `dynamic_update_slice` する）。対照は `nested-scan-ys-no-barrier.mlir`（外側の ys の barrier を外しただけ） |
| 再現のしかた | `iree-compile` した vmfb を `iree-run-module --device=local-task --function=main --input=3xf32=0.5,-0.25,0.9` で何度も実行し、`--device=local-sync` の出力と比べる。CPU に負荷が要る（`yes > /dev/null` を CPU の数だけ動かす）。`scripts/check-iree-repros.sh` が負荷の起動から比較までやる |
| 症状 | 外側の ys の1行（実測では外側の2回目までの書き込みのどれか）が、全要素 0（ys の初期値）のまま返る。carry は正しい。負荷のもとで実測 0.4〜2%（2000 回あたり 8〜31 回）。コンパイルは決定的（同じ vmfb の SHA-256） |
| 起きない条件 | `--device=local-sync`（0/300）、`--device=local-task --task_worker_count=1`（0/300）、外側の ys の barrier を外した形（2000 回、3000 回とも 0 回）。barrier を内側の while の後ろ（`dynamic_update_slice` の直前）に移しても直らない（1500 回で 11 回） |
| 原因の見立て | local-task の非同期な実行で、外側の ys の barrier の結果への書き込みが、内側の while の実行と順序づけられていない（ys の 0 埋めと in-place の書き込みの競合に見える）。未調査 |
| nabla の対応 | `src/scan.lisp` の `%scan-emit` は、本体の中に別の `stablehlo.while` がある scan（入れ子の外側）には ys の barrier を付けない。代償は、外側の ys が更新のたびに丸ごとコピーされること（#159 と同じ性能の問題が入れ子の外側にだけ戻る。外側の長さの2乗）。いちばん内側の scan は今までどおり barrier を付ける。形は `scan/nested-scan-keeps-the-ys-barrier-only-in-the-innermost-body`（`tests/scan-test.lisp`）が守る。nabla のテスト（`scan/iree-nested-*-matches-eager`）でも、回避策なしだと 100 回の PBT の中で 2% 前後の確率で落ちる |
| 「今も起きる」の確認 | `scripts/check-iree-repros.sh`（race-bug / race-control）。まれに起きる確率的な現象で、1 回の確認に数十秒かかり負荷も要るので、既定のテストスイートには入れていない |

### 下書き（英語）

````markdown
### What happened?

With the local-task HAL driver, a nested `stablehlo.while` whose outer body updates a
loop-carried buffer (through `optimization_barrier` + `dynamic_update_slice`) after running an
inner `stablehlo.while` occasionally returns one row of that buffer as zeros (its initial value).
It happens in about 1% of runs and only when the CPU is loaded. `--device=local-sync` and
`--device=local-task --task_worker_count=1` never show it (0/300), nor does the same module
without the `optimization_barrier` on the outer buffer (0/3000). The compiled vmfb is
bit-identical between runs.

### Steps to reproduce your issue

```sh
iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  nested-scan-ys-barrier-race.mlir -o m.vmfb
# with all cores busy (e.g. one `yes > /dev/null` per core), compare against local-sync:
for i in $(seq 2000); do
  iree-run-module --device=local-task --module=m.vmfb --function=main \
    --input=3xf32=0.5,-0.25,0.9
done | sort | uniq -c    # a second, zeroed variant of result[1] shows up
```

### What component(s) does this issue relate to?

Runtime

### Version information

`iree-base-compiler==3.11.0` from PyPI (`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`), runtime built from the same commit, Linux x86_64.
````
