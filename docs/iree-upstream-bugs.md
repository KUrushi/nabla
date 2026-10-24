# IREE 上流に報告するバグ（issue #73、#165、#199、#203、#220、#226、#243）

nabla が固定している IREE 3.11.0 で見つけたバグ8件（1〜5 はコンパイラ、6〜8 はランタイム）の、nabla を使わない最小の再現と、iree-org/iree にそのまま貼れる issue の下書き。**上流への報告はリポジトリのオーナーが行う**（このリポジトリの作業では iree-org/iree に何も投稿しない）。報告したら、各節の「上流 issue」の行にリンクを書き、nabla の issue（#73 / #165 / #243）にも同じリンクを記録する。

**起票前の注意**: iree-org/iree の既存 issue #24753 に付いた bot コメント（2026-10-19 に確認）によると、IREE は issue にも AI ツール利用方針を適用し、要約を起票者が自分で書くこと、LLM を使った部分（コマンド・再現 IR を含む）を理解して確認すること、相当に使った場合は issue の冒頭か末尾にその旨を書くことを求めている。これは bot コメントを読んだだけなので、投稿前に iree.dev の方針の原文（<https://iree.dev/developers/general/contributing/#ai-tool-use>）を確認すること。下の英語の下書きは出発点であり、投稿前にリポジトリのオーナーが読み直して自分の言葉に直す。

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
| 1 | `dot-general-k0.mlir` | `AnnotateDispatchesPass`（Flow） | SIGFPE（整数 0 除算、終了コード 136） | 決定的（24/24） | （ファイル無し。dot_general を出さずゼロ定数にする） | #62、#73 | 下書き済み・未報告（起票はリポジトリのオーナー）: `<link>` |
| 2 | `while-constant-carry.mlir` | `ScheduleAllocationPass`（Stream の AffinityAnalysis） | SIGSEGV（終了コード 139） | 非決定的。49/50（30/30 と 19/20） | `while-constant-carry-barrier.mlir`: 0/20 | #165 | 下書き済み・未報告（起票はリポジトリのオーナー）: `<link>` |
| 3 | `while-i1-carry-returned.mlir` | `ConvertToStreamPass`（`ScfWhileOpConversion`） | SIGSEGV（139）または `LLVM ERROR: out of memory` で SIGABRT（134） | 落ちること自体は決定的（33/33）、どちらのシグナルかは非決定的 | `while-i1-carry-not-returned.mlir`: 0/20（回避策ではなく、戻り値にしない対照） | #131、#165 | 下書き済み・未報告（起票はリポジトリのオーナー）: `<link>` |

## 1. K=0 の `stablehlo.dot_general` で AnnotateDispatches が整数 0 除算（#73）

- **状態**: 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>`
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

- **状態**: 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>`
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

- **状態**: 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>`
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

項目 4〜8 の上流の検索（2026-10-19、`gh search issues -R iree-org/iree`）の結果は、各節の「上流の状況」に書いた。重複と断定できる issue は無く、関連の候補は 8 の [#24753](https://github.com/iree-org/iree/issues/24753)（同じ2つのエラー）だけ。

## 4. f32 に convert した畳み込みで片方の operand が元の dtype に戻る（#203）

- **状態**: 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>`
- **上流の状況**（2026-10-19 検索）: 検索語 `optimization_barrier convolution bf16`、`optimization_barrier convolution`、`convolution bf16`、`compatible element type`、`stablehlo convolution convert`、`convert f32 convolution bf16`: 同じ症状の issue は無し。近いが別物: [#7612](https://github.com/iree-org/iree/issues/7612)（Open、古い mhlo の bf16 畳み込みが `hal.command_buffer.fill_buffer` で失敗。別の経路）
- **再現ファイル**: [`docs/iree-repros/conv-convert-operand-type.mlir`](iree-repros/conv-convert-operand-type.mlir)（対照: `conv-convert-operand-type-barrier.mlir`）
- **確認した環境**: Linux x86_64 VM（OrbStack、Rosetta）、2026-10-19。コンパイルエラーは決定的で、Rosetta に依らない
- **期待**: bf16 の2入力を `stablehlo.convert` で f32 にして f32 で畳み込み、結果を bf16 に戻す形がコンパイルできる
- **実際**: `expects lhs and rhs to have compatible element type. Got: 'bf16' and 'f32'`（クラッシュではなく通常のコンパイルエラー）。カーネルが `stablehlo.transpose`（入出力チャネルの入れ替え）を経ているときに出る。`reverse` + convert + convolution は通る。convert を transpose の前に置いても、片方の入力だけの `optimization_barrier` でも直らない。f16 も同じ
- **原因の見立て**: convert を畳み込みの operand に畳み込む（折り畳む）最適化が、片方の operand だけ元の dtype に戻すように見える（未調査）
- **nabla 側の回避策**: `src/primitives/conv.lisp` の `%conv-emit-lines` が、f32 に convert した2入力をまとめて `stablehlo.optimization_barrier`（多値）に通してから畳み込む。再現ファイルの `-barrier` が通ることで確かめた。`tests/iree/conv-grad-test.lisp` の `convolution/iree-bf16-convert-conv-bugs-still-reproduce` が「今も落ちる」ことを守っている（直れば失敗するので、回避策ごと消す）

### 下書き（英語）

**Title**: `[StableHLO] stablehlo.convolution on two stablehlo.convert-to-f32 operands fails verification: "expects lhs and rhs to have compatible element type. Got: 'bf16' and 'f32'"`

````markdown
### What happened?

Compiling a `stablehlo.convolution` whose two operands are `stablehlo.convert`s of bf16 tensors to f32
(with the result converted back to bf16) fails with a verifier error, although the input is valid
StableHLO:

```
error: expects lhs and rhs to have compatible element type. Got: 'bf16' and 'f32'
```

It looks as if one operand of the convolution is rewritten back to its pre-convert bf16 type
while the other keeps f32. In this (non-depthwise) shape it only happens when the kernel goes through a
`stablehlo.transpose` (the `[0, 1, 3, 2]` swap of the input/output channels); with a
`stablehlo.reverse` instead of the transpose it compiles. Moving the convert before the transpose, or putting only one of the two
operands through `stablehlo.optimization_barrier`, does not help. f16 behaves like bf16.
Passing both converted operands through a single (multi-value) `stablehlo.optimization_barrier`
makes it compile, so we use that as a workaround. This is an ordinary diagnostic, not a crash.

### Steps to reproduce your issue

```mlir
func.func @main(%a: tensor<2x5x6x4xbf16>, %w: tensor<3x3x3x4xbf16>) -> tensor<2x5x6x3xbf16> {
  %9 = stablehlo.transpose %w, dims = [0, 1, 3, 2] : (tensor<3x3x3x4xbf16>) -> tensor<3x3x4x3xbf16>
  %l = stablehlo.convert %a : (tensor<2x5x6x4xbf16>) -> tensor<2x5x6x4xf32>
  %r = stablehlo.convert %9 : (tensor<3x3x4x3xbf16>) -> tensor<3x3x4x3xf32>
  %acc = stablehlo.convolution(%l, %r) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x5x6x4xf32>, tensor<3x3x4x3xf32>) -> tensor<2x5x6x3xf32>
  %o = stablehlo.convert %acc : (tensor<2x5x6x3xf32>) -> tensor<2x5x6x3xbf16>
  func.return %o : tensor<2x5x6x3xbf16>
}
```

```sh
iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  conv-convert-operand-type.mlir -o /dev/null
```

### What component(s) does this issue relate to?

Compiler

### Version information

`iree-base-compiler==3.11.0` from PyPI
(`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`,
LLVM 23.0.0git, optimized build), Linux x86_64.

### Additional context

Found while adding bf16/f16 support to a Lisp frontend that emits StableHLO (compute in f32 for
precision, store in bf16). The same module with f32 inputs compiles. Workaround and a passing control:
both converted operands through one `stablehlo.optimization_barrier`. I have not investigated which
pass performs the fold; a guess is a convert-folding rewrite that only updates one operand.
````

## 5. O/G = 1 の depthwise 畳み込みを bf16 / f16 から f32 に convert すると tensor.collapse_shape で落ちる（#203）

- **状態**: 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>`
- **上流の状況**（2026-10-19 検索）: 検索語 `collapse_shape depthwise`、`tensor.collapse_shape expected collapsed type`、`depthwise`、`feature_group_count`: 同じ症状の issue は無し。近いが別物: [#19284](https://github.com/iree-org/iree/issues/19284)（Open、flux の text encoder で `tensor.collapse_shape` の検証が落ちる。型が違う原因）、[#17714](https://github.com/iree-org/iree/issues/17714)（Open、`feature_group_count` を持つ畳み込みの StableHLOToLinalgConvolution の assert。f32 の別の形）
- **再現ファイル**: [`docs/iree-repros/conv-depthwise-collapse-shape.mlir`](iree-repros/conv-depthwise-collapse-shape.mlir)（対照: `conv-depthwise-collapse-shape-barrier.mlir`）
- **確認した環境**: Linux x86_64 VM（OrbStack、Rosetta）、2026-10-19。コンパイルエラーは決定的で、Rosetta に依らない
- **期待**: コンパイルが通る（f32 のままなら通る）
- **実際**: `'tensor.collapse_shape' op expected collapsed type to be 'tensor<3x2x3xbf16>', but got 'tensor<3x2x3xf32>'`（rhs `[3, 2, 1, 3]`、`feature_group_count = 3`）。前方の畳み込みの時点で落ちる（nabla issue #197 で入った形）
- **原因の見立て**: #4 と同じ（convert の折り畳み後に dtype が食い違う）
- **nabla 側の回避策**: #4 と同じ `optimization_barrier`。同じテストが守っている

この2件は SIGSEGV ではなく通常のコンパイルエラー（終了コード 1）なので、`scripts/check-iree-repros.sh` では `error-bug` / `control` の行（`iree-compile` の終了コードで判定）で確かめる。

### 下書き（英語）

**Title**: `[StableHLO] depthwise stablehlo.convolution (O/G = 1) on operands converted from bf16 fails: tensor.collapse_shape expected collapsed type tensor<3x2x3xbf16>, got tensor<3x2x3xf32>`

````markdown
### What happened?

A depthwise `stablehlo.convolution` (`feature_group_count` = C, kernel `[kh, kw, 1, C]`) whose operands
are `stablehlo.convert`s of bf16 tensors to f32 fails with:

```
error: 'tensor.collapse_shape' op expected collapsed type to be 'tensor<3x2x3xbf16>', but got 'tensor<3x2x3xf32>'
note: see current operation: %3 = "tensor.collapse_shape"(%arg1) <{reassociation = [[0], [1], [2, 3]]}> : (tensor<3x2x1x3xbf16>) -> tensor<3x2x3xf32>
```

The collapse_shape that squeezes the kernel's `1` dimension takes the bf16 function argument as its
source (the convert was folded into it) but still declares an f32 result, so the verifier rejects it.
The same module with f32 inputs (no converts) compiles. f16 behaves like bf16. Passing both
converted operands through one multi-value `stablehlo.optimization_barrier` avoids it.

### Steps to reproduce your issue

```mlir
func.func @main(%a: tensor<1x5x6x3xbf16>, %w: tensor<3x2x1x3xbf16>) -> tensor<1x3x5x3xbf16> {
  %l = stablehlo.convert %a : (tensor<1x5x6x3xbf16>) -> tensor<1x5x6x3xf32>
  %r = stablehlo.convert %w : (tensor<3x2x1x3xbf16>) -> tensor<3x2x1x3xf32>
  %acc = stablehlo.convolution(%l, %r) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<1x5x6x3xf32>, tensor<3x2x1x3xf32>) -> tensor<1x3x5x3xf32>
  %o = stablehlo.convert %acc : (tensor<1x3x5x3xf32>) -> tensor<1x3x5x3xbf16>
  func.return %o : tensor<1x3x5x3xbf16>
}
```

```sh
iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  conv-depthwise-collapse-shape.mlir -o /dev/null
```

### What component(s) does this issue relate to?

Compiler

### Version information

`iree-base-compiler==3.11.0` from PyPI
(`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`,
LLVM 23.0.0git, optimized build), Linux x86_64.

### Additional context

Probably the same root cause as the non-depthwise report (a convert folded into a convolution operand
leaves the operand types inconsistent): conv-convert-operand-type.mlir fails with the
"compatible element type" verifier error. Both come from emitting f32 convolutions for bf16 tensors.
````

## 6. local-task device で入力が 162 個以上の関数を呼ぶと NULL を参照して落ちる（#199）

コンパイルは通り、**実行時**に落ちる（1〜5 はコンパイラが落ちる・エラーになる。6〜8 は実行時）。`main` の入力（テンソル）の個数だけで決まり、入力の形・演算には依らない。

| 項目 | 内容 |
| --- | --- |
| 状態 | 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>` |
| 上流の状況（2026-10-19 検索） | 検索語 `162 inputs`、`iree_task_call_initialize`、`task_call_initialize`、`local-task segfault`、`too many arguments local-task`: 該当なし |
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

## 7. local-task device で、ys を barrier に通して制御構造（while / if）をはさんで書くと、負荷のもとでまれに 0 の行が返る（#220）

コンパイルは通り、**実行時**にまれに結果が壊れる（クラッシュはしない）。CPU に負荷があるときだけ、1〜2% の実行で起きる。

| 項目 | 内容 |
| --- | --- |
| 状態 | 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>` |
| 上流の状況（2026-10-19 検索） | 検索語 `optimization_barrier`、`local-task race`、`local-task nondeterministic`、`dynamic_update_slice local-task`、`local-task wrong results`: 該当なし（`dynamic_update_slice` の既存 issue #9293 / #6013 は閉じていて別件） |
| 再現ファイル | `docs/iree-repros/nested-scan-ys-barrier-race.mlir`（外側の `stablehlo.while` の本体が ys `tensor<4x3xf32>` を `optimization_barrier` に通し、内側の `stablehlo.while` を回してから、その結果へ `dynamic_update_slice` する）と、内側を `stablehlo.if`（`cond*`）に替えた `nested-scan-ys-if-race.mlir`。対照はどちらも barrier を外しただけの `nested-scan-ys-no-barrier.mlir` / `nested-scan-ys-if-no-barrier.mlir` |
| 再現のしかた | `iree-compile` した vmfb を `iree-run-module --device=local-task --function=main --input=3xf32=0.5,-0.25,0.9` で何度も実行し、`--device=local-sync` の出力と比べる。CPU に負荷が要る（`yes > /dev/null` を CPU の数だけ動かす）。`scripts/check-iree-repros.sh` が負荷の起動から比較までやる |
| 症状 | ys の1行以上（1回の実行で 1 行目と 3 行目が 0 になったこともある）が、全要素 0（ys の初期値）のまま返る。carry は正しい。負荷のもとで実測 0.4〜2%（while 版は 2000 回あたり 8〜31 回、if 版は 600 回あたり 3〜7 回、負荷が弱いと 2000 回あたり 1 回）。コンパイルは決定的（同じ vmfb の SHA-256） |
| 起きない条件 | `--device=local-sync`（0/300）、`--device=local-task --task_worker_count=1`（0/300）、ys の barrier を外した形（while 版 2000 回・3000 回、if 版 2000 回、nabla の cond* を含む scan 3000 回とも 0 回）。barrier を内側の while の後ろ（`dynamic_update_slice` の直前）に移しても直らない（1500 回で 11 回） |
| 原因の見立て | local-task の非同期な実行で、ys の barrier の結果への書き込みが、間にはさんだ制御構造（while / if）の実行と順序づけられていない（ys の 0 埋めと in-place の書き込みの競合に見える）。未調査 |
| nabla の対応 | `src/scan.lisp` の `%scan-emit` は、本体の中に制御構造（別の `stablehlo.while` か `stablehlo.if`）がある scan には ys の barrier を付けない。代償は、ys が更新のたびに丸ごとコピーされること（#159 と同じ性能の問題が、そういう scan にだけ戻る。長さの2乗。実測で長さ 64・ys 16 MB のとき 788 ms 対 68 ms の 11 倍。測り直しと改善の検討は次の「nabla 側の回避策の代償と改善の検討（#233）」）。本体に制御構造の無い scan（いちばん内側の scan を含む）は今までどおり barrier を付ける。形は `scan/nested-scan-keeps-the-ys-barrier-only-in-the-innermost-body` と `scan/scan-with-cond-in-body-has-no-ys-barrier`（`tests/scan-test.lisp`）が守る。nabla のテスト（`scan/iree-nested-*-matches-eager`）でも、回避策なしだと 100 回の PBT の中で 2% 前後の確率で落ちる |
| 「今も起きる」の確認 | `scripts/check-iree-repros.sh`（race-bug / race-control。while 版と if 版の2組）。まれに起きる確率的な現象で、1 回の確認に数十秒かかり負荷も要るので、既定のテストスイートには入れていない |

### nabla 側の回避策の代償と改善の検討（#233）

**結論（2026-10-20 時点）: 既定の emit は変えない。上流の修正を待つ。** 下の候補 (b) はコピーを消せるが、競合を起こさないことをまだ確かめられていない（Rosetta の VM では競合がほとんど再現しない。while 版 0/200、if 版 1/200）。(b) を opt-in の特殊変数で入れることも今回は見送る（安全性が未検証の形は、静かに 0 の行が混ざる正しさの事故になりうる）。x86 ネイティブで 0/20000 が出たあとの PR で入れる。x86 ネイティブで確かめる手順を下に残す。

**代償の実測**（`scripts/bench-scan-ys.sh`。Linux x86_64 の VM（OrbStack、Rosetta、7 コア。性能の数値は参考値で、実行ごとに ±20% 揺れる）。IREE 3.11、local-task。本体は `h` を `tanh` で更新して各ステップの `h`（f32 [65536]）を ys に積む scan で、cond 列は本体を `cond*`（`reduce-sum` が 0.5 未満かで `tanh` か `tanh(tanh)`）にしたもの。ys は更新のたびにコピーされるため、1 ステップの時間が長さ L に比例して伸びる）:

| L | ys の大きさ | plain（制御構造なし。ys の barrier あり）の 1 ステップ | cond（ys の barrier なし。今の既定）の 1 ステップ |
| --- | --- | --- | --- |
| 16 | 4 MB | 1.3 ms | 1.9〜2.9 ms |
| 32 | 8 MB | 1.0 ms | 2.5 ms |
| 64 | 16 MB | 1.0〜1.5 ms | 4.3〜5.6 ms |
| 128 | 32 MB | 1.3〜1.6 ms | 4.8〜5.7 ms |

plain は L によらずほぼ一定（長さに線形）、cond は L とともに伸びる。2 乗になる兆候だが、時間だけでは断定できない: 2 乗の根拠は、`stream.cmd.copy` を数えて毎ステップ 16 MiB の全体コピーが出ること（下の候補の表）で、実測の時間は L を 64 から 128 にしても 2 倍にならず頭打ちになっている（Rosetta の参考値で、メモリ帯域・キャッシュの影響を分離できていない）。呼び出し全体では L=64・ys 16 MB で plain 67〜100 ms、cond 278〜358 ms（約 3〜4 倍）、L=128・ys 32 MB で 163〜206 ms 対 613〜728 ms。Issue の表（負荷のあるマシンで 788 ms 対 68 ms）より差が小さいのは、Rosetta のメモリ帯域と、この本体の計算（`tanh`）が重いためと推測する。Issue の数値は 7 回中の最良で本体も違うため、直接は比べられない。上の `67〜100 ms` のような範囲は、複数回の測定（`--reps 5`）や別の実行の中央値の幅（実行間の揺れ。p10-p90 ではない）で、`bench-scan-ys.sh` の表の `[p10-p90]` とは別物。W=4096（L=64 で ys 1 MB）では 17 ms 対 40 ms。使い方は `scripts/bench-scan-ys.sh` の冒頭、測るものの定義は `scripts/bench-scan-ys.lisp` の冒頭。

**候補の評価**（`scripts/bench-scan-ys.sh --variants plain,cond,scripts/scan-ys-candidates/{default,index-barrier,joint-barrier}.mlir.in --lengths 16,64,128 --widths 65536`。`default` は今の emit を同じテンプレートの形に書いたもので、cond 列と一致する（検算）。コピーは `iree-compile --compile-to=stream` の `stream.cmd.copy` で数えた）:

| 候補 | 内容 | ys 全体のコピー（L=64, ys 16 MB） | 1 ステップ（L=16 / 64 / 128） | 競合の安全性 |
| --- | --- | --- | --- | --- |
| 今の既定（default） | ys に barrier なし | 毎ステップ `stream.cmd.copy` 16 MiB | 2.4 / 4.9 / 5.9 ms（L に伸びる） | 競合は起きない（記録: 3000 回で 0） |
| (a) index-barrier | ys ではなく `dynamic_update_slice` の添字 `%bi` だけを `optimization_barrier` に通す | 消えない（16 MiB のコピーが残る。dispatch が 1 つ増えるだけ） | 3.2 / 4.7 / 5.4 ms（default と同じ） | 意味がない（コピーが消えない） |
| (b) joint-barrier | 制御構造の結果 `%ic` と ys のバッファ `%by` を 1 つの `optimization_barrier` に一緒に通し、結果の組へ `dynamic_update_slice` | **消える**（1 行ぶん 256 KiB のコピーだけ。16 MiB のコピーなし） | 2.7 / 2.6 / 2.7 ms（L によらず一定。plain との差は本体の `cond*` の費用） | **未検証** |

(b) は、barrier を制御構造の「後」に置き、その結果に data dependence を持たせて順序を決める形。以前の試し（barrier を内側の while の後ろ、`dynamic_update_slice` の直前に移す）は ys だけを barrier に通していたので直らなかったが（1500 回で 11 回）、(b) は制御構造の結果と一緒に通すところが違う。ただし競合の原因は未調査（上の「原因の見立て」）なので、これで直る理由も、直らない理由も分からない。

**(b) の一般形と未評価の形**: 束ねる値は、本体の全制御構造の全結果と、全 ys バッファ（`y_t` ではなく、書き込み先のバッファ）。テンプレートが覆っているのは「`y_t` が 1 つの制御構造の結果そのもので、制御構造が 1 つ、ys が 1 つ、入れ子なし」の形だけ。次の形は覆っておらず、コピーが消えるかも競合が起きないかも未評価: `y_t` が制御構造の結果でない形（たとえば dropout の rng の while が carry にだけ効く）、制御構造が複数ある本体、ys が複数ある scan、入れ子の scan。**この環境（Rosetta の VM）では競合がほとんど再現しない。`check-iree-repros.sh --candidates --race-runs 200`（7 コアに `yes` の負荷）で、race-bug は while 版 0/200、if 版 1/200（本来 0.4〜2%）、候補 (a)(b) の while 版・if 版 4 組はすべて 0/200 だった。0/200 は、1% の競合でも 13% の確率で起こる結果で、(b) が競合を起こさないことの根拠にならない。** race 用の形は `docs/iree-repros/nested-scan-ys-{,if-}joint-barrier-race.mlir`、(a) は `nested-scan-ys-{,if-}index-barrier-race.mlir`。

**(b') 「ys を carry でなく別の形で書き出す」は、StableHLO では表せないと判断した。** `stablehlo.while` が各回の値を外へ出す手段は carry だけで、ys を while の外で組み立てるには、各ステップの `y_t` をどこかに溜める必要がある。溜める先は結局 carry のバッファで、`dynamic_update_slice`（今と同じ）か、毎回 `concatenate` / `pad` で作り直すこと（さらに悪い、毎ステップ丸ごとのコピー）になる。ys を `K` 行ずつの小さなバッファに分けて外側の while で書き出す形（#178 の rng の emit と同じ考え）は、外側の本体に barrier と制御構造が同居するので、報告している競合と同じ構造になる。試していない。

**x86 ネイティブで確かめる手順**（別の環境、たとえば CI や手元の x86 Linux。Rosetta ではなく実機で）:

1. 先に race-bug が再現することを確かめる（再現しない環境では何も言えない）: `scripts/check-iree-repros.sh --runs 2 --race-runs 4000` が race-bug を `REPRODUCED`（負荷のもとで 0.4〜2%）にすること。
2. 候補を同じ負荷で走らせる: `scripts/check-iree-repros.sh --candidates --race-runs 20000`。`race-candidate` の行（a: index-barrier、b: joint-barrier、それぞれ while 版と if 版）の `mismatches` が 0 であること。時間は数時間かかりうる（race の行 8 本 × 20000 回、負荷つき。Rosetta の VM の 200 回で 20 分だった）。1 回の失敗確率が 0.4% なら 20000 回で 0 回になる確率は e^-80 より小さい。20000 回で 0 回なら、失敗確率は（95% の信頼で）約 0.015% 未満と言える。それより低い確率は言えない。
3. 0 だったら nabla の emit に反映する前に、`scripts/bench-scan-ys.sh --variants plain,cond,scripts/scan-ys-candidates/joint-barrier.mlir.in` で性能（コピーが消えること）を確かめ、`scan/iree-nested-*-matches-eager`（PBT。回避策なしだと 100 回の中で 2% 前後で落ちる）を繰り返し走らせる。
4. 新しい候補を足すには、`docs/iree-repros/` に race 用の形（`// INPUT: ...` の行つき）を置き、`scripts/check-iree-repros.sh` の `CANDIDATE_CASES` に `race-candidate <ファイル名>` を 1 行足す。性能の比較は `scripts/scan-ys-candidates/` に `@L@` / `@W@` のテンプレート（main の形は `bench-scan-ys.lisp` の冒頭）を置く。

**opt-in で切り替える案（実装していない）**: (b) を、特殊変数（たとえば `*scan-ys-joint-barrier*`、既定は nil）で選べるようにする。x86 ネイティブでの確認が済むまでは既定を nil に保つ。確認が済んだら既定を t にし、`scan/scan-with-cond-in-body-has-no-ys-barrier` と `scan/nested-scan-keeps-the-ys-barrier-only-in-the-innermost-body`（今は barrier が無いことを守っている）を新しい形に書き換える。上流が直ったら（`check-iree-repros.sh` の race-bug が再現しなくなったら）、回避策ごと消して plain の emit に戻す。

### 下書き（英語）

````markdown
### What happened?

With the local-task HAL driver, a `stablehlo.while` whose body updates a
loop-carried buffer (through `optimization_barrier` + `dynamic_update_slice`) after running an
inner `stablehlo.while` (or a `stablehlo.if`) occasionally returns one or more rows of that buffer as zeros (its initial value).
It happens in about 1% of runs and only when the CPU is loaded. `--device=local-sync` and
`--device=local-task --task_worker_count=1` never show it (0/300), nor does the same module
without the `optimization_barrier` on the outer buffer (0/3000). A `stablehlo.if` in place of the inner while behaves the same (nested-scan-ys-if-race.mlir). The compiled vmfb is
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

## 8. stablehlo.if の枝が入力をそのまま返す出力を持つと、もう片方の枝を選んだ実行時に `ref is null` で失敗する（#226）

コンパイルは通り、**実行時**に `iree-run-module` / `invoke` が失敗する（クラッシュではなく、終了コード 1 のエラー）。決定的。

| 項目 | 内容 |
| --- | --- |
| 状態 | 下書き済み・未報告（起票はリポジトリのオーナー）。上流 issue: `<link>`。Rosetta の VM で確認。報告前に x86 ネイティブで確かめる |
| 上流の状況（2026-10-19 検索） | [iree-org/iree#24753](https://github.com/iree-org/iree/issues/24753)（Open、2026-07-23 起票）が関連: JAX が出した StableHLO（i32 の入出力を素通しする関数）が 3.12.0rc で `OUT_OF_RANGE; indirect buffer reference slot 0 ...`（`command_buffer_validation.c:185`）、failure mode を反転すると `ref is null; while invoking native function hal.buffer_view.create` で失敗し、3.11.0 では通ると報告している。nabla の症状（3.11.0）と同じ2つのエラーだが、起きる版と最小の形が違うので重複とは断定できない。報告するときはこの issue を参照し、同じ根（素通しの出力のバッファ参照）かを IREE のメンテナに尋ねる。他の検索語（`stablehlo.if ref is null`、`stablehlo.if passthrough`、`scf.if yield function argument`、`hal.buffer_view.create`）に該当は無し。なお 3.12.0rc で nabla の再現が直っているかは未確認（IREE を上げるときの確認手順で見る） |
| 再現ファイル | `docs/iree-repros/if-passthrough-operands.mlir`（`stablehlo.if` の then が operand 3 つをそのまま `stablehlo.return` し、1 つだけ計算する。else は 4 つとも計算する）。対照は `if-passthrough-operands-select.mlir`（素通しの結果を if から外し、if の外の `stablehlo.select` にした形。nabla の回避策） |
| 再現のしかた | `iree-compile` した vmfb を `iree-run-module --device=local-task --function=main --input=f32=1.0 --input=f32=1.0 --input=f32=2.0 --input=f32=3.0 --input=f32=4.0` で実行する（最初の入力が pred 用で、`< 0.5` なら then）。`--device=local-sync` でも同じ。`scripts/check-iree-repros.sh` の fail-bug / fail-control の行 |
| 症状 | `iree-src/runtime/src/iree/vm/ref.h:223: INVALID_ARGUMENT; ref is null; while invoking native function hal.buffer_view.create`。素通しの結果を持つ枝**でない**側（else。p = 1.0）を選んだときだけ。素通しの枝を選ぶ側（p = 0.0）は正しい値が返る。結果が関数の戻り値でなく後続の演算で使われると、別のエラー `OUT_OF_RANGE; indirect buffer reference ...`（`command_buffer_validation.c:185`）になる |
| 条件 | 出力の並び（C = 両方の枝が計算、P = then が素通し、else は全て計算）で、`CPPP`・`CPPPP`・`CPPPPP`・`PCPPP` は失敗し、`PPPC`・`PPP`・`PPPP`・`CPP`・`PPC`・`CCPP`・`CCPPP`・`CPPPC`・`CPCPP`・`CPPCP`・`PCPC`・`CPCP` は通る。失敗した並びは、計算する出力が 1 つだけで、素通しが 3 つ以上あり、最後の出力が素通しのときに限るが、規則は未調査（Issue #226 の最初の報告は else が `a` も素通しする 4 出力） |
| 効かない回避策 | 素通しを枝の中で `optimization_barrier` に通す（then だけ・else だけ・両方）、operand を if の前で barrier に通す、if の結果を barrier に通す、`x + 0`（畳み込まれて素通しに戻る）。いずれも失敗が残る |
| 効く回避策 | (a) 素通しの代わりに、barrier に通した 0 を足すような畳み込まれない演算を枝の中に置く（通る。ただし dtype ごとに違う演算が要り、`x + 0` は -0.0 を +0.0 に変える）。(b) 採用: 素通しの結果を `stablehlo.if` から外し、if の外の `stablehlo.select(pred, then 側, else 側)` にする |
| 原因の見立て | `scf.if` の yield が関数の引数（のバッファ）をそのまま返すとき、結果の `hal.buffer_view` の参照が作られない（バッファのエイリアスの扱い）ように見える。未調査 |
| nabla の対応 | `src/primitives/cond.lisp` の `%cond-emit`（採用理由は下）。形は `cond/emit-*`（`tests/cond-test.lisp`）が、IREE での実行は `cond-passthrough/*` と `pytree-control/jit-cond-with-tree-operands-matches-eager`（`tests/iree`）が守る。「今も落ちること」は `cond-passthrough/iree-known-bug-passthrough-operands-fail-at-run-time` が子プロセスで守り、直れば失敗する |
| 確認した環境 | OrbStack の Rosetta 上の Linux x86_64 VM（`iree-run-module` はネイティブ x86 と同じ結果になるはずだが、CI での確認は未） |

### nabla の回避策の選び方

(A) 素通しを枝の中で barrier に通す案は効かなかった（上の表）。(B) select にする案を採った。素通しの出力は次のように書き換える。

| 出力 | `stablehlo.if` の結果 | if の外 |
| --- | --- | --- |
| どちらの枝も計算 | そのまま | なし |
| 片方の枝だけ素通し | 計算する側の値（素通しの枝は 0 の定数を返す） | `select(pred, 素通しの入力, 計算した値)`（素通しが else なら逆順） |
| 両方の枝が素通し | 外す | `select(pred, then の入力, else の入力)` |

結果が 1 つも残らないときは `stablehlo.if` 自体を出さない（その `cond*` だけを本体に持つ scan は、`src/scan.lisp` の本体のテキストの判定で「制御構造なし」となり ys の barrier が付くが、制御構造が無いので 7 の競合の条件を満たさず安全）。理由: (1) 動く。(2) dtype に依らず、値を変えない（`select` は NaN も -0.0 も整数もビットのまま選ぶ。`x + 0` の類は dtype ごとに演算が違い、-0.0 を壊す）。(3) pred は実行時の値なので、`select(c, x, x)` のように畳み込まれない。(4) emit だけで済み、eqn の形・eager・jvp・vmap は変わらない。コピーの量は、素通しの出力 1 つにつき、素通しの枝の側で 0 の定数の生成（選ばれたときだけ）と、if の後の `select` 1 回（入力 2 つを読み、1 つ書く）。素通しでない出力には何も増えない。

### 下書き（英語）

````markdown
### What happened?

With `llvm-cpu` and either the local-task or the local-sync driver, a `stablehlo.if` whose
region returns three function arguments unchanged (pass-through results), next to one computed
result, fails at run time when the *other* region is taken:

    INVALID_ARGUMENT; ref is null; while invoking native function hal.buffer_view.create

The module compiles without diagnostics. The region that returns the pass-through values runs
fine. Wrapping the pass-through values in `stablehlo.optimization_barrier` (inside the region,
or on the operands before the `if`) does not help; computing something that cannot be folded
(e.g. adding a zero that went through a barrier) does, and so does moving the pass-through results out of the `if`
into `stablehlo.select`. When the `if` results feed further ops instead of being returned
directly, the failure is `OUT_OF_RANGE; indirect buffer reference ...` from
`command_buffer_validation.c`.

### Steps to reproduce your issue

```sh
iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  if-passthrough-operands.mlir -o m.vmfb
iree-run-module --device=local-task --module=m.vmfb --function=main \
  --input=f32=0.0 --input=f32=1.0 --input=f32=2.0 --input=f32=3.0 --input=f32=4.0   # then region: OK
iree-run-module --device=local-task --module=m.vmfb --function=main \
  --input=f32=1.0 --input=f32=1.0 --input=f32=2.0 --input=f32=3.0 --input=f32=4.0   # else region: ref is null
```

```mlir
func.func @main(%p: tensor<f32>, %a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>)
    -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  %half = stablehlo.constant dense<0.5> : tensor<f32>
  %pred = stablehlo.compare LT, %p, %half : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %k = stablehlo.constant dense<0.9> : tensor<f32>
  %r0, %r1, %r2, %r3 = "stablehlo.if"(%pred) ({
    %t0 = stablehlo.multiply %a, %k : tensor<f32>
    stablehlo.return %t0, %b, %c, %d : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }, {
    %e0 = stablehlo.multiply %a, %k : tensor<f32>
    %e1 = stablehlo.multiply %b, %k : tensor<f32>
    %e2 = stablehlo.multiply %c, %k : tensor<f32>
    %e3 = stablehlo.multiply %d, %k : tensor<f32>
    stablehlo.return %e0, %e1, %e2, %e3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
  return %r0, %r1, %r2, %r3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}
```

Whether it fails depends on the layout of the results (C = computed in both regions, P = passed
through by the first region, the second region computes everything): `CPPP`, `CPPPP`, `CPPPPP`
and `PCPPP` fail; `PPPC`, `PPP`, `PPPP`, `CPP`, `PPC`, `CCPP`, `CCPPP`, `CPPPC`, `CPCPP`, `CPPCP`, `PCPC`
and `CPCP` work.

### What component(s) does this issue relate to?

Runtime (possibly the compiler's handling of `scf.if` results that alias function arguments)

### Version information

`iree-base-compiler==3.11.0` from PyPI (`IREE compiler version 3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25`), runtime built from the same commit, Linux x86_64.

### Additional context

Possibly related: #24753 reports the same two error messages (`ref is null` in
`hal.buffer_view.create`, and `indirect buffer reference ... OUT_OF_RANGE`) on 3.12.0rc and says
3.11.0 works. Our module fails on 3.11.0, and we have not tried 3.12.0rc, so this may be a
different trigger of the same symptom. Verified on an x86_64 Linux VM running under Rosetta
(not yet on native x86).
````

## 9. 非最終軸で正規化する layer-norm の jit した grad が、まれにコンパイルに失敗する（#241）

- 現象: `layer-norm` の `:reduction-axes` が最後の軸を含まない（例: 入力 `[2 3 4 5]` で `(1 2)`、`[4 3 5]` で `-2`）モデルの `grad` を IREE の local（llvm-cpu）で jit すると、約 3 割の確率で `One or more operations with large vector sizes (8192 bytes) were found`（`vector.contract`）でコンパイルが失敗する。同じプログラムを繰り返すと通ることもあり、入力の値には依存しない。最後の軸で正規化する場合は 30 回中 0 回、順伝播の jit は毎回通る。
- 再現: `tests/fixtures/layers/flax-layers.lisp` の `layer-norm-reduction-axes` のケースを `(nabla:jit (nabla:grad loss :argnums '(0 1)))` で 10 回コンパイルする（x86_64 の VM で 4/10 失敗、`layer-norm-reduction-middle-axis` は 2/10）。
- 状態: 最小 MLIR 未作成・未報告。Rosetta の VM で観測したもので、x86 ネイティブでの頻度は未確認。回避策として `tests/nn/layers-iree-test.lisp` の 2 ケースは grad の jit を飛ばして eager の grad で比べている（`*layers-iree-flaky-grad-cases*`）。
- group-norm（#260）: 群の大きさ 1（instance norm。統計を取る軸が入力の最後の軸（チャンネル）を含まない）の `grad` の jit も同じエラー（`large vector sizes (16384〜32768 bytes)`）で失敗する。CI（GitHub Actions の x86_64 ネイティブ）で 3 回中 3 回で、確率的ではないように見える。統計を `[バッチ 群 要素]` に transpose して最後の軸で縮約する形にしても、群の大きさが 1 だと transpose がチャンネルの軸を動かすので回避できなかった（群の大きさが 2 以上のケースは通る）。`tests/nn/group-norm-iree-test.lisp` の `*group-norm-iree-failing-grad-cases*` が grad の jit を飛ばしている
