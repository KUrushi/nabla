# 転置を含む dot_general の速さ（issue #265）

IREE の CPU（iree-local）では、縮約する軸が lhs の先頭にある `dot_general`（`contracting_dims = [0] x [0]`）が、
同じ大きさの普通の dot より遅い。この形は `dot` の転置ルールが重みの勾配に必ず作る（MLP の W1 の勾配
`xᵀ · dz`）。#193 の測定（Colab の CPU）では約 2 倍だった。

この文書は、nabla の emit で直せるかを測った結果をまとめる。結論は **emit を変えない**:

- 「transpose + 普通の dot」を出しても、IREE が同じ形に畳むので何も変わらない（コードが同じになる）
- transpose を `optimization_barrier` に通すと畳まれずに速くなる。しかし core の emit に入れると
  PJRT（XLA CPU）の学習ステップが遅くなり、iree-local の小さなモデルも遅くなる

測定スクリプトは `scripts/profile-call-overhead.lisp` の `profile-ops`（使い方は CLAUDE.md の「コマンド」）。

## 測定条件

- 2026-10-31、OrbStack の x86_64 VM（Apple Silicon の上の Rosetta 2 によるエミュレーション。`/proc/cpuinfo` は
  `VirtualApple @ 2.50GHz`、7 コア）。IREE 3.11.0（`third_party/iree.lock` のコミット）、PJRT CPU プラグイン 0.0.1、SBCL 2.2.9
- iree-local は nabla の `compile-flags` の `:local` のまま（`--iree-llvmcpu-target-cpu=host`）。Rosetta の上では host が
  `westmere` になり、AVX を使わないコードになる。**絶対値はネイティブの x86 の約 10 倍遅い**ので、比だけを見る
- 各項目 100 回の invoke（入力を送った後の呼び出しだけ。出力の読み出しを含めない）の中央値 [p10-p90]、ms

## 1 つの演算（profile-ops）

大きさは large の MLP（N1024 D256 H512 C10）の W1 の勾配に合わせた。「正規形」は縮約軸が lhs の最後・rhs の先頭にある形。

| 演算 | iree-local | pjrt-cpu |
| --- | --- | --- |
| 正規形の dot 256x1024 · 1024x512 | 28.70 [27.25-30.11] | 2.40 [2.21-2.63] |
| `[0] x [0]`（1024x256, 1024x512 → 256x512） | 45.82 [44.16-48.68] | 2.48 [2.31-2.75] |
| lhs を `stablehlo.transpose` してから正規形の dot | 47.59 [46.00-49.61] | 2.47 [2.31-2.73] |
| transpose → `optimization_barrier` → 正規形の dot | 30.29 [29.11-31.28] | 2.53 [2.36-2.84] |
| 正規形の dot 1024x512 · 512x256 | 28.16 [25.07-32.50] | 2.37 [2.28-2.49] |
| `[1] x [1]`（1024x512, 256x512 → 1024x256。入力の勾配の形） | 27.40 [25.12-29.29] | 2.45 [2.10-2.80] |
| rhs を transpose してから正規形の dot | 27.94 [25.46-30.43] | 2.42 [2.06-2.70] |
| transpose だけ（1024x256） | 0.235 [0.216-0.257] | 0.136 [0.133-0.143] |

- iree-local の `[0] x [0]` は正規形の 1.6 倍遅い。rhs を転置して読む `[1] x [1]` は正規形と変わらない
- transpose を明示しても `[0] x [0]` と同じ速さ（下の「IREE の中で起きていること」のとおり、同じコードになる）
- barrier に通すと、transpose（0.24 ms）と正規形の dot の和に近い 30.3 ms になる
- pjrt-cpu はどの形でも差が無い（XLA は転置をレイアウトで吸収する）

## IREE の中で起きていること

`iree-compile --compile-to=<段>` で途中の IR を見た（フラグは `:local` と同じ）。

- StableHLO を linalg に下ろす入力の変換で、`[0] x [0]` の `dot_general` は「lhs の転置（`linalg.generic`）+
  `linalg.matmul`」になる。nabla が transpose を明示した形とここで同じになる
- 大域最適化（`--compile-to=global-optimization` の出力で初めて現れる）で、その転置が matmul に畳まれ、`indexing_maps`
  の lhs が `(d2, d0)`（lhs を転置して読む）の `linalg.matmul` 1 つの dispatch になる。`[0] x [0]` と transpose + dot の flow の IR はバイト単位で同じで、
  vmfb の違いはソース名などのメタデータ（25 バイト）だけだった
- 遅いのは、この「lhs を転置して読む matmul」の llvm-cpu のコード生成。rhs を転置して読む matmul（`[1] x [1]`）は遅くない
- barrier があると転置は畳まれず、転置の dispatch と正規形の matmul の dispatch の 2 つになる

## コンパイラのフラグ

`NABLA_PROFILE_IREE_COMPILE_FLAGS` で `:local` のフラグの後ろに足して測った。MLP の invoke は large の学習ステップ
（`profile-mlp` の `mlp/invoke`）。

| フラグ | 正規形の dot | `[0] x [0]` | `[1] x [1]` | large の MLP の invoke |
| --- | --- | --- | --- | --- |
| なし | 28.70 | 45.82 | 27.40 | 95.85 |
| `--iree-opt-level=O3` | 30.26 | 47.83 | 27.44 | コンパイルできない（※） |
| `--iree-opt-data-tiling=true` | 38.25 | 38.53 | 38.83 | 84.05 |
| `--iree-llvmcpu-enable-ukernels=all` | 30.55 | 48.19 | 27.13 | 94.50 |

- ※ `func.func op exceeded stack allocation limit of 32768 bytes for function. Got 122880 bytes`（MLP の出力側の
  `matvec_like_1024x10` の dispatch）
- data tiling は転置の有無の差を消すが、正規形の dot も 1.3 倍遅くなる。MLP 全体では 12% 速いが、Rosetta の上の
  westmere のコードでの結果なので、ネイティブの AVX2 / AVX-512 で測るまで既定にはしない
- 大域最適化の転置の伝播を止める `--iree-global-opt-propagate-transposes=false` を `iree-compile` に渡すと、転置は畳まれず、
  barrier を入れたときと同じ「転置の dispatch + 正規形の matmul」になる。ただし nabla が使う埋め込み API のセッションの
  フラグとしては `option not found` で受け付けられず（セッションに登録されない大域のオプションと思われる。未確認）、
  nabla からは測れていない

## emit に barrier を入れたときの学習ステップ

core の `%dot-emit-lines` で、縮約軸が lhs の末尾にない `dot_general` の lhs を `[バッチ 自由 縮約]` に transpose し、
`optimization_barrier` に通してから正規形の `dot_general` にする変更を試した（small の golden テスト・IREE での
数値の一致・mutation testing まで通した上で採らなかった）。main と交互に 2 回ずつ測った `mlp/invoke`:

| 設定 | main #1 | 変更後 #1 | main #2 | 変更後 #2 |
| --- | --- | --- | --- | --- |
| iree-local large | 92.62 | 68.18 | 90.24 | 67.72 |
| iree-local small（N16 D2 H8 C2） | 0.371 | 0.507 | 0.360 | 0.531 |
| pjrt-cpu large | 5.87 | 8.47 | 7.00 | 10.73 |

- iree-local の large は 26% 速くなる（W1 の勾配の分）
- しかし pjrt-cpu の large は 45% 遅くなる。XLA は barrier のせいで転置をレイアウトに吸収できず、実体化する
- iree-local の small も 40% 遅くなる（転置の dispatch が 1 つ増える固定費）

core の emit は全バックエンドで共有するので、PJRT を遅くする変更は入れない。バックエンドの能力を問う総称関数を
足して IREE だけで barrier を入れる案は、core が IREE の性質を知ることになるので採らない。

## 今後

- IREE を上げたら `profile-ops` で測り直す（lhs を転置して読む matmul のコード生成が直っていれば不要になる）
- ネイティブの x86（AVX2 / AVX-512）で、data tiling と barrier の両方を測り直す。Rosetta の westmere のコードとは
  比が変わりうる
- IREE の側だけで直す候補は 2 つ。どちらも core と PJRT を変えない。上の比がネイティブで再現してから考える
  - 埋め込み API の大域のオプション（`ireeCompilerSetupGlobalCL`）で `--iree-global-opt-propagate-transposes=false` を
    渡す。プロセス全体のコンパイルに効き、ほかの転置の伝播（融合のための）も止まるので、学習ステップ全体で測る
  - nabla/iree の `backend-compile` で、縮約軸が lhs の末尾にない `dot_general` の行を transpose + barrier に書き換える。
    small のような小さな形では転置の dispatch の固定費で遅くなるので、大きさの閾値が要る
