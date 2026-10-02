# フェーズ2 報告書: grad と PJRT 比較

フェーズ2（親 issue #90、「jvp + transpose 方式の `grad` / `value-and-grad` と、2層 MLP の学習、PJRT バックエンドと IREE の比較」）で作ったもの・得た知見をまとめ、フェーズ3（vmap と制御構造）への引き継ぎ材料にする。形式は [phase1-report.md](phase1-report.md) に合わせてある。

## 1. 範囲と結論

#90 の完了条件（計画の「見落としやすい完了条件」を含む）に対する達成状況:

| 完了条件 | 判定 | 検証方法 |
| --- | --- | --- |
| jvp + transpose 方式の `grad` / `value-and-grad` があり、`jit` と合成できる | 達成 | `tests/iree/grad-test.lisp`（`(jit (grad f))` の IREE 実行、defjit・jitted-function との合成）、`tests/ad/` の PBT（中心差分・内積テスト）。全微分可能プリミティブが jvp / transpose ルールを持つことは `tests/primitives/registry-test.lisp` が検査 |
| 2層 MLP を手書きループで学習でき、JAX の SGD と一致する | 達成 | `tests/iree/mlp-train-test.lisp` の `train/mlp-matches-jax-fixture`（各ステップの損失とパラメータを JAX フィクスチャと比較）、PBT（seed を変えて80ステップで損失が半分未満）、2ステップ目以降 `*jit-miss-count*` 不変、`examples/mlp.lisp` の実行 |
| PJRT (XLA CPU) バックエンドで同じ StableHLO をコンパイル・実行できる | 達成 | `tests/pjrt/executable-test.lisp`（compile / load / invoke、`(jit f :backend :pjrt)`、fingerprint に sha256 と API 版、ディスクキャッシュ、フェーズ0フィクスチャ） |
| 同じ StableHLO を PJRT にも流し、コンパイル時間と学習ステップ時間を IREE と実測比較した表がある | 達成（CPU のみ。CUDA は未測定） | 本書 §4。`scripts/bench-backends.sh`。出力の読み書きは `tests/bench-test.lisp`（small）、スクリプトを小さな設定で1回走らせるのは `tests/iree/bench-test.lisp` / `tests/pjrt/bench-test.lisp`（medium） |
| core は PJRT の名前を知らない | 達成 | `tests/backend-test.lisp` の `backend/core-sources/do-not-mention-iree`（`src/` を再帰して "iree" と "pjrt" を検査。#99 の再帰化と #100 の "pjrt" 検索の両方を残した） |
| 既定スイート（small + medium）が CI で通る | 達成（ローカル） | PR #97〜#112 の CI がすべて green。このブランチの CI は PR 作成後。ローカルでは `NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_PJRT=1 scripts/run-tests.sh` が通る |
| 微分できる各プリミティブに jvp ルールが、線形な各プリミティブに transpose ルールが揃っている | 達成 | `tests/primitives/registry-test.lisp`（全微分可能プリミティブの jvp、線形プリミティブの transpose の存在を検査） |
| GPU（CUDA）での数値一致と計測 | 未測定 | このマシンに GPU が無い（#12 から引き続き）。ベンチは `--cuda` で測れる形にしてあり、GPU が無ければ「未測定」と出す |

結論: CPU 上の完了条件はすべて達成した。GPU に関する項目だけが未測定のまま残る。

## 2. 作ったもの

PR 番号は GitHub 上のもの。stacked PR で、下から順に積んである。

### 2.1 自動微分

- **#76（PR #97）**: 自動微分のテスト支援。f64 の中心差分・内積テスト（随伴性）のヘルパー。
- **#77a / #77b / #77c（PR #99 / #101 / #102）**: 骨格。`primitive` に `jvp` / `transpose` スロットと `def-jvp-rule` / `def-transpose-rule`、symbolic zero（`make-symbolic-zero` / `instantiate-zero` / `add-tangents`）、`%call-with-fresh-trace`、`inline-graph`、`dce-graph`、`jvp-graph`。
- **#82（PR #105）**: `linearize`（jvp を非線形部分と線形部分に分割）と `transpose-graph`、vjp。
- **#80 / #81（PR #106 / #107）**: 要素演算・形状・縮約・dot-general の jvp ルールと `stop-gradient`。
- **#83 / #84（PR #109 / #110）**: 線形プリミティブの transpose ルールと dot-general の transpose ルール。
- **#86（PR #111）**: 公開 API の `grad` / `value-and-grad`、jit との合成、全ルール検査。
- **#88（PR #112）**: 2層 MLP の学習 end-to-end（`examples/mlp.lisp` の `make-mlp-train-step`、JAX フィクスチャ）。
- **mutation runner の拡張（PR #104）**: `def-jvp-rule` / `def-transpose-rule` / `def-jvp-partials` を変異対象にした。フェーズ1 §5 の留保（変異体の数が少ない）への対処の続きで、PR #95（全箇所への変異と条件・削除・リテラルの演算子）の上に積んだ。

### 2.2 PJRT

- **#79（PR #98）**: `nabla/ffi-support`。シグナルハンドラと浮動小数点トラップの保護を `nabla/iree` から共有の場所へ移した。
- **#78（PR #100）**: PJRT プラグインの固定取得（`third_party/pjrt.lock`、`scripts/fetch-pjrt.sh`、`docs/pjrt-setup.md`）と `nabla/pjrt` の骨格。
- **#85（PR #103）**: クライアント・デバイス・device-array（`to-device` / `to-host`）。
- **#87（PR #108）**: コンパイル・ロード・実行と `backend-fingerprint`（プラグインの sha256 + PJRT API 版）。`(jit f :backend :pjrt)`。
- **#89（本 PR）**: 計測スクリプトと本書。#88 の上に積み、#87（PR #108）のブランチを merge している。

## 3. 知見

### 3.1 設計判断

- **symbolic zero**: ゼロの接線・余接線は配列ではなく `symbolic-zero`（aval だけ持つ）で表す。全ゼロの eqn は変換側で飛ばし、`add-tangents` はゼロなら他方をそのまま返す。ゼロが実体化される（`instantiate-zero`）のは、出力に出すときと、`select` の片側のように値が要る場所だけ。transpose でも、ゼロの余接線の eqn を飛ばす（#82 で修正）。
- **`:i1` の接線**: 接線空間は自明なので、非 float の接線は常に symbolic zero。`jvp-graph` の既定の `nonzero` は「float の invar だけ」で、非 float に接線を渡すと `autodiff-error`。出力の個数は「主値 ++ 接線」で揃え、`:i1` の接線は全 false の配列で実体化する（compare は symbolic zero、select は pred の接線を無視、非 float への convert は zero）。
- **DCE の位置**: `jvp-graph` の中では DCE しない。`linearize` が接線部分に `dce-graph` をかける。jvp の出力は主値と接線が混ざっているので、ここで消すと主値の使い道が分からなくなるため。
- **stop-gradient**: `stablehlo.optimization_barrier` として出す。微分では定数扱いにしつつ、値は変えない（logsumexp の最大値の引き算で使う。`examples/mlp.lisp`）。
- **接線は線形にしか流さない**: jvp ルールは接線を、その被演算子について線形なプリミティブにしか流さない（接線同士の積、接線への exp / log / tanh / max / min は禁止）。transpose が各ルールの局所的な転置で済むのはこの制約のおかげ。
- **入れ子**: `grad` は f を aval で新しいトレースにトレースし、変換後に外側へ `inline-graph` する。f が外側のトレーサを閉包で捕まえると `tracing-error`。

### 3.2 PJRT と XLA

- **XLA CPU は bf16 の reduce を逐次 bf16 丸めで累積する**: IREE は（`reduce` の emit で f32 累積に直しているので）f32 で累積するが、XLA は要素ごとに bf16 に丸めて足す。同じ StableHLO でも bf16 の `reduce_sum` は両者で一致しない。テストは、許容誤差を緩める代わりに「逐次 bf16 丸めの参照実装」と比べる（#87 のレビューで rtol 9 倍の緩和を撤回した）。
- **PJRT のシグナル調査**: CPU プラグインの dlopen / Plugin_Initialize / Client_Create / BufferFromHost / ToHost、および Compile / Serialize / Load / Execute の前後で、全シグナル（1..64）の処分を比較したところ変化は無かった（XLA は LLVM のシグナル登録をしない）。IREE のような「世界を止めた1点での登録」は不要。`with-lisp-signal-handlers-preserved` と浮動小数点トラップのマスクは多重防御として残してあり、子プロセスのテストが GC スレッド並走下での処分と FP モードの不変を毎回検査する。実験の記録は `docs/pjrt-setup.md`。
- **空の CompileOptions は SIGABRT**: `PJRT_Client_Compile` に空のオプションを渡すとプラグインが CHECK で落ちる。`num_replicas = num_partitions = 1` と `compile_portable_executable = 1` だけの最小の protobuf を手書きで渡す（`src/pjrt/executable.lisp`）。
- **`PJRT_Buffer_Type_INVALID` でのバッファ作成も abort する**: エラー変換のテストは小さい dst_size の ToHostBuffer で行う。
- **SBCL の GC ロック餓死**: 別スレッドが sleep 無しの tight loop で `(gc :full t)` を回すと、main スレッドが進めず固まる（`src/ffi-support/signals.lisp` 冒頭の餓死リスク3）。PJRT のクライアント寿命管理は、Lisp の mutex を finalizer スレッドが GC 停止中に保持しうることを避けるため、バッファの finalizer が整数と構造体だけを捕まえてアトミックに数える方式にし、子プロセステストは GC の間に sleep を入れる。
- **プラグインの sha256 は約2.4秒**: 260 MB のファイルを読む。プロセスにつき1回だけ計算してキャッシュする（§4 の「初期化」の行）。

### 3.3 mutation runner の対象拡張

フェーズ1 §5 の留保（変異演算子と粒度の少なさ）に対し、PR #95 で全箇所への変異と条件・削除・リテラルの演算子を足し、PR #104 で `def-jvp-rule` などのルール定義マクロの本体を変異対象にした。#81 での実測は `rules-shape.lisp` で total=6 killed=5 survived=1 で、生き残った1つは reduce-max の `ind` を 1 から 2 にする、スケールに不変な等価変異だった。

## 4. IREE と PJRT の実測比較

### 4.1 測定条件

- iree-local: date=2026-10-01 17:24; cpu=Intel(R) Xeon(R) Processor @ 2.80GHz; cores=4; sbcl=2.2.9.debian; disk-cache=disabled (nabla:*compile-cache-directory* = NIL); threads=nabla sets no thread count (IREE local-task default); OMP_NUM_THREADS=unset XLA_FLAGS=unset; iree-commit=e4a3b0405d7d23554da26403658d0e8c3c5ecf25; iree-compiler-revision=3.11.0rc20260316 @ e4a3b0405d7d23554da26403658d0e8c3c5ecf25
- pjrt-cpu: date=2026-10-01 17:25; cpu=Intel(R) Xeon(R) Processor @ 2.80GHz; cores=4; sbcl=2.2.9.debian; disk-cache=disabled (nabla:*compile-cache-directory* = NIL); threads=nabla sets no thread count (XLA CPU default); OMP_NUM_THREADS=unset XLA_FLAGS=unset; pjrt-plugin=/root/.local/share/nabla/pjrt-0.0.1/cpu/xla_cpu_pjrt.so; pjrt-plugin-sha256=88c3f28c5900f48ad4a82cab9e79bba9537809743395e341485f8794268820dc; pjrt-api=0.81

- 計測は `scripts/bench-backends.sh --cuda`（既定の設定: small / medium / large、ステップ200回・ウォームアップ20回、コンパイル時間は3回の繰り返し）で、backend ごとに別の SBCL プロセスで測った。
- MLP のサイズ（N = バッチ、D = 入力次元、H = 隠れ層、C = クラス数）: small N16 D2 H8 C2、medium N256 D64 H128 C10、large N1024 D256 H512 C10。f32。モデルは `examples/mlp.lisp` の `make-mlp-train-step`（`(jit (value-and-grad loss :argnums (0 1 2 3)))` と Lisp 側の SGD）。
- ディスクキャッシュは無効（`nabla:*compile-cache-directory*` を NIL）。コンパイル時間の各繰り返しでは新しい jit した関数を作るので、インメモリの jit キャッシュも効かない。
- **測定中、同じ4コアのマシンで他のジョブ（別の Driver のテスト）が走っていた**。絶対値はノイズを含み、p10-p90 の幅が広い項目がある。桁と傾向を見る表で、絶対値を保証するものではない。

### 4.2 結果

セルはミリ秒。n > 1 の項目は「中央値 [p10-p90]」（ステップは n = 200、コンパイルの各段は n = 3）。`stage/*` は jit パイプラインの各段を単独で測ったもの、`jit/first-call` は新しい jit した関数の初回呼び出し全体（trace + emit + compile + load + to-device + invoke + to-host）、`stage/backend-compile` は IREE ではコンパイラ（MLIR → vmfb）、PJRT では `PJRT_Client_Compile` + `PJRT_Executable_Serialize`、`stage/backend-load` は IREE では vmfb のロード、PJRT では `PJRT_Executable_DeserializeAndLoad`。`step/full` は SGD の更新込みの1ステップ、`step/jitted-call` は jit した関数の呼び出しだけ（to-device / to-host を含む）。CUDA（iree-cuda / pjrt-cuda）はこのマシンに GPU が無く**未測定**（表からは省いた。`scripts/bench-backends.sh --cuda` は GPU が無いと「未測定」と出す）。

#### 初期化（プロセスにつき1回。ms）

| 項目 | iree-local | pjrt-cpu |
| --- | --- | --- |
| init/compiler-load | 13.537 | - |
| init/device-create | 16.317 | - |
| init/plugin-load | - | 25.358 |
| init/client-create | - | 21.479 |
| init/plugin-sha256 | - | 2420.255 |

#### small（ms。n > 1 は 中央値 [p10-p90]）

| 項目 | iree-local | pjrt-cpu |
| --- | --- | --- |
| size/stablehlo-text | 4901 chars | 4901 chars |
| size/compiled | 27826 bytes | 46223 bytes |
| stage/trace | 0.334 [0.318-6.976] | 0.419 [0.349-11.183] |
| stage/emit-stablehlo | 0.286 [0.233-0.308] | 0.310 [0.209-0.480] |
| stage/backend-compile | 1035.684 [1010.112-1039.127] | 140.524 [132.319-157.657] |
| stage/backend-load | 0.617 [0.528-3.387] | 4.430 [4.121-4.490] |
| jit/first-call | 1125.758 [1094.788-1136.256] | 130.737 [129.490-182.402] |
| jit/second-call | 0.734 [0.728-2.333] | 0.516 [0.461-0.535] |
| step/full | 0.520 [0.380-0.631] | 0.257 [0.197-0.427] |
| step/jitted-call | 0.373 [0.334-0.431] | 0.305 [0.224-0.605] |

#### medium（ms。n > 1 は 中央値 [p10-p90]）

| 項目 | iree-local | pjrt-cpu |
| --- | --- | --- |
| size/stablehlo-text | 5069 chars | 5069 chars |
| size/compiled | 45578 bytes | 60023 bytes |
| stage/trace | 0.397 [0.384-0.574] | 0.547 [0.507-0.558] |
| stage/emit-stablehlo | 0.275 [0.260-0.388] | 0.397 [0.293-0.415] |
| stage/backend-compile | 1333.405 [1321.695-1507.947] | 200.410 [179.056-206.466] |
| stage/backend-load | 0.569 [0.512-0.602] | 7.411 [5.691-8.272] |
| jit/first-call | 1531.229 [1503.033-1561.129] | 176.743 [174.776-186.273] |
| jit/second-call | 1.302 [1.211-1.460] | 1.341 [1.319-1.574] |
| step/full | 1.755 [1.347-2.324] | 1.683 [1.431-2.278] |
| step/jitted-call | 0.968 [0.790-1.443] | 1.232 [0.993-1.602] |

#### large（ms。n > 1 は 中央値 [p10-p90]）

| 項目 | iree-local | pjrt-cpu |
| --- | --- | --- |
| size/stablehlo-text | 5129 chars | 5129 chars |
| size/compiled | 46242 bytes | 50997 bytes |
| stage/trace | 0.435 [0.428-0.563] | 0.441 [0.383-0.702] |
| stage/emit-stablehlo | 0.352 [0.267-0.375] | 0.273 [0.253-0.291] |
| stage/backend-compile | 1602.872 [1567.869-1715.599] | 161.095 [150.173-180.243] |
| stage/backend-load | 0.578 [0.526-0.700] | 7.126 [5.738-8.960] |
| jit/first-call | 1589.354 [1527.090-1747.579] | 194.480 [170.311-212.232] |
| jit/second-call | 21.831 [21.096-23.212] | 6.957 [6.564-8.123] |
| step/full | 26.811 [23.258-31.431] | 11.047 [9.361-13.875] |
| step/jitted-call | 21.949 [19.502-26.093] | 5.445 [4.247-6.958] |

### 4.3 読み取れること

- **コンパイル時間は XLA（PJRT）の方が約 6〜10 倍短い**: `stage/backend-compile` は IREE が約 1.0〜1.6 秒、PJRT が約 0.14〜0.2 秒。原因は切り分けていない（候補: IREE の既定の最適化パイプラインとフラグ、ELF リンク。XLA:CPU も LLVM で JIT するので「LLVM を使うから」は差の説明にならない）。ロード（`backend-load`）は逆に IREE の方が速い（約 0.6 ms 対 4〜7 ms）。初回の `jit` 呼び出し全体は、IREE が約 1.1〜1.6 秒、PJRT が約 0.13〜0.19 秒。
- **PJRT にはプロセスにつき1回の初期化がある**: プラグインの sha256（約 2.4 秒、fingerprint 用。**プロセスにつき1回だけのコストで、2回目以降はキャッシュされる**）が支配的で、dlopen とクライアント作成は数十 ms。ディスクキャッシュを使う場合の初回の `backend-fingerprint` で一度だけ払う。
- **学習ステップは小さいモデルでは同程度、大きいと XLA が速い**: small / medium では `step/full` が IREE 0.5 / 1.8 ms、PJRT 0.26 / 1.7 ms でほぼ同じ（to-device / to-host と SGD の更新といったホスト側の仕事が効く）。large（N1024 H512）では IREE 約 27 ms、PJRT 約 11 ms と XLA が約 2.4 倍速い。IREE の `llvm-cpu` のスレッド・タイリングの設定は既定のまま（nabla は何も設定していない）なので、調整で縮まる可能性はあるが、今回は調べていない。
- **「どちらが優れている」とは言えない**: 1台の共有マシン、1つのモデル、既定の設定での比較。結論を出すには、コンパイルフラグ（IREE の `--iree-opt-level` など）とスレッド設定を揃えた測り直しが要る。

## 5. 既知の制限と積み残し

- **`pjrt-module` の finalizer はフェーズ2の後に追加した**（#120）: 報告書の計測時点では `backend-unload` を呼ばないと PJRT の実行体が解放されなかった。現在は GC でも解放される（明示的な `backend-unload` との二重解放は起きない）。
- **argnums がリストの勾配は jit の出力にできない**: 勾配のリストは jit の出力（フラットな多値）にならないので、`examples/mlp.lisp` のように多値に直す。
- **`grad` は jitted-function の `:backend` を無視する**: `grad` が jit した関数をトレース中に呼ぶと、その関数は展開されるだけで `:backend` は見られない（外側の `jit` の backend が使われる）。
- **PJRT で `:i1` は未対応**（`unsupported-dtype`）。複数デバイス・replica、CUDA プラグインでの動作も未検証。
- **GPU が無く未測定**: #12（local/cuda の数値一致）と、§4 の CUDA の計測。GPU のある環境で `scripts/bench-backends.sh --cuda` と `NABLA_TEST_SIZES=large NABLA_REQUIRE_CUDA=1 scripts/run-tests.sh` を実行する。
- **計測のノイズ**: §4.1 のとおり共有マシンでの1回の測定。フラグ・スレッド数を揃えた比較や、定期的な計測は積み残し。
- **SBCL の GC ロック餓死**: 別スレッドが sleep 無しの tight loop で `(gc :full t)` を回すと main スレッドが進まない（§3.2）。対策は子プロセステストの GC 間の sleep だけで、`src/ffi-support/signals.lisp` のリスク3は残っている。
- **#68 / #73**: フェーズ1から引き続き（in-process コンパイラのメモリ破壊の根本原因、IREE 上流への報告）。

## 6. フェーズ3（vmap と制御構造）への引き継ぎ

- **ルールの拡張点**: `primitive` のスロットと `def-*-rule` は、バッチ化ルール用のスロットを同じ形で足せる。jvp / transpose の「ルールは `src/ad/rules-*.lisp` に置く」という配置を踏襲すると、プリミティブ定義との衝突が避けられる。
- **graph → graph 変換の挿入点**: `%jit-cache-lookup-or-compile` の trace と emit の間（フェーズ1 §7）。`vmap` も `grad` と同じ形で書ける。
- **入れ子のトレース**: `%call-with-fresh-trace` と `inline-graph` が、`grad` の中の `grad`（高階微分）や `vmap` との合成の足場になる。ただし閉包で外側のトレーサを捕まえる関数は未対応（`tracing-error`）。
- **制御構造**: `if` は両枝評価の `select`（フェーズ1）のまま。ループ・条件分岐の微分は、`while` / `cond` に相当するプリミティブを入れるフェーズ3の仕事。
- **backend の追加**: `backend` プロトコル（`src/backend.lisp`）は IREE と PJRT の2つの実装で確かめられた。
- **計測**: `scripts/bench-backends.sh` はモデルを `examples/mlp.lisp` に固定している。モデルを差し替えられるようにすると、フェーズ4（Flax 相当）以降の回帰計測に使える。
