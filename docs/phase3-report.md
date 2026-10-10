# フェーズ3 報告書: vmap と制御構造と PRNG

フェーズ3（親 issue #124、「`vmap`、`cond*` / `while-loop` / `scan` の制御構造とその自動微分、明示的なキー渡しの PRNG」）で作ったもの・得た知見をまとめ、フェーズ4（Flax 相当）への引き継ぎ材料にする。形式は [phase2-report.md](phase2-report.md) に合わせてある。

## 1. 範囲と結論

#124 の完了条件に対する達成状況:

| 完了条件 | 判定 | 検証方法 |
| --- | --- | --- |
| `vmap` があり、`jit` / `grad` / 別の `vmap` と合成できる | 達成 | `tests/vmap-test.lisp` ほか `tests/vmap-*-test.lisp`（PBT: 「各要素に f を適用して積み直した結果」と一致）、`tests/per-example-test.lisp`（`jit (vmap (grad f))`、`grad` の中の `vmap`、`vmap (vmap f)` を eager と IREE で確認） |
| 全プリミティブにバッチ化ルールがある | 達成 | 要素演算（#128）、形状・縮約・dot-general（#129）、制御構造（#140）、PRNG 関連（#136）。`tests/primitives/registry-test.lisp` の `registry/every-registered-primitive-has-a-batch-rule-or-is-excluded` が検査（#142 で追加。27個すべてがルールを持ち、除外リストは空） |
| `cond*` / `while-loop` / `scan` が eager・`jit`（IREE）で動く | 達成 | `tests/cond-test.lisp` / `while-loop-test.lisp` / `scan-test.lisp` と `tests/iree/` の medium テスト（eager との一致） |
| 制御構造を `grad` できる | 達成（`while-loop` の逆モードを除く） | `cond*` の jvp・transpose（#134）、`scan` の jvp（#135）・partial eval† と transpose（#139）。中心差分・内積テストの PBT。`while-loop` の逆モードは JAX と同じく対象外（§4） |
| 制御構造を `vmap` できる | 達成 | #140。条件がバッチされる `cond*` / `while-loop` を含む PBT |
| per-example 勾配が JAX と一致する | 達成 | #138。`tests/fixtures/per-example/`（JAX 0.10.2、x64 無効）と f32 の許容誤差で一致 |
| 明示的なキー渡しの PRNG が eager・`jit`・`vmap` で動く | 達成 | #133（`rng-bit-generator`）と #136（`prng-key` / `split` / `fold-in` / `uniform` / `normal`）。eager == IREE == PJRT（XLA CPU）のビット一致、統計検定 |
| RNN を `scan` で学習できる（end-to-end） | 達成 | #141。`tests/iree/rnn-train-test.lisp`、JAX（`jax.lax.scan`）フィクスチャと損失の軌跡が一致 |
| 定型の `do` ループを `scan` に展開する | 達成（`do` のみ。`dotimes` / `loop` / `do*` は対象外） | `tests/loop-scan-test.lisp`（PBT: 展開した `scan` と Lisp の `do` の一致）。§3.11 |
| GPU（CUDA）での数値一致と計測 | 未測定 | このマシンに GPU が無い（#12 から引き続き） |

結論: CPU 上の完了条件はすべて達成した。既定スイート（small + medium、IREE・PJRT 必須）は各 PR で通した（#136 の PR #156 のブランチで small 140212 checks / medium 1493 checks。このブランチ（#142）の最終実行は #156 と #163 を取り込んだ後の small 140473 checks / medium 1587 checks）。

## 2. 作ったもの

PR 番号は GitHub 上のもの。stacked PR で、下から順に積み、親が squash merge されるたびに `main` を merge して取り込んだ。

- **#143**: 並行作業用のアンカーコメント（`src/package.lisp`・`nabla.asd`・README・用語集の共有ファイルに、issue ごとの書き足し場所を置いた。#142 で撤去した）。
- **#126（PR #144）**: 整数 dtype `:i32` / `:u32` / `:u64`。eager は StableHLO と同じ 2 の補数で折り返す。
- **#125（PR #145）**: `vmap` の骨格（`src/vmap.lisp`）、`primitive` の `batch` スロットと `def-batch-rule`、`vmap-error` / `no-batch-rule`。
- **#127（PR #146）**: サブグラフ†を持つ eqn、複数出力の eqn（`:multiple-outputs`）、closure conversion†（`%trace-subgraph`）、StableHLO のリージョン出力。
- **#128 / #129（PR #147 / #148）**: 要素演算（`broadcast_batcher` 相当）と、形状・縮約・`dot-general` のバッチ化ルール。
- **#131（PR #149）**: `while-loop`（`stablehlo.while`）。
- **#130（PR #150）**: `cond*`（`stablehlo.if`）。
- **mutation runner（PR #151）**: `def-batch-rule` を変異対象にした。
- **#138（PR #152）**: per-example 勾配（`examples/mlp.lisp` の `make-per-example-grad`）。`src` の変更は無く、合成が既存のルールだけで動くことを JAX フィクスチャで確かめた。
- **#133（PR #153）**: `rng-bit-generator`（Threefry-2x32、eager == IREE == PJRT のビット一致）。
- **#132（PR #154）**: `scan`（`:i32` のカウンタを持つ `stablehlo.while` に落とす）。
- **#140（PR #155）**: `cond*` / `while-loop` / `scan` のバッチ化ルール（`src/ad/rules-batch-control.lisp`）。
- **#134（PR #157）**: `cond*` の jvp・transpose、`while-loop` の jvp（前進のみ）。
- **#135（PR #158）**: `scan` の jvp。
- **#139（PR #160）**: `scan` の partial eval（`*partial-eval-rules*`）と transpose。これで `grad` が `scan` を通る。
- **#141（PR #161）**: RNN の end-to-end 学習（`examples/rnn.lisp`）。
- **#136（PR #156）**: PRNG の公開 API（`prng-key` / `split` / `fold-in` / `uniform` / `normal` / `prng-error`）。`shift-right-logical` / `bitwise-or` / `bitcast-convert` の内部プリミティブとバッチ化ルール、`rng-bit-generator` のバッチ化も含む。
- **#137（PR #163）**: `with-tracing` の中の定型の `do` ループを `scan` に展開する（`src/loop-scan.lisp`）。
- **フォローアップ #159**: `scan` の `ys` が IREE で O(length × |ys|) になる問題（§5）。

## 3. 設計上の判断

### 3.1 ルールの置き場所と形

- **バッチ化ルール**は jvp / transpose と同じく `src/ad/rules-batch-*.lisp` に置き、`def-batch-rule`（`src/vmap.lisp`）で書く。`primitive` 構造体の最後に `batch` スロットがあり、`defprimitive` の最後のキー `:batch` で設定できる（`%existing-rule` で再定義しても引き継がれる）。`def-batch-rule` は `(name (args batch-dims &rest param-lambda-list) body)` の形で、body は `(values outs out-dims)` を返す。**outs と out-dims は、単一出力のプリミティブでも常にリスト**にする（複数出力と同じ形に揃えて、変換側の場合分けを無くすため）。`batch-dims` は各引数のバッチ軸（整数）か `nil`。全部 `nil` の eqn は変換側が短絡してルールに来ない。
- **`vmap` は `grad` と同じ形の graph → graph 変換**として、新しい外側のトレースの中でルールを呼びながら書き直す。バッチされていない値だけを入力とする演算は、バッチ化ルールを呼ばずにそのまま残す（不要な複製をしない）。
- **複数出力の eqn は、個数ではなくプリミティブのフラグ `:multiple-outputs t` で切り替える**（`primitive` の `multiple-outputs-p`）。フラグがあるとき `abstract-eval` は aval のリスト、`eager` は配列のリスト、`emit` は名前と aval をリストで受けて `%8, %9 = ...` の左辺を自分で書く。`%trace-eqn` は単一出力のまま（複数出力に使うと `tracing-error`）で、`%trace-eqn*` が常にトレーサのリストを返す。inline / jvp / transpose / vmap は `%trace-eqn*` で書く。
- **複数出力の jvp ルールの規約**は、単一出力と別にした: `(rule primals tangents &rest params)` → `(values primal-outs tangent-outs)`（`%jvp-multiple-output-rule`）。単一出力のルールは変換側が先に作った出力を受け取る形だが、複数出力のルールは主値の eqn も自分で足す（先に足すと `while-loop` / `scan` / `cond*` のような高階プリミティブが2回走る eqn になるため）。
- **`%jvp-graph-core` に一本化した**: 第2値が本当の「出力の接線が非ゼロか」で、出力は `(or force out-nonzero)`。`while-loop` と `scan` の不動点†計算が同じ関数を使う。

### 3.2 サブグラフと closure conversion

- eqn の params の値に、閉じた `graph`（外側の var を参照せず、定数は自分の定数表に持つ）を持たせる。`eval-graph` / `inline-graph` / `dce-graph` と jvp / transpose の変換は、サブグラフの中身を書き換えずに素通しする。中身の変換は各プリミティブのルールの仕事。
- 本体は `with-tracing` が作った `traceable-function`。`(%trace-subgraph fn avals)` → `(values graph captured-outer-tracers)`。`%trace` に `parent` と `captured` を足し、`%resolve-tracer` は、今のトレースのトレーサならそのまま、祖先のトレーサなら新しい invar に持ち上げ（EQ でメモ化。明示的な invar の**後ろに**最初に使った順で足す）、それ以外は `tracing-error`。呼び出し側は captured を eqn の invars の末尾に足す。
- `%call-with-fresh-trace`（`grad` が使う）は `parent = nil` のまま。`grad` の「外側のトレーサを閉包で捕まえると `tracing-error`」の既知の制限は変えていない。`vmap` も同じ。
- StableHLO のリージョン: `(%stablehlo-region-lines graph &key arg-names)`。`arg-names` を渡すとブロック引数を出さず外側の SSA 名に結びつける（`stablehlo.if` の枝用）。渡さないと `^bb0(...)` を出す（`while` の cond / body 用）。リージョン内の SSA 名には `emit-stablehlo` ごとのカウンタから作る接頭辞 `%s<k>_` を付けて衝突を避ける。

### 3.3 `cond*` と `select`

- 公開名は `cond*`（CL の `cond` と衝突するため）、プリミティブ名は `:cond`。`(cond* pred then-fn else-fn &rest operands)`。
- **`with-tracing` の `if` は `select` のまま**にした。`if` の条件は要素ごとの `:i1` 配列でありうるので、要素ごとの意味を保つ `select` が要る。`cond*` は rank 0 の `:i1` の条件で片枝だけを実行する（実行時に選ばれなかった枝を評価しない）。使い分けは利用者が明示する。
- **両枝は同一の入力シグネチャを持つ**（JAX と同じ）: eqn の invars は `pred ++ operands ++ captures`（captures は両枝の捕捉値の和集合）で、両枝のサブグラフの invars はどちらも `operands ++ captures`（pred は枝の入力ではない）。片方の枝が使わない捕捉値の位置には使われない入力が置かれる（`%cond-unify-branches`）。当初は枝ごとに引数を持つ形だったが、`:num-operands` と `%cond-split-args` が要らなくなり、jvp / transpose / vmap のルールが単純になった（#130 のレビュー対応）。
- 条件がバッチされる `vmap` では、片枝だけを実行する性質は失われ、両枝を評価して `select` で選ぶ（JAX の `_cond_batching_rule` と同じ）。

### 3.4 carry† の受け渡し

- `while-loop` と `scan` の carry・xs・ys は**リスト**で受け渡す。`(while-loop cond-fn body-fn init-list)` → carry のリスト。`(scan f init-list xs-list &key length reverse)` → `(values carry-list ys-list)`で、`f` は `(carry-list x-list) → (values carry-list y-list)`。リストでないものは文書化したコンディションで拒否する。
- 理由: リストは既定の PyTree なので、フェーズ4で PyTree を入れたとき、リストを PyTree に一般化するだけで API を変えずに済む（§6）。

### 3.5 `while-loop`

- cond / body をサブグラフにし、cond / body が閉包で捕まえた値は、**素通しで返す追加の carry**にする（StableHLO の while は閉包を持てないが、リージョンは外側の SSA 値を直接参照できるため emit としては単純）。body の出力で捕捉値の位置が「入力そのまま」であることは EQ で検査する不変条件（`abstract-eval`）。
- jvp は前進モードのみ。接線を持つ carry の集合を**不動点まで広げる**（最初は接線がゼロの carry も、本体を通ると非ゼロになりうる。JAX の `_while_loop_jvp` と同じ）。
- **逆モード（`grad`）は対応しない**。反復回数がトレース時に分からないと、各反復の残差を保存できない（JAX の `while_loop` と同じ）。`grad` が通ると `:while-loop` を含む `autodiff-error`。固定回数で逆モードが要るときは `scan` を使う。
- vmap: 「バッチされる carry の集合」を不動点まで広げる。条件がバッチされると、どれかの要素の条件が真の間回し、条件が偽になった要素の carry は `select` で据え置く。

### 3.6 `scan`

- params は JAX と同じ形: `:num-consts :num-carry :length :reverse :body`。本体の入力は consts ++ carry ++ x_t、出力は carry ++ y_t。`f` が閉包で捕まえた外側の値は consts になる。**ホストの配列を閉包で捕まえたものは、本体のサブグラフの定数**（constants 表）になる。ループ不変なので consts と同じ扱いにでき、partial eval でも「ループ不変な残差」と認識される（§3.7）。
- StableHLO では `:i32` のカウンタを carry の先頭に足した `stablehlo.while` に落とし、x_t は `dynamic_slice`、y_t は `dynamic_update_slice` で読み書きする。長さ 0 は `while` を出さず素通し。
- jvp は JAX の `_scan_jvp` と同じ: 本体を `jvp-graph` した、主値と接線を一緒に回す1つの `scan`。接線を持つ carry の集合は不動点まで広げる。
- vmap: consts は元のバッチ軸のまま、バッチされる carry は軸 0、`xs` のバッチ軸は走査の軸（先頭）とぶつからないよう 1 に動かし、`ys` も 1 に出る。

### 3.7 `scan` の逆モード（partial eval と transpose）

- `linearize-graph` に、プリミティブごとの partial eval フック `*partial-eval-rules*`（`set-partial-eval-rule`、`src/ad/partial-eval.lisp`）を足した。jvp した graph を接線への依存で主値側と線形側に分ける前に、このフックで eqn を置き換える。`:scan` のルールは JAX の `_scan_partial_eval` に倣い、jvp した1つの scan を「主値と各ステップの残差を `ys` として積む scan」と「残差を `xs`、ループ不変な残差を `consts` として受ける、接線について線形な scan」に分ける。ループ不変な残差と、既知の `xs` / `ys` と同じ値の残差は積み直さず、外側の値を転送する。
- `cond*` はこのフックを使わず、**jvp ルールの中で**主値の `:cond`（残差を枝の出力として出す）と、接線について線形な `:cond` の2つの eqn に分ける。分岐の中は残差の選び方が枝で違うだけで、分割が eqn 1つで閉じるため。
- `:scan` の transpose は線形な scan を `reverse` を反転した scan にする（carry の余接線は carry、`xs` の余接線は `ys`、consts の余接線は carry に足し込む和）。carry が線形入力に依存しない scan（主値と接線が混ざった scan）の transpose は `autodiff-error`。

### 3.8 PRNG

- **`rng-bit-generator`（#133）**: `stablehlo.rng_bit_generator`（`THREE_FRY`）に対応する複数出力のプリミティブ。状態は `ui64[2]` で、`[0]` = 鍵（下位32ビット = key0、上位32ビット = key1）、`[1]` = カウンタ。新しい状態は鍵を保ち、カウンタを生成した64ビット単位の個数だけ進める。eager は IREE の lowering（`StableHLOToLinalgRandom`）を写した Threefry-2x32 で、**eager == IREE（local）== PJRT（XLA CPU）が23通りの形状 × `:u32` / `:u64` でビット単位で一致する**。XLA 自身の実装ではなく IREE の lowering を写したのは、IREE との一致を保証するため。
- **キー**: `:u32` の `(2)`。状態 `ui64[2]` への写像は `[k0 | k1<<32, カウンタ]`（`bitcast-convert` で2語を1語にする）。`uniform` / `normal` / `split` はカウンタ 0 から、`fold-in` はカウンタ 2^32 + data から始める（引く量が 2^32 未満なら重ならない）。ビットは常に1次元で作って reshape する（多次元の `:u32` の配置と状態の進みが形に依存するため）。
- **JAX とビット単位では一致しない**: JAX は `threefry_2x32` を直接呼ぶ（nabla は `rng_bit_generator` の上に組んでいる）。統計的性質（平均・分散・KS 検定・裾・相関）で検査し、JAX 0.10.2 の既知の答え（zero 状態と形 `(3 3 3)` の `:u32`）は `rng-bit-generator` のテストにある。
- `uniform` は仮数部トリック（f32 は23ビット、f64 は52ビット）。`normal` は erf の逆関数（Giles の近似。f32 は JAX の f32 と同じ単精度用の係数、f64 は XLA の f64 と同じ倍精度用の係数）。相対誤差の実測は f32 で最大 2.0e-7、f64 で最大 4.9e-16（#166 で f64 用の近似を足した。それまでは f64 でも単精度用を使い、裾で最大 12% 過小評価していた）。Box–Muller は sin / cos のプリミティブが無いので採用しなかった。
- **バッチされた rng の emit は while で回す**: プリミティブは状態の先頭にバッチ次元を許す（`ui64[..., 2]`、各行は単独呼び出しとビット一致）が、StableHLO は `ui64[2]` しか受けないので、emit は `stablehlo.while` で1行ずつ `dynamic_slice` → `rng_bit_generator` → `dynamic_update_slice` を回す（#164）。当初（#136）は行ごとに slice → `rng_bit_generator` → concatenate に展開しており、コンパイル時間が行数で急に増えた（§4.2）。

### 3.9 整数 dtype

- `:i32 :u32 :u64`。THREE_FRY の状態が `ui64[2]` のため `:u64` が要る。整数は既存の `*dtypes*`（PBT の浮動小数点用）に入れず、`*integer-dtypes*` を別に作る（整数が exp / log の PBT に流れ込まないように）。
- 整数の接線は常に symbolic zero。`div` `exp` `log` `tanh` `dot-general` は整数を `primitive-error` で拒否する（暗黙の型昇格もしない）。
- **float → int の `convert` は、飽和と NaN → 0 を StableHLO 側で明示する**（clamp と select。IREE の `fptosi` は範囲外・NaN が未定義で eager と食い違った）。**整数 → `:bf16` は f32 を経由し、間に `optimization_barrier`† を置く**（IREE CPU が `__truncsfbf2` のリンクに失敗する。barrier が無いと2つの convert が最適化で畳まれて同じ失敗になる）。

### 3.10 PRNG の API

公開 API は `prng-key`（整数のシードから `[上位32ビット 下位32ビット]`）、`split`、`fold-in`、`uniform`、`normal`、`prng-error`。eager・`jit`・`vmap` で動く。キーは「使う（`uniform` / `normal`）か `split` するか」のどちらか一方にだけ使う。詳細は §3.8。

### 3.11 `do` ループの `scan` への展開（#137）

`with-tracing` の最初の処理（`macroexpand-all` より前、展開前のフォームに対して）で、**定型の `do` だけ**を `scan` にする（`src/loop-scan.lisp` の表）。`do` は展開されると `block` / `tagbody` / `setq` になって元の形が分からなくなるため。対応する形は、カウンタの step が `(1+ i)` / `(+ i 1)` / `(+ 1 i)`、終了条件が `(>= i N)` か `(= i N)`、本体のフォームが無いもの（宣言だけ可）。carry の step は純粋な式（`setq` は書けない）で、step の無い変数は不変。カウンタは `:i32` の carry で、上限 `N` は1回だけ評価する（トレース時に決まる整数）。反復回数は `(>= i N)` なら `max(0, N - INIT)`、`(= i N)` で `N < INIT` なら `scan-length-error`。`dotimes` / `loop` / `do*` は従来どおり `unsupported-form`（`dotimes` は carry を `setq` でしか渡せず、`loop` の `for ... = ... then ...` は更新と終了判定の順序が `do` と違うため）。`quote` / バッククォートの中と、`flet` / `labels` / `macrolet` の定義リストは見ない（呼び出し側が `do` の構文に読めると `do` 扱いになる制限は README に記載）。

### 3.12 IREE 3.11 のバグと回避策

フェーズ3で IREE 3.11（固定コミット）のコンパイラのバグを3つ踏んだ。

1. **定数で初期化された carry を持つ `stablehlo.while` で SIGSEGV / SIGBUS**（`ScheduleAllocationPass` の Stream AffinityAnalysis、`walkTransitiveUses`）。cond を駆動する carry の初期値が `stablehlo.constant` で、ほかに carry が2つ以上あり、少なくとも1つが rank 1 以上のとき、非決定的に落ちる（単体の `iree-compile` で10回中6回）。**回避策**: while のオペランドのうち定数のもの（`%while-barrier-lines`、`*stablehlo-constant-names*`）と、`scan` のカウンタ・ys の0初期値を、while の前の `stablehlo.optimization_barrier` に通す。**ルール**: 定数の carry を持ちうる while / scan を emit するコードは、必ず定数オペランドを `optimization_barrier` に通す。長さ 1 の `scan` は IREE が `scf.for` に変換し、barrier があると型不一致でコンパイルに失敗するので barrier を付けない（長さ 1 ではクラッシュしない）。守るテストは、定数の carry を持つ生の StableHLO が今も落ちることを子プロセスで確かめる medium テスト（直れば失敗するので、回避策ごと消す）と、barrier 付きを10回続けてコンパイルするテスト。PR #157 の CI で見つかり、それ以前の「負荷によるメモリフォルトのフレーク」もこのバグだった。
2. **`:i1` の carry が比較由来のとき、その while の結果が戻り値になるとコンパイラがプロセスごと落ちる**（LLVM の out of memory / メモリフォルト）。`:i32` に通す・barrier・`select` のどれでも直らない。戻り値にしない場合（継続判定にだけ使う）は動く。**回避策は無い**。docstring・`docs/stablehlo-ops.md` に制限を書き、子プロセスのテストで IREE に残っていることを守る（#131）。eager と PJRT は影響を受けない。
3. **整数 → bf16 の `convert` のリンク失敗**と、**float → int の範囲外**の挙動差（§3.9）。

## 4. 測定

#### 4.1 `scan` の `ys`（#159、IREE local、f32）

（#159 で解決）IREE 3.11 はループの carry を本体で使うたびに丸ごとコピーしていたため、ys を持つ scan は長さに対して2乗で遅かった（n=1000, w=1024 で ys あり 1638 ms、ys なし 31 ms。n=4000 で約 39.7 秒）。ys のバッファを本体の先頭で `optimization_barrier` に通す回避で、n=4000 でも ys なしと同じ桁になった。原因と実測は `docs/stablehlo-ops.md` の scan の節。PJRT（XLA CPU）では未測定。

#### 4.2 バッチされた rng のコンパイル時間（#136）

IREE local、バッチされた rng の eqn 1つ（bits は `(4)` の `:u32`）のコンパイル時間。行ごとの展開（#136）: 32行 4.0 秒 / 64行 6.9 秒 / 256行 42.8 秒（MLIR 179 KB。同じ機械で測り直すと 3.9 / 9.5 / 35.2 秒）。while 化（#164）後: 0.79 / 0.80 / 0.85 秒（MLIR 約 2.6 KB、行数に依らない）。代わりに IREE の実行時間は増えた: `vmap` した `uniform` を jit して 1回あたり、256 キー × 4 要素で 0.4 → 7.2 ms、256 キー × 1024 要素で 3.2 → 56.8 ms、64 キー × 16384 要素で 32.8 → 57.6 ms（#164 の PR の時点の測定。ループ1回ごとの起動と、`dynamic_update_slice` がビットのバッファ全体をコピーするため。§4.1 と同じ #159）。PJRT（XLA CPU）は同じ3通りで 0.4 → 0.0（タイマーの分解能以下）/ 4.0 → 4.0 / 35.6 → 24.0 ms で、遅くならない。

その後、ビットのバッファを `scan` の ys と同じく `optimization_barrier` に通して in-place に書くようにした（#172 のレビュー。初期値も `%scan-ys-init-lines` で作る）。同じ機械・同じ方法（10回の平均の3回のうちの最短、`NABLA_CACHE_DIR` は空の一時ディレクトリ）で測り直すと、バッファ全体をコピーしていたとき 40 / 99 / 72 ms が、in-place で 40 / 46 / 20 ms になった（行ごとに展開していた main は 0.8 / 2.8 / 8.0 ms）。256 キーで要素数を 4 から 1024 に増やしたときの差は 59 ms から 4〜7 ms に減り、ビットの量に比例する費用はほぼ消えた。残りは小さなバッファでも変わらない**ループ1回ごとの起動の費用**（この機械では1行あたり約 0.16 ms）で、行数に比例する。#164 の PR の時点より起動の費用が大きい機械だったため、256 キー × 1024 要素の絶対値（46 ms）は目標の 15 ms に届いていない。

#### 4.3 RNN の学習（#141、IREE local）

誤差の実測: JAX の f32 と f64 の損失の相対誤差は 30 ステップ目で 3.3e-7、nabla（IREE）と JAX は最大 6.8e-7、eager と IREE は最大 8.5e-7。許容は 1ステップ目と5ステップ後が既定の f32（rtol 1e-5 / atol 1e-6）、10ステップ目以降は rtol 3e-5（学習率の 1e-4 の誤差は 4.7e-5、勾配の 1e-3 の誤差は 7.9e-5 以上で検出できる）。medium の新しい3テストは合計約 5.8 秒。

#### 4.4 mutation testing

`def-batch-rule` を変異対象に加え（PR #151）、ルールを足した PR ごとに実行した。主な結果: #130 は 41 中 34 kill（生存は defun 先頭行の同値のみ）、#131 は 48/48、#135 は 16/16、#133 は `--max-per-def 15` で 58/58、#126 は 255 中 234 kill（生存は SSA 補助名の等価変異）、#136 は 82/91（生存9件は等価か別の有効値）。
#137 は初回が 47/49（43 kill、4 timeout、生存2件）で、生存は `(signed-byte 32)` → 31 の境界だった（カウンタが 2^31-1 まで届くテストで kill）。レビュー対応後は 55/56 で、生存は不完全な定義のガード1件（テストを足したが mutation は再実行していない）。

## 5. 既知の制限と積み残し

- 以下のうち、#164 / #165 / #166 は #142 の時点で起こしたフォローアップの issue。
- **#159: `scan` の `ys` が IREE で O(length × |ys|)**（§4.1）。解決済み（ys のバッファを `optimization_barrier` に通す）。
- **バッチされた rng の emit は while 化済み（#164）**。コンパイル時間は行数に依らなくなった。ビットのバッファは in-place に書くので、IREE の実行時間に残るのはループ1回ごとの起動の費用で、行数に比例する（§4.2）。
- **`while-loop` の逆モード（`grad`）は対応しない**（§3.5）。`scan` で書く。
- **IREE 3.11 の `:i1` carry のバグは回避策が無い**（§3.12 の 2。比較由来の `:i1` の carry を持つ while の結果を jit の戻り値にしない）。
- f64 の `normal` の裾が f32 並みの精度だった件は、#166 で倍精度用の erf の逆関数の近似を足して解消した（§3.8）。
- ~~**PJRT では `:i1` が未対応のまま**~~ → #166 (b) で対応した（`:i1` を `PRED` に写し、ホストの BIT 配列と1要素1バイトの間を IREE と同じく詰め直す）。`:i1` の往復、jit の `:i1` の入出力、`cond*` の `pred` を引数で渡す形と比較由来の `:i1` の carry を返す `while-loop`（IREE 3.11 では落ちる形）を `tests/pjrt/i1-test.lisp` が PJRT と eager の一致で確かめる。
- **PRNG は JAX とビット単位で一致しない**（§3.8）。`rng_bit_generator` の `:i32` のビットと Philox / `DEFAULT` は未対応。
- **`vmap` / `grad` の「外側のトレーサを閉包で捕まえると `tracing-error`」は変わらない**（`cond*` / `while-loop` / `scan` の本体は closure conversion で捕まえられるが、`grad` / `vmap` の `f` は不可）。
- **`vmap` / `grad` は `f` がリストを返せない**（`(with-tracing ... (values-list ...))` で包む。フェーズ4の PyTree で解消する見込み）。
- **`(vmap f)` / `(grad f)` は呼ぶたびに新しい関数オブジェクト**を作り、ループの中で `(jit (vmap f))` を作ると毎回コンパイルされる（jit キャッシュのキーが関数の同一性のため）。
- **`while-loop` の捕捉値の重複は取り除いた**（#166 の d。2026-10-04）。cond と body が同じトレーサを捕まえても、オペランド（StableHLO の while の carry）には1回だけ現れる（`cond*` と同じく、両方のサブグラフの invars を「carry、捕捉値の和集合」に揃える）。
- **#166 にまとめた小さな積み残し**: f64 `normal` の裾（解消済み）、PJRT の `:i1`、`while-loop` の捕捉値の dedupe、eager の `scan` の EQ な carry、`scan` の bf16 生配列、IREE の `scan` テストの追加、`vmap` の多数決の軸。PJRT の `:i1` と dedupe（解消済み）は上の項目のとおり。
- **GPU が無く未測定**: #12 と CUDA の計測（フェーズ2から引き続き）。
- **#73**: フェーズ1から引き続き（IREE 上流への報告。#68 の in-process コンパイラのメモリ破壊は閉じている）。
- **#165**: 上の IREE 3.11 の while のバグ2つ（§3.12 の 1 と 2）を上流へ報告し、回避策を消す時期を決める。

## 6. 進め方の知見（プロセス）

- **stacked PR と squash merge の相性**: 親が squash merge されると、子ブランチには親の元のコミットが残り、`main` には内容の同じ別のコミットができる。子を rebase すると履歴を書き換えて force push が要り、CI が走らなくなることがあった。**`git merge origin/main`（merge commit）で取り込む**ことで、force push を避けた。rebase してよいのは一度も push していないときだけで、push するときは先に PR の base を `main` に変えてから force push する（逆順だと GitHub が CI を起動しない）。
- **CI は、`main` と衝突している PR では `pull_request` のワークフローが起動しない**。push 前に `git merge-tree --write-tree origin/main HEAD` で衝突を確かめる。
- **複数の Driver が4コアのマシンを共有する**: テストと mutation testing は別々のロック（`/tmp/nabla-tests.lock`、`/tmp/nabla-mutate.lock`）の中で実行した。
- **`defprimitive` の `:emit` などに `#'fn` を書くと mutation testing が変異させられない**（変異の対象になる本体の式が `defprimitive` の中に無いため）。本体は `defprimitive` の中に `lambda` で書くか、`def-*-rule` を使う。
- **core の docstring に実行系の名前（`iree` / `pjrt`）を書くと、medium の `core-sources` テスト（`do-not-mention-iree`）が落ちる**。IREE 固有の制限は `docs/stablehlo-ops.md` に書く。
- アンカーコメント（#143）で、複数 Driver が共有ファイル（`package.lisp`・`nabla.asd`・README・用語集）を並行して書き換えても衝突しにくくした（役目を終えたので #142 で撤去）。

## 7. フェーズ4（Flax 相当）への引き継ぎ

- **PyTree を入れると、制御構造と `vmap` の API は「リスト」から「PyTree」に一般化される**。`while-loop` / `scan` の carry・xs・ys、`cond*` の operands と枝の戻り値、`vmap` の引数・戻り値・`in-axes` / `out-axes`（今は整数か `nil`、またはリスト）は、PyTree のフラット化・復元（`flatten` / `unflatten`）を挟んで、内部ではこれまでどおりフラットなリストを扱う。リストを既定の PyTree にしてあるので、今の呼び出しはそのまま動く（C4）。新しく必要になるのは、`defmodule` の構造体を carry や `vmap` の引数にできること、`vmap` / `grad` の出力にリストを返せるようにすること（§5）、`in-axes` を PyTree と同じ構造で指定できること。`scan` の `ys` が PyTree になると、`ys` の積み方（§5 の #159）の最適化を PyTree の葉ごとに考える必要がある。
- **`defmodule` の init / dropout は PRNG のキーを使う**。`split` と `fold-in` でパラメータごとのキーを作り、`uniform` / `normal` で初期化する。キーは `:u32` の `(2)` の配列で PyTree の葉にできる。`vmap` でキーをバッチして独立な乱数を作る使い方は動く（ただしバッチされた rng は行数が多いとコンパイルが重い。§5）。dropout のように `jit` の中で毎ステップキーを更新するときは、キーを引数に取って新しいキーを返す形にする（`jit` のキャッシュに乗せるため、キーは動的な引数）。
- **`scan` で層の積み重ね（`nn.scan` 相当）やループを書ける**。本体のパラメータは consts（ループ不変）か xs（層ごとに別）で渡す。長い系列の `ys` に注意（§5）。
- **バッチ化ルール・partial eval の拡張点**: 新しいプリミティブには `defprimitive` の `:batch` か `def-batch-rule` を書く（`src/ad/rules-batch-*.lisp`）。主値と接線を分ける特別な処理が要る高階プリミティブは `set-partial-eval-rule` を使う。
- **`optimization_barrier` のルール**: 定数の carry を持ちうる `while` を emit するプリミティブを足すときは、定数オペランドを `%while-barrier-lines` に通す（§3.12）。
- **per-example 勾配**（`vmap` の `in-axes` で、パラメータをバッチしない）は、PyTree のパラメータでは `in-axes` の指定が PyTree になる。
- **CUDA の未測定**: GPU のある環境で `NABLA_TEST_SIZES=large NABLA_REQUIRE_CUDA=1 scripts/run-tests.sh`（#12）と `scripts/bench-backends.sh --cuda` を実行する。
