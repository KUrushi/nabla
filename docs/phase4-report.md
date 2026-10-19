# フェーズ4 報告書: PyTree と Flax 相当のモデル記述層

フェーズ4（親 issue #194、「PyTree を入れて `jit` / `grad` / `vmap` / 制御構造の引数・戻り値を木に一般化し、その上に `defmodule`・レイヤー・Optax 相当のオプティマイザ・safetensors のチェックポイントを積み、小さな Transformer を学習する」）で作ったもの・得た知見をまとめ、フェーズ5（Grain 相当。データ供給）への引き継ぎ材料にする。形式は [phase3-report.md](phase3-report.md) に合わせてある。

## 1. 範囲と結論

#194 の完了条件に対する達成状況:

| 完了条件 | 判定 | 検証方法 |
| --- | --- | --- |
| 小さな Transformer（decoder-only、2層）を `defmodule` で書き、`jit` した学習ステップ（`value-and-grad` + Adam）で学習でき、最初の数ステップの損失とパラメータが Flax + Optax のフィクスチャと IREE `local` で数値一致し、損失が減る（#209） | 達成 | PR #234。`tests/nn/transformer-test.lisp` の medium が Flax / Optax と一致（損失の最大差 3.6e-6、パラメータ 1.2e-6 以下。`attn.key.bias` だけ別の基準。§4.1）、100 ステップで損失が 2.6 から 0.065 まで下がり（最初の 5% 未満）、同じ条件の再実行とチェックポイントからの再開がビット単位で一致する |
| `defmodule` で定義したモデル（入れ子のサブモジュールを含む）を、PyTree として `jit` / `grad` / `vmap` / `scan` にそのまま渡せる（#199〜#202） | 達成 | `jit` / `grad`: PR #218（`tests/pytree-function-test.lisp`、`tests/nn/module-transform-test.lisp`。`(jit (grad loss))` が同じ型のモジュールを返す）。`vmap`: PR #224（`tests/vmap-pytree-test.lisp`、`tests/nn/module-vmap-test.lisp`）。`scan`: PR #231（`tests/pytree-control-test.lisp`、`tests/nn/module-control-test.lisp`。`defmodule` の RNN の状態を carry にした scan の eager・`grad`・IREE） |
| レイヤー（dense / conv2d / layer-norm / embedding / multi-head-attention / dropout / sequential）の順伝播と勾配が Flax のフィクスチャと数値一致する（#204、#207、#208） | 達成（conv は PR #232 に依存） | dense / layer-norm / embedding / dropout / sequential: PR #222（Flax 0.12.8 のフィクスチャ 11 ケース。eager と IREE の jit）。conv: PR #232（10 ケース。名前は N 次元の `conv`。§3.9）。multi-head-attention: PR #230（7 ケース。`dense-general` は PR #229、`dot-general` の export は PR #228） |
| オプティマイザ（sgd / adam / adamw、chain、clip-by-global-norm、学習率スケジュール）が Optax のフィクスチャと数値一致する（#205） | 達成 | PR #225。`tests/nn/optimizer-test.lisp`（Optax 0.2.8 で 11 ケースを 6 ステップ、4 つのスケジュール）。`tests/nn/optimizer-train-test.lisp` の medium（jit した学習ステップが eager と一致、2 ステップ目から再コンパイルしない、チェックポイントの往復） |
| モデルとオプティマイザ状態を safetensors に保存・復元でき、Python の `safetensors` が書いたファイルを読める（#198、#206） | 達成 | PR #213（`tests/nn/safetensors-test.lisp`。Python 0.4.5 が書いたファイルを読める）、PR #219（`tests/nn/checkpoint-test.lisp`。全 8 dtype のランダムな木の往復がビット単位で一致、Flax の `flatten_dict` 形式の名前で Python が書いたファイルを復元して順伝播が一致）。モデルとオプティマイザの状態を 1 ファイルに入れた往復は PR #225 と PR #234 のテスト |
| README にチュートリアル（モデル定義から学習・保存まで）がある（#210） | 達成（PR #238 に依存） | PR #238。`examples/tutorial.lisp` と README の節。`tests/nn/tutorial-test.lisp` が (1) README のコードと例のファイルの一致、(2) 損失が下がること、(3) 保存・復元した続きが止めずに続けた学習とビット単位で一致すること、を確かめる |
| 既定のテストスイート（small + medium）が CI で通る | 達成 | 各 PR で CI（`NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_PJRT=1`）が通った。この報告書のブランチの最終実行は §4.5 |

結論: 完了条件はすべて達成した。目安は 1〜2ヶ月だったが、子 issue 17 本（#195〜#211）は、PR #212〜#236 のうちの 19 本と、#232（conv）、#238（チュートリアル）、この報告書の PR で、2026-10-19 の1日のうちに、stacked PR と Advisor の検証（§6）を使って進めた。IREE 3.11 のバグを 4 件踏み、回避策と最小の再現を入れた（§4.3）。小さな積み残しは §5 のフォローアップにした。

## 2. 作ったもの

PR 番号は GitHub 上のもの。stacked PR で、下から順に積み、親が squash merge されるたびに子の base を `main` に付け替えた。mutation testing は変更した行にかけ、PR の説明にある殺した数 / 変異体の数を併記する（生き残りは等価変異か、エラーメッセージのラベル文字列が大半）。

### 2.1 core（PyTree と変換）

- **#195（PR #212）**: PyTree の土台（`src/pytree.lisp`）。`defpytree` / `tree-flatten` / `tree-unflatten` / `tree-leaves` / `tree-structure` / `tree-map` / `treedef=`、`pytree-error`。mutation 36/40。
- **#196（PR #214）**: `sqrt` / `rsqrt` プリミティブ（jvp・バッチ化つき）。JAX のフィクスチャ 4 dtype × 159 行（±0・±inf・NaN を含む）。26/26。
- **#197（PR #215）**: `convolution` プリミティブの順方向（channels-last 固定、1D〜3D、strides / padding / 両方の dilation / feature group）。公開 API は `conv-general-dilated`。JAX のフィクスチャ 8 ケース。104/104。
- **#199（PR #218）**: `jit` / `grad` / `value-and-grad` の引数と戻り値を PyTree に。jit のキャッシュのキーに treedef を入れた（§3.1）。`iree-too-many-inputs`（§4.3）。51/55。
- **#203（PR #217）**: `convolution` の jvp / transpose / バッチ化ルール、内部プリミティブ `:rev`、`:batch-group-count`。JAX の勾配のフィクスチャ 10 ケース。bf16 / f16 の回避策（§4.3）。280 中 269 から、再実行で 162/163（最後の1体は到達しないガードで、ガードごと削除）。
- **#200（PR #224）**: `vmap` の引数・戻り値・`in-axes` / `out-axes` を PyTree に（§3.2）。論理を変える変異体はすべて殺した。
- **#201（PR #231）**: `cond*` / `while-loop` / `scan` を PyTree に（§3.3）。24 件中 23 件を殺し、タイムアウトの1件はテストを直して検出できるようにした。再実行で 31/35（残りはエラーメッセージのラベル）。
- **#208 の core 側（PR #228）**: `dot-general`（バッチ軸・縮約軸を明示する）を公開した。
- **#204 の core 側（PR #221）**: `truncated-normal`（JAX と同じ方法。範囲は ±5 に限る）と `value-aval`（トレーサ・配列・device-array の aval を返す）。53/56。

### 2.2 nabla/nn（モデル記述の層）

- **#198（PR #213）**: `nabla/nn` システムの新設と safetensors の読み書き（`src/nn/safetensors.lisp`、ヘッダの JSON は自前の最小の読み書き `src/nn/json.lisp`）。193/207 と 111/132（生き残りは同値か到達できない変異）。
- **#202（PR #216、完了条件の「モデルを jit / grad に渡す」は PR #218 のテストで満たした）**: `defmodule`（`src/nn/module.lisp`。§3.4）。34/34。
- **#206（PR #219）**: PyTree のチェックポイント（`src/nn/checkpoint.lisp`。§3.5）。44/44。
- **#204（PR #222）**: `dense` / `layer-norm` / `embedding` / `dropout` / `sequential` / `activation` と `relu` / `gelu` / `softmax` / `log-softmax`（`src/nn/layers.lisp`。§3.6）。78/79。
- **#205（PR #225）**: オプティマイザ（`src/nn/optimizer.lisp`。§3.7）。255 件中 231 件以上。
- **#208（PR #229、#230）**: `dense-general` とマスクの部品（`make-causal-mask` / `make-attention-mask` / `combine-masks`）、`multi-head-attention`（§3.8）。3 ファイル全体で 119 件中 117 件（残りは等価変異）。
- **#209（PR #234）**: 小さな Transformer の例（`examples/transformer.lisp`）と Flax / Optax との一致テスト。`src/` は変えていない。続いて PR #235 が、#234 の CI で見つかった `clip-by-global-norm` の PBT の誤り（期待する係数が 1 を超えうる）を直した。実装の誤りではなく性質の書き方の誤り。
- **#207（PR #232）**: `conv`（Flax の `nn.Conv`。N 次元。§3.9）。58/58。
- **#210（PR #238）**: README のチュートリアルと `examples/tutorial.lisp`。README の「現状」と ロードマップの表も更新した。あわせて、Advisor が見つけた「FiveAM の root suite に落ちて一度も走っていないテスト」を防ぐ `suite/every-loaded-test-belongs-to-a-nabla-suite`（`tests/support-test.lisp`）を足した（§6）。

### 2.3 IREE の回避策とテストの修正（#220）

- **#220（PR #236）**: 入れ子の scan のテストが、ときどき落ちていた。原因は許容誤差ではなく IREE 3.11 の local-task の実行時の競合で、本体に制御構造を含む scan では ys の barrier を外した（§4.3）。check-it の回帰 id の重複とファイルの共有の修正、`regressions/ids-are-unique` も含む。mutation は `src/scan.lisp:243-330` で 12/19（この PR に関係する変異体はすべて殺し、生き残りは既存の行。#237）。

## 3. 設計上の判断

### 3.1 jit のキャッシュのキーと treedef（#199）

- キーは「関数の同一性 + **動的引数の葉の `aval`** + **動的引数の treedef**（`treedef-spec` を EQUAL で比べる）+ 静的引数 + backend」。同じ aval の葉でも木の形（リストの長さ、`defmodule` のクラスや `:static` スロットの値）が違えば別のコンパイルになる。**戻り値の treedef はキーに入れず、キャッシュのエントリに持つ**（同じ入力の形からは同じ形の戻り値ができるため）。登録した型の補助データを EQUAL で比べられる値に限ったのは、この比較のため。
- Lisp の実数の葉は rank 0 の配列にする（`double-float` は `:f64`、それ以外は `:f32`）。配列・device array・実数以外の葉は `jit-error`。
- 要素型 `t` の `simple-vector` は配列ではなく PyTree の節になった（以前も配列としては `dtype-mismatch` だったので壊れる呼び出し元は無い）。型特化した配列と文字列は葉のまま（#195 で決めた）。
- `grad` は、微分する引数ごとにその引数の treedef で勾配を組み立て直す。多値の戻り値は「値のリスト」という 1 つの木として平坦化する。木を受けて返す関数を「葉だけを受けて返す関数」に変える内部の部品 `%flatten-arguments` / `%flatten-function` / `%unflatten-results` を、`vmap` と制御構造も使い回す。

### 3.2 `vmap` の軸は木の接頭辞（#200）

- `in-axes` / `out-axes` は **木の接頭辞**。整数か `nil` を部分木の全体に配って葉ごとの軸にする（`%vmap-broadcast-axes`。JAX の `flatten_axes`）。一番外側のリストは引数ごと・戻り値ごとの指定なので、フェーズ3までの呼び出し（`'(nil nil 0 0)`）はそのまま動く。1 つのリストを返す `f` の軸は `'((a b))` か整数 1 つで指定する。
- **軸の木では `nil` は葉**（バッチしない）で、空の部分木ではない。PyTree 本体では `nil` は空の部分木だが、軸の木ではこの意味にしないと「バッチしない」を書けない。曖昧さが無いことを Advisor が 16 個の反例で確かめた。
- モデルをバッチしない・データをバッチする per-example 勾配は `(vmap (grad loss) :in-axes '(nil 0 0))` で書ける。結果は同じクラスのモジュールで、各スロットが軸 0 に積まれる。
- 木の形が合わないエラー（`vmap-error`）は、treedef が呼ぶまで分からないので関数を呼んだ時点で、引数の個数や軸の型の誤りは `vmap` を呼んだ時点で出る。軸の木はコピーして持つ。

### 3.3 制御構造の PyTree（#201）

- `cond*` / `while-loop` / `scan` は、木を `tree-flatten` で平らにして**既存のフラットな経路をそのまま通し**、結果を `tree-unflatten` で組み立て直す。プリミティブ・eqn の形・jvp / transpose / partial eval / バッチ化のルール・IREE の回避策・`loop-scan.lisp` は変えていない。フェーズ3で「carry・xs・ys・operands はリスト」にしておいた効果で、リストは既定の節なので既存の呼び出しがそのまま動いた。
- 本体の戻り値の treedef が init と違えば `while-loop-carry-mismatch` / `scan-carry-mismatch`、`cond*` の両枝の戻り値の treedef が違えば `cond-error`。メッセージに両方の形を出す。木の不正（`pytree-error`）は各構文のエラーに変えて報告するが、**枝や本体の中で起きた `pytree-error` はそのまま伝える**（戻り値の flatten だけを狭い `handler-case` で包む。Advisor の指摘）。
- `scan` の `ys` の木の形は本体が返した y の treedef に従う。`defmodule` の RNN の状態を carry にして `grad` が通る。

### 3.4 `defmodule` のスロットの種類（#202）

- スロットは `:param`（省略時）/ `:state`（学習しない状態）/ `:static`（層の幅などのハイパーパラメータ。**葉でなく treedef の補助データ**。EQUAL で比べられる値に限る）。子の葉は `:param` / `:state` の宣言順。構造体は定義時に `defpytree` へ登録する。`:state` に勾配を流さないために `map-state`（`stop-gradient` 用）を持つ。継承・述語・copier は生成しない。
- **スロットは MOP で列挙せず、`defmodule` が定義時に持つ表（`*defined-modules*`、`module-slots` で引ける）から取る**。#194 の設計の「MOP でスロットを列挙」からの変更で、`defmodule` は自分でスロットをすべて知っているため。チェックポイントのパス名（§3.5）が宣言順の表を使う。
- 呼び出しの名前は総称関数 `call`（`apply` は CL の関数で再定義できず、Flax の `apply` とも意味が違う）。**`:call` の本体は `with-tracing` の本体に展開される**ので、配列にもトレーサにも使え、`grad` / `vmap` の中で呼べる。引数は必須のシンボルだけ（dropout のキーや `deterministic` も位置引数で渡す）。1 回目の検証で「`:call` の本体がトレースされず、トレーサの計算ができない」が blocking の指摘として見つかった。
- 初期化は `init-<name>`。PRNG キーを `split-keys` で分けてサブモジュールに渡す。ロジックは `defmodule` の外の `defun` に置く（mutation testing の runner が `defmodule` の中を変異させられないため）。

### 3.5 チェックポイントのパス名とその版（#206）

- 葉を `"."` でつないだパス名で safetensors に保存し、ひな形（配列・`device-array`・`aval`）に名前で対応づけて復元する。リスト・ベクタの子は添字、`defmodule` の子はスロット名（Lisp が大文字にする名前だけ小文字にし、`|Dense_0|` のように大文字小文字を保った名前は保つ。Flax の `flatten_dict` の `params.Dense_0.kernel` と一致する）、`defmodule` 以外の `defpytree` 型は `flatten` が返す子の添字で入れ子も辿る。名前の重複・空のスロット名・順序の食い違いは `checkpoint-error`。
- **このパス名の決め方はファイル形式の一部**で、メタデータ `nabla.checkpoint_version = "1"` に結びつく。`"nabla."` 接頭辞のキーは予約。新しい版のファイルは `:unsupported-version`、版の無いファイル（Python が書いたもの）は読める。**chain の状態のパス名は位置で決まる**ので、chain に変換を挿すと古いチェックポイントは読めなくなる（README に書いた）。
- 復元は名前・形・dtype を厳密に照合する（`:missing-entries` / `:unexpected-entries` / `:shape-mismatch` / `:dtype-mismatch`）。`(unsigned-byte 16)` の葉の bf16 / f16 は、保存でも復元でも `:dtypes` か `device-array` / `aval` の dtype で決める（f16 のファイルを bf16 のモデルに黙って読み込まないため）。
- モデルとオプティマイザ状態は 1 つの木（`defmodule`）にまとめて 1 ファイルに入れる。プレフィックスの引数は持たない。

### 3.6 レイヤー（#204）

- 名前・形・初期化は Flax Linen の既定に合わせた。**スロット名がチェックポイントのパス名になる**ので Flax の名前（kernel / bias / scale / embedding）を使う。`dense` の kernel は `lecun_normal`（`truncated-normal` × 1/(0.8796·√in)）、`layer-norm` は epsilon 1e-6 と「速い分散」（桁落ちするので Flax のフィクスチャとの許容誤差だけ rtol 1e-4 にした）。
- **embedding は gather でなく one-hot と `dot` で引く**。gather は jvp・scatter-add の transpose・バッチ化ルールまで要り、#204 の範囲を超えるため。計算量とメモリは id 数 × vocab × features に比例するので、語彙の大きなモデルには gather が要る（#240）。範囲外の id は 0 の行になる（Flax は NaN）。
- **dropout は反転 dropout で、マスクは常に f32 の乱数で作る**。bf16 / f16 のモデルでも比較が f32 で行われ、f32 と f64 で同じマスクになる。1 回目の検証で、bf16 の dropout が落ちる件が見つかった。`rate = 1` は inf / NaN の入力でもちょうど 0 を返し、勾配も 0。`deterministic` は Lisp の真偽値で、jit では静的引数。
- 活性化関数 `relu` / `gelu`（tanh 近似）/ `softmax` / `log-softmax` は `nabla.nn` の関数で、core のプリミティブにはしない。
- `sequential` は引数 1 つの層だけを受け付ける（Flax と同じ）。`init-sequential` はキーを層ごとに分ける。

### 3.7 オプティマイザ（#205）

- Optax と同じ「`init` と `update` の組」（`optimizer` 構造体、`defoptimizer`、`chain`）。`update` は純粋関数で、学習ステップ全体を `jit` できる。組み込みは `sgd` / `adam` / `adamw` / `chain` / `clip-by-global-norm` / `scale` / `scale-by-adam` / `scale-by-schedule` / `momentum-trace`（`cl:trace` と衝突するので改名）/ `add-decayed-weights` と 4 つの学習率スケジュール。`chain` は `check-it:chain` と衝突するので、両方を `:use` するパッケージには `(:shadowing-import-from #:nabla.nn #:chain)` が要る。
- **ステップ数 `count` は状態の中の rank 0 の `:i32` の配列**（Optax の `safe_increment` と同じく飽和する）。Lisp の整数にすると jit がステップごとに再コンパイルする。状態の型は `defmodule`（`adam-state` など）で、`count` / `mu` / `nu` / `trace` はすべて `:state`。葉の順は Optax の `tree_leaves` と同じ。
- core に `cos` / `expm1` が無いので、cosine decay とバイアス補正 `1 - b^t` は級数で計算した（格子上の直接のテストで守る）。プリミティブを足して置き換えるのは #223。
- `clip-by-global-norm` はノルムに `where` を 2 重にかける（JAX の書き方）。勾配がすべて 0 でも、微分した値に NaN が出ない。検証で、docstring が「NaN が出ない」と事実と違うことを書いていたのが見つかり、直した。スカラーは葉の dtype（`value-aval`）に合わせる（f32 / f64）。

### 3.8 multi-head attention（#208）

- **PR を 3 本に分けた**（#228 core の `dot-general`、#229 `dense-general` とマスク、#230 attention）。差分が大きすぎるという Advisor の指摘による。
- スロットは Flax と同じ `query` / `key` / `value` / `out`（`dense-general`）で、チェックポイントのパス名は Flax の `flatten_dict` と一致する。`dense-general` が縮約する入力の軸の数は `:static` の `contract-axes` に持つ（bias の rank の差から推論するのをやめた。Advisor の指摘）。
- 先頭に任意個のバッチ軸を置ける。スコアは先頭のバッチ軸とヘッドをバッチ軸にした `dot-general`。q は `sqrt(head_dim)` で割ってから dot を取る。隠す位置は select で dtype の有限の最小値にして、max を引く softmax を使う。全部隠した行は一様な重みになり NaN は出ない。bf16 / f16 でマスクを付けると落ちる件が検証で見つかり、直した。
- **dropout は要素ごとに独立なマスク**（Flax の `broadcast_dropout=False` 相当）。Flax の既定（batch とヘッドでマスクを共有）とは違うが、nabla の PRNG は JAX とビット単位で一致しないので値は比べられない（#241）。

### 3.9 conv レイヤー（#207）

- 名前は **`conv`**（issue の題は conv2d だが、Flax の `nn.Conv` と同じく 1D / 2D / 3D を同じ関数で受ける）。スロットは `kernel`（HWIO）と `bias`、`strides` / `padding` / `kernel-dilation` / `feature-group-count` は `:static`。入力は NHWC で、バッチ次元の無い入力も受け付ける（画像ごとの `vmap` に使える）。kernel は lecun_normal（fan_in = カーネルの面積 × in/G）。
- padding は `:same` / `:valid` か明示の `((low high) ...)`。SAME は呼び出しのたびに入力の形から JAX の `padtype_to_pads` と同じ式で計算する。`make-conv` で手で作ったモジュールも、呼び出し時に init と同じ正規化と検査を通る。文字列のパディングと入力の dilation・use_bias・CIRCULAR / CAUSAL は対応しない（#241）。

### 3.10 core に足した公開 API

`sqrt` の書き換えと `rsqrt`、`conv-general-dilated`、`dot-general`（`nabla:dot-general (a b &key lhs-contracting rhs-contracting lhs-batch rhs-batch)`。出力の軸の順はバッチ → lhs の残り → rhs の残り。eager でも形状推論を通す）、`truncated-normal`（±5 に限る。それより広いと f32 で erf(hi/√2) が 1 に丸まる）、`value-aval`、PyTree 一式（`defpytree` / `tree-*` / `treedef=` / `pytree-error`）。`nabla.iree` の `iree-too-many-inputs`。公開 API の一覧は README を正とする。

## 4. 測定

### 4.1 Transformer の end-to-end（#209、IREE local）

構成: vocab 12、長さ 8、d_model 16、ヘッド 2、MLP 32、pre-LN ブロック 2 つ（葉 38 個）。課題は `x1..x4 SEP x1..x4` のコピー（損失は SEP 以降の 4 か所）。`chain(clip-by-global-norm 1, adam 1e-2)`。初期値は numpy で作って safetensors で両側に渡す（nabla の PRNG は JAX とビット単位で一致しない）。

| 比べるもの | Flax / Optax との最大差 |
| --- | --- |
| 20 ステップの損失 | 3.6e-6 |
| 1 ステップ後のパラメータ（`attn.key.bias` を除く） | 1.5e-7 |
| 5 ステップ後のパラメータ（`attn.key.bias` を除く） | 1.2e-6 |

許容誤差は損失・パラメータとも rtol 1e-4 / atol 1e-5。

- **`attn.key.bias` だけ別の基準で比べる理由**: この勾配は数学的に 0 になる（キーのロジットに行ごとの定数を足しても softmax で打ち消される）。そのため Adam が浮動小数点の雑音を学習率くらいの大きさの更新に増幅する。値でなく差の上限 |Δ| ≤ 2 × lr × steps で抑える。Adam の eps をテストのために変えて誤魔化すのは、Advisor の指摘でやめた（eps は既定の 1e-8）。attention 単体のテストでは、この勾配が「絶対値が 1e-5 未満」であることで比べる。
- 損失は 100 ステップで 2.6 から 0.065 に下がる。時間は最初の呼び出し（トレースとコンパイルを含む）が約 4〜7 秒（PR 内の測定は約 7 秒）、その後は 1 ステップ約 3 ms。
- jit した学習ステップの入力は 117 個（パラメータ 38 + mu 38 + nu 38 + count 1 + バッチ 2）で、local-task の上限 161 に収まる（テストで確かめる）。Adam の場合、パラメータの葉は 50 個程度までが目安（#242）。
- チェックポイントから再開した学習が、止めずに続けた学習とビット単位で一致する。

### 4.2 フィクスチャの規模と許容誤差

どれも JAX 0.10.2 / Flax 0.12.8 / Optax 0.2.8（x64 無効）で作り、f32 の既定の許容誤差（rtol 1e-5 / atol 1e-6）を使う。例外は次のとおり。

| 対象 | ケース | 例外 |
| --- | --- | --- |
| sqrt / rsqrt（PR #214） | 4 dtype × 159 行 | NaN・±inf・±0 は厳密、ほかは dtype の許容誤差。非正規数は XLA CPU が 0 に丸めるので入れていない |
| convolution の順方向（PR #215） | 8 ケース | なし |
| convolution の勾配（PR #217） | 10 ケース（`batch_group_count = 2` を含む） | bf16 / f16 は f32 に上げた eager と比べる |
| dense / layer-norm / embedding / dropout / sequential（PR #222） | 11 ケースの順伝播と勾配 | layer-norm だけ rtol 1e-4（速い分散の桁落ち） |
| conv レイヤー（PR #232） | 10 ケース（SAME の非対称パディング、偶数カーネル、グループ、depthwise、dilation、1D、バッチ次元なしを含む） | bf16 のテストだけ緩めた |
| multi-head attention（PR #230） | 7 ケース（因果、padding で全部隠した行を含む、cross、2 つのバッチ次元） | `key.bias` の勾配だけ max\|actual\| < 1e-5 |
| オプティマイザ（PR #225） | 11 ケースを 6 ステップ + 4 つのスケジュール | なし |
| Transformer（PR #234） | 20 ステップ | §4.1 |

### 4.3 フェーズ4で見つけた IREE 3.11 のバグ

詳細・最小の再現・上流への報告の下書きは `docs/iree-upstream-bugs.md`。上流への報告はまだ。#243 で項目 4〜8 の英語の下書きと上流の既存 issue の検索（重複と断定できる issue は無し、8 に関連の [#24753](https://github.com/iree-org/iree/issues/24753)）、`scripts/check-iree-repros.sh` の 4・5 の行を足した。起票はリポジトリのオーナー（下書き済み・未報告）。

1. **local-task device で入力が 162 個以上の関数を呼ぶと NULL を参照して落ちる**（項目 6、#199）。入力の個数だけで決まり、形・演算に依らない（手書きのモジュールでも 162 個で落ちる）。`:local-sync` は動く。`nabla.iree:invoke` が呼ぶ前に `iree-too-many-inputs` を signal する（`+local-task-max-inputs+` = 161。ドライバを文字列で指定しても検査する）。もう 1 つの制限として、入力 170 個の要素ごとの加算の連鎖は、ロード時に `binding count 51 > 32` で失敗する。Advisor が、テストを 8 葉に制限して隠していたのを見つけた。根本対処は #242。
2. **bf16 / f16 を f32 に convert した畳み込みの 2 件**（項目 4・5、#203）。(a) 転置したカーネルを convert して畳み込むと、片方の operand の dtype が元に戻ってコンパイルエラー（lhs の勾配）。(b) O/G = 1 の depthwise で `tensor.collapse_shape` の型エラー。回避策は、f32 に直した 2 入力をまとめて `stablehlo.optimization_barrier` に通してから畳み込む。f32 の emit は変わらない。「今も落ちること」を `convolution/iree-bf16-convert-conv-bugs-still-reproduce` が守る。Advisor が bf16 / f16 の lhs の勾配が落ちる（14 件）ことを見つけた。
3. **local-task の実行時の競合**（項目 7、#220 / PR #236）。scan の本体が ys を `optimization_barrier` に通したあと、制御構造（`stablehlo.while` / `stablehlo.if`）をはさんで in-place の `dynamic_update_slice` をすると、CPU に負荷があるとき ys の行が 0 のまま返る（0.3〜2%。`cond*` の本体では 0.3%。local-sync と worker 数 1 では起きない。コンパイル結果は決定的）。**テストの許容誤差の問題ではなかった**。回避策は、本体に制御構造がある scan では ys の barrier を付けないこと。代償は、外側の長さについて 2 乗の費用（L=64・ys 16 MB で 788 ms 対 68 ms の 11 倍。#233）。
4. **`stablehlo.if` の枝が入力をそのまま返すと `ref is null`**（項目 8、#226）。片方の枝が入力を素通しし、もう片方の枝を選ぶと、invoke で失敗する（local-task・local-sync の両方。出力の並びによる）。#201 の検証で見つかった。**解決済み（#226）**: 素通しの出力を `stablehlo.if` の結果から外し、if の外の `stablehlo.select` にする（`%cond-emit`。barrier に通す案は効かなかった）。最小の再現は `docs/iree-repros/if-passthrough-operands.mlir`。

### 4.4 mutation testing

各 PR の説明に書いた結果は §2 に併記した。まとめると、新規のファイルは 34/34（module）、44/44（checkpoint）、58/58（conv）、78/79（layers）、104/104（conv プリミティブ）、53/56、26/26 と、生き残りはほぼ等価変異（`0.0d0` と `0`、級数の項数、境界の `<=`、シンボルの `equal` / `eq`）だった。テストで本当の抜けが見つかった例: `convolution` のバッチ化ルールの rhs のバッチ軸の位置（ケースを足した）、`scan` の emit の `while` ブロック引数の型・戻り値の名前の個数（medium の IREE のテストでは捕まるが small では捕まらない。#237）。`defmodule` の中は runner が変異させられないので、ロジックは `defun` に出す（§3.4）。

### 4.5 既定のテストスイート

各 PR の最終実行は、PR #225 で small 143980 checks / medium 3287 checks。この報告書のブランチの最終実行（#232 と #238 を取り込んだ tree に、この PR の docs と README を足したもの）は、下のとおり。

`NABLA_REQUIRE_IREE=1 NABLA_REQUIRE_PJRT=1 scripts/run-tests.sh`（空の `NABLA_CACHE_DIR`）で small 144453 checks / medium 3432 checks がすべて通った（失敗 0、スキップ 0）。Transformer のテストの実測は、1 回目の呼び出しが 6.1 秒、その後 3.0 ms/ステップ、5 ステップ後のパラメータの最大差 1.2e-6。

## 5. 既知の制限と積み残し

- 以下のうち、#223 / #226 / #227 / #233 / #237 は各 PR の検証で起こしたフォローアップで、#239〜#245 は #211 で起こした。
- **#223: `expm1` / `log1p` / `sin` / `cos` のプリミティブが無い**。cosine decay と Adam のバイアス補正は級数で計算している（§3.7）。足して置き換える。
- **#226: `stablehlo.if` の枝が入力を素通しすると IREE で失敗する**（§4.3 の 4）。素通しの出力を `stablehlo.select` にして直し、#201 の IREE の `cond*` のテストに素通しのケースを戻した（解消済み）。
- **#227: `while-loop` の carry に Lisp の実数を渡せない**（`scan` / `cond*` の operands は実数を葉にできる）。
- **#233: 本体に制御構造がある scan の ys が、更新のたびに丸ごとコピーされる**（外側の長さの 2 乗。§4.3 の 3）。レイヤーを scan で回して本体に dropout や `cond*` があると効く。
- **#237: scan の emitter の mutation の生き残り**（`while` のブロック引数の型・戻り値の名前の個数・カウンタの型。medium では捕まるが small のテストが無い）。
- **#239: nn が core の内部シンボル `nabla::%pytree-node-entry` を使っている**。子を返す公開関数（`tree-children`）を export して移す。
- **#240: gather プリミティブが無く、embedding が one-hot + `dot`**（§3.6）。
- **#241: Flax との差の残り**: `use_bias`、`broadcast_dropout` の既定、conv の入力 dilation と CIRCULAR / CAUSAL・`conv_transpose`、embedding の `attend`、layer-norm の `use_scale` など、バッチ正規化 / グループ正規化。
- **#242: jit の入力が 161 個を超えるモデルは local-task で学習できない**（葉が約 50 個を超える Adam の学習ステップ）。dtype ごとに葉を連結する案。
- **#243: フェーズ4の IREE のバグ（項目 4〜7）と #226 の上流への報告**（下書き済み・未報告。起票はリポジトリのオーナー）。#73（フェーズ1）と #165（フェーズ3の while のバグ 2 件）が未報告のまま残っているので、まとめて進める。
- **#244: チェックポイントは全体をメモリに読む**。葉ごとの遅延読み込みと、データパイプラインの状態の保存（フェーズ5の前提）。
- **#245: check-it の回帰 datum の文字列のバグ**。#236 が `check-it::datum` に足した防御のメソッドの扱い。
- **`while-loop` の逆モード（`grad`）は引き続き対応しない**（フェーズ3から）。`vmap` / `grad` の「外側のトレーサを閉包で捕まえると `tracing-error`」も変わらない。
- **nn の API は `defmodule` の `:call` が必須引数のみ**。可変個の引数や省略可能な引数（Flax の `mask=None`）は、位置引数に `nil` を渡して表す。
- **GPU**: フェーズ4の機能（Transformer の学習、conv、attention）は CUDA で測っていない。large テストとベンチの手順はフェーズ2・3のまま（`NABLA_REQUIRE_CUDA=1`、`scripts/colab-gpu-check.sh`）。
- **チェックポイントのパス名の互換性**: chain に変換を挿すと状態のパス名が変わる（§3.5）。版 1 のファイルを将来読み続けるかは、版を上げるときに決める。

## 6. 進め方の知見（プロセス）

- **stacked PR と squash merge**: フェーズ3と同じく、親が squash merge されると子に親の元のコミットが残る。子の base を `main` に付け替え、`main` を取り込んで衝突を解く。フェーズ4は 1 日で PR が連続して入り、**`nabla.asd`・`src/package.lisp`・README・CLAUDE.md・用語集の衝突が並行作業の最大のコスト**だった（フェーズ3の並行作業用のアンカーコメントは撤去済みで、フェーズ4では置かなかった。複数の Driver が同じ行の近くに追記するため、`main` を取り込むたびに手で解いた）。差分が大きい PR は 3 本に分ける（#208）。
- **Advisor（Fable）の敵対的検証が、実際のバグを毎回見つけた**。全 PR で行い、判定は ship / revise。revise になった PR は #214・#216・#217・#218・#213 など。例:
  - `clip-by-global-norm` の docstring が「勾配に NaN が出ない」と事実と違った（#225）。
  - bf16 の dropout が落ちる（#222）、bf16 / f16 にマスクを付けた attention が `CASE-FAILURE` で落ちる（#230）、bf16 / f16 の lhs の勾配が IREE でコンパイルできない（#217）。
  - 入力 162 個以上で IREE が落ちるのを、テストを 8 葉に制限して隠していた（#218）。Advisor が nabla を使わない手書きモジュールで原因を絞った。
  - **#220 の競合は 2 回「やり直し」になった**。最初は許容誤差の問題と診断され、次に while だけの問題と診断されていた。Advisor が (1) 許容誤差ではなく実行時の競合であること、(2) while だけでなく if でも起きることを見つけた。
  - `defmodule` の `:call` の本体がトレースされない（#216）、`nb:rsqrt` を export しているのに定義がコミットに入っていなかった（#214）。
  - **doc-rot を防ぐはずのテストが FiveAM の root suite に落ちて、一度も走っていなかった**（#210 / PR #238）。`every-loaded-test-belongs-to-a-nabla-suite` を足し、ロードしたテストがどれも nabla のスイートに属すことを検査する。ドキュメントを守るテストは、実際に走っていることを確かめるテストとセットで書く。
  - Driver がテストを先に書いていない（#215）のも、Advisor が申告どおりか確かめた。
- **Sonnet の Driver が各 issue を実装し、Advisor が検証する**分業で、1 日に PR を出し続けた。Driver が PR の説明に「確認」と「検証（Advisor）」の節（何を確かめ、何を直したか）を書く形式にそろえた。
- **CI の落ち方は再実行で済ませず原因を調べた**。#235 は #234 の CI で見つかった PBT の性質の書き方の誤り（実装は正しい）で、seed (15 6 13) を回帰ケースにして直した。#220 は「ときどき落ちる」テストの原因が IREE の競合で、負荷をかけて local-task と local-sync を比べて特定した（1000〜3000 回の実行）。許容誤差を緩めて隠していたら見つからなかった。check-it が失敗した回帰データを文字列のまま積むバグ（`NO-APPLICABLE-METHOD CHECK-IT::DATUM`）も、この調査の副産物。
- **PR の説明の末尾のフッター**: PR の説明の末尾に付く `Generated by Claude Code` の行は、GitHub 連携が PR を作るときに自動で付けるもので、Driver が書いたものではない。squash merge した `main` のコミットメッセージには入らないことを確かめた（フェーズ4の PR はすべて署名なし）。コミットメッセージに署名を付けない運用は守れている。
- **mutation testing の運用**: `defmodule` の中と、`defprimitive` に `#'fn` を書いた本体は変異させられない（フェーズ3から）。nn のロジックは `defun` に出す。レビューで指摘された生き残りは、テストを直して殺し、殺せないものは等価変異として PR に理由を書く。
- **テストの id と回帰ファイル**: PBT の回帰 id はテストごとに一意にする（`regressions/ids-are-unique`）。共有していると、あるテストが保存した seed を別のテストが再生して、別の型のデータで落ちる。

## 7. フェーズ5（Grain 相当。データ供給）への引き継ぎ

- **`nabla/data` システムはまだ無い**（`nabla.asd` にコメントがあるだけで、`src/data/` も無い）。フェーズ5の最初の作業で、`nabla/data` と `nabla/data/tests` を新設する（`nabla/nn` を作った #198 / PR #213 が手本。`scripts/run-tests.sh` への追加と CI も）。データ層は core にも nn にも依存しない形を保つ。
- **バッチは PyTree**: 学習ステップの引数は `(model opt-state batch)` で、バッチは配列のリスト・ベクタ・`defmodule` など何でもよい。`jit` はバッチの treedef と葉の aval をキャッシュのキーに入れるので、**バッチの形（最後の端数のバッチを含む）が変わると再コンパイルされる**。データローダは固定の形のバッチを返す（端数は捨てるか詰める）。
- **学習ステップは `jit` した純粋関数**（チュートリアルと `examples/transformer.lisp`）。データの読み込みは Lisp 側のループで、`to-device` を挟んで渡す。1 回の呼び出しの内訳（to-device / invoke / to-host）は `scripts/profile-call-overhead.lisp` で測れる（#190）。先読み（prefetch）で入力の転送と計算を重ねたいときの比較対象になる。
- **再開できる学習**: チェックポイントは「モデル + オプティマイザ状態」を 1 つの木に入れて往復できる（ビット単位で一致する再開をテスト済み）。データローダの位置（エポック、シャッフル用の PRNG キー、バッチ内の位置）も、整数と `:u32` のキーの葉を持つ `defmodule` にすれば同じ木に入れられるはず。方針は #244 で決める。**シャッフルは PRNG のキーから決定的に作る**（`fold-in` でエポックを混ぜる）と、位置の保存が小さく済む。
- **PRNG**: `split` / `fold-in` / `uniform` / `normal` / `truncated-normal`。バッチされた rng の emit は K 行ずつの while（フェーズ3）。データ拡張で `vmap` してキーをバッチするとコンパイル時間は行数に依らないが、実行時間は行数 / 16 に比例する。
- **並列**: 処理系は SBCL で、並列は lparallel を使う（CLAUDE.md の構成）。finalizer は別スレッドで走るので、`device-array` を並列のワーカーから作るときの解放の順序（CLAUDE.md の device-array の節）に注意する。
- **safetensors の読み書きは使い回せる**（データセットの配列の保存にも）。ただし今はファイル全体をメモリに読む（#244）。
- **README のチュートリアルの手書きのループ**は、データローダの最初の利用者になる。フェーズ5の例は、このチュートリアルの `make-batch` を置き換える形にするとよい。
- **残っている IREE の制限**（§4.3）は、データを入れても変わらない。特に学習ステップの入力が 161 個を超えない（#242）こと。
- GPU の未測定は、フェーズ3までと同じ手順で Colab で測る。
