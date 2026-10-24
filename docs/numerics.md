# 数値の安定性: 設計とテストの方針

nabla の演算と損失関数が、浮動小数点の丸めで「値は合っているのに勾配が壊れる」種類の不具合を起こさないための方針。きっかけは Barron, Gnanapragasam, Tschannen「A Fix for Softmax and Cross-Entropy」（arXiv 2610.31642、<https://jonbarron.info/softmaxfix/>）で、JAX・PyTorch などの標準の softmax / log-softmax / 交差エントロピーにある 2 つのバグを、nabla では起こさないようにした（`src/nn/layers.lisp`、`src/nn/losses.lisp`）。

この文書の構成: 1 節で 2 つのバグを説明し、2 節で nabla での直し方を、3 節で同じ種類の問題を防ぐ設計上の約束を、4 節でテストの方針を、5 節で残っている危険の一覧を述べる。新しい演算を足すときやレビューするときの手順は `.claude/skills/nabla-numerics` スキルにある。

## 1. 2 つのバグ

記号: ロジット z（長さ K）、z_max = max z、s = Σ_{j≠max} exp(z_j − z_max)（最大値以外の exp の和）、eps は dtype の丸めの単位（bf16 で 2^-8、f32 で 2^-24、f64 で 2^-53）。

### 1.1 Softmax Collapse†

標準の実装は `logsoftmax(z) = (z − z_max) − log Σ exp(z − z_max)` で、和の中の最大値の項は exp(0) = 1。正解のロジットが他より ln(1/eps)（**精度の崖**、bf16 で約 5.5、f32 で約 16.6、f64 で約 36.7）以上大きいと、`1 + s` が 1 に丸まり、

- 値: log-softmax の正解の位置と損失が **ちょうど 0** になる（正しくは −log1p(s) ≈ −s）
- 勾配: 正解の位置の勾配 `p_t − 1` が **ちょうど 0** になる。他の位置の勾配 p_j = exp(z_j − z_max) は 0 にならないので、**勾配の和が 0 でなくなる**

Adam のように勾配を大きさで正規化するオプティマイザは、この非対称な勾配を学習率の大きさの更新に増幅する（grokking の「slingshot」、正解率の崩壊）。nabla の修正前の実装でも、f32 で z = [20, 0, 0] のとき損失が `-0.0`、勾配が `[0, 2.06e-9, 2.06e-9]` だった。

逆モードでは、`stop-gradient` をかけた最大値で平行移動するので、正解の位置の余接線が `−1 + exp(0)/fl(1 + s) = −1 + 1` という打ち消し合いで作られる。値と勾配の両方が、**1 との足し引きで小さな量を作っている**ことが原因。

### 1.2 Softmax Cancellation†

蒸留や KL 正則化の交差エントロピー `−Σ p log q`（p = softmax(z_T) は教師、q = softmax(z_S) は生徒）の勾配は q − p。素朴に自動微分すると、

- ほぼ等しい確率 q と p を別々に丸めてから引くので、z_S が z_T に近いと桁を失う（f32 で差 1e-7 のとき、成分の符号まで変わる）
- 順伝播の `softmax(z_T)` と、逆伝播で作られる `exp(log-softmax(z_S))` が別の演算の並びなので、**z_S = z_T でも勾配が 0 にならない**（nabla の修正前の実装では、Σq = 1 が厳密に成り立つ入力でも 3e-8 程度の勾配が残った。逆伝播の p が `e · (1/S)` で、順伝播の `e / S` と丸めが違うため）

Adam は ε より大きいこの「雑音の床」を一定の大きさの更新にするので、教師で初期化した生徒が教師から離れていく。

## 2. nabla での修正

### 2.1 log1p / expm1 プリミティブ（core）

`nabla:log1p`（`stablehlo.log_plus_one`）と `nabla:expm1`（`stablehlo.exponential_minus_one`）は #252（issue #223）で入った。小さな x で `log(1 + x)` / `exp(x) − 1` が失う桁を失わない。eager は double の Kahan の形。f32 / bf16 / f16 の emit は `log_plus_one` / `exponential_minus_one` をそのまま出す（IREE は f32 で正確に多項式近似する）。

f64 は、IREE（llvm-cpu）が多項式近似せず MLIR の lowering が `log(1 + x)` / `exp(x) − 1` をそのまま出す（`log1p(1e-12)` が `1.0000889e-12`）ので、emit が Goldberg / Kahan の式（`u = 1 + x` を `optimization_barrier` に通し、`u = 1 ? x : log(u)·x/(u − 1)`）を log と exp で出す。barrier が無いと XLA（PJRT）が `(x + 1) − 1` を x に畳む。どちらのバックエンドでも eager とビット単位で一致し、−0 の符号も保つことを medium テストで守っている（`tests/iree/unary-test.lisp`、`tests/pjrt/log1p-expm1-test.lisp`）。この問題は Fable の敵対的レビューが見つけた。

### 2.2 log-softmax / softmax（`src/nn/layers.lisp`）

論文の式 (5) を、同点の最大値に対称にして使う。M を最大値の集合（同点を含む）、c = |M| として、

```
x_max   = peak + Σ_M (x − peak) / c      ; peak は stop-gradient した最大値。値は peak と厳密に等しい
shifted = 0 (c = 1 の最大値の位置)、それ以外は x − x_max
rest    = Σ_{shifted が定数 0 でない位置} exp(shifted)
logsoftmax = shifted − log1p(c = 1 なら rest、同点なら rest − 1)
softmax    = exp(logsoftmax)
```

- 最大値がただ 1 つのとき、その位置は `x − x_max` ではなく **定数 0**。最大値の x は差 x_j − x_max だけを通して現れるので、自動微分はその勾配を **他の勾配の和の符号を変えたもの** として計算する（打ち消し合いが無く、和は 0 のまま）
- x_max には勾配を流す（`stop-gradient` にすると上の経路が消えて崩れる）。どんな x_max についても `logsoftmax(x) = (x − x_max) − log Σ exp(x − x_max)` は恒等式なので、M を定数とみなした微分は正しい（2 階以上も）
- 同点（c ≥ 2）では p ≤ 1/2 で桁落ちしないので、定数 0 の置き換えはしない
- 論文の JAX の実装は argmax の one-hot を使うが、nabla では使わない。iota を host の配列で作ると行列の大きさの定数が StableHLO に埋め込まれる（attention のスコア `(b h q k)` で大きい）ため。同点に対称なので、置換に対しても対称になる
- `log1p` の引数は、**log1p の前で** select する（後で選ぶと、選ばれない log1p(−1) の勾配が inf × 0 = NaN になる。3.4 節）
- NaN を含む行と全部が −inf の行は NaN（JAX と同じ）。−inf の要素は log-softmax が −inf、softmax が 0、勾配 0

代償: reduce が 2 回（最大値・和）から 4 回（最大値・同点の数・同点の和・残りの和）になり、`softmax` は `exp(log-softmax)` なので、IREE の local-task で attention の形 `(8 12 128 128)` の softmax が約 1.25〜3 倍、勾配が約 1.6 倍遅くなった（Fable の測定。`(32 1000)` は変わらない）。正しさを優先し、速い経路は持たない。

バックエンドの限界: IREE は非正規数を 0 に切り捨てるので、f32 で勾配が 0 に丸まらないのは余裕が約 87 まで（eager は非正規数を保つので約 103 まで）。

論文の例 z = [9, −9, −9]（f32）で、log-softmax `[-3.046e-8, -18, -18]`、交差エントロピーの勾配 `[-3.046e-8, 1.523e-8, 1.523e-8]` が論文の値とビット単位で一致する。

### 2.3 soft-target-cross-entropy（`src/nn/losses.lisp`）

教師の **ロジット** を受け取り、論文の式 (9) で q − p をロジットの差から対数空間で計算し、stop-gradient で勾配を差し替える。

```
delta = z_S − z_T（教師の確率が 0 の位置は 0）
d     = delta − c                    ; c は教師の最大値の位置の delta（同点は平均）
s     = log(q/p) = d − log1p(Σ p expm1(d)) + log1p(−Q_u)
u     = q − p = p expm1(s)（|s| ≥ 1 と p = 0 の位置は q − p）
g     = u − q Σ u                        ; 和が 0
損失  = −Σ p sg(log q) + Σ (sg(g) + (q − sg(q))/2) (z_S − sg(z_S))
```

論文から足したもの:

- **中心化の位置**: 論文の c = Σ p delta は Σ p = 1 の丸めで eps × C の誤差を入れ、z_S = z_T + 定数（q = p）でも勾配が 0 にならない。教師の最大値の位置の delta なら定数のずれはちょうど消える（どんな c でも s の値は数学的に同じ。mutation testing の調査で見つけた）
- **Q_u の補正**: 教師の確率が 0 の位置（−inf のロジット）があると、Σ p exp(delta) から生徒のその位置の確率の和 Q_u = Σ_{p=0} q が抜けるので、s に log1p(−Q_u) を足す。論文の式は p > 0 の行だけを考えていて、そのままだと勾配がずれる（テストが見つけた）
- **2 階微分**: 論文の stop-gradient の形だけだとヘッセ行列が 0 になる。値が 0・1 階微分も 0 の項 (q − sg(q))(z_S − sg(z_S))/2 を足すと、2 階微分が (∂q_j/∂z_i + ∂q_i/∂z_j)/2 = diag(q) − q q^T になる（3 階以上は正しくない。docstring に書いてある）
- **教師は定数**: 教師のロジットには勾配を流さない（混合 2 階微分まで正しくするのは割に合わない）
- **±inf の扱い**: `z − sg(z)` は ±inf で NaN になるので、その位置は 0（勾配も 0）。教師と生徒が同じ位置で −inf のとき、損失と勾配は有限。生徒だけが −inf の位置（教師の確率が正）は、中心化のずらしから外して NaN の中間値を作らず、損失は +inf、その位置の勾配は 0、他の位置は q − p
- **アンダーフローした教師**: 有限の教師のロジットでも softmax が 0 にアンダーフローした位置（余裕が約 103 以上、IREE では約 87 以上）は、−inf と同じく目標の確率 0 として扱う（値は正しい）

z_S = z_T なら勾配はちょうど 0（f32 / f64 / bf16、−inf を含む行も）。論文の例 z_T = [2, 1, 0]、z_S = z_T + [−1e-7, 1e-7, 0] で厳密な勾配 `[-4.596e-8, 4.144e-8, 4.513e-9]` と一致する。

正解がクラス（one-hot）の交差エントロピーは、修正した `log-softmax` で `−Σ y log-softmax(z)` と書けば Collapse を起こさない（`examples/transformer.lisp`）。ただし −inf のロジットと y = 0 の積は 0 × −inf = NaN になるので、−inf を入れるなら where で 0 にする。

## 3. 設計の約束（同じ種類の問題を防ぐ）

### 3.1 小さな量は直接作る。1 との足し引きで作らない

1 − p、log(1 + x)、exp(x) − 1、p − q のような「1 や大きな値との差・和で小さな量を作る」式は、小さな量の桁を失う（absorption† と catastrophic cancellation†）。小さな量を表す式が別にあれば、そちらを使う。

- log(1 + x) → `log1p`、exp(x) − 1 → `expm1`、log(1 − p) → `log1p(−p)`
- 1 − p_max（softmax の最大値）→ 最大値以外の和 Σ_{j≠max} p_j
- q − p（近い確率）→ ロジットの差から対数空間で
- 「和から 1 つ引く」（Σ − x_k）→ x_k を除いた和を直接（x_k が和の大部分を占めるとき）

### 3.2 微分の経路の式も数値として読む

値の式が正しくても、自動微分が作る逆伝播の式で桁落ちしうる。新しい合成関数を書いたら、逆伝播で各入力に集まる余接線がどんな和になるか（打ち消し合う大きな項があるか）を確かめる。

- 平行移動のための最大値に `stop-gradient` をかけると、逆伝播で最大値の位置の余接線が `−1 + 1` の打ち消しになる。平行移動の量は、**値に現れない形**（定数 0 に置き換えた位置）にするか、勾配が差だけを通るようにする
- 打ち消し合いが避けられない数学的な恒等式（Σ g = 0 など）は、構造で保証する（片方を他の和の符号違いとして作る）

### 3.3 同じ量は同じ式で計算する

順伝播と逆伝播で同じ数学的な量（確率 p など）を別の演算の並びで計算すると、丸めが違って「厳密に 0 になるはずの差」が 0 にならない。損失関数が勾配を差し替えるときは、比べる両方を同じ関数・同じ演算の並びで作る（`soft-target-cross-entropy` は q も p も同じ `%log-softmax` / `%softmax` から作る）。

### 3.4 特異な関数の前で選ぶ（double where）

`where(c, f(x), y)` は選ばれない側の f(x) も計算し、その微分も計算する。f が log・除算・sqrt・log1p などで x が特異点（0、−1）にあると、選ばれない側の局所微分が inf になり、余接線 0 との積が NaN になる。**f の入力の前で** 安全な値に select する（`log1p(where(c, rest, rest − 1))` のように）。±inf の値に `x − sg(x)` を作るのも同じ種類の罠（inf − inf = NaN）。

### 3.5 端の値の意味を決めて docstring に書く

−inf（マスク）、+inf、NaN、全部が −inf の行、長さ 1 の軸、同点を、どう扱うかを決めて docstring とテストに書く。nabla は JAX に合わせる（NaN を伝える、全部が −inf の行は NaN）。attention のマスクは −inf でなく dtype の最小の有限値（Flax と同じ）なので、全部隠れた行は一様な重みになる。

### 3.6 低精度で内部の和を広げる

bf16 / f16 の reduce と dot は f32 で累積する（`docs/stablehlo-ops.md`、issue #54 / #63）。新しい演算で和を取るときも、この経路（`reduce-sum`）を通す。精度の崖は dtype ごとに違うので、テストも dtype ごとに崖の向こう側を見る。

## 4. テストの方針

性質の一覧は `.claude/skills/nabla-testing/references/properties.md` の「数値の安定性」、手順は `nabla-numerics` スキル。要点:

1. **崖の向こう側を生成する**: 入力の生成器を「普通の値」だけにしない。精度の崖（ln(1/eps)）をまたぐ余裕、アンダーフローの近く（f32 で約 88）、同点、−inf / NaN を混ぜる。bf16 は崖が 5.5 なので、余裕 6〜30 を見る
2. **参照は別の式で、高い精度で**: 参照（期待値）は実装と別のコード（ループで書いた double の計算や有理数）で作る。参照も崖を越えるので、素朴な式の double では f64 の崖（36.7）を越えた入力で同じバグを持つ。参照の式自体が小さな量を直接作っていることを確かめる
3. **絶対誤差の床を置かない**: `allclose` の atol は、0 に丸まったバグ（期待値 1e-9 に対して 0）を通してしまう。勾配が小さくなる領域では、成分ごとの相対誤差で比べるか、床を最大の成分に比例させる
4. **構造の不変量を見る**: 勾配の和が 0（|Σ g| ≤ 4 n eps Σ |g|）、平行移動に不変、置換に対して対称、同点の位置の勾配が等しい、ヘッセ行列とベクトルの積が対称
5. **ちょうど 0 になるべきものは厳密に比べる**: z_S = z_T の勾配は `zerop` で比べる（許容誤差で比べると雑音の床を見逃す）
6. **2 階微分まで見る**: grad の grad（ヘッセ行列とベクトルの積）も崖の向こう側で参照と比べる
7. **jit でも見る**: バックエンドの数学関数の近似と融合で結果が変わりうるので、IREE で jit した結果も参照と比べる（medium、`tests/nn/softmax-numerics-iree-test.lisp`）
8. **テストが素朴な実装で落ちることを確かめる**: 新しい性質のテストは、修正前（または素朴な式）の実装に差し替えて落ちることを一度確かめる（mutation testing の手動版。`nabla-testing` の作業の流れの 2）

## 5. 残っている危険の一覧

修正していないが、同じ種類の問題を持つ箇所。直すときはこの文書の方針で直し、ここから消す。

| 場所 | 危険 | 現状と判断 |
| --- | --- | --- |
| `src/nn/layers.lisp` の `layer-norm`、`src/nn/batch-norm.lisp` の `batch-norm` | 「速い分散」 mean(x²) − mean(x)² は、平均が標準偏差より大きい（μ/σ ≈ 1e3）と f32 で数 % の誤差（catastrophic cancellation） | Flax の既定と同じ式（Transformer を Flax と数値で比べるため）。負になる分は 0 に切っている。2 パスの分散に変えると Flax と食い違う |
| `src/nn/layers.lisp` の `gelu` | tanh 近似の微分の 1 − tanh² が \|x\| > 9 で 0 に丸まる（真は約 4 e^{-2\|x\|}） | 勾配が 1e-8 以下の領域で、影響は小さい |
| `examples/mlp.lisp` | 素朴な logsumexp（Collapse を起こす） | フェーズ2のベンチマークと JAX のフィクスチャのため core だけで書いたまま。コメントで nabla/nn の log-softmax を勧めている |
| 利用者が書く one-hot の交差エントロピー | −inf のロジットと y = 0 の積が NaN | 2.3 節の末尾。整数ラベルの交差エントロピーの関数は持たない（必要になったら足す） |
