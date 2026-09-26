# 守らせる性質の一覧と書き方

対象ごとに、テストで確かめる性質を並べる。新しい対象を実装するときは、似た対象の性質をもとに新しい性質を作り、この一覧に追加する。

## 目次

1. 性質の見つけ方
2. 対象ごとの性質
3. 書き方のひな形
4. 共通の生成器と比較関数

## 1. 性質の見つけ方

よい性質が思いつかないときは、次の型に当てはめて考える。

| 型 | 考え方 | nabla での例 |
| --- | --- | --- |
| 別の実装と比べる（オラクル） | 同じ計算を別の方法で行い、結果を比べる | eager 実装と jit の結果、自動微分と中心差分 |
| 往復（ラウンドトリップ） | 変換して元に戻すと、元の値になる | `unflatten(flatten(x)) = x`、safetensors の保存と読み込み |
| 不変量 | 操作の前後で変わらない量がある | 形状推論の結果と実際の出力の形状、`shuffle` しても要素の集合は同じ |
| 代数的な法則 | 線形性、結合法則、交換法則など | transpose ルールの線形性、`vmap` と要素ごとの適用の一致 |
| 変換の可換性 | 2つの変換を入れ替えても結果が同じ | `jit(grad f) = grad f`、`jit(vmap f) = vmap f` |
| 決定性 | 同じ入力からは、何度実行しても同じ結果 | PRNG、データローダの順序 |

「結果が例外を出さない」だけの性質は弱い。mutation testing で変異体がほとんど生き残るので、値まで確かめる性質を組み合わせる。

## 2. 対象ごとの性質

### プリミティブ（`defprimitive`）

新しいプリミティブを足すときは、少なくとも次の3つを書く。

- 形状推論の `aval`† = 実際の出力の形状と dtype
- eager 実装の結果 = jit（IREE `local`）の結果（dtype ごとの許容誤差で比べる）
- 生成した StableHLO† のテキストを IREE がコンパイルできる（medium）

JAX のフィクスチャがあれば、eager 実装の結果 = JAX の結果 も確かめる。

### 自動微分（jvp† / transpose† / vjp† / grad）

- `grad f` の結果 = 中心差分†（f64 で計算）の結果
- 内積テスト†: ランダムな `u`, `v` について `<vjp(u), v> = <u, jvp(v)>`
- transpose ルールの線形性: `T(a·u + b·w) = a·T(u) + b·T(w)`
- transpose ルールの随伴性: `<T(u), v> = <u, L(v)>`（`L` は元の線形演算）
- jvp の線形性: 接線 `v` について `jvp(a·v) = a·jvp(v)`

中心差分の刻み幅 `h` は f64 で `1e-6` 程度にし、許容誤差は `rtol 1e-4` 程度に緩める。差分近似そのものに誤差があるためで、この理由をテストに書く。

### vmap（バッチ化ルール†）

- `vmap(f)(xs)` = `xs` の各要素に `f` を適用して積み上げた結果
- `in-axes` を変えても（軸の位置をずらしても）結果が一致する
- バッチ次元のない引数（`in-axes` が `nil`）は、全要素に同じ値として渡される
- `vmap(vmap f)` = 2重ループで各要素に `f` を適用した結果

### jit とキャッシュ

- `jit(f)(x)` = `f(x)`（eager）
- 同じキー（関数・`aval`・静的引数・ターゲット）で2回呼ぶと、2回目はコンパイルしない
- `aval`、静的引数、ターゲットのどれかが違えば、別のエントリとしてコンパイルする
- 関数を再定義したら、古いキャッシュを使わない

### 変換の合成

- `jit(grad f)` = `grad f`
- `jit(vmap f)` = `vmap f`
- `vmap(grad f)` = 各要素に `grad f` を適用して積み上げた結果（per-example 勾配）

### トレースと IR

- `with-tracing` でトレースした graph を eager で評価した結果 = 元の Lisp 関数の結果
- 対応していない形式（`setq` など）を含むコードは、決まったコンディションを出す

### PyTree†

- `unflatten(flatten(x))` = `x`
- `tree-map #'identity x` = `x`
- `tree-leaves x` の長さ = `flatten` した葉の数
- 同じ構造の木に `tree-map` した結果は、元と同じ構造を持つ

### safetensors

- 保存して読み込むと、名前・dtype・形状・値がすべて元と一致する
- bf16 / f16 のビット列が変わらない

### PRNG†

- 同じキーからは同じ値が出る
- `split` した子キーどうしの値は重ならない（小さな標本で衝突がない）
- `vmap` の中で使っても、各要素の結果 = 同じキーで個別に呼んだ結果

### データ層（Grain 相当）

- 同じシード・エポックなら同じ順序
- ワーカー数を変えても出力の順序と値は同じ
- `iterator-state` を保存して `restore-iterator` で再開すると、続きが中断しなかった場合と一致する
- 1エポックで各インデックスがちょうど1回ずつ出る（`shuffle` しても要素の集合は同じ）

## 3. 書き方のひな形

以下はひな形で、関数名は実装に合わせて読み替える。

check-it（b79c9103665be3976915b56b570038f03486e62f）の `check-it` マクロは
`(check-it generator test &key examples shrink-failures random-state
regression-id regression-file)`。次の2点に注意する。

- `:regression-file` は `:regression-id` も一緒に渡さないと何もしない。
  失敗例を保存したいテストには、必ずどちらも書く
- `:regression-id` はマクロが渡された式をそのまま `quote` するので、
  **クォートせずにシンボルを書く**（`:regression-id foo/bar`。
  `:regression-id 'foo/bar` と書くと、渡る値が `foo/bar` というシンボル
  ではなく `(quote foo/bar)` というリストになり、型エラーになる）
- `regression-path` が渡したファイルを無ければ作るので、`:regression-file`
  には毎回 `(regression-path "名前")` を渡してよい

### 別の実装と比べる性質

```lisp
(test primitive/exp/eager-matches-jit
  "exp の eager 実装と jit（IREE local）の結果が dtype ごとの許容誤差で一致する。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64)))
                (lambda (spec)
                  (let ((x (make-random-array spec)))
                    (allclose (nb:exp x)
                              (funcall (nb:jit #'nb:exp) x)
                              :dtype (array-spec-dtype spec))))
                :regression-id primitive/exp/eager-matches-jit
                :regression-file (regression-path "primitive-exp"))))
```

### 往復の性質

```lisp
(test pytree/flatten-roundtrip
  "unflatten(flatten(x)) は x に戻る。"
  (is (check-it (generator (pytree-spec))
                (lambda (tree)
                  (multiple-value-bind (leaves treedef) (nb:tree-flatten tree)
                    (tree-equal* tree (nb:tree-unflatten treedef leaves))))
                :regression-id pytree/flatten-roundtrip
                :regression-file (regression-path "pytree-roundtrip"))))
```

### 内積テスト

```lisp
(test grad/tanh/dot-product
  "tanh の vjp と jvp が転置の関係にある: <vjp(u), v> = <u, jvp(v)>。"
  (is (check-it (generator (array-spec :dtypes '(:f64)))
                (lambda (spec)
                  (let ((x (make-random-array spec))
                        (u (make-random-array spec))
                        (v (make-random-array spec)))
                    (approx= (inner (nb:vjp #'nb:tanh x u) v)
                             (inner u (nb:jvp #'nb:tanh x v))
                             :dtype :f64)))
                :regression-id grad/tanh/dot-product
                :regression-file (regression-path "grad-tanh-dot"))))
```

テスト名は `<対象>/<演算や関数>/<性質>` の形にする。docstring には性質を1文で書く。失敗したとき、テスト名と docstring だけで何が壊れたか分かるようにするため。

check-it の generator DSL（`(generator ...)` の中で使える形式）は主に次のとおり:
`(integer lo hi)`、`(real lo hi)`、`(list g)`、`(tuple g...)`、`(or g...)`、
`(guard pred g)`、`(map fn g...)`、`(chain ((v g)...) body)`、
`(struct type :slot g ...)`。名前付きの生成器は `def-generator` で作る
（`tests/support/array-spec.lisp` の `array-spec` を参照）。

`check-it:*num-trials*`（既定100）が試行回数のノブで、mutation testing の
runner はこれを下げて実行する。fiveam も同名の `*num-trials*` を export
しているので、check-it と fiveam を両方 `:use` するパッケージは
`(:shadowing-import-from #:check-it #:*num-trials*)` で check-it 側を選ぶ。

`(real lo hi)` には既知のバグがある（0 <= lo < hi のように lo と hi が
同符号だと、常に幅0の範囲になり `(random 0.0)` で落ちる）。0 未満から
0 以上をまたぐ範囲か、`(real 0 hi)` の形にして、境界が消えないよう
生成した値に小さな定数を足すなどして避ける。

## 4. 共通の生成器と比較関数

テスト用の共通部品は `tests/support/` に置き、各テストで作り直さない。

| 部品 | 役割 |
| --- | --- |
| `array-spec` 生成器 | rank 0〜4、各次元 1〜8 の形状と dtype の組を作る。`:dtypes` で候補を絞れる |
| `make-random-array` | spec と固定シードから配列を作る。`:domain` で定義域（正の数だけ、など）を指定できる |
| `pytree-spec` 生成器 | リスト・ベクタ・`defmodule` 構造体を入れ子にした木を作る |
| `allclose` / `approx=` | dtype ごとの既定の許容誤差で比べ、失敗時に最大誤差とその位置を出力する |
| `regression-path` | `tests/regressions/<名前>.lisp` のパスを返す |

部品を追加・変更したら、この表も直す。
