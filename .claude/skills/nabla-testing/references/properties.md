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

### IREE バインディング（`nabla/iree`）

CFFI の生バインディング自体は性質を書きにくいので mutation testing の対象外（CLAUDE.md）。ここでの性質は主に「壊れた入力でプロセスが落ちない」「結果が決定的」の2つ。

- `compile-stablehlo` は、手書きの StableHLO フィクスチャから非空の vmfb（ZIP local-file-header シグネチャで始まるバイト列）を返す
- `compile-stablehlo` は、文法の誤った StableHLO（`(string)` 生成器で作ったランダムな断片を関数本体に埋め込む）に対して、必ず少なくとも1つの `:error` 診断を含む `iree-compile-error` を signal し、プロセスは落ちない
- 同じ入力を繰り返しコンパイルしても、結果のバイト列は毎回一致し（決定性）、メモリ使用量（RSS）が際限なく増え続けない（リークの疎通確認）
- `compile-flags` は同じ `target` に対して毎回同じフラグのリストを返す（決定性）。未知の `target` はエラーになる
- IREE の共有ライブラリが要るテストは `skip-unless-iree`（`tests/iree/support.lisp`）でスキップし、`NABLA_REQUIRE_IREE=1`（CI）ならスキップの代わりに失敗させる
- `make-device` を local ドライバ（`local-task` / `local-sync`）で繰り返し作成・解放しても失敗せず、`release-device` は idempotent（二重解放しても何も起きない）
- `(driver-names)` に含まれない任意のドライバ名を `make-device` に渡すと、必ず `iree-status-error`（`code` が `:not-found`）が signal される
- `compile-stablehlo` でコンパイルした vmfb を `session-append-module` でロードすると、`session-function-names` に元の StableHLO の関数名（`main` など）が含まれ、`session-lookup-function` は存在する関数を linkage `IREE_VM_FUNCTION_LINKAGE_EXPORT`（2）で見つけ、存在しない関数には `iree-status-error`（`code` が `:not-found`）を signal する
- 壊れた（vmfb として無意味な）バイト列を `session-append-module` に渡すと `iree-status-error` を signal し、プロセスは落ちない
- `session-append-module`（インメモリ）と `session-append-module-from-file`（`iree-compile` の CLI が書いたファイル）は同じ vmfb を同じように呼び出せる（2つの経路の交差確認）
- `buffer-view-allocate-copy` で作った入力から `call-invoke` した結果を `buffer-view-read-into` で読み戻すと、期待する数値に一致する（許容誤差つき、`allclose :dtype :f32`）。要素型は `buffer-view-element-type` で手計算した `IREE_HAL_ELEMENT_TYPE_*` の定数どおりに decode される

### device-array（`nabla/iree`）

`to-device` / `to-host` は CFFI の生バインディングではなく nabla.iree の公開 API なので、他の対象と同じく値まで確かめる性質を書く（mutation testing の対象外は生バインディングと FFI オーケストレーションだけ）。

- `to-device` してから `to-host` すると、f32 / bf16 のどの形状（rank 0〜4）でも元の値が変わらない（`allclose`、bf16 はビット列そのものが `equalp` で一致する）
- `to-device` した `device-array` の `device-array-aval` は `array-aval` と `equalp` で一致し、`to-host` した結果の `array-dimensions` は元の shape と一致する
- `(unsigned-byte 16)` の配列を `:dtype` なしで `to-device` に渡すと `nabla:dtype-mismatch` が signal される。displaced な配列を渡すと `type-error` が signal される
- `release-device-array` は idempotent。解放後の `device-array` を `to-host` に渡すと `iree-object-released`（kind `:device-array`）が signal される
- `device-array` は生成時に自分のデバイスを retain しているので、`release-device` で device オブジェクト自身を解放した後でも、生きている `device-array` の `to-host` は正しい値を返し、`release-device-array` もクラッシュしない

### 実行（`invoke`、`nabla/iree`）

`invoke` も CFFI の生バインディングではなく公開 API なので、値まで確かめる（`invoke` 自体は FFI オーケストレーションなので mutation testing の対象外）。期待値は `tests/support/reference.lisp` の `reference-add` / `reference-matmul` / `reference-reduce-sum` で計算する。

- 手書きの StableHLO フィクスチャ（要素ごとの加算、`dot_general` による行列積、`reduce` による総和）を `compile-stablehlo` → `session-append-module` → `invoke` の順に実行した結果は、対応する `reference-*` の期待値と `allclose :dtype :f32` で一致し、結果の `device-array-aval` も期待した shape・dtype と一致する
- StableHLO の宣言と違う形状・dtype・個数の引数を `invoke` に渡すと、必ず `iree-status-error`（`code` が `:invalid-argument`）が signal される（IREE の `hal.buffer_view.assert` と vm の呼び出しチェックがクラッシュの前に検出する）
- 解放済みの `device-array` を `invoke` に渡すと `iree-object-released`（kind `:device-array`）が signal される
- `invoke` を呼んだ `session` の device とは別の device で作った `device-array` を渡すと、plain `error` が signal される

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

`(real lo hi)` には既知のバグがある（`real-generator-function` が
`new-low` を `hi` の絶対値と `lo` の符号から計算するため、`lo` と `hi` が
同じ非0の符号のとき（例: `(real 2 5)`）に範囲の幅が消え、`(random 0.0)` で
落ちる。`lo` が 0 のとき（`(signum 0)` = 0）は壊れない）。加えて両端は
`check-it::*size*`（既定 10）でクランプされるので、`(real -100 100)` のような
範囲を書いても実際に出るのは `[-10, 10]` の値になる。0 未満から 0 以上を
またぐ範囲か `(real 0 hi)` の形にし、境界が消えないよう生成した値に
小さな定数を足すなどして避ける。

## 4. 共通の生成器と比較関数

テスト用の共通部品は `tests/support/` に置き、各テストで作り直さない。

| 部品 | 役割 |
| --- | --- |
| `array-spec` 生成器 | rank 0〜4、各次元 1〜8 の形状と dtype の組を作る。`:dtypes` で候補を絞れる |
| `make-random-array` | spec と固定シードから配列を作る。`:domain` で定義域（正の数だけ、など）を指定できる |
| `pytree-spec` 生成器（未実装） | リスト・ベクタ・`defmodule` 構造体を入れ子にした木を作る予定（フェーズ1で PyTree を実装するときに追加する。それまでは `tests/support/` に存在しない） |
| `uniform-integer` / `uniform-real` 生成器 | `check-it::*size*` にクランプされない、指定した範囲全体から一様に選ぶ整数・実数の生成器。下の「check-it の落とし穴」を読んでから `(integer lo hi)` / `(real lo hi)` の代わりに使う |
| `allclose` / `approx=` | dtype ごとの既定の許容誤差で比べ、失敗時に最大誤差とその位置を出力する |
| `regression-path` | `tests/regressions/<名前>.lisp` のパスを返す |
| `skip-unless-iree`（`tests/iree/support.lisp`） | IREE の共有ライブラリが無ければテストをスキップし（`NABLA_REQUIRE_IREE=1` なら失敗させる）、あれば何もしない |
| `stablehlo-fixture`（`tests/iree/support.lisp`） | `tests/fixtures/stablehlo/<名前>.mlir` の内容を文字列で返す |
| `with-device-arrays`（`tests/iree/support.lisp`） | 複数の `device-array` を束縛して本体を評価し、終わったら逆順に `release-device-array` する（finalizer が無い #11 より前の期間、テストごとのリークを防ぐ） |
| `reference-add` / `reference-matmul` / `reference-reduce-sum`（`tests/support/reference.lisp`） | 素朴なループで計算する参照実装。DOUBLE-FLOAT で計算し `(simple-array double-float shape)` を返す。`nabla/iree` の `invoke` の期待値として使う |

部品を追加・変更したら、この表も直す。

### check-it の (integer lo hi) / (real lo hi) の落とし穴

check-it 組み込みの `(integer lo hi)` / `(real lo hi)` は、指定した `lo` /
`hi` を無視して `check-it::*size*`（既定 10）に値をクランプする
（`check-it` の `int-generator-function` / `real-generator-function` の
実装が、内部で `(min (abs limit) *size*) を取っているため）。たとえば
`(generator (integer 0 1023))` は、見た目には 0..1023 の一様分布に見え
るが、実際には 0..10 の値しか生成しない。

この落とし穴は check-it のドキュメントには書かれておらず、生成された
値の分布を実際に確認しない限り気づけない。テストは「落ちないから正し
い」と誤解しやすく、実際に nabla のこの PBT（f16 の非正規化数の仮数、
`make-random-array` の乱数シード）がこれに引っかかり、意図した範囲の
1% 未満しか検査していないのに全部パスしていた。

対策として `lo` / `hi` が `check-it::*size*`（既定 10）を超えうる範囲を
使いたいときは、必ず `tests/support/uniform-generator.lisp` の
`uniform-integer` / `uniform-real`（または `make-uniform-integer-generator`
/ `make-uniform-real-generator`）を使う。これらは check-it の generator
プロトコル（`generate` / `shrink`）だけを自前で実装し、`*size*` による
クランプを経由しない。新しい PBT で `(integer ...)` / `(real ...)` を
書くときは、範囲の上限が 10 を大きく超えないか、超えるならこちらを
使っているかを必ず確認する。

`array-spec` 生成器も、rank と各次元の範囲を選ぶのにこの `uniform-integer`
を使っている。`:max-rank` / `:max-dim` に 10 より大きい値を渡しても、
実際にその範囲まで rank や次元が届くことを
`support/array-spec/respects-larger-than-ten-max-rank-and-max-dim`
（`tests/support-test.lisp`）で確かめている。

### check-it のもう1つの落とし穴: 同じ Lisp イメージ内での再実行

check-it は失敗例を見つけると、`:regression-file` に書き出すのと同時に、
`(get regression-id 'regression-cases)` という plist にも生の文字列
（`(format nil "~S" value)`）をそのまま `push` する（`check-it.lisp` の
`save-regression`）。一方、regression ファイルを `load` して過去の失敗例
を再生するときは `regression-case` マクロ（`regression-case%`）経由で
`datum` アクセサを持つ `REGRESSION-CASE` オブジェクトとして登録される。

そのため、同じ SBCL プロセス（同じ Lisp イメージ）の中で、あるテストが
新しい失敗例を見つけて保存した「あと」に、同じテストフォームをもう一度
評価すると、2回目の実行は `(get regression-id 'regression-cases)` の中に
`REGRESSION-CASE` オブジェクトと生の文字列が混在した状態で
`(datum regression-case)` を呼ぶことになり、生の文字列に対しては
`datum` の実装（メソッド）が無いため `NO-APPLICABLE-METHOD` で落ちる。
これは check-it 側の実装の非対称性が原因で、nabla 側のコードの不具合
ではない。`scripts/run-tests.sh` のように毎回新しい SBCL プロセスを
起動する通常の実行では、プロセス起動時点でこの plist が空なので問題に
ならない。SLIME / REPL などで同じイメージのまま同じテストを何度も
`(fiveam:run! ...)` し直すときにだけ注意する（プロセスを再起動するか、
`(remprop 'テスト名 'check-it::regression-cases)` で一旦クリアする）。
