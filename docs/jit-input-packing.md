# jit の入力の連結（issue #242）

IREE 3.11 の local-task device は、入力が 162 個以上の関数を呼ぶと落ちる（`docs/iree-upstream-bugs.md` の 6）。
jit した関数の入力は引数の PyTree の葉の個数なので、Adam の学習ステップ（パラメータ + mu + nu + count + バッチ）は
パラメータの葉が 50 個を超えると上限を超える。この文書は、その回避策として jit に入れた「入力の連結」の仕組みと
費用をまとめる。

## 仕組み

- 実行系は総称関数 `backend-max-inputs`（`src/backend.lisp`）で入力の上限を申告する。既定は `nil`（上限なし）。
  nabla/iree は local-task のとき（`:local` の既定、`:local-task`、文字列 `"local-task"`）だけ 161 を返し、
  local-sync と CUDA は `nil`。core は IREE の名前を知らないまま
- jit（`src/jit.lisp` の `%jit-call`）は、葉の個数がこの上限を **超えるときだけ** 連結する。上限の内側では
  何もしないので、StableHLO もディスクキャッシュ（vmfb）も今までと同じ
- 連結の計画（`%jit-pack-plan`）は、dtype の初出順に「その dtype の葉の位置の昇順のリスト」を並べたもの。
  葉の aval と上限だけで決まる。jit キャッシュのキーには上限（`backend-max-inputs` の値）を入れる
- 呼び出しのたび、葉を dtype ごとにホストで 1 本の 1 次元配列にコピーする（`%jit-pack-leaves`、row-major の順。
  device array の葉はいったん `to-host` する）。入力の個数は dtype の種類の数になる（学習ステップなら f32 と i32 の 2 個）
- コンパイルする graph の先頭で、内部プリミティブ `:slice`（`stablehlo.slice`、`src/primitives/slice.lisp`）と
  `reshape` で葉に戻す（`%jit-unpack-leaves`）。`:slice` は jit の一番外側の graph にだけ現れるので、
  jvp / transpose / バッチ化のルールは持たない
- 出力は連結しない。local-task は出力 170 個・300 個でも落ちない（`input-limit/local-task-runs-many-outputs`）
- コンパイルに失敗したときの `use-eager` も、連結した入力で graph を評価する
- dtype の種類の数そのものが上限を超えるときは `jit-error`（161 では起きない）

device array の葉をデバイス上で連結する方法（連結用の小さな関数をもう 1 つコンパイルする）は採らなかった。
葉が上限を超えるのは学習ステップのような大きな木で、その入力はたいてい前のステップの戻り値（ホストの配列）
だからで、ホストでのコピーの費用は下の表のとおり小さい。ただし葉を device array のまま持ち回る使い方では、
連結するたびにデバイス → ホスト → デバイスの往復が入る（その費用は測っていない）。

## 費用

測定: CI（GitHub Actions の ubuntu-24.04、2026-10-24、PR #267 の run 40084201841）で、
`tests/nn/transformer-test.lisp` の `transformer/deep-model-past-the-input-limit-trains-on-iree-local-task` が
出力に書いた値（テストは値を検査しない。目安）。モデルは `examples/transformer.lisp` のブロック 4 つ
（幅 16、2 ヘッド、MLP 32、パラメータの葉 70 個）、バッチ 8、clip + Adam。学習ステップの入力は 213 個
（パラメータ・mu・nu の 210 個と正解の f32 211 個、count とトークンの i32 2 個。連結すると f32 と i32 の 2 本）、出力は 212 個。

| 項目 | 値 |
| --- | --- |
| ホストでの連結だけ（`%jit-pack-leaves`、入力 213 個 → 2 個、100 回の平均） | 0.070 ms/ステップ |
| local-task（連結あり）の学習ステップ（30 ステップの平均。ホストでのバッチ作りと連結を含む） | 3.4 ms/ステップ |
| local-sync（連結なし。上限を避ける従来の回避策）の同じステップ | 4.2 ms/ステップ |
| eager との差（2 ステップ後のパラメータ、attn.key.bias 以外の最大） | 3.5e-6 |

読み方:

- ホストでの連結はステップ時間の約 2% で、無視できる大きさ。コピーする量はパラメータと Adam の状態の
  合計に比例するので、大きなモデルでは比例して増える
- IREE 側の slice の費用は単独では測れない（連結しない local-task は入力 213 個で落ちるので、同じ
  ドライバで比べる相手が無い）。ステップ全体では local-sync（連結なし）より速いので、slice と reshape を足した費用は
  local-task の並列実行の利得の内側に収まっている。
  比べるなら、上限の内側のモデル（入力 117 個のブロック 2 つ）で `backend-max-inputs` を小さく返す
  backend を作り、同じ local-task で連結あり・なしを測る（未測定）
- 測った値は CI の共有ランナーの 1 回分で、ばらつきの幅は測っていない

再測定のしかた: `NABLA_REQUIRE_IREE=1 scripts/run-tests.sh` の出力の `deep transformer:` の行を見る。
