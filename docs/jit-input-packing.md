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
  葉の aval と上限だけで決まり、jit キャッシュのキーに入れる
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
だからで、ホストでのコピーの費用は下の表のとおり小さい。

## 費用

<!-- COST -->
