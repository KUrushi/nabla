---
name: nabla-testing
description: nabla（Common Lisp × IREE の深層学習ライブラリ）のテスト戦略。property-based testing（FiveAM + check-it）と自前の mutation testing で自動テストを組み立てる手順、テストサイズ、許容誤差、守らせる性質の一覧を定める。nabla のコードを書く・直す・レビューするとき、テストを追加・修正するとき、プリミティブ・自動微分・vmap・jit・StableHLO 出力・PyTree・PRNG・データローダを実装するとき、テストが落ちた・フレーキーなときには、ユーザーが「テスト」と言っていなくても必ずこのスキルを使うこと。
---

# nabla のテスト戦略

nabla の自動テストは **property-based testing†（PBT）** と **mutation testing†** の2本柱で組み立てる。PBT で「どんな入力でも成り立つ性質」を大量の入力で確かめ、mutation testing で「そのテストが本当にバグを見つけられるか」を確かめる。

この2つを組み合わせる理由: 深層学習ライブラリのバグは「特定の形状や値のときだけ数値がずれる」ものが多く、人が選んだ数個の例では見逃しやすい。PBT はそこを広く探す。ただし PBT は性質が弱いと何も見つけないので、mutation testing で性質の強さを測る。

例ベースのテスト（具体的な入力と期待値を並べるテスト）は、次の2つに限る。

- JAX で生成した期待値との数値一致フィクスチャ
- PBT が見つけた失敗例の回帰テスト

例外として、CFFI の生バインディング（`nabla/iree` の `iree_*` 埋め込み C API 呼び出しなど）の疎通確認は、性質を書きにくいので例ベースでよい（手書きのフィクスチャがコンパイルできる、既知の誤った入力が決まったコンディションを出す、など）。IREE の共有ライブラリが要る medium テストは `skip-unless-iree`（`tests/iree/support.lisp`）でスキップする。

† の付いた用語は `docs/glossary.md` に説明がある。

## 作業の流れ

コードを書く・直すときは、この順で進める。

1. **性質を決める**: 変更する対象が満たすべき性質を [references/properties.md](references/properties.md) の一覧から選ぶ。一覧にない対象なら、同じ考え方で新しい性質を作り、一覧にも追加する
2. **テストを先に書く**: 性質をテストとして書き、実装前に**失敗することを確認する**。失敗しないテストは、何も確かめていない可能性がある
3. **実装する**: テストが通るまで実装する
4. **既定のスイートを実行する**: small + medium のテストをすべて実行して通ることを確認する
5. **mutation testing をかける**: プリミティブ・変換のルール・IR・StableHLO 出力・nn / data の純粋関数を変えたときは、変更した行に mutation testing をかける。手順は [references/mutation.md](references/mutation.md)
6. **報告する**: 実行したテストと結果、生き残った変異体への対処を報告に書く。テストを実行できなかったときは、そのことを隠さず書く

## テストのサイズ

Google のテストサイズ†の分類で、テストの置き場所と実行タイミングを決める。小さいテストほど速く安定しているので、できるだけ小さいサイズで書く。

| サイズ | このプロジェクトでの範囲 | 実行タイミング | 目安の割合 |
| --- | --- | --- | --- |
| small | 1プロセス内で完結し、FFI・ファイル・スレッドを使わない。IR、変換、形状推論、eager 実装、PyTree など | 毎回（既定） | 約 80% |
| medium | 1台のマシン内。IREE の `local`（CPU）での実行、ファイル I/O、lparallel のワーカー | 毎回（既定） | 約 15% |
| large | GPU（`cuda`）での実行、JAX フィクスチャの再生成、学習の end-to-end | 手動または定期実行 | 約 5% |

GPU を使うテストは別スイートに分け、既定のスイートは GPU のないマシンでも通るようにする。

## よいテストの条件

Google の *Software Engineering at Google* のテストの章の考え方にならう。

- **ハーメティック†である**: ネットワーク、現在時刻、実行順序、他のテストの結果に依存しない。乱数のシードは固定するか、失敗時に出力する。そうしないと、失敗を再現できない
- **フレーキー†なテストを放置しない**: たまに落ちるテストは、見つけた時点で原因を直す。再実行で通ったことにしない。放置すると、本物の失敗も無視されるようになる
- **公開 API を通してテストする**: 内部関数（`nb::` で呼ぶもの）を直接テストしない。内部を変えただけで壊れるテストは、リファクタリングの邪魔になる
- **振る舞いをテストする**: 「どの関数が何回呼ばれたか」ではなく「結果がどうなったか」を確かめる。本物の実装 → フェイク† → モックの順に優先する（例: GPU の代わりに IREE の `local` バックエンドを使う）
- **テストの中にロジックを書かない**: テスト本体に `loop` や `if` で期待値を計算するコードを書かない。期待値の計算が複雑なら、それは性質として表す
- **DRY より DAMP†**: 多少重複しても、1つのテストを読むだけで何を確かめているか分かるように書く
- **失敗メッセージだけで原因が分かるようにする**: 入力（縮小後）、期待値、実際の値、許容誤差、シードを出力する

## Property-based testing の決まりごと

- フレームワークは FiveAM、値の生成器は check-it。check-it の結果を FiveAM の `is` で受ける（`(is (check-it gen #'prop))`）
- check-it が見つけた失敗例は、`regression-file` で `tests/regressions/` に保存してコミットする。次回からは毎回その例も実行される
- 浮動小数点の比較はテキストの一致ではなく、許容誤差つきの数値の一致で行う。`|actual - expected| <= atol + rtol * |expected|` を満たせば一致とする（rtol / atol†）

| dtype | rtol | atol |
| --- | --- | --- |
| f64 | `1e-12` | `1e-12` |
| f32 | `1e-5` | `1e-6` |
| bf16 / f16 | `1e-2` | `1e-3` |
| 総和・内積をバックエンド間で比べる | `0` | `accumulation-atol`（2·n·u·Σ\|項\|。`tests/support/dtypes.lisp`、issue #12） |

許容誤差を緩めるときは、理由をテストのコメントに書く。理由の例: 総和の順序が違うため（JAX と比べるとき、CNN では 1e-5〜1e-4 の差が出ることが分かっている）。

生成器の方針:

- 配列は rank 0〜4、各次元 1〜8 の小さな形状にする。大きな形状はバグの発見率をあまり上げずに、テストを遅くする
- dtype も生成器で選ぶ。f32 だけで確かめると、dtype ごとの分岐のバグを見逃す
- 定義域がある演算（`log`, `sqrt`, 除算、`acos` など）には定義域内の値だけを渡す。定義域外の値は、NaN を返すことを確かめる別の性質で扱う
- 中心差分†を使う性質では、微分できない点（`abs` の 0、`relu` の 0、`max` の同値）を避ける
- 失敗したらシードと縮小（shrinking†）後の入力を出力する

どの対象にどの性質を書くか、書き方のひな形は [references/properties.md](references/properties.md) を読む。

## Mutation testing の決まりごと

- Common Lisp には実用的な既存ツールがないので、リポジトリ内の自前 runner（`tools/mutate/`）を使う
- Google の運用にならい、**PR で変更した行だけ**に変異をかける。全体にかけると時間がかかりすぎ、結果も読み切れない
- 生き残った変異体は1つずつ対処する。まず、その変異体を殺す性質を足す。等価変異体†だと判断したら、理由をつけて除外リストに記録する
- mutation score† の目安は 80% 以上。ただし数字を上げることが目的ではなく、生き残った変異体から「どんなテストが足りないか」を学ぶことが目的

変異演算子、対象範囲、arid node† の扱い、除外リストの書き方、runner の仕様は [references/mutation.md](references/mutation.md) を読む。

## コマンド

```sh
# 依存の準備（初回のみ。apt は root で実行、check-it / optima は git clone）
scripts/setup-lisp-deps.sh

# テスト（既定は small + medium。CPU だけで動き、GPU は不要）
scripts/run-tests.sh                      # NABLA_TEST_SIZES=small,medium が既定
NABLA_TEST_SIZES=large scripts/run-tests.sh

# mutation testing（既定は main から HEAD までの git diff で変わった行が対象）
tools/mutate/run.sh
tools/mutate/run.sh src/core/foo.lisp:10-40           # ファイル・行範囲を指定する
tools/mutate/run.sh --system nabla --base main --trials 20 --timeout 300
```

IREE のソースビルド（`scripts/build-iree.sh` / `scripts/verify-iree.sh`）のコマンドは
CLAUDE.md の「コマンド」を見る。mutation testing の詳しいオプションは
[`tools/mutate/README.md`](../../../tools/mutate/README.md) を見る。
