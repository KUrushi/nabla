# mutation score の実測（issue #70）

issue #70 は、`tools/mutate` の粒度（定義ごとに1体 → 変異をかけられる箇所ごとに1体）と
変異演算子（4種類 → 9種類。一覧は [`tools/mutate/README.md`](../tools/mutate/README.md) の
「変異演算子」）を増やし、「mutation score 80%」を意味のある基準にするためのもの。
runner 側の変更は先に入っている。この文書は、その runner でコアの中心のファイル
（`src/eval.lisp`・`src/primitives/*.lisp`・`src/jit.lisp`・`src/stablehlo.lisp`）を測った結果と、
生き残った変異体への対処を記録する。

## 1. 変異体の数と定義の数

`tools/mutate/run.sh --dry-run FILE` で数えた変異体の数（間引かない全数）と、ファイルの中の
変異可能なトップレベルの定義（`defun` / `defmethod` / `defmacro` / `defprimitive` など）の数。
以前の runner は定義1つにつき高々1体しか作らなかったので、変異体の数は定義の数を超えなかった。

| ファイル | 定義 | 変異体（全数） |
| --- | ---: | ---: |
| `src/eval.lisp` | 7 | 22 |
| `src/primitives/arith.lisp` | 30 | 10 |
| `src/primitives/bits.lisp` | 13 | 71 |
| `src/primitives/common.lisp` | 19 | 52 |
| `src/primitives/compare.lisp` | 20 | 100 |
| `src/primitives/cond.lisp` | 5 | 15 |
| `src/primitives/dot.lisp` | 17 | 63 |
| `src/primitives/reduce.lisp` | 19 | 73 |
| `src/primitives/rng.lisp` | 12 | 257 |
| `src/primitives/shape-common.lisp` | 7 | 15 |
| `src/primitives/shape.lisp` | 3 | 37 |
| `src/primitives/stop-gradient.lisp` | 4 | 3 |
| `src/primitives/unary.lisp` | 20 | 10 |
| `src/jit.lisp` | 21 | 48 |
| `src/stablehlo.lisp` | 23 | 63 |
| 計 | 220 | 839 |

全体では変異体が定義の約3.8倍ある。`arith.lisp` / `unary.lisp` / `stop-gradient.lisp` だけは
定義より少ないが、これらの定義はほとんどが `common.lisp` の共通部分を呼ぶだけの1行の
委譲（`(%binary-numeric-abstract-eval :add in-avals)` など）で、変異をかけられる箇所が
StableHLO の演算名と算術演算子しか無いため。中身の論理は `common.lisp` 側で変異を受けている。

## 2. 測り方

- コマンド: `tools/mutate/run.sh --trials 5 --timeout 120 --max-per-def 25 FILE`
  （テストは既定の `nabla/tests` の small + medium）。生き残った変異体は runner が既定の試行回数で
  自動的に再確認する
- `--max-per-def 25` は、変異体の多い定義を1つの定義あたり25体まで等間隔に間引く。
  間引きが効いたのは `dot.lisp`（63 → 46 体）と `rng.lisp`（257 → 127 体。
  Threefry や u32 の並べ替えなど、算術の箇所が多い定義がある）だけで、ほかのファイルは全数
- `nabla/tests` は IREE を使わない。IREE で実行して初めて分かる変異（StableHLO の型・形の
  綴り）は、`nabla/iree/tests` のテストを `--test-form` で選んで走らせる第2段で測った。
  第1段で生き残った変異体のうち、IREE で実行しないと区別できないものが集中していた
  `%rng-emit-batched`（`src/primitives/rng.lisp:181`）について、
  `--test-system nabla/iree/tests --test-form` で
  `iree/prng/batched-rng-bit-generator-matches-eager` だけを走らせた（間引かない32体）
- 4コアのマシンで、ほかの作業と並行して2本ずつ走らせた。1ファイルあたり30秒〜20分、
  全体で約2時間（1回分）

## 3. 結果

mutation score は `(殺した数 + タイムアウト数) / (全数 - 除外数)`。「前」は issue #70 の
テストを足す前、「後」はテストと除外リスト（`tools/mutate/exclusions.lisp`）を足した後。

| ファイル | 実行した変異体 | 前: 殺した（うちタイムアウト） / 生存 | 前の score | 後: 殺した（うちタイムアウト） / 除外 / 生存 | 後の score |
| --- | ---: | --- | ---: | --- | ---: |
| `src/eval.lisp` | 22 | 21 / 1 | 95% | 22 / 0 / 0 | 100% |
| `src/primitives/arith.lisp` | 10 | 10 (2) / 0 | 100% | 10 (2) / 0 / 0 | 100% |
| `src/primitives/bits.lisp` | 71 | 64 / 7 | 90% | 64 / 7 / 0 | 100% |
| `src/primitives/common.lisp` | 52 | 48 / 4 | 92% | 52 / 0 / 0 | 100% |
| `src/primitives/compare.lisp` | 100 | 86 (6) / 14 | 86% | 85 (6) / 15 / 0 | 100% |
| `src/primitives/cond.lisp` | 15 | 15 / 0 | 100% | 15 / 0 / 0 | 100% |
| `src/primitives/dot.lisp` | 46 | 45 / 1 | 98% | 46 / 0 / 0 | 100% |
| `src/primitives/reduce.lisp` | 73 | 70 / 3 | 96% | 70 / 3 / 0 | 100% |
| `src/primitives/rng.lisp` | 127 | 104 / 23 | 82% | 110 / 6 / 11（第2段がすべて殺す） | 91%（両段で 100%） |
| `src/primitives/shape-common.lisp` | 15 | 12 / 3 | 80% | 13 / 2 / 0 | 100% |
| `src/primitives/shape.lisp` | 37 | 32 (1) / 5 | 86% | 37 (2) / 0 / 0 | 100% |
| `src/primitives/stop-gradient.lisp` | 3 | 3 / 0 | 100% | 3 / 0 / 0 | 100% |
| `src/primitives/unary.lisp` | 10 | 10 / 0 | 100% | 10 / 0 / 0 | 100% |
| `src/jit.lisp` | 48 | 46 (3) / 2 | 96% | 48 (3) / 0 / 0 | 100% |
| `src/stablehlo.lisp` | 63 | 53 / 10 | 84% | 61 / 2 / 0 | 100% |
| 計 | 692 | 619 (12) / 73 | 89% | 646 (13) / 35 / 11 | 98%（両段で 100%） |

第2段（`%rng-emit-batched` を IREE で実行）: 前は32体中 殺した 25 / 生存 7、後は
殺した 25 / 除外 6 / 生存 1。後の生存1体（補助の名前の元にする出力の番号を `(subseq … 2)` に
する変異体）は、第1段に足した `prng/rng-batched-emit-defines-each-ssa-name-once` が殺すので、
両段を合わせると生き残る変異体は無い。第1段の `rng.lisp` で生き残る `%rng-emit-batched` の
変異体（StableHLO の型の綴り）は、逆にすべて第2段が殺す。

タイムアウトは、変異で `select` や比較の向きが変わり、`while-loop` を使うテストが止まらなく
なったものが多い（タイムアウトは殺したとみなす）。マシンの負荷によって、同じ変異体が
「殺した」と「タイムアウト」の間を行き来する。

注意: 前の測定の `compare.lisp` で「殺した」になった `%convert-aux-name "clamped"` の変異体は、
変異と関係の無い medium テスト（子プロセスで nabla を読み直す
`load/nabla-emits-no-redefinition-warning`）が並行作業の影響で落ちたためで、本当は生き残る
（等価変異体として除外した）。

## 4. 生き残った変異体への対処

### 4.1 性質を足して殺したもの

| 変異体 | 足した・強めたテスト |
| --- | --- |
| `eval-graph`: 複数出力の eager が返す配列の個数の照合を消す | `eval/multiple-output-eager-with-wrong-count-signals-primitive-error`（テスト用プリミティブ `%test-multiple-eager-count`） |
| `%quiet-nan`: canonical quiet NaN のビット列を変える（4体） | `primitives/log/negative-is-canonical-quiet-nan`（PBT。f32 / f64 の任意の負の値の log のビット列） |
| `dot-general`: 次元の範囲の下限を -1 に緩める | `dot-general/out-of-range-index-signals-primitive-error` に -1 を追加 |
| Threefry-2x32: 鍵とカウンタの和を差にする、カウンタの上位語の取り出しを1ビットずらす、鍵の取り出しを31ビットにする | `primitives/rng/matches-threefry-2x32-known-answers`（Random123 / JAX の既知の答え3組。今までの既知の答えは鍵が 0 だけだった） |
| rng: カウンタの 2^64 での折り返しを 65 ビットにする（u64 と u32 の新しい状態） | `primitives/rng/counter-wraps-around-2-to-the-64`（PBT） |
| rng: バッチされた状態の eager の出力の要素型 | `primitives/rng/batched-state-shape-and-dtype-match-abstract-eval`（PBT） |
| `%rng-emit-batched`: 補助の名前を出力の名前の一部だけから作る | `prng/rng-batched-emit-defines-each-ssa-name-once` |
| `%shape-check-shape-param`: 負の次元を許す | `shape/reshape/invalid-shape-signals-primitive-error` に要素数 0 の例を追加 |
| `broadcast-in-dim`: 範囲・重複・入力の個数の検査を消す、範囲の下限を -1 にする | `shape/broadcast-in-dim/invalid-dims-signal-primitive-error` に次元の合う入力の例を追加、`shape/broadcast-in-dim/wrong-arity-signals-primitive-error` |
| `transpose`: 入力の個数の検査を消す | `shape/transpose/wrong-arity-signals-primitive-error` |
| `jit`: `:static-args` の下限を -1 に緩める | `jit/errors-on-static-arg-out-of-range` に -1 を追加 |
| `%jit-execute`: 呼ぶ関数名 `"main"` を変える | フェイクの実行系（`tests/support/fake-backend.lisp`）の `backend-invoke` が、本物と同じく `"main"` 以外の名前をエラーにするようにした |
| `%stablehlo-f32-bits` / `%stablehlo-f64-bits`: 下位ビットを落とす（5体） | `stablehlo/non-finite-literal-is-the-exact-bit-pattern`（PBT。任意のペイロードの NaN と ±inf） |
| `graph-eqn-for-diagnostic`: N = 0 を範囲外とみなす | `stablehlo/graph-eqn-for-diagnostic-finds-referenced-eqn` に `eqn-0` を追加 |
| `%stablehlo-region-lines`: arg-names の個数の検査を消す | `subgraph/region-lines-rejects-arg-names-of-the-wrong-length` |

### 4.2 等価変異体として除外したもの

理由は `tools/mutate/exclusions.lisp` の各エントリの `:reason` にある（このファイルでは要約だけ）。

- 補助の SSA 名のタグを `""` にする（`compare.lisp` の `%convert-aux-name` 13体、
  `reduce.lisp` の `%reduce-aux-name` 3体、`rng.lisp` の `%rng-emit-batched` の行ごとの名前 6体）:
  名前が一意のまま綴りが変わるだけ
- `bits.lisp` の7体: シフト量 = ビット幅の境界（どちらでも 0）、直前の cond の節が捕まえる
  `=` の場合の `>` / `>=`、後で必ず切り捨てられる上位ビット、すべて上書きされる初期値
- `compare.lisp` の select の shape の検査の削除2体: 3つの検査のうち1つは残り2つから従う
  （等しさの推移律）。変わるのはエラーメッセージの文言だけ
- `rng.lisp` の2体: 和の下位32ビットに影響しない ldb の幅、行ごとの一時配列の要素型
- `shape-common.lisp` の2体: すべて上書きされるストライドの初期値
- `stablehlo.lisp` の2体: 直後の `incf` が同じく誤用をエラーにする防御的な検査、
  常に 0 以上の値に対する下限

### 4.3 残り（積み残し）

この測定の範囲では、両段を合わせて生き残った変異体は無い。ただし次は測っていない。

- `--max-per-def 25` で間引いた `dot.lisp` の17体と `rng.lisp` の130体
- `nabla/iree/tests` による第2段は `%rng-emit-batched` だけ。ほかの emit の変異は
  `nabla/tests` の StableHLO のフィクスチャとの比較だけで殺している（今回はそれで足りた）
- 第2段で IREE の全 medium テストを変異体ごとに走らせると1体あたり数分かかるので、
  対象の emit に対応するテストを `--test-form` で選ぶ必要がある

## 5. 測定で分かった運用上の注意

- **同じチェックアウトで runner を並行に走らせない**。runner は変異体ごとに
  `tests/regressions/` をスナップショット・復元するが、別の runner の変異体が書いた
  regression-case を自分のスナップショットに取り込み、復元のときに書き戻してしまう。
  測定の後は `git checkout -- tests/regressions/` で戻すこと（今回の測定でも、変異体由来の
  regression-case が約100ファイルに残った）。並行に測りたいときは別のワークツリーを使う
- 子プロセスで nabla を読み直す medium テスト（`load/…`）は、測定中にソースを編集すると
  落ちることがある。「殺した」のに理由が分からない変異体は、ログで落ちたテストを確かめる
