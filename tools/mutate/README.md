# nabla-mutate

nabla 用の最小限の mutation testing† runner。詳しい考え方は
[`.claude/skills/nabla-testing/references/mutation.md`](../../.claude/skills/nabla-testing/references/mutation.md)
にある。ここでは使い方を書く。

## 使い方

```sh
# main から HEAD までの git diff で変わった .lisp の行が対象（既定）
tools/mutate/run.sh

# ファイルを指定する（そのファイル全体が対象）
tools/mutate/run.sh src/core/foo.lisp

# 行範囲まで指定する
tools/mutate/run.sh src/core/foo.lisp:10-40

# オプション
tools/mutate/run.sh --system nabla --base main --trials 20 --timeout 300
```

- `--system NAME`: `nabla.mutate:run` の `:system`（記録用。既定 `nabla`）
- `--base REF`: 既定の行範囲を計算する基準（既定 `main`）
- `--trials N`: 変異体1つあたりの check-it の試行回数（既定 20。生き残った
  変異体は既定の試行回数で自動的に再確認される）
- `--timeout SEC`: 変異体1つあたりのテストのタイムアウト秒数（既定 300）
- `--test-system SYSTEM`: `nabla.mutate:run` を呼ぶ前に、この ASDF システム
  を読み込む（テストの対象になるコードや、テスト実行関数を含むシステム）。
  省略すると既定で `<--system>/tests`（`--system` の既定は `nabla` なので
  通常は `nabla/tests`）を読み込む
- `--test-form FORM`: これを `eval` した結果をテスト実行関数として使う
  （渡さなければ `nabla.mutate:default-test-function` を使う。これは
  `NABLA.TESTS.SUPPORT:RUN-TESTS` を実行時に探すので、`nabla/tests` が
  ロードされている必要がある。`nabla` 以外のシステムを対象にするときは、
  `--test-system` と `--test-form` を明示的に渡すこと）

## 終了コード

`tools/mutate/run.sh` の終了コード:

| コード | 意味 |
| --- | --- |
| 0 | mutation score†（`(殺した数 + タイムアウト数) / (全数 - 除外数)`）が 0.8 以上 |
| 1 | mutation score が 0.8 未満 |
| 2 | `FILE[:START-END]` に存在しないファイルを指定した |
| 3 | 変異させられる定義が1つも見つからなかった（`total=0`） |

3 と 0 を区別しているのは、`total=0`（対象範囲が空、たいてい `--base` や
`FILE[:START-END]` の指定ミス）のときも `nabla.mutate:mutation-score` 自体は
（除外数と全数が一致するので）1 を返し、score だけを見ると「殺した」のと
区別が付かないため。CI が「何も変異していない」のを「変異はすべて殺した」
と取り違えないよう、`run.sh` はこの場合だけ別のコードで抜ける。
`nabla.mutate:run` を直接 Lisp から呼ぶときは、`report-mutants` の長さで
同じことを自分で確認すること。

## 変異演算子

`.claude/skills/nabla-testing/references/mutation.md` の「2. 変異演算子」
にある表のうち、最初の4つ（算術演算子の入れ替え・比較の境界・定数の置き換え・
`if` の分岐の入れ替え）を実装している。残りはフェーズ1以降で追加する。

この最小版では、1つの変異可能な定義（`defun` / `defmethod` / `defmacro` /
`defprimitive`）につき、演算子を上の順で試して最初に適用できたものを
1つだけ使う。将来、行ごとの粒度に細かくする余地がある。

`defmethod` の specialized lambda list（`((x (eql 0)) ...)` のように
specializer を含むもの）は、演算子を問わずまるごと arid† として扱い、
中には決して降りない。specializer の中の値（`(eql 0)` の `0` など）を
変異させると、再評価時に元のメソッドとは違う specializer の組を持つ
別のメソッドが新しく増えてしまい、後始末（変異体の評価後に元の
`defmethod` を評価し直すこと）でも消えずに残ってしまうため
（`remove-method` していないので、元の specializer に戻す再定義は
「新しいメソッドを足す」だけで、変異体のメソッドを置き換えない）。

## 変異体どうしの隔離（regression 状態）

nabla のテストは check-it の `:regression-id` / `:regression-file` で、
見つかった失敗例を `tests/regressions/` の下のファイルとシンボルの
plist（`check-it::regression-cases`）の両方に記録する
（`tests/support/regression.lisp` の `regression-path` を見よ）。
mutation testing 中はほぼ確実にどこかの変異体でテストを失敗させるので、
何もしないと変異体ごとにこの記録が増え続け、次の2つの問題を起こす。

1. **ディスクに副作用が残る**: 変異体が生成したデタラメな値
  （境界値の変異でたまたま失敗した入力など）が `tests/regressions/*.lisp`
  に書き込まれ、コミットされてしまう
2. **偽の kill / false survive**: ある変異体（M1）が記録した regression-case
  が、`check-it::regression-cases` の plist に残ったまま次の変異体（M2）の
  実行に引き継がれる。check-it は `regression-id` ごとに、通常のランダム
  生成の前に記録済みの regression-case をすべて再生するので、M2 が単体
  では絶対に落ちない性質でも、M1 の記録したケースを再生して落ちてしまう
  （逆に、M1 由来のケースのせいで本来生き残るはずの変異体が「殺された」
  ことになる、という向きの誤りにもなる）

これを防ぐため、runner は変異体1体（正確には baseline チェックと
`%evaluate-mutant` の呼び出し）ごとに、`*regression-directories*`
（既定 `("tests/regressions/")`、`nabla.mutate:run` の
`:regression-directories` で上書きできる）以下のファイルの中身と、
`check-it::regression-cases` を持つすべてのシンボルの plist を
呼び出し前にまるごとスナップショットし、呼び出しが成功しても失敗しても
（`unwind-protect`）呼び出し後に必ず元へ戻す
（`nabla.mutate::%isolate-mutant-side-effects`）。

- nabla-mutate は nabla のコアシステムに依存しない独立したツールという
  設計（`nabla-mutate.asd` 参照）なので、この仕組みは check-it の
  regression-case の記録先（ファイルとシンボルの plist）という一般的な
  知識だけを使い、`tests/support/regression.lisp` のような nabla 側の
  関数名やパッケージ名には一切依存しない
- `*regression-directories*` に含めていないディレクトリへの書き込みは
  隔離されない。nabla 以外のプロジェクトで runner を使うときは、
  `:regression-directories` を実際の regression ファイルの置き場所に
  合わせて渡すこと
- ファイルはテキストとして丸ごと退避・復元する（regression ファイルは
  Lisp のソースなので、この前提で問題ない）。新しく作られたファイルは
  削除され、書き換えられた・消されたファイルは元の内容に戻る
- plist の隔離は `do-all-symbols` で image 全体を舐めて
  `check-it::regression-cases` を持つシンボルを探すので、check-it が
  ロードされていなければ何もしない

## 除外リストの書き方

`tools/mutate/exclusions.lisp` はトップレベルに1つ、次の形の plist の
リストを持つ。

```lisp
(:file "src/core/primitives/add.lisp"
 :mutation "(* 1 x) -> (/ x 1)"
 :reason "x に 1 をかけても割っても値が変わらない等価変異体")
```

- `:file`: 変異体のファイルパスの末尾と一致すること（`namestring` の suffix）
- `:mutation`: `"<変異前> -> <変異後>"` という形の文字列。変異前・変異後の
  トップレベルの定義全体を `PRIN1` で印字したものを `" -> "` でつないだ
  ものと完全に一致すること。手で書くのは大変なので、runner の出力
  （`report` の各行の `orig=` / `mut=`、または `%mutation-string`）から
  そのままコピーする
- `:form`（省略可）: 変異前の定義を `PRIN1` した文字列の部分文字列である
  こと。手がかり用で、無くても照合できる
- `:reason`: 記録のためだけで、照合には使わない

`tools/mutate/exclusions.lisp` には、サンプル（下記）の `clamp` の下限
チェックで実際に見つかった等価変異体が1件、例として入っている。

## サンプル（`tools/mutate/sample/`）

runner 自身を確かめるための、nabla にも nabla-mutate にも依存しない
自己完結したサンプル。

- `nabla-mutate-sample`: `clamp` / `mean`（`src/sample.lisp`）
- `nabla-mutate-sample/weak-tests`: わざと弱いテスト（エラーが出ないことしか
  見ない）
- `nabla-mutate-sample/strong-tests`: 値と境界を確かめる強いテスト

デモ:

```sh
# 弱いテスト: 生き残る変異体がある（mean の `/` → `*` が生き残る）
tools/mutate/run.sh --test-system nabla-mutate-sample/weak-tests \
  --test-form '(lambda () (funcall (find-symbol "RUN-TESTS" "NABLA.MUTATE.SAMPLE.WEAK-TESTS")))' \
  tools/mutate/sample/src/sample.lisp

# 強いテスト: 除外リストにある clamp の等価変異体を除いて全滅する
tools/mutate/run.sh --test-system nabla-mutate-sample/strong-tests \
  --test-form '(lambda () (funcall (find-symbol "RUN-TESTS" "NABLA.MUTATE.SAMPLE.STRONG-TESTS")))' \
  tools/mutate/sample/src/sample.lisp
```

`nabla-mutate/tests` の `RUN-REPORTS-SURVIVORS-WITH-WEAK-TESTS-AND-NONE-WITH-STRONG-TESTS`
テストが、これと同じことを自動テストとして確かめている。

## nabla-mutate 自身のテストを実行する

```sh
CL_SOURCE_REGISTRY="$(pwd)//:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:" \
  sbcl --non-interactive \
    --eval '(require :asdf)' \
    --eval '(asdf:load-system "nabla-mutate/tests")' \
    --eval '(uiop:quit (if (nabla.mutate.tests:run-tests) 0 1))'
```

（`asdf:test-system "nabla-mutate"` でも同じことができる。）

## 既知の制限

- reader マクロのコメント（`;` によるインラインコメント）は、`read` で
  読んだあと再度 `PRIN1` するときに失われる。除外リストの `:mutation` は
  この再印字後の文字列なので、ソースの見た目とは空白や改行が異なる
- `read-source-forms` は対象ファイルを読む間 `*read-eval*` を `NIL` に
  束縛する（変異対象のソースを読み込むだけの目的で `#.` を実際に
  評価したくないため）。そのため `#.`（read-eval）を含むファイルは、
  そこで読み込みを打ち切る（そこより前の定義は対象になる）
- 変異は1つの定義につき1つだけ。同じ定義の中に複数の変異可能な箇所が
  あっても、演算子ごとに最初の1箇所しか試さない
- CFFI のバインディング（`nabla/iree`、`nabla/pjrt` の foreign 関数定義）
  は対象外（`.claude/skills/nabla-testing/references/mutation.md` の
  「3. 対象と除外」を見よ）。ファイルを絞ることで対象から外すこと
- `defmacro` の変異は既に展開・コンパイル済みの呼び出し元には効かない
  （`%evaluate-mutant` は変異させたマクロ定義を再 `eval` するだけで、
  それを使っているコードを再コンパイルはしない）。そのため `defmacro`
  の変異体は、呼び出し元がその後で評価・コンパイルされない限りほぼ
  必ず survived になる。マクロを変異対象から外したいときは
  `*mutable-definition-heads*` から `DEFMACRO` を除く
- `:constant` 演算子は `&optional` / `&key` のデフォルト値
  （`(defun f (x &optional (y 0)) ...)` の `0` など）も置き換える対象にする
- `nabla-mutate/tests` の PBT は、check-it 組み込みの `(integer lo hi)` /
  `(real lo hi)` generator をそのまま使っている。この generator は
  `check-it::*size*`（既定 10）で値をクランプするため、たとえば
  `(integer -50 50)` と書いても実際には -10..10 前後しか生成されない
  （詳しくは
  [`tests/support/uniform-generator.lisp`](../../tests/support/uniform-generator.lisp)
  のコメントと
  [`.claude/skills/nabla-testing/references/properties.md`](../../.claude/skills/nabla-testing/references/properties.md)
  を見よ）。`nabla-mutate` は `nabla` のコアシステムに依存しない独立した
  ツールという設計（`nabla-mutate.asd` 参照）なので、この PR では
  `nabla.tests.support:uniform-integer` / `uniform-real` に依存させず、
  この制限として文書化するだけにとどめる。境界の近くまで広く探索したい
  テストを `nabla-mutate/tests` に足すときは、この制限を踏まえて
  generator を選ぶこと
