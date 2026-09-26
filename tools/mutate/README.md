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
  `NABLA.TESTS.SUPPORT:RUN-TESTS` を実行時に探すので、#4（テスト基盤）が
  マージされていないと動かない。それまでは `nabla` 以外を対象にするときも
  含め、`--test-system` と `--test-form` を渡すこと）

## 終了コード

mutation score†（`(殺した数 + タイムアウト数) / (全数 - 除外数)`）が
0.8 以上なら 0、そうでなければ 1。殺せる変異体が1つもない
（全数と除外数が同じ）ときは 1 として扱う。

## 変異演算子

`.claude/skills/nabla-testing/references/mutation.md` の「2. 変異演算子」
にある表のうち、最初の4つ（算術演算子の入れ替え・比較の境界・定数の置き換え・
`if` の分岐の入れ替え）を実装している。残りはフェーズ1以降で追加する。

この最小版では、1つの変異可能な定義（`defun` / `defmethod` / `defmacro` /
`defprimitive`）につき、演算子を上の順で試して最初に適用できたものを
1つだけ使う。将来、行ごとの粒度に細かくする余地がある。

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
- `#.`（read-eval）などファイルの残りを読み進められない reader マクロを
  含むファイルは、そこで読み込みを打ち切る（そこより前の定義は対象になる）
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
