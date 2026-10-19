# check-it 上流に報告するバグ（issue #245、#220）

nabla が固定している check-it（`scripts/setup-lisp-deps.sh` の `b79c9103665be3976915b56b570038f03486e62f`）で見つけたバグ1件の、nabla を使わない最小の再現と、DalekBaldwin/check-it にそのまま貼れる issue の下書き。**上流への報告はユーザーが行う**（このリポジトリの作業では check-it の GitHub に何も投稿しない）。報告したら、「上流 issue」の行にリンクを書き、nabla の issue #245 にも記録し、`tests/support/regression.lisp` の `datum` メソッドの近くのコメントも更新する。

固定コミットを上げる・防御を外す条件は `.claude/skills/nabla-testing/SKILL.md` の「check-it の固定コミットを上げる条件」にある。

## 共通の環境

| 項目 | 値 |
| --- | --- |
| check-it | `b79c9103665be3976915b56b570038f03486e62f`（"version info"、2015-07-05）。2026-10-20 に `git ls-remote origin HEAD` で確かめた時点で、上流の HEAD もこのコミット（修正された版は無い） |
| SBCL | 2.6.7（macOS arm64）。バグは処理系に依存しない（総称関数にメソッドが無いだけ） |
| 確認日 | 2026-10-20 |

## 1. 保存した失敗例が、同じプロセスの次の実行では文字列のまま再生され NO-APPLICABLE-METHOD で落ちる

- **状態**: 未報告（ユーザーの確認待ち）。上流 issue: `<link>`
- **再現ファイル**: [`docs/check-it-repros/string-datum.lisp`](check-it-repros/string-datum.lisp)（`sbcl --non-interactive --load docs/check-it-repros/string-datum.lisp`。防御を入れた対照は末尾に `with-defense`）
- **期待**: 失敗例を保存したあと、同じプロセスで同じ `regression-id` の `check-it` を走らせると、保存した値で再生される（別プロセスでファイルを load し直したときと同じ）
- **実際**: `NO-APPLICABLE-METHOD`（総称関数 `CHECK-IT::DATUM` に文字列のメソッドが無い）で落ち、本当の失敗が隠れる
- **原因**: `src/check-it.lisp` の `save-regression`（130 行目）が `(push string (get regression-id 'regression-cases))` で生の文字列を積む。再生側の `check-it%`（102-104 行目）は積まれたものを `regression-case` オブジェクトとみなして `(datum regression-case)` を呼ぶ。`datum` は `regression-case` クラスの reader（15-17 行目）で、文字列のメソッドは無い。ファイルから load した場合は `regression-case%`（22-31 行目）が `read-from-string` した値を持つ `regression-case` オブジェクトを積むので、問題は同一プロセス内の再生だけで起きる
- **修正案**: 130 行目で `regression-case%` と同じく `regression-case` オブジェクトを積む（`(push (make-instance 'regression-case :name regression-id :datum (read-from-string string) :timestamp (get-universal-time)) ...)`）か、`datum` に文字列のメソッドを足す
- **nabla 側の回避策**: `tests/support/regression.lisp` で `(defmethod check-it::datum ((datum string)) (read-from-string datum))` を定義する（#220）。回帰 id をテストごとに分けること、`regressions/ids-are-unique`（`tests/regressions.lisp`）は経路 (a) の再発を防ぐが、経路 (b) は防げない
- **nabla での再現結果（2026-10-20）**:

  | 経路 | 防御なし | 防御あり |
  | --- | --- | --- |
  | (a) テスト A が失敗して保存 → 同じプロセスで別のテスト B が同じ id で再生 | B が `NO-APPLICABLE-METHOD` | B は通る |
  | (b) 同じテストが失敗して保存 → 同じプロセスでもう一度実行 | 2 回目が `NO-APPLICABLE-METHOD`（失敗が隠れる） | 2 回目は本来の失敗として報告される |
  | 対照: 保存したファイルを load し直してから再生 | 通る（`regression-case` オブジェクトが積まれる） | 通る |

### 下書き（英語）

**Title**: `Failure saved via :regression-file is pushed as a raw string; replaying it in the same process signals NO-APPLICABLE-METHOD on DATUM`

````markdown
### What happened?

When a `check-it` run with `:regression-id` / `:regression-file` finds a failure, `save-regression` pushes the *raw string* onto the `regression-cases` property of the id symbol. If `check-it` is run again in the same image with the same `:regression-id` (for example, re-running the same test from the REPL, or a second test that shares the id), the replay loop treats that entry as a `regression-case` object and calls `(datum regression-case)`. There is no `DATUM` method for strings, so the replay signals `NO-APPLICABLE-METHOD` and the real failure is hidden.

Entries loaded from the regression file are fine: `regression-case%` stores a `regression-case` object whose datum is `(read-from-string datum)`. Only the entry pushed at save time is a string.

### Steps to reproduce

```lisp
(asdf:load-system "check-it")
(defpackage #:repro (:use #:cl #:check-it))
(in-package #:repro)

(defun always-fail (x) (declare (ignore x)) nil)

(let ((file "/tmp/repro-regressions.lisp"))
  (with-open-file (s file :direction :output :if-exists :supersede)
    (format s "(in-package #:repro)~%"))
  (flet ((run ()
           (check-it (generator (integer)) #'always-fail
                     :regression-id my-id :regression-file file)))
    (run)    ; fails and saves; pushes the string "N" onto (get 'my-id 'check-it::regression-cases)
    (run)))  ; => NO-APPLICABLE-METHOD for CHECK-IT::DATUM (called with "N")
```

The same happens when the second run is a different test that uses the same `:regression-id`. Reloading the regression file in a fresh image and running again works.

### Expected behavior

The second run replays the saved value (same as when the file is loaded in a new process) and reports the failure.

### Actual behavior

`NO-APPLICABLE-METHOD` signalled by `CHECK-IT::DATUM` with a string argument.

### Cause

`src/check-it.lisp`, `save-regression` (line 130): `(push string (get regression-id 'regression-cases))`. The replay loop in `check-it%` (lines 102-104) calls `(datum regression-case)` on each entry, but `datum` is only a reader of class `regression-case` (lines 15-17).

### Suggested fix

Push a `regression-case` object at save time, as `regression-case%` does when the file is loaded:

```lisp
(push (make-instance 'regression-case
                     :name regression-id
                     :datum (read-from-string string)
                     :timestamp (get-universal-time))
      (get regression-id 'regression-cases))
```

(or add a `datum` method for strings that calls `read-from-string`).

### Version information

- check-it: b79c9103665be3976915b56b570038f03486e62f (current HEAD)
- SBCL 2.6.7, macOS arm64 (the bug is implementation-independent)
````
