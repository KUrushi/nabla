;;;; diff.lisp -- git diff から変更された行の範囲を取り出す
;;;;
;;;; `git diff <base-ref>...HEAD -U0 -- '*.lisp'` の出力を読み、
;;;; ハンクの `+` 側（新しいファイルの行番号）だけを (pathname start end)
;;;; のリストにする。純粋な削除（追加行が0行）のハンクは対象にしない。

(in-package #:nabla.mutate)

(defun %parse-hunk-header (line)
  "'@@ -a,b +c,d @@' 形式の行から (values new-start new-count) を返す。
'+c' のみ（1行だけの追加）のときは new-count = 1 とみなす。
形式に合わなければ NIL。"
  (let ((plus-pos (search "+" line)))
    (unless plus-pos (return-from %parse-hunk-header nil))
    (let* ((rest (subseq line (1+ plus-pos)))
           (space-pos (position #\Space rest))
           (spec (if space-pos (subseq rest 0 space-pos) rest))
           (comma-pos (position #\, spec)))
      (if comma-pos
          (values (parse-integer spec :end comma-pos)
                  (parse-integer spec :start (1+ comma-pos)))
          (values (parse-integer spec) 1)))))

(defun %diff-lines (base-ref)
  (multiple-value-bind (output error-output exit-code)
      (uiop:run-program (list "git" "diff" "--unified=0"
                               (format nil "~A...HEAD" base-ref)
                               "--" "*.lisp")
                         :output '(:string :stripped nil)
                         :error-output :string
                         :ignore-error-status t)
    (declare (ignore error-output))
    (unless (member exit-code '(0 1))
      (error "git diff failed (exit code ~D)" exit-code))
    (uiop:split-string output :separator '(#\Newline))))

(defun ranges-from-git-diff (&key (base-ref "main"))
  "BASE-REF から HEAD までの間に `.lisp` ファイルで変更された行の範囲を
(pathname start end) のリストとして返す。純粋な削除だけのハンクは含めない。"
  (let ((current-file nil)
        (ranges nil))
    (dolist (line (%diff-lines base-ref))
      (cond
        ((and (>= (length line) 6) (string= line "+++ b/" :end1 6))
         (let ((path (subseq line 6)))
           (setf current-file (unless (string= path "/dev/null")
                                 (uiop:merge-pathnames* path (uiop:getcwd))))))
        ((and current-file (>= (length line) 2) (string= line "@@" :end1 2))
         (multiple-value-bind (start count) (%parse-hunk-header line)
           (when (and start count (> count 0))
             (push (list current-file start (+ start count -1)) ranges))))))
    (nreverse ranges)))
