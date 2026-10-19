;;;; check-it の回帰データ (文字列のまま積まれる) の最小再現。
;;;; 使い方（docs/upstream-bugs-check-it.md）:
;;;;   sbcl --non-interactive --load docs/check-it-repros/string-datum.lisp [with-defense]
;;;; with-defense を付けると nabla の防御（tests/support/regression.lisp の datum の文字列メソッド）を入れる。
;;;; 作業ファイルは一時ディレクトリに作る。
;;;; check-it: DalekBaldwin/check-it b79c9103665be3976915b56b570038f03486e62f
(require :asdf)
(asdf:initialize-source-registry
 `(:source-registry (:tree ,(namestring (merge-pathnames ".local/share/nabla/lisp-deps/" (user-homedir-pathname)))) :inherit-configuration))
(asdf:load-system "check-it")
(defpackage #:repro (:use #:cl #:check-it))
(in-package #:repro)

(defvar *with-defense* (member "with-defense" sb-ext:*posix-argv* :test #'string=))
(when *with-defense*
  (defmethod check-it::datum ((datum string)) (read-from-string datum)))

(defvar *dir* (merge-pathnames "check-it-repro/" (uiop:temporary-directory)))
(defun fresh-file (name)
  (let ((p (merge-pathnames (format nil "~A.lisp" name) *dir*)))
    (ensure-directories-exist p)
    (with-open-file (s p :direction :output :if-exists :supersede)
      (format s "(in-package #:repro)~%"))
    p))

(defun always-fail (x) (declare (ignore x)) nil)
(defun always-pass (x) (declare (ignore x)) t)
(defvar *check-it-output* (make-broadcast-stream))

(defun run-check (gen fn id file)
  "check-it% を直接呼ぶ (マクロ展開と同じ)。(values 結果 エラー) を返す。"
  (handler-case
      (let ((check-it::*check-it-output* (make-broadcast-stream)))
        (values (check-it::check-it% 'fn gen fn :regression-id id :regression-file file) nil))
    (error (e) (values :signaled (type-of e)))))

(defun report (label thunk)
  (multiple-value-bind (r e) (funcall thunk)
    (format t "~&~A: result=~S ~@[error=~S~]~%" label r e)))

(format t "~&== defense: ~:[absent~;present~] ==~%" *with-defense*)

;; 経路 (a): テスト A が失敗して保存、同じプロセスで別のテスト B が同じ id で再生
(let ((file (fresh-file "a")) (gen (generator (integer))))
  (report "(a) test A fails and saves" (lambda () (run-check gen #'always-fail 'repro::shared-id file)))
  (format t "~&    plist entry type: ~S~%" (type-of (first (get 'repro::shared-id 'check-it::regression-cases))))
  (report "(a) test B replays same id " (lambda () (run-check gen #'always-pass 'repro::shared-id file))))

;; 経路 (b): 同じテストが失敗 → 同じプロセスでもう一度実行
(let ((file (fresh-file "b")) (gen (generator (integer))))
  (report "(b) run 1 (fails, saves)  " (lambda () (run-check gen #'always-fail 'repro::own-id file)))
  (report "(b) run 2 (same test again)" (lambda () (run-check gen #'always-fail 'repro::own-id file))))

;; 対照: 別のプロセスの想定 (ファイルを load し直す) では文字列は積まれない
(let ((file (fresh-file "c")) (gen (generator (integer))))
  (setf (get 'repro::fresh-id 'check-it::regression-cases) nil)
  (run-check gen #'always-fail 'repro::fresh-id file)
  (setf (get 'repro::fresh-id 'check-it::regression-cases) nil)
  (load file)
  (format t "~&(control) after reload from file: entry type ~S~%"
          (type-of (first (get 'repro::fresh-id 'check-it::regression-cases))))
  (report "(control) replay after reload" (lambda () (run-check gen #'always-pass 'repro::fresh-id file))))
