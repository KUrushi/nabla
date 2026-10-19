;;;; IREE 3.11 の local-task device は、入力が 162 個以上の関数を呼ぶと NULL を参照して
;;;; プロセスごと落ちる（issue #199、docs/iree-upstream-bugs.md の 6）。nabla.iree はその呼び出しを
;;;; 拒否する（iree-too-many-inputs）。守らせる性質:
;;;; - local-task で入力 170 個の jit した関数を呼ぶと、落ちずに iree-too-many-inputs を signal する。
;;;; - 境界の 161 個は動く。
;;;; - local-sync device（make-backend :iree :driver :local-sync）なら 170 個でも動き、結果は eager と一致する。
;;;; - バグが IREE に残っていること（直ったら失敗する。検査と回避策を外す合図）: 最小の再現
;;;;   （docs/iree-repros/）を iree-run-module で local-task に流すと 162 個で落ち、161 個は通る。
;;;;   落ちる呼び出しはテストプロセスの外（子プロセス）で行う。

(in-package #:nabla.iree.tests)

(defun %input-limit-arrays (n)
  (loop for i below n collect (make-random-array (make-array-spec '(3) :f32) :seed i)))

(defun %input-limit-jit-call (backend n)
  "N 個の rank 1 の配列のリストを、葉ごとに tanh(x+x) する関数を BACKEND で jit して呼んだ結果。"
  (let* ((f (pytree-map-function))
         (jf (nb:jit f :backend backend))
         (arrays (%input-limit-arrays n)))
    (unwind-protect
         (values (funcall jf arrays) (funcall f arrays))
      (nb::%jit-cache-forget f))))

(define-iree-test input-limit/local-task-refuses-170-inputs-instead-of-crashing
    "local-task device で入力 170 個の jit した関数を呼ぶと、iree-too-many-inputs を signal する
（プロセスは落ちない）。エラーは個数と上限を持つ。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local))
        (nb:*compile-cache-directory* nil))
    (handler-case (progn (%input-limit-jit-call backend 170)
                         (fail "iree-too-many-inputs にならなかった"))
      (nabla.iree:iree-too-many-inputs (condition)
        (is (= 170 (nabla.iree:iree-too-many-inputs-count condition)))
        (is (= nabla.iree:+local-task-max-inputs+ (nabla.iree:iree-too-many-inputs-limit condition)))
        (is (search "local-sync" (princ-to-string condition)))))
    (gc-and-run-finalizers)))

(define-iree-test input-limit/local-task-named-by-string-also-refuses-170-inputs
    "ドライバを文字列 \"local-task\" で指定した backend でも、入力 170 個は iree-too-many-inputs に
なる（キーワードでない指定で検査をすり抜けてプロセスが落ちないこと）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local :driver "local-task"))
        (nb:*compile-cache-directory* nil))
    (signals nabla.iree:iree-too-many-inputs (%input-limit-jit-call backend 170))
    (gc-and-run-finalizers)))

(define-iree-test input-limit/local-task-runs-161-inputs
    "境界の入力 161 個は local-task でも動き、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local))
        (nb:*compile-cache-directory* nil))
    (multiple-value-bind (result expected) (%input-limit-jit-call backend 161)
      (is (= 161 (length result)))
      (is (every (lambda (r e) (allclose r e :dtype :f32)) result expected)))
    (gc-and-run-finalizers)))

(define-iree-test input-limit/local-sync-runs-170-inputs
    "local-sync device なら入力 170 個の関数も動き、eager と一致する（回避策）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local :driver :local-sync))
        (nb:*compile-cache-directory* nil))
    (multiple-value-bind (result expected) (%input-limit-jit-call backend 170)
      (is (= 170 (length result)))
      (is (every (lambda (r e) (allclose r e :dtype :f32)) result expected)))
    (gc-and-run-finalizers)))

(defun %iree-bin (name)
  (namestring (merge-pathnames (format nil "bin/~A" name) (nabla.iree::iree-home))))

(defun %run-repro-on-local-task (file)
  "docs/iree-repros/FILE を iree-compile してから iree-run-module --device=local-task で
子プロセスとして実行し、終了コードを返す。入力の個数は main の引数の個数から数える。"
  (let* ((source (namestring (asdf:system-relative-pathname "nabla" (format nil "docs/iree-repros/~A" file))))
         (vmfb (namestring (uiop:tmpize-pathname #p"/tmp/nabla-input-limit.vmfb")))
         (inputs (with-open-file (stream source)
                   (loop for line = (read-line stream)
                         when (search "func.func @main" line)
                           return (count #\: line :end (position #\) line)))))
         (compile-code
           (nth-value 2 (uiop:run-program
                         (list (%iree-bin "iree-compile")
                               "--iree-hal-target-device=local"
                               "--iree-hal-local-target-device-backends=llvm-cpu"
                               source "-o" vmfb)
                         :ignore-error-status t :output nil :error-output nil))))
    (unwind-protect
         (progn
           (unless (zerop compile-code) (return-from %run-repro-on-local-task :compile-failed))
           (nth-value 2 (uiop:run-program
                         (append (list (%iree-bin "iree-run-module")
                                       "--device=local-task" (format nil "--module=~A" vmfb)
                                       "--function=main")
                                 (loop repeat inputs collect "--input=3xf32=1,2,3"))
                         :ignore-error-status t :output nil :error-output nil)))
      (ignore-errors (delete-file vmfb)))))

(define-iree-test input-limit/iree-known-bug-local-task-162-inputs-crashes
    "IREE 3.11 のバグが残っていること: 入力 162 個の最小の再現は local-task で落ち（終了コード 128
超）、161 個は通る。直ると失敗する——そのときは +local-task-max-inputs+ の検査
（src/iree/execute.lisp）と docs/iree-upstream-bugs.md の 6 を外す。"
  (skip-unless-iree :library :both)
  (unless (and (probe-file (%iree-bin "iree-run-module"))
               (probe-file (%iree-bin "iree-compile")))
    (skip "iree-compile / iree-run-module が無い")
    (return-from iree-test))
  (is (eql 0 (%run-repro-on-local-task "local-task-161-inputs.mlir")))
  (is (< 128 (%run-repro-on-local-task "local-task-162-inputs.mlir"))))
