;;;; PJRT クライアントの作成が SBCL のシグナルハンドラ・GC と両立すること
;;;; （issue #85）。真っさらな子 SBCL プロセスで確かめる（このプロセス自身では
;;;; 他のテストが既にプラグインをロード済みで、「初回」の挙動を見られない）。

(in-package #:nabla.pjrt.tests)

(defparameter *signal-check-script*
  '("(require :asdf)"
    "(asdf:load-system \"nabla/pjrt\")"
    ;; 全シグナル（1..64）の処分を、ロード・クライアント作成・転送・GC の
    ;; 前後で比べる。別スレッドが GC を回し続ける中で往復させる（GC の間に
    ;; 少し眠る。眠らない tight loop だと、SBCL の GC ロックが公平でないため
    ;; 他のスレッドが進めなくなる餓死が起きうる。src/ffi-support/signals.lisp
    ;; の「残る課題」3）。
    "(defvar *t0* (get-internal-real-time))"
    "(let* ((handlers (lambda () (loop for s from 1 below 65
                                      collect (nabla.ffi-support::%signal-handler-address s))))
           (before (funcall handlers))
           (stop nil)
           (gc-thread (sb-thread:make-thread
                       (lambda () (loop until stop do (sb-ext:gc :full t) (sleep 0.01)))))
           (backend (nabla:make-backend :pjrt))
           (x (make-array '(2 2) :element-type 'single-float
                                 :initial-contents '((1.0 2.0) (3.0 4.0)))))
      (dotimes (i 200)
        (let ((y (nabla:to-device x backend)))
          (unless (equalp x (nabla:to-host y)) (sb-ext:exit :code 3 :abort t))
          (nabla.pjrt:release-device-array y)))
      (setf stop t)
      (sb-thread:join-thread gc-thread)
      (dotimes (i 5) (sb-ext:gc :full t))
      (format t \"ELAPSED-MS=~D~%\" (round (* 1000 (- (get-internal-real-time) *t0*))
                                           internal-time-units-per-second))
      (format t \"CHANGED=~S~%\"
              (loop for b in before for a in (funcall handlers) for s from 1
                    unless (eql a b) collect s))
      (sb-ext:exit :code 0))"))

(defun %run-signal-check-child ()
  (let* ((args (list* "--non-interactive" "--disable-debugger"
                      (loop for form in *signal-check-script* append (list "--eval" form))))
         (env (append (%forward-env-vars (list* "NABLA_PJRT_HOME" *child-sbcl-forwarded-env-vars*))
                      (list (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))
         (output (make-string-output-stream))
         (process (sb-ext:run-program "timeout" (list* "-k" "5" "240" "sbcl" args)
                                      :search t :environment env
                                      :output output :error output)))
    (values (sb-ext:process-exit-code process) (get-output-stream-string output))))

(define-pjrt-test client/create-leaves-signal-handlers-alone-and-survives-gc
  "真っさらな子プロセスで、プラグインのロード・PJRT_Plugin_Initialize・
PJRT_Client_Create・to-device / to-host の往復を、別スレッドが
sb-ext:gc :full t を回し続ける中で行っても、(1) 全シグナルの処分が1つも
変わらない（LLVM のシグナルハンドラ登録が起きない。SIGUSR2 が奪われると
\"no SP known for thread\" で落ちる）、(2) プロセスが落ちずに終了する。
CPU プラグインのこの範囲では登録が起きないことを実地に確かめたので、
IREE の ensure-compiler-loaded のような「世界を止めた1点での登録」は要らない
（src/pjrt/client.lisp 冒頭）。コンパイル・ロード・実行を含む検査は tests/pjrt/executable-test.lisp。"
  (skip-unless-pjrt :kind :cpu)
  (multiple-value-bind (exit-code output) (%run-signal-check-child)
    (is (= 0 exit-code) "child exited ~D (124 = timeout), output:~%~A" exit-code output)
    (is (search "CHANGED=NIL" output)
        "signal dispositions changed, output:~%~A" output)))
