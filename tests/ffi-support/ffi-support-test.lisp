;;;; nabla/ffi-support のテスト（issue #79）。
;;;;
;;;; 完了条件「nabla/iree 以外のシステムから nabla/iree をロードせずに
;;;; これらを使える」を確かめる。このシステムは nabla/iree に依存しない
;;;; ので、ここで本物の外部ライブラリを呼ぶ必要は無く、libc の signal(2) と
;;;; 浮動小数点演算で、保護が効くことだけを見る。IREE を使った統合の
;;;; テストは tests/iree/{compiler,float-traps,finalizer}-test.lisp にある。

(in-package #:nabla.ffi-support.tests)

(test (ffi-support-loads-without-iree :suite :nabla.medium)
  "真っさらな子 SBCL で nabla/ffi-support だけをロードしても、IREE の
パッケージ（NABLA.IREE）は作られず、マクロが使える。（このプロセスには
nabla/iree/tests が nabla/iree をロード済みのことがあるので、子プロセスで
確かめる。環境は %child-source-registry などで明示的に組み立てる。）"
  (let* ((output (make-string-output-stream))
         (process
           (sb-ext:run-program
            "sbcl"
            (list "--non-interactive" "--disable-debugger"
                  "--eval" "(require :asdf)"
                  "--eval" "(asdf:load-system \"nabla/ffi-support\")"
                  "--eval" "(format t \"IREE-PACKAGE=~A IREE-LOADED=~A MACRO=~A~%\"
                              (and (find-package \"NABLA.IREE\") t)
                              (asdf:component-loaded-p \"nabla/iree\")
                              (nabla.ffi-support:with-all-float-traps-masked
                                (nabla.ffi-support:with-lisp-signal-handlers-preserved
                                  :ok)))")
            :search t :output output :error output
            :environment (append (%forward-env-vars *child-sbcl-forwarded-env-vars*)
                                 (list (format nil "CL_SOURCE_REGISTRY=~A"
                                               (%child-source-registry))))))
         (text (get-output-stream-string output)))
    (is (eql 0 (sb-ext:process-exit-code process)) "~A" text)
    (is (search "IREE-PACKAGE=NIL IREE-LOADED=NIL MACRO=OK" text) "~A" text)))

(defun %opaque (x)
  "コンパイラの定数畳み込みを避けるための恒等関数。"
  x)
(declaim (notinline %opaque))

(defun %handler (signo)
  (nabla.ffi-support::%signal-handler-address signo))

(defun %ignore-sigusr1 ()
  (cffi:foreign-funcall "signal" :int sb-unix:sigusr1
                                 :pointer (cffi:make-pointer 1) :pointer))

(test (with-lisp-signal-handlers-preserved/restores-changed-disposition :suite :nabla.medium)
  "本体が SIGUSR1 の処分を SIG_IGN に変えても、抜けるときに元へ戻る。"
  (let ((before (%handler sb-unix:sigusr1)))
    (nabla.ffi-support:with-lisp-signal-handlers-preserved
      (%ignore-sigusr1)
      (is (/= before (%handler sb-unix:sigusr1))))
    (is (= before (%handler sb-unix:sigusr1)))))

(test (with-lisp-signal-handlers-preserved/restores-on-non-local-exit :suite :nabla.medium)
  (let ((before (%handler sb-unix:sigusr1)))
    (ignore-errors
     (nabla.ffi-support:with-lisp-signal-handlers-preserved
       (%ignore-sigusr1)
       (error "boom")))
    (is (= before (%handler sb-unix:sigusr1)))))

(test (call-with-world-stopped/returns-thunk-value-and-rejects-nesting :suite :nabla.medium)
  (is (eql :ok (nabla.ffi-support::%call-with-world-stopped (lambda () :ok))))
  (signals error
    (sb-sys:without-gcing
      (nabla.ffi-support::%call-with-world-stopped (lambda () :unreachable)))))

(test (with-all-float-traps-masked/masks-all-traps :suite :nabla.medium)
  "マスクの中では 0 除算・オーバーフロー・NaN 演算が例外にならず、
外では従来どおり例外になる（トラップの設定が戻る）。"
  (let ((zero (%opaque 0.0))
        (huge (%opaque most-positive-single-float)))
    (nabla.ffi-support:with-all-float-traps-masked
      (is (sb-ext:float-infinity-p (/ 1.0 zero)))
      (is (sb-ext:float-infinity-p (* huge 2.0)))
      (is (sb-ext:float-nan-p (- (/ 1.0 zero) (/ 1.0 zero)))))
    (signals arithmetic-error (%opaque (/ 1.0 zero)))))
