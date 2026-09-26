;;;; nabla-mutate.asd -- 自前の mutation testing runner
;;;;
;;;; nabla のコアシステムには依存しない。独立に読み込み・実行できる。

(defsystem "nabla-mutate"
  :description "nabla 用の最小限の mutation testing runner"
  :author "nabla contributors"
  :license "MIT"
  :depends-on ("alexandria" "uiop")
  :in-order-to ((test-op (test-op "nabla-mutate/tests")))
  :pathname "src"
  :components ((:file "package")
               (:file "reader" :depends-on ("package"))
               (:file "mutate" :depends-on ("package"))
               (:file "diff" :depends-on ("package"))
               (:file "exclusions" :depends-on ("package"))
               (:file "runner" :depends-on ("package" "reader" "mutate" "diff" "exclusions"))))

(defsystem "nabla-mutate/tests"
  :description "nabla-mutate 自身のテスト"
  :depends-on ("nabla-mutate" "fiveam" "check-it")
  :pathname "tests"
  :components ((:file "package")
               (:file "reader-tests" :depends-on ("package"))
               (:file "mutate-tests" :depends-on ("package"))
               (:file "diff-tests" :depends-on ("package"))
               (:file "runner-tests" :depends-on ("package")))
  :perform (test-op (op c)
             (unless (uiop:symbol-call "NABLA.MUTATE.TESTS" "RUN-TESTS")
               (error "nabla-mutate/tests failed"))))
