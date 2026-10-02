;;;; nabla のロードで警告が出ないこと（issue #116）。
;;;;
;;;; 同名の関数を2か所で defun すると、ロードのたびに
;;;; 「redefining NABLA::FOO in DEFUN」の STYLE-WARNING が出る。ロード順で
;;;; 振る舞いが変わる危険の目印なので、真っさらな子 SBCL で nabla を
;;;; ロードして出力に REDEFINING が無いことを確かめる。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(test load/nabla-emits-no-redefinition-warning
  "子 SBCL で nabla を強制再ロードしても、出力に「redefining NABLA...」が現れない。"
  (let* ((output (make-string-output-stream))
         (process
           (sb-ext:run-program
            "sbcl"
            (list "--non-interactive" "--disable-debugger"
                  "--eval" "(require :asdf)"
                  "--eval" "(asdf:load-system \"nabla\" :force '(\"nabla\"))"
                  "--eval" "(format t \"LOAD-DONE~%\")")
            :search t :output output :error output
            :environment (append (%forward-env-vars *child-sbcl-forwarded-env-vars*)
                                 (list (format nil "CL_SOURCE_REGISTRY=~A"
                                               (%child-source-registry))))))
         (text (get-output-stream-string output)))
    (is (eql 0 (sb-ext:process-exit-code process)) "~A" text)
    (is (search "LOAD-DONE" text) "~A" text)
    ;; 依存ライブラリ（ironclad の BLOCK-LENGTH など）自身の警告は対象外。
    (is (null (search "REDEFINING NABLA" (string-upcase text))) "~A" text)))
