;;;; nabla/iree/tests が共有するテスト部品。
;;;;
;;;; skip-unless-iree: IREE の共有ライブラリが無い環境ではテストをスキップし
;;;; (CI では NABLA_REQUIRE_IREE=1 で失敗にする)、あるときは何もしない。
;;;; stablehlo-fixture: tests/fixtures/stablehlo/ 配下の手書き StableHLO を
;;;; 文字列として読む。*vmfb-magic*: vmfb が実際に先頭に持つ ZIP の
;;;; local-file-header シグネチャ（フラットバッファの識別子ではない。
;;;; IREE は既定で「polyglot zip」形式の vmfb を出す）。

(in-package #:nabla.iree.tests)

(defparameter *vmfb-magic* #(#x50 #x4B #x03 #x04)
  "IREE が出す vmfb（polyglot zip）の先頭4バイト（ZIP local-file-header
シグネチャ \"PK\\3\\4\"）。フラットバッファ自体の識別子ではない。")

(defmacro define-iree-test (name docstring &body body)
  "fiveam:test 相当だが、本体を (block iree-test ...) でくるみ、常に
:nabla.medium スイートに登録する。skip-unless-iree はこの block から
return-from して、IREE が無い環境ではテストの残りを実行しない。

このファイルの中の IREE を使うテストは、fiveam:test の代わりに必ずこちらを
使う。スイートを :nabla.medium 固定にしているのは、ASDF がこのシステムを
ロードする過程でファイルをまたいで fiveam::*suite* の値が（このファイルの
in-suite を経由せずに）変わることがあり、周囲の *suite* に頼ると登録先の
スイートが不安定になるため。"
  `(fiveam:test (,name :suite :nabla.medium) ,docstring
     (block iree-test
       ,@body)))

(defmacro skip-unless-iree (&key (library :both))
  "(iree-available-p :library LIBRARY) が偽なら、NABLA_REQUIRE_IREE 環境変数が
設定されていれば fiveam:fail で失敗させ、無ければ fiveam:skip でこのテストを
スキップして呼び出し元の test 本体（define-iree-test の block）から return
する。真ならなにもしない。"
  `(unless (iree-available-p :library ,library)
     (if (let ((value (sb-ext:posix-getenv "NABLA_REQUIRE_IREE")))
           (and value (plusp (length value))))
         (fiveam:fail "IREE libraries required but not found under ~A" (nabla.iree::iree-home))
         (progn
           (fiveam:skip "IREE libraries not found under ~A (set NABLA_IREE_HOME or run scripts/build-iree.sh)"
                        (nabla.iree::iree-home))
           (return-from iree-test)))))

(defun stablehlo-fixture (name)
  "tests/fixtures/stablehlo/NAME.mlir の内容を文字列として返す。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/~A.mlir" name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (let ((count (read-sequence text stream)))
          (subseq text 0 count))))))

;;; tests/regressions/ は nabla/tests の tests/regressions.lisp が全ファイルを
;;; load しているが、そちらは nabla/iree/tests に依存しない（システム構成が
;;; 独立している）ため、nabla/tests を単体でロードする時点では
;;; nabla.iree.tests パッケージがまだ無く、この iree-*.lisp のファイルは
;;; 読み飛ばされる（tests/regressions.lisp 参照）。そのため、この
;;; パッケージ自身の regression ファイル（"iree-" で始まる名前。
;;; regression-path の呼び出し側の慣習）は、ここで自分でロードし直す。
(dolist (path (sort (uiop:directory-files
                      (asdf:system-relative-pathname "nabla" "tests/regressions/")
                      "iree-*.lisp")
                     #'string<
                     :key #'namestring))
  (load path))
