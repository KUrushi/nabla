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

(defun gc-and-run-finalizers ()
  "(sb-ext:gc :full t) してから (sb-kernel:run-pending-finalizers) する
（#11）。SBCL 2.2.9 は finalizer を別スレッド（finalizer thread）で
非同期に実行するため、gc :full t の直後に確認しても、ほとんどの
finalizer はまだ実行されていない（この環境での計測では約3%）。
sb-kernel:run-pending-finalizers はキューに溜まった finalizer を
呼び出しスレッドで同期的に実行するので、これを続けて呼ぶことで
テストから確実に観測できる。それでも保守的なスタックルート（GC が
「もしかしたらポインタかもしれない」ビット列をルートとして残すこと）
のせいで、run-pending-finalizers まで呼んでも少数のオブジェクトが
実行されずに生き残ることがある（この環境での計測では 10000 個中 1個
程度、つまり 9999 個は実行された）。finalizer 系のテストは、この
わずかな残留を許容する形で許容量を設定すること（0 で比較しない）。"
  (sb-ext:gc :full t)
  (sb-kernel:run-pending-finalizers))

(defmacro with-device-arrays ((&rest bindings) &body body)
  "BINDINGS の各 (VAR FORM) を、書いた順に評価して VAR に束縛し、BODY を
評価してから、束縛した順とは逆順に release-device-array する（テスト専用の
ヘルパー。finalizer は #11 まで無いので、ここで明示的に解放しないと
テストごとに IREE のバッファがリークする）。BODY やどれかの FORM が
非局所脱出しても、それまでに束縛が済んだ VAR はすべて解放する。"
  (let ((vars (mapcar #'first bindings)))
    `(let ,(mapcar (lambda (var) (list var nil)) vars)
       (unwind-protect
            (progn
              ,@(mapcar (lambda (binding) `(setf ,(first binding) ,(second binding)))
                        bindings)
              ,@body)
         ,@(mapcar (lambda (var) `(when ,var (release-device-array ,var)))
                   (reverse vars))))))

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
