;;;; with-temporary-directory: テストが使い捨てのディレクトリを作って、
;;;; 終わったら消すためのマクロ（issue #10 のディスクキャッシュのテストで
;;;; ~/.cache に触らないために使う）。

(in-package #:nabla.tests.support)

(defmacro with-temporary-directory ((var) &body body)
  "(UIOP:TEMPORARY-DIRECTORY) の下に nabla-test-<乱数>/ という名前の
ディレクトリを作って VAR に束縛し、BODY を評価してから UNWIND-PROTECT で
そのディレクトリを丸ごと削除する（UIOP:DELETE-DIRECTORY-TREE、既に
無くなっていてもエラーにしない）。"
  (let ((dir (gensym "DIR")))
    `(let ((,dir (uiop:ensure-directory-pathname
                  (merge-pathnames
                   (format nil "nabla-test-~D/" (random most-positive-fixnum (make-random-state t)))
                   (uiop:temporary-directory)))))
       (ensure-directories-exist ,dir)
       (unwind-protect
            (let ((,var ,dir))
              ,@body)
         (uiop:delete-directory-tree ,dir :validate t :if-does-not-exist :ignore)))))
