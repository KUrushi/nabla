;;;; nabla/data/image のパッケージ（issue #283）。
;;;;
;;;; 画像ファイル（PNG / JPEG）のリストから作るデータソース。デコードは外部のライブラリ
;;;; （PNG は png-read、JPEG は cl-jpeg）に任せる。nabla/data 本体を純 Lisp の軽い層の
;;;; ままにするため、画像のライブラリへの依存はこの別システムに閉じ込める。nabla の
;;;; システムのうち nabla/data にだけ依存する（nabla/nn と実行系のシステムを知らない）。

(defpackage #:nabla.data.image
  (:use #:cl)
  (:export
   #:image-decode-error
   #:image-decode-error-reason
   #:image-decode-error-pathname
   #:image-decode-error-detail
   #:read-image
   #:image-source
   #:make-image-source)
  (:documentation
   "nabla/data/image: PNG / JPEG のデコードと、画像ファイルのリストから作る data-source。"))
