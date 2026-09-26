;;;; nabla コアパッケージ。
;;;;
;;;; フェーズ0（IREE 疎通）の時点での公開 API は、dtype と aval（issue #7）
;;;; だけ。トレース・IR・プリミティブは後続のフェーズで追加する。

(defpackage #:nabla
  (:nicknames #:nb)
  (:use #:cl)
  (:export
   ;; dtype
   #:dtype
   #:dtype-element-type
   #:dtype-byte-width
   #:array-dtype
   #:dtype-mismatch
   #:dtype-mismatch-element-type
   #:dtype-mismatch-dtype
   ;; aval
   #:aval
   #:make-aval
   #:aval-p
   #:aval-shape
   #:aval-dtype
   #:aval-rank
   #:aval-size
   #:aval-byte-length
   #:array-aval)
  (:documentation
   "nabla のコアパッケージ。JAX 相当のトレース・IR・変換（jit / grad / vmap）を持つ。
ニックネームは NB。フェーズ0時点の公開シンボルは dtype と aval のみ。"))
