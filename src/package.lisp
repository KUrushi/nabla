;;;; nabla コアパッケージ。
;;;;
;;;; フェーズ0（IREE 疎通）の時点では、コアの公開 API はまだない。
;;;; トレース・IR・プリミティブは後続のフェーズで追加する。

(defpackage #:nabla
  (:nicknames #:nb)
  (:use #:cl)
  (:documentation
   "nabla のコアパッケージ。JAX 相当のトレース・IR・変換（jit / grad / vmap）を持つ。
ニックネームは NB。フェーズ0時点では公開シンボルはまだない。"))
