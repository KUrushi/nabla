;;;; nabla コアパッケージ。
;;;;
;;;; フェーズ0（実行系との疎通）の時点での公開 API は、dtype と aval（issue #7）
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
   #:array-aval
   ;; backend プロトコル（issue #9）
   #:backend
   #:make-backend
   #:find-backend
   #:backend-target
   #:backend-fingerprint
   #:backend-compile
   #:backend-load
   #:backend-unload
   #:backend-invoke
   #:to-device
   #:to-host
   #:device-array-aval
   #:backend-error
   #:backend-not-available
   #:backend-not-available-kind
   ;; :i1 dtype と TO-DEVICE の未対応 dtype（issue #37、u3）
   #:unsupported-dtype
   #:unsupported-dtype-dtype
   ;; vmfb ディスクキャッシュ（issue #10）
   #:*compile-cache-directory*
   ;; IR と defprimitive（issue #29、u1a）
   #:defprimitive
   #:primitive-name
   #:var-aval
   #:eqn-prim
   #:eqn-params
   #:eqn-invars
   #:eqn-outvars
   #:graph-invars
   #:graph-eqns
   #:graph-outvars
   #:graph-constants
   #:unknown-primitive
   #:unknown-primitive-name
   #:primitive-error
   ;; graph の印字と読み込み（issue #29、u1b）
   #:print-graph)
  (:documentation
   "nabla のコアパッケージ。JAX 相当のトレース・IR・変換（jit / grad / vmap）を持つ。
ニックネームは NB。フェーズ0時点の公開シンボルは dtype・aval・backend
プロトコルのみ。"))
