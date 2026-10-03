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
   #:print-graph
   ;; graph の eager 評価（issue #39、e0）
   #:eval-graph
   #:graph-input-mismatch
   #:primitive-not-evaluable
   #:primitive-not-evaluable-name
   ;; トレーサ（issue #32、t1）
   #:with-tracing
   #:trace-to-graph
   #:traceable-function
   #:unsupported-form
   #:unsupported-form-form
   #:unsupported-form-path
   #:tracing-error
   ;; 自動微分のコンディション（issue #77、77a）
   #:autodiff-error
   #:no-jvp-rule
   #:no-jvp-rule-name
   #:no-transpose-rule
   #:no-transpose-rule-name
   ;; grad / value-and-grad（issue #86）
   #:grad
   #:value-and-grad
   #:grad-requires-scalar-output
   #:grad-requires-scalar-output-aval
   ;; if を select に、配列レベルの公開 API（issue #32、t2）
   #:dot
   #:reshape
   #:transpose
   #:broadcast-in-dim
   #:reduce-sum
   #:reduce-max
   #:convert
   #:where
   #:stop-gradient
   ;; StableHLO テキスト emitter（issue #33、wave 3 s1）
   #:emit-stablehlo
   #:primitive-not-emittable
   #:primitive-not-emittable-name
   ;; jit とインメモリのコンパイルキャッシュ（issue #34、wave 4 j1）
   #:jit
   #:jit-error
   #:*default-backend*
   ;; defjit、compile-error のリスタート、end-to-end jit テスト（issue #34、wave 4 j2）
   #:defjit
   #:jit-compile-error
   #:jit-compile-error-condition
   #:jit-compile-error-graph
   #:jit-compile-error-eqn
   #:jit-compile-error-eqn-index
   #:use-eager
   #:recompile

   ;; フェーズ3 anchor: issue #127（サブグラフを持つ eqn）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #126（整数 dtype）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #125（vmap の骨格）。この issue のexportはこの下に足す
   #:vmap
   #:vmap-error
   #:no-batch-rule
   #:no-batch-rule-name



   ;; フェーズ3 anchor: issue #128（要素演算のバッチ化ルール）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #129（形状演算・縮約・dot-general のバッチ化ルール）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #130（cond プリミティブ）。この issue のexportはこの下に足す
   #:cond*
   #:cond-error



   ;; フェーズ3 anchor: issue #131（while-loop プリミティブ）。この issue のexportはこの下に足す
   #:while-loop
   #:while-loop-error
   #:while-loop-argument-error
   #:while-loop-carry-mismatch
   #:while-loop-carry-mismatch-expected
   #:while-loop-carry-mismatch-actual
   #:while-loop-condition-error



   ;; フェーズ3 anchor: issue #132（scan プリミティブ）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #133（rng-bit-generator）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #134（cond / while-loop の jvp）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #135（scan の jvp）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #136（PRNG の公開 API）。この issue のexportはこの下に足す
   #:prng-key
   #:split
   #:fold-in
   #:uniform
   #:normal
   #:prng-error



   ;; フェーズ3 anchor: issue #137（dotimes / loop を scan に展開）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #138（per-example 勾配）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #139（scan の linearize と transpose）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #140（制御構造のバッチ化ルール）。この issue のexportはこの下に足す



   ;; フェーズ3 anchor: issue #141（RNN の e2e）。この issue のexportはこの下に足す
   )
  (:documentation
   "nabla のコアパッケージ。JAX 相当のトレース・IR・変換（jit / grad / vmap）を持つ。
ニックネームは NB。公開シンボルの一覧は README.md の「公開 API」を正とする。"))
