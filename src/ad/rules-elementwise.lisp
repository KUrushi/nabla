;;;; ad/rules-elementwise: 要素ごとのプリミティブの jvp ルール（issue #77、77c。
;;;; #80 がこのファイルに他の要素演算を足す）。
;;;;
;;;; ルールを書く人への制約: 接線は、その被演算子について線形なプリミティブ
;;;; にしか流さない。接線どうしの積や、接線への exp / log / tanh / max / min /
;;;; compare / reduce-max は禁止。mul / div / dot-general は片側だけが接線に
;;;; なるように書く（係数は主値だけから作る）。transpose（#78）はルールが
;;;; 出す接線側の eqn をこの線形性に頼って逆向きにたどるため。
;;;; ルールは配列を計算せず、現在のトレースに eqn を足すコードとして書く。

(in-package #:nabla)

(def-jvp-rule add (primals out tangents)
  (declare (ignore primals out))
  (add-tangents (first tangents) (second tangents)))

(def-jvp-rule neg (primals out tangents)
  (declare (ignore primals out))
  (%t-neg (first tangents)))

;;; transpose ルール（issue #82。add-tangents が足す add と、jvp ルールの neg を
;;; 転置できるようにするため、ここに置く。#83 は add / neg を扱わない）。

(def-transpose-rule add (ct invars)
  ;; 線形な入力にだけ ct をそのまま流す（既知の入力が混ざってもよい。JAX と同じ）。
  (mapcar (lambda (v) (and (undefined-primal-p v) ct)) invars))

(def-transpose-rule neg (ct invars)
  (declare (ignore invars))
  (list (%t-neg ct)))
