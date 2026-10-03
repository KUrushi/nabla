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

;;; --- 以下は issue #80。JAX の jax._src.lax.lax の jvp ルールを写す。 ---

(def-jvp-rule sub (primals out tangents)
  (declare (ignore primals out))
  (destructuring-bind (ta tb) tangents
    (cond ((symbolic-zero-p tb) ta)
          ((symbolic-zero-p ta) (%t-neg tb))
          (t (%t-sub ta tb)))))

;; d(x·y) = y·tx + x·ty。係数は主値だけ。
(def-jvp-partials mul
  (lambda (primals out) (declare (ignore out)) (second primals))
  (lambda (primals out) (declare (ignore out)) (first primals)))

;; d(x/y) = tx / y − (x/y / y)·ty。tx は主値 y で割る（線形）。
(def-jvp-rule div (primals out tangents)
  (destructuring-bind (x y) primals
    (declare (ignore x))
    (destructuring-bind (tx ty) tangents
      (add-tangents (if (symbolic-zero-p tx) tx (%t-div tx y))
                    (if (symbolic-zero-p ty)
                        ty
                        (%t-mul (%t-neg (%t-div out y)) ty))))))

(def-jvp-partials exp
  (lambda (primals out) (declare (ignore primals)) out))

;; d log(x) = t / x。
(def-jvp-rule log (primals out tangents)
  (declare (ignore out))
  (%t-div (first tangents) (first primals)))

;; d tanh(x) = (1 − tanh²(x))·t。
(def-jvp-partials tanh
  (lambda (primals out) (declare (ignore primals)) (%t-sub 1 (%t-mul out out))))

(defun %balanced-eq (x z y)
  "JAX の _balanced_eq: X が Z（max / min の出力）と等しい要素で、Y も Z と
等しければ 0.5、そうでなければ 1、X が Z と等しくなければ 0（係数のトレーサ）。"
  (%t-select (%t-compare x z :eq)
             (%t-select (%t-compare y z :eq) (%lift-number 0.5 z) (%lift-number 1 z))
             (%lift-number 0 z)))

(def-jvp-partials max
  (lambda (primals out) (%balanced-eq (first primals) out (second primals)))
  (lambda (primals out) (%balanced-eq (second primals) out (first primals))))

(def-jvp-partials min
  (lambda (primals out) (%balanced-eq (first primals) out (second primals)))
  (lambda (primals out) (%balanced-eq (second primals) out (first primals))))

(def-jvp-rule convert (primals out tangents &key dtype)
  (declare (ignore primals))
  (let ((tangent (first tangents)))
    ;; 入力が整数・:i1 のときの接線はゼロでルールは呼ばれない（jvp-graph）ので、DTYPE だけ見ればよい。
    (if (%float-dtype-p dtype)
        (%trace-eqn :convert (list tangent) :dtype dtype)
        (make-symbolic-zero (tracer-aval out)))))

(def-jvp-rule compare (primals out tangents &key direction)
  (declare (ignore primals tangents direction))
  (make-symbolic-zero (tracer-aval out)))

(def-jvp-rule select (primals out tangents)
  ;; pred（:I1）の接線は常にゼロで無視する。ゼロの枝は値側の dtype で 0 にする。
  (destructuring-bind (pred-tangent on-true on-false) tangents
    (declare (ignore pred-tangent))
    (if (and (symbolic-zero-p on-true) (symbolic-zero-p on-false))
        (make-symbolic-zero (tracer-aval out))
        (%t-select (first primals) (instantiate-zero on-true) (instantiate-zero on-false)))))

(def-jvp-rule stop-gradient (primals out tangents)
  (declare (ignore primals tangents))
  (make-symbolic-zero (tracer-aval out)))

;;; --- 以下は issue #83: 線形なプリミティブの transpose ルール。JAX の
;;; jax._src.lax.lax の _sub_transpose / _convert_element_type_transpose_rule /
;;; _select_transpose_rule / _mul_transpose / _div_transpose を写す。 ---

(def-transpose-rule sub (ct invars)
  ;; 線形な入力にだけ流す。b 側は符号が反転する。
  (destructuring-bind (a b) invars
    (list (and (undefined-primal-p a) ct)
          (and (undefined-primal-p b) (%t-neg ct)))))

(def-transpose-rule convert (ct invars &key dtype)
  ;; 余接線を元の入力の dtype に戻す。
  (let ((from (aval-dtype (undefined-primal-aval (first invars)))))
    (list (if (eq from dtype) ct (%trace-eqn :convert (list ct) :dtype from)))))

(def-transpose-rule select (ct invars)
  ;; pred は既知の主値。線形な値側の入力ごとに、選ばれた側にだけ ct を流す
  ;; （選ばれなかった側は 0）。
  (destructuring-bind (pred on-true on-false) invars
    (when (undefined-primal-p pred)
      (error 'autodiff-error
             :format-control "select の pred は既知の主値でなければならない（線形な入力にはできない）"))
    (let ((zero (instantiate-zero (make-symbolic-zero (tracer-aval ct)))))
      (list nil
            (and (undefined-primal-p on-true) (%t-select pred ct zero))
            (and (undefined-primal-p on-false) (%t-select pred zero ct))))))

(def-transpose-rule mul (ct invars)
  ;; 片側が既知の係数、もう片側が線形。両方が線形なら非線形なのでエラー。
  (destructuring-bind (x y) invars
    (cond ((and (undefined-primal-p x) (undefined-primal-p y))
           (error 'autodiff-error
                  :format-control "mul の両方の入力が線形な入力のとき、転置できない（接線どうしの積）"))
          ((undefined-primal-p x) (list (%t-mul ct y) nil))
          (t (list nil (%t-mul x ct))))))

(def-transpose-rule div (ct invars)
  ;; 線形なのは被除数だけ: d(x / y) の x 側は ct / y。除数が線形ならエラー。
  (destructuring-bind (x y) invars
    (declare (ignore x))
    (when (undefined-primal-p y)
      (error 'autodiff-error
             :format-control "div の除数が線形な入力のとき、転置できない"))
    (list (%t-div ct y) nil)))
