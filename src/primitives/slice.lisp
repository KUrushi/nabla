;;;; slice（静的な範囲の切り出し。issue #242）。stablehlo.slice の strides が全部 1 の形。
;;;;
;;;; jit が backend の入力の上限（BACKEND-MAX-INPUTS）を超える葉を dtype ごとに 1 本の
;;;; 1 次元バッファへ連結したとき、graph の先頭で葉を切り出すのに使う内部プリミティブ
;;;; （export しない。src/jit.lisp の %JIT-UNPACK-LEAVES）。jit の一番外側の graph にだけ
;;;; 現れるので、jvp / transpose / バッチ化のルールは持たない。値を解釈しない構造演算なので、
;;;; shape.lisp の他の演算と同じく %SHAPE-EAGER-FILL で書く。

(in-package #:nabla)

(defprimitive slice (:start :limit)
  :abstract-eval
  (lambda (in-avals &key start limit)
    (%shape-check-arity :slice in-avals 1)
    (%shape-check-shape-param :slice in-avals :start start)
    (%shape-check-shape-param :slice in-avals :limit limit)
    (let* ((in (first in-avals))
           (shape (aval-shape in)))
      (unless (and (= (length start) (length shape)) (= (length limit) (length shape))
                   (every #'<= start limit shape))
        (error 'primitive-error :name :slice :in-avals in-avals
               :format-control "start ~S・limit ~S は形 ~S の範囲（0 <= start <= limit <= 次元）でない"
               :format-arguments (list start limit shape)))
      (make-aval (mapcar #'- limit start) (aval-dtype in))))
  :emit
  (lambda (in-names in-avals out-name out-aval &key start limit)
    (format nil "~A = stablehlo.slice ~A [~{~{~D:~D~}~^, ~}] : (~A) -> ~A"
            out-name (first in-names) (mapcar #'list start limit)
            (tensor-type-string (first in-avals)) (tensor-type-string out-aval)))
  :eager
  (lambda (arrays in-avals &key start limit)
    (let* ((in-shape (aval-shape (first in-avals)))
           (out-shape (mapcar #'- limit start))
           (in-strides (%shape-strides in-shape))
           (out-strides (%shape-strides out-shape)))
      (%shape-eager-fill
       out-shape (aval-dtype (first in-avals))
       (lambda (out-index)
         (%shape-row-major-index
          (mapcar #'+ (%shape-subscripts out-index out-shape out-strides) start)
          in-shape in-strides))
       (first arrays)))))
