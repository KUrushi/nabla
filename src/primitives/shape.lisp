;;;; shape: reshape / broadcast-in-dim / transpose プリミティブ（issue #31 p4）。
;;;;
;;;; 3つとも値を解釈しない構造だけの演算なので、eager 実装は
;;;; %SHAPE-EAGER-FILL（shape-common.lisp）に「出力の row-major index から
;;;; 入力の row-major index を計算する関数」を渡すだけで書ける。

(in-package #:nabla)

;;; reshape

(defun %reshape-abstract-eval (in-avals &key shape)
  (%shape-check-arity :reshape in-avals 1)
  (unless (and (listp shape) (every (lambda (d) (typep d '(integer 0))) shape))
    (error 'primitive-error :name :reshape :in-avals in-avals
           :format-control "shape は非負整数のリストでなければならない: ~S"
           :format-arguments (list shape)))
  (let* ((in (first in-avals))
         (out-size (reduce #'* shape :initial-value 1)))
    (unless (= (aval-size in) out-size)
      (error 'primitive-error :name :reshape :in-avals in-avals
             :format-control "要素数が一致しない: ~S（要素数 ~S） → ~S（要素数 ~S）"
             :format-arguments (list (aval-shape in) (aval-size in) shape out-size)))
    (make-aval shape (aval-dtype in))))

(defun %reshape-emit (in-names in-avals out-name out-aval &key shape)
  (declare (ignore shape))
  (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
          out-name (first in-names) (tensor-type-string (first in-avals)) (tensor-type-string out-aval)))

(defun %reshape-eager (arrays in-avals &key shape)
  ;; reshape は row-major の要素列をそのまま保つ。出力・入力どちらも
  ;; 同じ row-major 順序で並んでいるので、出力の row-major index i は
  ;; そのまま入力の row-major index i になる。
  (%shape-eager-fill shape (aval-dtype (first in-avals)) #'identity (first arrays)))

(defprimitive reshape (:shape)
  :abstract-eval #'%reshape-abstract-eval
  :emit #'%reshape-emit
  :eager #'%reshape-eager)

;;; broadcast-in-dim

(defun %broadcast-in-dim-abstract-eval (in-avals &key shape dims)
  (%shape-check-arity :broadcast-in-dim in-avals 1)
  (let* ((in (first in-avals))
         (in-shape (aval-shape in))
         (out-rank (length shape)))
    (unless (= (length dims) (length in-shape))
      (error 'primitive-error :name :broadcast-in-dim :in-avals in-avals
             :format-control "dims の長さ ~S が operand の rank ~S と一致しない"
             :format-arguments (list (length dims) (length in-shape))))
    (unless (every (lambda (d) (typep d `(integer 0 (,out-rank)))) dims)
      (error 'primitive-error :name :broadcast-in-dim :in-avals in-avals
             :format-control "dims ~S は出力の rank ~S の範囲外を含む"
             :format-arguments (list dims out-rank)))
    (unless (= (length dims) (length (remove-duplicates dims)))
      (error 'primitive-error :name :broadcast-in-dim :in-avals in-avals
             :format-control "dims ~S に重複がある" :format-arguments (list dims)))
    (loop for operand-dim in in-shape
          for target-dim in dims
          for target-size = (nth target-dim shape)
          unless (or (= operand-dim 1) (= operand-dim target-size))
          do (error 'primitive-error :name :broadcast-in-dim :in-avals in-avals
                    :format-control "operand の次元 ~S は1でも出力の次元 ~S（サイズ ~S）とも一致しない"
                    :format-arguments (list operand-dim target-dim target-size)))
    (make-aval shape (aval-dtype in))))

(defun %broadcast-in-dim-emit (in-names in-avals out-name out-aval &key shape dims)
  (declare (ignore shape))
  (format nil "~A = stablehlo.broadcast_in_dim ~A, dims = [~{~D~^, ~}] : (~A) -> ~A"
          out-name (first in-names) dims (tensor-type-string (first in-avals)) (tensor-type-string out-aval)))

(defun %broadcast-in-dim-eager (arrays in-avals &key shape dims)
  (let ((in-shape (aval-shape (first in-avals)))
        (dtype (aval-dtype (first in-avals)))
        (array (first arrays)))
    (%shape-eager-fill
     shape dtype
     (lambda (out-index)
       (let ((out-subscripts (%shape-subscripts out-index shape)))
         (%shape-row-major-index
          (loop for operand-dim in in-shape
                for target-dim in dims
                collect (if (= operand-dim 1) 0 (nth target-dim out-subscripts)))
          in-shape)))
     array)))

(defprimitive broadcast-in-dim (:shape :dims)
  :abstract-eval #'%broadcast-in-dim-abstract-eval
  :emit #'%broadcast-in-dim-emit
  :eager #'%broadcast-in-dim-eager)

;;; transpose

(defun %transpose-abstract-eval (in-avals &key perm)
  (%shape-check-arity :transpose in-avals 1)
  (let* ((in (first in-avals))
         (in-shape (aval-shape in))
         (rank (length in-shape)))
    (unless (and (= (length perm) rank)
                 (equal (sort (copy-list perm) #'<) (loop for i below rank collect i)))
      (error 'primitive-error :name :transpose :in-avals in-avals
             :format-control "perm ~S は rank ~S の permutation でない"
             :format-arguments (list perm rank)))
    (make-aval (mapcar (lambda (p) (nth p in-shape)) perm) (aval-dtype in))))

(defun %transpose-emit (in-names in-avals out-name out-aval &key perm)
  (format nil "~A = stablehlo.transpose ~A, dims = [~{~D~^, ~}] : (~A) -> ~A"
          out-name (first in-names) perm (tensor-type-string (first in-avals)) (tensor-type-string out-aval)))

(defun %transpose-eager (arrays in-avals &key perm)
  (let* ((in-shape (aval-shape (first in-avals)))
         (dtype (aval-dtype (first in-avals)))
         (array (first arrays))
         (out-shape (mapcar (lambda (p) (nth p in-shape)) perm))
         (rank (length in-shape)))
    (%shape-eager-fill
     out-shape dtype
     (lambda (out-index)
       (let ((out-subscripts (%shape-subscripts out-index out-shape))
             (in-subscripts (make-list rank)))
         (loop for p in perm
               for s in out-subscripts
               do (setf (nth p in-subscripts) s))
         (%shape-row-major-index in-subscripts in-shape)))
     array)))

(defprimitive transpose (:perm)
  :abstract-eval #'%transpose-abstract-eval
  :emit #'%transpose-emit
  :eager #'%transpose-eager)
