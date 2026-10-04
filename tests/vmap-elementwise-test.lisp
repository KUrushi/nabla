;;;; 要素演算のバッチ化ルールの性質（issue #128）。
;;;;
;;;; 守らせる性質: 各プリミティブ f について「vmap f = バッチ軸で切り出した各要素に f を
;;;; eager で適用して積み直したもの」（期待値は tests/support/vmap.lisp の REFERENCE-VMAP）。
;;;; バッチ軸の位置・どの引数がバッチされるか・out-axes をランダムにする。もう1つの
;;;; 性質は、バッチされた引数が全部同じ軸で、バッチされていない引数も無いときは、
;;;; 余分な transpose / broadcast-in-dim の eqn が1つも生成されないこと。軸が食い違うときは
;;;; 多数決の軸へ揃え、少数派の引数だけを transpose すること（issue #166）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; (名前 関数 引数の種類のリスト 出力の dtype)。:pred は :i1（BIT）の配列、:float は f64。
(defparameter *vmap-elementwise-cases*
  (list
   (list :add (nb:with-tracing (x y) (+ x y)) '(:float :float) :f64)
   (list :sub (nb:with-tracing (x y) (- x y)) '(:float :float) :f64)
   (list :mul (nb:with-tracing (x y) (* x y)) '(:float :float) :f64)
   (list :div (nb:with-tracing (x y) (/ x y)) '(:float :float) :f64)
   (list :max (nb:with-tracing (x y) (max x y)) '(:float :float) :f64)
   (list :min (nb:with-tracing (x y) (min x y)) '(:float :float) :f64)
   (list :neg (nb:with-tracing (x) (- x)) '(:float) :f64)
   (list :exp (nb:with-tracing (x) (exp x)) '(:float) :f64)
   (list :log (nb:with-tracing (x) (log x)) '(:float) :f64)
   (list :tanh (nb:with-tracing (x) (tanh x)) '(:float) :f64)
   (list :compare (nb:with-tracing (x y) (< x y)) '(:float :float) :i1)
   (list :select (nb:with-tracing (p x y) (nb:where p x y)) '(:pred :float :float) :f64)
   (list :convert (nb:with-tracing (x) (nb:convert x :f32)) '(:float) :f32)
   (list :stop-gradient (nb:with-tracing (x) (nb:stop-gradient x)) '(:float) :f64)))

(defun %ew-inner-shape (rank seed)
  (loop for i below rank collect (1+ (mod (floor seed (expt 5 i)) 4))))

(defun %ew-axis-or-nil (code rank)
  "CODE から 0..RANK のバッチ軸か NIL（CODE mod (RANK+2) = RANK+1）を選ぶ。"
  (let ((v (mod code (+ rank 2))))
    (and (<= v rank) v)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *vmap-elementwise-integer-primitives*
    '(:add :sub :mul :max :min :neg :compare :select :convert :stop-gradient)
    "整数 dtype も受け付けるプリミティブ（div exp log tanh は浮動小数点専用なので入れない）。"))

(defun %ew-array (kind inner size axis seed &optional (dtype :f64))
  (let ((shape (if axis
                   (append (subseq inner 0 axis) (list size) (nthcdr axis inner))
                   inner)))
    (ecase kind
      (:float (if (member dtype *integer-dtypes*)
                  (make-random-array (make-array-spec shape dtype) :seed seed)
                  (make-random-array (make-array-spec shape dtype) :seed seed :domain :positive)))
      (:pred (let* ((a (make-array shape :element-type 'bit))
                    (r (make-random-array (make-array-spec shape :f64) :seed seed)))
               (dotimes (i (array-total-size a) a)
                 (setf (row-major-aref a i) (if (plusp (row-major-aref r i)) 1 0))))))))

(defun %ew-axes (codes kinds rank)
  "各引数のバッチ軸（NIL か整数）。すべて NIL のときは先頭の引数を軸 0 でバッチする。"
  (let ((axes (loop for kind in kinds for code in codes collect (%ew-axis-or-nil code rank))))
    (if (notany #'identity axes) (cons 0 (rest axes)) axes)))

(defun %ew-property (f kinds out-dtype &optional (dtype :f64))
  "OUT-DTYPE が :F64 のときは出力が入力と同じ dtype DTYPE（compare は :I1、convert は :F32 を渡す）。"
  (lambda (case)
    (destructuring-bind (rank size seed c1 c2 c3 code-out) case
      (let* ((inner (%ew-inner-shape rank seed))
             (axes (%ew-axes (list c1 c2 c3) kinds rank))
             (out (mod code-out (1+ rank)))
             (args (loop for kind in kinds for axis in axes for i from 0
                         collect (%ew-array kind inner size axis (+ seed i) dtype)))
             (expected (first (reference-vmap f args :in-axes axes :out-axes out)))
             (actual (apply (nb:vmap f :in-axes axes :out-axes out) args)))
        (let ((out-dtype (if (eq out-dtype :f64) dtype out-dtype)))
          (if (or (eq out-dtype :i1) (member out-dtype *integer-dtypes*))
              (equalp actual expected)
              (allclose actual expected :dtype out-dtype)))))))

(defun %ew-generator ()
  (generator (tuple (uniform-integer :lo 0 :hi 3)
                    (uniform-integer :lo 1 :hi 4)
                    (uniform-integer :lo 0 :hi 10000)
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99))))

(defmacro %def-elementwise-pbt (name)
  `(progn
     (%def-elementwise-float-pbt ,name)
     (%def-elementwise-integer-pbt ,name)))

(defmacro %def-elementwise-float-pbt (name)
  `(test ,(intern (format nil "VMAP/~A-MATCHES-PER-SLICE-REFERENCE" name))
     ,(format nil "~(~A~): バッチ軸の位置・バッチされる引数の部分集合・out-axes をランダムにした vmap が、要素ごとに適用して積み直した参照実装と一致する。" name)
     (destructuring-bind (name f kinds out-dtype) (assoc ,name *vmap-elementwise-cases*)
       (declare (ignore name))
       (is (check-it (%ew-generator)
                     (%ew-property f kinds out-dtype)
                     :regression-id ,(intern (format nil "VMAP/~A-MATCHES-PER-SLICE-REFERENCE" name))
                     :regression-file (regression-path "vmap-elementwise"))))))

(defmacro %def-elementwise-integer-pbt (name)
  "NAME が整数も受け付けるプリミティブのときだけ、各整数 dtype の PBT を定義する。"
  (when (member name *vmap-elementwise-integer-primitives*)
    `(test ,(intern (format nil "VMAP/~A-INTEGER-MATCHES-PER-SLICE-REFERENCE" name))
       ,(format nil "~(~A~): 整数 dtype（*integer-dtypes*）でも、vmap が要素ごとに適用して積み直した参照実装と完全に一致する。" name)
       (destructuring-bind (name f kinds out-dtype) (assoc ,name *vmap-elementwise-cases*)
         (declare (ignore name))
         (dolist (dtype *integer-dtypes*)
           (is (check-it (%ew-generator)
                         (%ew-property f kinds (if (eq out-dtype :f64) dtype out-dtype) dtype)
                         :regression-id ,(intern (format nil "VMAP/~A-INTEGER-MATCHES-PER-SLICE-REFERENCE" name))
                         :regression-file (regression-path "vmap-elementwise"))))))))

(%def-elementwise-pbt :add)
(%def-elementwise-pbt :sub)
(%def-elementwise-pbt :mul)
(%def-elementwise-pbt :div)
(%def-elementwise-pbt :max)
(%def-elementwise-pbt :min)
(%def-elementwise-pbt :neg)
(%def-elementwise-pbt :exp)
(%def-elementwise-pbt :log)
(%def-elementwise-pbt :tanh)
(%def-elementwise-pbt :compare)
(%def-elementwise-pbt :select)
(%def-elementwise-pbt :convert)
(%def-elementwise-pbt :stop-gradient)

;;; --- 余分な eqn が生成されない ---

(defun %ew-eqn-names (f in-axes out-axes avals)
  (mapcar (lambda (e) (nb::primitive-name (nb::eqn-prim e)))
          (nb:graph-eqns (nb:trace-to-graph (nb:vmap f :in-axes in-axes :out-axes out-axes) avals))))

(defun %ew-aval (kind shape)
  (nb:make-aval shape (if (eq kind :pred) :i1 :f64)))

(test vmap/elementwise-same-axis-adds-no-transpose-or-broadcast
  "バッチされた引数が全部同じ軸（0..rank）で、バッチされていない引数が無いとき、graph は
元のプリミティブ1つだけ（transpose も broadcast-in-dim も無い）。"
  (dolist (case *vmap-elementwise-cases*)
    (destructuring-bind (name f kinds out-dtype) case
      (declare (ignore out-dtype))
      (dolist (axis '(0 1 2))
        (let ((avals (loop for kind in kinds collect (%ew-aval kind (append (subseq '(2 3) 0 axis) '(5) (nthcdr axis '(2 3)))))))
          (is (equal (list name)
                     (%ew-eqn-names f axis axis avals))
              "~S axis ~D" name axis))))))

(test vmap/elementwise-different-axes-transposes-to-the-common-position
  "バッチ軸の位置が違う2引数は、片方だけ transpose して揃える（transpose 1つ + 元のプリミティブ）。
バッチされていない引数は broadcast-in-dim 1つで足す。"
  (let ((f (nb:with-tracing (x y) (+ x y))))
    (is (equal '(:transpose :add)
               (%ew-eqn-names f '(0 1) 0 (list (nb:make-aval '(5 2 3) :f64) (nb:make-aval '(2 5 3) :f64)))))
    (is (equal '(:broadcast-in-dim :add)
               (%ew-eqn-names f '(1 nil) 1 (list (nb:make-aval '(2 5 3) :f64) (nb:make-aval '(2 3) :f64)))))))

(test vmap/select-with-only-the-predicate-batched
  "select の条件だけがバッチされるとき、on-true / on-false は broadcast-in-dim で揃えてから選ぶ。"
  (let* ((f (nb:with-tracing (p x y) (nb:where p x y)))
         (g (nb:vmap f :in-axes '(0 nil nil)))
         (p (make-array '(2 3) :element-type 'bit :initial-contents '((1 0 1) (0 1 1))))
         (x (make-array '(3) :element-type 'double-float :initial-contents '(1d0 2d0 3d0)))
         (y (make-array '(3) :element-type 'double-float :initial-contents '(-1d0 -2d0 -3d0)))
         (graph (nb:trace-to-graph g (list (nb:array-aval p) (nb:array-aval x) (nb:array-aval y)))))
    (is (equal '(:broadcast-in-dim :broadcast-in-dim :select)
               (mapcar (lambda (e) (nb::primitive-name (nb::eqn-prim e))) (nb:graph-eqns graph))))
    (is (equalp (make-array '(2 3) :element-type 'double-float
                                   :initial-contents '((1d0 -2d0 3d0) (-1d0 2d0 3d0)))
                (funcall g p x y)))))

(test vmap/elementwise-rule-rejects-all-unbatched-arguments
  "バッチされた引数が1つも無いまま共通ルールが呼ばれたら、黙って壊れず VMAP-ERROR にする。"
  (let ((rule (nb::primitive-batch (nb::find-primitive :add)))
        (avals (list (nb:make-aval '(2) :f64) (nb:make-aval '(2) :f64))))
    (nb::%call-with-fresh-trace
     avals
     (lambda (x y)
       (signals nb:vmap-error (funcall rule (list x y) (list nil nil)))
       (nb::%trace-eqn :add (list x y))))))

;;; --- バッチ軸は多数決で揃える（issue #166 (e)） ---

(defun %ew-eqns-before-primitive (f in-axes avals name)
  "F を IN-AXES で vmap した graph のうち、元のプリミティブ NAME の eqn より前にある
eqn の名前のリスト（引数を揃えるために足された eqn）。"
  (let ((names (%ew-eqn-names f in-axes 0 avals)))
    (subseq names 0 (position name names))))

(test vmap/elementwise-aligns-to-the-majority-axis
  "3引数のうち2つが軸 1 でバッチされているとき、残りの1つだけを transpose する。"
  (let ((f (nb:with-tracing (p x y) (nb:where p x y))))
    (is (equal '(:transpose)
               (%ew-eqns-before-primitive
                f '(1 1 0) (list (nb:make-aval '(2 5 3) :i1) (nb:make-aval '(2 5 3) :f64)
                                 (nb:make-aval '(5 2 3) :f64))
                :select)))))

(test vmap/elementwise-transposes-only-the-minority-operands
  "どの引数がどの軸でバッチされても、引数を揃える transpose の数は「バッチされた引数の数 -
最も多くの引数が共有する軸の引数の数」で、broadcast-in-dim の数はバッチされていない引数の数。"
  (dolist (case *vmap-elementwise-cases*)
    (destructuring-bind (name f kinds out-dtype) case
      (declare (ignore out-dtype))
      (is (check-it
           (%ew-generator)
           (lambda (c)
             (destructuring-bind (rank size seed c1 c2 c3 code-out) c
               (declare (ignore code-out))
               (let* ((inner (%ew-inner-shape rank seed))
                      (axes (%ew-axes (list c1 c2 c3) kinds rank))
                      (avals (loop for kind in kinds for axis in axes
                                   collect (%ew-aval kind (if axis
                                                              (append (subseq inner 0 axis) (list size) (nthcdr axis inner))
                                                              inner))))
                      (batched (remove nil axes))
                      (majority (reduce #'max (mapcar (lambda (a) (count a batched)) batched)))
                      (added (%ew-eqns-before-primitive f axes avals name)))
                 (and (= (count :transpose added) (- (length batched) majority))
                      (= (count :broadcast-in-dim added) (count nil axes))
                      (= (length added) (+ (- (length batched) majority) (count nil axes)))))))
           :regression-id vmap/elementwise-transposes-only-the-minority-operands
           :regression-file (regression-path "vmap-elementwise"))
          "~S" name))))

(test vmap/elementwise-common-axis-tie-picks-the-smallest-axis
  "最も多くの引数が共有する軸が同数で複数あるときは、引数の順序に依らず最も小さい軸を選ぶ。
呼び出し側の BATCH-DIMS のリストは壊さない。"
  (is (= 0 (nb::%elementwise-common-axis (list 2 0 nil))))
  (is (= 0 (nb::%elementwise-common-axis (list 0 2))))
  (is (= 1 (nb::%elementwise-common-axis (list 2 1 1 2))))
  (is (= 1 (nb::%elementwise-common-axis (list nil 3 1))))
  (let ((batch-dims (list 2 nil 1 0)))
    (is (= 0 (nb::%elementwise-common-axis batch-dims)))
    (is (equal '(2 nil 1 0) batch-dims))))

(test vmap/elementwise-common-axis-majority-beats-the-first-argument
  "先頭の引数の軸より、より多くの引数が共有する軸を選ぶ。"
  (is (= 0 (nb::%elementwise-common-axis (list 2 0 0))))
  (is (= 1 (nb::%elementwise-common-axis (list nil 0 1 1))))
  (is (= 2 (nb::%elementwise-common-axis (list 0 2 2)))))
