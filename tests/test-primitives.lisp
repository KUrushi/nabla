;;;; テスト専用プリミティブ（issue #29、u1a）。
;;;;
;;;; 名前は %TEST- 接頭辞にして、wave 2 が defprimitive する本物の
;;;; プリミティブ（add など）と衝突させない。整数・整数リスト・キーワード
;;;; の3種の params を網羅する（%TEST-RESHAPE = 整数リスト、%TEST-REDUCE =
;;;; 整数、%TEST-CONVERT = キーワード）。ABSTRACT-EVAL のみを持ち（EMIT /
;;;; EAGER は省略。u1a のスコープ外）、make-eqn / check-graph の配管を
;;;; 確かめるためだけに使う。%TEST-TWO-PARAMS はパラメタが2つあるときに
;;;; make-eqn が呼び出し順ではなく宣言順に正規化することを確かめるための
;;;; プリミティブ。

(in-package #:nabla.tests)

;; EAGER は f32 / f64 だけに対応する（bf16 / f16 のビット演算までは u1a の
;; スコープ外。EVAL-GRAPH の PBT は (GRAPH-RECIPE :DTYPES '(:F32 :F64)) で
;; bf16 / f16 のレシピを生成しないようにして避ける。issue #39、e0）。

(defun %eager-map (array element-type shape fn)
  "ARRAY の各要素に FN を適用した、SHAPE・ELEMENT-TYPE を持つ新しい配列を
返す。単項の %TEST- プリミティブの EAGER が共通で使う小さなヘルパー
（issue #39、e0）。"
  (let ((result (make-array shape :element-type element-type)))
    (dotimes (i (array-total-size array) result)
      (setf (row-major-aref result i) (funcall fn (row-major-aref array i))))))

(nb:defprimitive %test-neg ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager
  (lambda (arrays in-avals)
    (declare (ignore in-avals))
    (let ((array (first arrays)))
      (%eager-map array (array-element-type array) (array-dimensions array) #'-))))

(nb:defprimitive %test-add ()
  :abstract-eval
  (lambda (in-avals)
    (destructuring-bind (a b) in-avals
      (unless (equal (nb:aval-shape a) (nb:aval-shape b))
        (error 'nb:primitive-error :name :%test-add :in-avals in-avals
               :format-control "shape が一致しない: ~S / ~S"
               :format-arguments (list (nb:aval-shape a) (nb:aval-shape b))))
      (unless (eq (nb:aval-dtype a) (nb:aval-dtype b))
        (error 'nb:primitive-error :name :%test-add :in-avals in-avals
               :format-control "dtype が一致しない: ~S / ~S"
               :format-arguments (list (nb:aval-dtype a) (nb:aval-dtype b))))
      a))
  :eager
  (lambda (arrays in-avals)
    (declare (ignore in-avals))
    (destructuring-bind (a b) arrays
      (let ((result (make-array (array-dimensions a) :element-type (array-element-type a))))
        (dotimes (i (array-total-size a) result)
          (setf (row-major-aref result i) (+ (row-major-aref a i) (row-major-aref b i))))))))

(nb:defprimitive %test-reshape (:shape)
  :abstract-eval
  (lambda (in-avals &key shape)
    (let ((in (first in-avals)))
      (unless (= (nb:aval-size in) (reduce #'* shape :initial-value 1))
        (error 'nb:primitive-error :name :%test-reshape :in-avals in-avals
               :format-control "要素数が一致しない: ~S → ~S" :format-arguments (list (nb:aval-shape in) shape)))
      (nb:make-aval shape (nb:aval-dtype in))))
  :eager
  (lambda (arrays in-avals &key shape)
    (declare (ignore in-avals))
    (let ((array (first arrays)))
      (%eager-map array (array-element-type array) shape #'identity))))

(nb:defprimitive %test-convert (:dtype)
  :abstract-eval
  (lambda (in-avals &key dtype)
    (nb:make-aval (nb:aval-shape (first in-avals)) dtype))
  :eager
  (lambda (arrays in-avals &key dtype)
    (declare (ignore in-avals))
    (let* ((array (first arrays))
           (element-type (nb:dtype-element-type dtype)))
      (%eager-map array element-type (array-dimensions array)
                  (lambda (x) (coerce x element-type))))))

(nb:defprimitive %test-reduce (:axis)
  :abstract-eval
  (lambda (in-avals &key axis)
    (let* ((in (first in-avals))
           (shape (nb:aval-shape in)))
      (unless (< -1 axis (length shape))
        (error 'nb:primitive-error :name :%test-reduce :in-avals in-avals
               :format-control "axis ~S が shape ~S の範囲外" :format-arguments (list axis shape)))
      (nb:make-aval (append (subseq shape 0 axis) (subseq shape (1+ axis))) (nb:aval-dtype in))))
  :eager
  (lambda (arrays in-avals &key axis)
    (declare (ignore in-avals))
    ;; AXIS 次元に沿って足し合わせる。入力の各添字 (i0 i1 ...) から AXIS 番目
    ;; を除いた添字が、出力の対応する要素になる。
    (let* ((array (first arrays))
           (dims (array-dimensions array))
           (rank (length dims))
           (element-type (array-element-type array))
           (result (make-array (append (subseq dims 0 axis) (subseq dims (1+ axis)))
                                :element-type element-type
                                :initial-element (coerce 0 element-type))))
      (labels ((walk (dim in-idx out-idx)
                 (if (= dim rank)
                     (let ((in-idx (reverse in-idx))
                           (out-idx (reverse out-idx)))
                       (incf (apply #'aref result out-idx) (apply #'aref array in-idx)))
                     (dotimes (i (nth dim dims))
                       (walk (1+ dim) (cons i in-idx)
                             (if (= dim axis) out-idx (cons i out-idx)))))))
        (walk 0 '() '()))
      result)))

;; :EAGER を持たないプリミティブ（PRIMITIVE-NOT-EVALUABLE のテスト用、
;; issue #39、e0）。
(nb:defprimitive %test-no-eager ()
  :abstract-eval (lambda (in-avals) (first in-avals)))

;; ABSTRACT-EVAL と食い違う shape を返す壊れた :EAGER（EVAL-GRAPH の
;; post-eqn 不変量チェックのテスト用、issue #39、e0）。
(nb:defprimitive %test-bad-eager ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager
  (lambda (arrays in-avals)
    (declare (ignore in-avals))
    (let ((array (first arrays)))
      (make-array (append (array-dimensions array) '(1)) :element-type (array-element-type array)))))

;; ABSTRACT-EVAL 通りの shape だが要素型が食い違う配列を返す壊れた
;; :EAGER（(ARRAY-AVAL RESULT (AVAL-DTYPE OUT-AVAL)) 自身が DTYPE-MISMATCH
;; を signal する経路のテスト用。EVAL-GRAPH はこれも PRIMITIVE-ERROR に
;; まとめる必要がある。issue #39、e0）。
(nb:defprimitive %test-bad-dtype-eager ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager
  (lambda (arrays in-avals)
    (declare (ignore in-avals))
    (let ((array (first arrays)))
      (make-array (array-dimensions array) :element-type '(unsigned-byte 16) :initial-element 0))))

(nb:defprimitive %test-two-params (:a :b)
  :abstract-eval (lambda (in-avals &key a b) (declare (ignore a b)) (first in-avals)))

;; ir-print.lisp の READ-GRAPH が params 中の T を CL:T に正規化することを
;; 確かめるための、フラグ（真偽値）パラメタを持つテスト専用プリミティブ
;; （issue #29、u1b）。
(nb:defprimitive %test-flag (:keep)
  :abstract-eval (lambda (in-avals &key keep) (declare (ignore keep)) (first in-avals)))

;;; --- jvp ルール（issue #77、77c）。 ---
;;;
;;; %test-neg / %test-add / %test-reshape / %test-convert / %test-reduce は
;;; どれも入力について線形（convert は丸めを除いて線形）なので、接線は
;;; 同じプリミティブを接線に適用するだけ。ランダムな graph-recipe の jvp
;;; 変換のテストが使う。%TEST-NO-EAGER にはわざと jvp ルールを付けない
;;; （NO-JVP-RULE のテスト用）。

(nb::def-jvp-rule %test-neg (primals out tangents)
  (declare (ignore primals out))
  (nb::%trace-eqn :%test-neg (list (first tangents))))

(nb::def-jvp-rule %test-add (primals out tangents)
  (declare (ignore primals out))
  (nb::add-tangents (first tangents) (second tangents)))

(nb::def-jvp-rule %test-reshape (primals out tangents &key shape)
  (declare (ignore primals out))
  (nb::%trace-eqn :%test-reshape (list (first tangents)) :shape shape))

(nb::def-jvp-rule %test-convert (primals out tangents &key dtype)
  (declare (ignore primals out))
  (nb::%trace-eqn :%test-convert (list (first tangents)) :dtype dtype))

(nb::def-jvp-rule %test-reduce (primals out tangents &key axis)
  (declare (ignore primals out))
  (nb::%trace-eqn :%test-reduce (list (first tangents)) :axis axis))

;; わざと主値と aval の違う接線（f64 なら f32、それ以外なら f64）を返す
;; jvp ルールを持つプリミティブ（autodiff-error のテスト用）。
(nb:defprimitive %test-bad-jvp ()
  :abstract-eval (lambda (in-avals) (first in-avals)))

(nb::def-jvp-rule %test-bad-jvp (primals out tangents)
  (declare (ignore primals))
  (nb::%trace-eqn :%test-convert (list (first tangents))
                  :dtype (if (eq (nb:aval-dtype (nb::tracer-aval out)) :f64) :f32 :f64)))
