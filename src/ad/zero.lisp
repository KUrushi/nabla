;;;; zero: 自動微分の symbolic zero と接線の基本操作（issue #77、77a）。
;;;;
;;;; symbolic zero は「値がゼロと分かっている接線・余接線」を、配列も eqn も
;;;; 作らずに表す内部オブジェクト。変換（jvp / transpose）はこれを伝播させ、
;;;; 実体が要るとき（graph の出力など）にだけ INSTANTIATE-ZERO で作る。
;;;; ここのシンボルはすべて内部（export しない）。

(in-package #:nabla)

(defstruct (symbolic-zero (:constructor make-symbolic-zero (aval)) (:copier nil))
  "値がゼロと分かっている接線（または余接線）。AVAL はその接線が持つはずの
配列の aval。配列は持たない。"
  (aval nil :type aval :read-only t))

(defmethod print-object ((zero symbolic-zero) stream)
  (print-unreadable-object (zero stream :type t)
    (let ((aval (symbolic-zero-aval zero)))
      (format stream "~A[~{~D~^,~}]" (dtype-mlir-name (aval-dtype aval)) (aval-shape aval)))))

(defstruct (undefined-primal (:constructor make-undefined-primal (aval)) (:copier nil))
  "transpose ルールの INVARS で、「まだ値の無い線形入力」の印。AVAL はその
入力の aval。既知の主値（トレーサ）と区別するために使う。"
  (aval nil :type aval :read-only t))

(defun tangent-aval (tangent)
  "TANGENT（TRACER または SYMBOLIC-ZERO）の aval を返す。"
  (etypecase tangent
    (symbolic-zero (symbolic-zero-aval tangent))
    (tracer (tracer-aval tangent))))

(defun instantiate-zero (tangent)
  "TANGENT が SYMBOLIC-ZERO なら、現在のトレース（*CURRENT-TRACE*）に、同じ
aval を持つゼロのトレーサを作って返す: rank 0 の定数 0 を登録し、rank が 1
以上なら BROADCAST-IN-DIM で shape まで広げる（%LIFT-NUMBER と同じ形）。
TRACER ならそのまま返す（eqn は足さない）。:I1 は %SCALAR-ARRAY が数値との
対応を持たず拒否するので、rank 0 の bit 配列 0（false）を直接登録する
（jvp の :I1 出力の接線は常に全 false）。"
  (etypecase tangent
    (tracer tangent)
    (symbolic-zero
     (let ((aval (symbolic-zero-aval tangent)))
       (if (eq (aval-dtype aval) :i1)
           (let ((scalar (%lift-constant (make-array '() :element-type 'bit :initial-element 0)
                                         (make-aval '() :i1) *current-trace*)))
             (if (plusp (length (aval-shape aval)))
                 (%trace-eqn :broadcast-in-dim (list scalar) :shape (aval-shape aval) :dims '())
                 scalar))
           (%lift-number-to 0 (aval-dtype aval) (aval-shape aval)))))))

(defun add-tangents (a b)
  "接線（余接線）A と B（それぞれ TRACER または SYMBOLIC-ZERO）の和を返す。
片方が SYMBOLIC-ZERO なら、もう片方をそのまま返す（eqn は足さない。両方
ゼロなら B）。A と B の aval が一致しなければ AUTODIFF-ERROR。どちらもトレーサなら現在のトレースに :ADD の eqn を足す
（%T-ADD）。jvp ルールの接線の加算と、transpose の余接線の加算が共用する。"
  (unless (equalp (tangent-aval a) (tangent-aval b))
    (error 'autodiff-error
           :format-control "接線の aval が一致しない: ~S と ~S"
           :format-arguments (list (tangent-aval a) (tangent-aval b))))
  (cond
    ((symbolic-zero-p a) b)
    ((symbolic-zero-p b) a)
    (t (%t-add a b))))
