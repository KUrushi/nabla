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
TRACER ならそのまま返す（eqn は足さない）。:I1 の aval は
%SCALAR-ARRAY が TRACING-ERROR にする。"
  (etypecase tangent
    (tracer tangent)
    (symbolic-zero
     (let* ((aval (symbolic-zero-aval tangent))
            (dtype (aval-dtype aval))
            (constant (%lift-constant (%scalar-array 0 dtype) (make-aval '() dtype) *current-trace*)))
       (if (plusp (aval-rank aval))
           (%trace-eqn :broadcast-in-dim (list constant) :shape (aval-shape aval) :dims '())
           constant)))))

(defun add-tangents (a b)
  "接線（余接線）A と B（それぞれ TRACER または SYMBOLIC-ZERO）の和を返す。
片方が SYMBOLIC-ZERO なら、もう片方をそのまま返す（eqn は足さない。両方
ゼロなら B）。どちらもトレーサなら現在のトレースに :ADD の eqn を足す
（%T-ADD）。jvp ルールの接線の加算と、transpose の余接線の加算が共用する。"
  (cond
    ((symbolic-zero-p a) b)
    ((symbolic-zero-p b) a)
    (t (%t-add a b))))
