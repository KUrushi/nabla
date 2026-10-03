;;;; primitives/bits: 乱数ビットを浮動小数点にするためのビット演算プリミティブ
;;;; （issue #136）。shift-right-logical / bitwise-or / bitcast-convert。
;;;;
;;;; shift-right-logical と bitwise-or は整数 dtype（:i32 :u32 :u64）専用の2入力の
;;;; 要素演算（shape と dtype が一致）。bitcast-convert はビット列の再解釈で、
;;;; dtype が :f32 :f64 :i32 :u32 :u64 の間で変わる。幅が違うときは StableHLO と同じく
;;;; 末尾の次元が増減する（狭い → 広い: 末尾の次元（長さ = 幅の比）が消える。
;;;; 広い → 狭い: 末尾に長さ = 幅の比の次元が付く。要素の並びはリトルエンディアン、
;;;; つまり先頭の要素が下位ビット）。
;;;;
;;;; いずれも整数・ビット列の演算なので微分しない（jvp ルールは要らない。入力の接線は
;;;; 常に symbolic zero）。バッチ化ルールは src/ad/rules-batch-rng.lisp。

(in-package #:nabla)

;;; --- shift-right-logical / bitwise-or ---

(defun %check-integer-pair (name in-avals)
  "2入力・整数 dtype・shape/dtype が一致していることを確かめて、最初の AVAL を返す。"
  (%check-arity name in-avals 2)
  (let ((result (%check-same-avals name in-avals)))
    (unless (integer-dtype-p (aval-dtype result))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "dtype ~S は整数（:i32 :u32 :u64）でなければならない"
             :format-arguments (list (aval-dtype result))))
    result))

(defun %shift-right-logical-element (x amount dtype)
  "整数 DTYPE の X を、AMOUNT（同じ dtype を符号なしと見たもの）ビット論理右シフトする。
AMOUNT がビット幅以上なら 0（StableHLO と同じ）。"
  (let* ((bits (integer-dtype-bits dtype))
         (n (ldb (byte bits 0) amount)))
    (if (>= n bits)
        0
        ;; :i32 は最上位ビットが立つ値（シフト量 0 など）が負の範囲に戻るよう折り返す
        (wrap-integer (ash (ldb (byte bits 0) x) (- n)) dtype))))

(defun %shift-right-logical-eager (arrays in-avals)
  (let ((out (%check-integer-pair :shift-right-logical in-avals)))
    (%elementwise-eager (lambda (x n) (%shift-right-logical-element x n (aval-dtype out)))
                        arrays in-avals out)))

(defun %bitwise-or-eager (arrays in-avals)
  (%elementwise-eager #'logior arrays in-avals (%check-integer-pair :bitwise-or in-avals)))

(defprimitive shift-right-logical ()
  :abstract-eval (lambda (in-avals) (%check-integer-pair :shift-right-logical in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%emit-elementwise "shift_right_logical" in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%shift-right-logical-eager arrays in-avals)))

(defprimitive bitwise-or ()
  :abstract-eval (lambda (in-avals) (%check-integer-pair :bitwise-or in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%emit-elementwise "or" in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%bitwise-or-eager arrays in-avals)))

;;; --- bitcast-convert ---

(defparameter *bitcast-dtypes* '(:f32 :f64 :i32 :u32 :u64)
  "bitcast-convert が扱う dtype。")

(defun %dtype-bits (dtype)
  (* 8 (dtype-byte-width dtype)))

(defun %bitcast-result-shape (name in-avals shape in-dtype out-dtype)
  "bitcast-convert の出力の shape（幅が違うときの末尾の次元の増減はファイル冒頭のとおり）。"
  (let ((in-bits (%dtype-bits in-dtype))
        (out-bits (%dtype-bits out-dtype)))
    (flet ((fail (control &rest arguments)
             (error 'primitive-error :name name :in-avals in-avals
                                     :format-control control :format-arguments arguments)))
      (cond
        ((= in-bits out-bits) shape)
        ((> in-bits out-bits) (append shape (list (/ in-bits out-bits))))
        (t (let ((ratio (/ out-bits in-bits)))
             (unless (and shape (= (car (last shape)) ratio))
               (fail "~S から ~S への bitcast は末尾の次元が ~D でなければならない: ~S"
                     in-dtype out-dtype ratio shape))
             (butlast shape)))))))

(defun %bitcast-abstract-eval (in-avals &key dtype)
  (%check-arity :bitcast-convert in-avals 1)
  (let* ((in (first in-avals)))
    (unless (and (member dtype *bitcast-dtypes*) (member (aval-dtype in) *bitcast-dtypes*))
      (error 'primitive-error :name :bitcast-convert :in-avals in-avals
             :format-control "dtype は ~S のどれかでなければならない（入力 ~S・出力 ~S）"
             :format-arguments (list *bitcast-dtypes* (aval-dtype in) dtype)))
    (make-aval (%bitcast-result-shape :bitcast-convert in-avals (aval-shape in)
                                      (aval-dtype in) dtype)
               dtype)))

(defun %element-to-bits (x dtype)
  "DTYPE の1要素 X のビット列（符号なし整数）。"
  (ecase dtype
    (:f32 (ldb (byte 32 0) (sb-kernel:single-float-bits x)))
    (:f64 (logior (ash (ldb (byte 32 0) (sb-kernel:double-float-high-bits x)) 32)
                  (sb-kernel:double-float-low-bits x)))
    ((:i32 :u32 :u64) (ldb (byte (integer-dtype-bits dtype) 0) x))))

(defun %bits-to-element (bits dtype)
  "ビット列 BITS（符号なし整数）を DTYPE の要素にする。"
  (ecase dtype
    (:f32 (sb-kernel:make-single-float (wrap-integer bits :i32)))
    (:f64 (sb-kernel:make-double-float (wrap-integer (ash bits -32) :i32) (ldb (byte 32 0) bits)))
    ((:i32 :u32 :u64) (wrap-integer bits dtype))))

(defun %bitcast-eager (arrays in-avals &key dtype)
  "要素をビット列にして、リトルエンディアンで切り分け／結合して並べ直す。"
  (let* ((in-aval (first in-avals))
         (in-dtype (aval-dtype in-aval))
         (out-aval (%bitcast-abstract-eval in-avals :dtype dtype))
         (in-bits (%dtype-bits in-dtype))
         (out-bits (%dtype-bits dtype))
         (source (first arrays))
         (result (make-array (aval-shape out-aval) :element-type (dtype-element-type dtype))))
    (with-ieee-arithmetic
      (cond
        ((= in-bits out-bits)
         (dotimes (i (array-total-size result))
           (setf (row-major-aref result i)
                 (%bits-to-element (%element-to-bits (row-major-aref source i) in-dtype) dtype))))
        ((> in-bits out-bits)
         (let ((ratio (/ in-bits out-bits)))
           (dotimes (i (array-total-size source))
             (let ((bits (%element-to-bits (row-major-aref source i) in-dtype)))
               (dotimes (j ratio)
                 (setf (row-major-aref result (+ (* i ratio) j))
                       (%bits-to-element (ldb (byte out-bits (* j out-bits)) bits) dtype)))))))
        (t
         (let ((ratio (/ out-bits in-bits)))
           (dotimes (i (array-total-size result))
             (let ((bits 0))
               (dotimes (j ratio)
                 (setf (ldb (byte in-bits (* j in-bits)) bits)
                       (%element-to-bits (row-major-aref source (+ (* i ratio) j)) in-dtype)))
               (setf (row-major-aref result i) (%bits-to-element bits dtype))))))))
    result))

(defprimitive bitcast-convert (:dtype)
  :abstract-eval (lambda (in-avals &key dtype) (%bitcast-abstract-eval in-avals :dtype dtype))
  :emit (lambda (in-names in-avals out-name out-aval &key dtype)
          (declare (ignore dtype))
          (format nil "~A = stablehlo.bitcast_convert ~A : (~A) -> ~A"
                  out-name (first in-names) (tensor-type-string (first in-avals))
                  (tensor-type-string out-aval)))
  :eager (lambda (arrays in-avals &key dtype) (%bitcast-eager arrays in-avals :dtype dtype)))
