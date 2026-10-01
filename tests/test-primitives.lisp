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

;;; --- %test-mul（issue #82）。 ---
;;;
;;; 要素ごとの積。jvp / transpose ルールを持つ唯一の「片側が定数」の線形
;;; プリミティブ: ルールは、接線を片方の被演算子にだけ流し、もう片方は主値の
;;; ままにする（接線どうしの積は作らない）。transpose はどちらが
;;; UNDEFINED-PRIMAL かで係数を選ぶ。グラフ全体としては非線形（x * y）になる
;;; ので、graph-recipe の :binary-prims に明示したときだけ生成される。

(nb:defprimitive %test-mul ()
  :abstract-eval
  (lambda (in-avals)
    (destructuring-bind (a b) in-avals
      (unless (equalp a b)
        (error 'nb:primitive-error :name :%test-mul :in-avals in-avals
               :format-control "aval が一致しない: ~S / ~S" :format-arguments (list a b)))
      a))
  :eager
  (lambda (arrays in-avals)
    (declare (ignore in-avals))
    (destructuring-bind (a b) arrays
      (let ((result (make-array (array-dimensions a) :element-type (array-element-type a))))
        (dotimes (i (array-total-size a) result)
          (setf (row-major-aref result i) (* (row-major-aref a i) (row-major-aref b i))))))))

(nb::def-jvp-rule %test-mul (primals out tangents)
  (declare (ignore out))
  (destructuring-bind (a b) primals
    (destructuring-bind (ta tb) tangents
      (nb::add-tangents
       (if (nb::symbolic-zero-p ta)
           ta
           (nb::%trace-eqn :%test-mul (list ta b)))
       (if (nb::symbolic-zero-p tb)
           tb
           (nb::%trace-eqn :%test-mul (list a tb)))))))

;; jvp ルールはあるが transpose ルールの無いプリミティブ（NO-TRANSPOSE-RULE の
;; テスト用）。接線は同じプリミティブにそのまま流す。
(nb:defprimitive %test-no-transpose ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager
  (lambda (arrays in-avals)
    (declare (ignore in-avals))
    (first arrays)))

(nb::def-jvp-rule %test-no-transpose (primals out tangents)
  (declare (ignore primals out))
  (nb::%trace-eqn :%test-no-transpose (list (first tangents))))

;;; --- transpose ルール（issue #82）。 ---
;;;
;;; neg / add / reshape / convert / reduce は線形なので、transpose は
;;; 「出力の余接線を入力の形に戻す」演算になる。

(defun %test-undefined-aval (invar)
  (nb::undefined-primal-aval invar))

(nb::def-transpose-rule %test-neg (ct invars)
  (declare (ignore invars))
  (list (nb::%trace-eqn :%test-neg (list ct))))

(nb::def-transpose-rule %test-add (ct invars)
  (unless (every #'nb::undefined-primal-p invars)
    (error 'nb:autodiff-error :format-control "%test-add は両方の入力が線形のときだけ転置できる"))
  (list ct ct))

(nb::def-transpose-rule %test-reshape (ct invars &key shape)
  (declare (ignore shape))
  (list (nb::%trace-eqn :%test-reshape (list ct) :shape (nb:aval-shape (%test-undefined-aval (first invars))))))

(nb::def-transpose-rule %test-convert (ct invars &key dtype)
  (declare (ignore dtype))
  (list (nb::%trace-eqn :%test-convert (list ct) :dtype (nb:aval-dtype (%test-undefined-aval (first invars))))))

(nb::def-transpose-rule %test-reduce (ct invars &key axis)
  ;; 縮約した軸に沿って余接線を複製する（broadcast-in-dim の dims は
  ;; 縮約された軸以外の出力の軸）。
  (let* ((shape (nb:aval-shape (%test-undefined-aval (first invars))))
         (dims (loop for i below (length shape) unless (= i axis) collect i)))
    (list (nb::%trace-eqn :broadcast-in-dim (list ct) :shape shape :dims dims))))

(nb::def-transpose-rule %test-mul (ct invars)
  (destructuring-bind (a b) invars
    (cond ((and (nb::undefined-primal-p a) (nb::undefined-primal-p b))
           (error 'nb:autodiff-error :format-control "%test-mul の両方の入力が線形（非線形な使い方）"))
          ((nb::undefined-primal-p a) (list (nb::%trace-eqn :%test-mul (list ct b)) nil))
          ((nb::undefined-primal-p b) (list nil (nb::%trace-eqn :%test-mul (list ct a))))
          (t (error 'nb:autodiff-error :format-control "%test-mul に線形な入力が無い")))))

;; jvp の接線の加算（ADD-TANGENTS）と transpose の余接線の加算は実プリミティブの
;; add を使うので、%test- の graph を転置するには add の transpose ルールも要る。
;; 実プリミティブの線形ルールは #83 が src/ad/ に書く。それまでのつなぎ（#83 が
;; 同じ意味のルールで上書きする）。
(nb::def-transpose-rule add (ct invars)
  (unless (every #'nb::undefined-primal-p invars)
    (error 'nb:autodiff-error :format-control "add は両方の入力が線形のときだけ転置できる"))
  (list ct ct))

;; わざと規約に反する transpose ルールを返すプリミティブ（transpose 変換の検査の
;; テスト用）。MODE: :short = 長さの違うリスト、:missing = 線形入力の余接線が NIL、
;; :wrong-aval = 入力と aval の違う余接線（dtype を変える）。
(nb:defprimitive %test-bad-transpose (:mode)
  :abstract-eval (lambda (in-avals &key mode) (declare (ignore mode)) (first in-avals))
  :eager (lambda (arrays in-avals &key mode) (declare (ignore in-avals mode)) (first arrays)))

(nb::def-jvp-rule %test-bad-transpose (primals out tangents &key mode)
  (declare (ignore primals out))
  (nb::%trace-eqn :%test-bad-transpose (list (first tangents)) :mode mode))

(nb::def-transpose-rule %test-bad-transpose (ct invars &key mode)
  (declare (ignore invars))
  (ecase mode
    (:short '())
    (:missing (list nil))
    (:wrong-aval (list (nb::%trace-eqn :%test-convert (list ct) :dtype :f32)))))
