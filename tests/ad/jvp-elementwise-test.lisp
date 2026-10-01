;;;; 要素ごとのプリミティブの jvp ルールの性質（issue #80）。
;;;;
;;;; 対象: sub mul div exp log tanh max min convert compare select（と合成）。
;;;; 各ルールを、1つの共通の表（*ELEMENTWISE-JVP-CASES*）に対する3つの性質で
;;;; 確かめる:
;;;;   1. f64 の jvp の接線が central-difference-jvp と許容誤差で一致する
;;;;   2. 接線について線形（jvp(a·v) = a·jvp(v)、jvp(v+w) = jvp(v)+jvp(w)）
;;;;   3. 接線の出力の aval が主値の出力の aval と一致する（f32 / f64。ルールが
;;;;      返した接線の aval は jvp-graph も検査するが、ここでは各ルールについて
;;;;      PBT で確かめる）
;;;; 加えて max / min の同値（0.5 ずつ）・compare / convert / select の
;;;; 接線の扱いは固定の例で確かめる。
;;;; 定義域: log / div の分母は正の値、max / min / select は2入力が必ず
;;;; 離れた値（中心差分の刻み幅で比較の向きが変わらない）になるよう作る。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *elementwise-jvp-cases*
  (list
   (list :name :sub :n 2 :domains '(:any :any) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (- x y)))
   (list :name :mul :n 2 :domains '(:any :any) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (* x y)))
   (list :name :div :n 2 :domains '(:any :positive) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (/ x y)))
   (list :name :exp :n 1 :domains '(:any) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x) (exp x)))
   (list :name :log :n 1 :domains '(:positive) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x) (log x)))
   (list :name :tanh :n 1 :domains '(:any) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x) (tanh x)))
   (list :name :max :n 2 :domains '(:any :distinct) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (max x y)))
   (list :name :min :n 2 :domains '(:any :distinct) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (min x y)))
   (list :name :select :n 2 :domains '(:any :distinct) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (nb:where (< x y) (* x y) (exp y))))
   (list :name :composite :n 2 :domains '(:any :any) :dtypes '(:f32 :f64)
         :fn (nb:with-tracing (x y) (tanh (* x (exp y)))))
   ;; 出力が :i1 / 別 dtype のもの: 中心差分の対象外（:cd nil）。
   (list :name :compare :n 2 :domains '(:any :any) :dtypes '(:f32 :f64) :cd nil
         :fn (nb:with-tracing (x y) (< x y)))
   (list :name :convert-up :n 1 :domains '(:any) :dtypes '(:f32) :cd nil
         :fn (nb:with-tracing (x) (nb:convert x :f64)))
   (list :name :convert-down :n 1 :domains '(:any) :dtypes '(:f64) :cd nil
         :fn (nb:with-tracing (x) (nb:convert x :f32)))
   (list :name :convert-bf16 :n 1 :domains '(:any) :dtypes '(:f32) :cd nil
         :fn (nb:with-tracing (x) (nb:convert x :bf16))))
  "jvp ルールのテスト表。各要素は plist: :NAME :N（入力数）:DOMAINS（入力ごとの
定義域 :ANY / :POSITIVE / :DISTINCT（直前の入力から 0.1 以上離す））:DTYPES
（試す入力の dtype）:FN（with-tracing した関数）:CD（NIL なら中心差分の対象外、
既定は対象）。")

(defun %ew-case-name (case) (getf case :name))

(defun %ew-inputs (case shape dtype seed &key tangent)
  "CASE の各入力の決定的な乱数配列（定義域つき）。TANGENT が真なら接線用（定義域なし）。"
  (let ((arrays (loop for domain in (getf case :domains)
                      for i from 0
                      collect (make-random-array (make-array-spec shape dtype)
                                                 :seed (+ seed (* 7 i) (if tangent 1000 0))
                                                 :domain (if (eq domain :positive) :positive :any)))))
    (if tangent
        arrays
        (let ((previous nil))
          (loop for array in arrays
                for domain in (getf case :domains)
                collect (setf previous (if (eq domain :distinct)
                                           (%separate-from previous array)
                                           array)))))))

(defun %separate-from (base array)
  "ARRAY の各要素を、BASE の対応する要素から 0.1 以上離れた値にした新しい配列。"
  (let ((result (make-array (array-dimensions array) :element-type (array-element-type array))))
    (dotimes (i (array-total-size array) result)
      (let ((offset (row-major-aref array i)))
        (setf (row-major-aref result i)
              (+ (row-major-aref base i)
                 (* (if (minusp offset) -1 1) (+ 1/10 (abs offset)))))))))

(defun %ew-graph (case dtype shape)
  (nb:trace-to-graph (getf case :fn)
                     (loop repeat (getf case :n) collect (nb:make-aval shape dtype))))

(defun %ew-shape (seed)
  (loop for k from 1 to (mod seed 3) collect (1+ (mod (+ seed k) 3))))

(defun %ew-eval (graph arrays)
  (multiple-value-list (apply #'nb:eval-graph graph arrays)))

(defun %ew-close-p (actual expected &key rtol atol)
  (and (= (length actual) (length expected))
       (every (lambda (a e) (allclose a e :dtype (if (eq (array-element-type a) 'double-float) :f64 :f32)
                                          :rtol rtol :atol atol))
              actual expected)))

(defmacro %do-ew-cases ((case-var &key (filter t)) &body body)
  `(dolist (,case-var (remove-if-not (lambda (c) (declare (ignorable c)) ,filter) *elementwise-jvp-cases*))
     ,@body))

(test jvp-elementwise/tangent-matches-central-difference-f64
  "各ルールの接線が f64 の中心差分と許容誤差で一致する（compare / convert は対象外）。"
  (%do-ew-cases (case :filter (getf c :cd t))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((shape (%ew-shape seed))
                           (graph (%ew-graph case :f64 shape))
                           (primals (%ew-inputs case shape :f64 seed))
                           (tangents (%ew-inputs case shape :f64 seed :tangent t))
                           (result (%ew-eval (nb::jvp-graph graph) (append primals tangents)))
                           (n-out (length (nb:graph-outvars graph))))
                      (%ew-close-p (subseq result n-out)
                                   (central-difference-jvp graph primals tangents)
                                   :rtol *autodiff-rtol* :atol *autodiff-atol*)))
                  :regression-id jvp-elementwise/tangent-matches-central-difference-f64
                  :regression-file (regression-path "jvp-elementwise-central-difference"))
        "~S" (%ew-case-name case))))

(test jvp-elementwise/tangent-is-linear
  "接線について線形: jvp(3·v) = 3·jvp(v)、jvp(v + w) = jvp(v) + jvp(w)（主値は固定）。"
  (%do-ew-cases (case :filter (getf c :cd t))
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let* ((shape (%ew-shape seed))
                           (graph (%ew-graph case :f64 shape))
                           (jvp (nb::jvp-graph graph))
                           (n-out (length (nb:graph-outvars graph)))
                           (primals (%ew-inputs case shape :f64 seed))
                           (v (%ew-inputs case shape :f64 seed :tangent t))
                           (w (%ew-inputs case shape :f64 (+ seed 500) :tangent t)))
                      (flet ((tangent-of (tangents)
                               (subseq (%ew-eval jvp (append primals tangents)) n-out))
                             (scaled (arrays) (mapcar (lambda (a) (%scale-array a 3)) arrays)))
                        (and (%ew-close-p (tangent-of (scaled v)) (scaled (tangent-of v))
                                          :rtol 1d-10 :atol 1d-12)
                             (%ew-close-p (tangent-of (mapcar #'%sum-array v w))
                                          (mapcar #'%sum-array (tangent-of v) (tangent-of w))
                                          :rtol 1d-10 :atol 1d-12)))))
                  :regression-id jvp-elementwise/tangent-is-linear
                  :regression-file (regression-path "jvp-elementwise-linear"))
        "~S" (%ew-case-name case))))

(test jvp-elementwise/tangent-aval-matches-primal-aval
  "jvp graph の出力の後半（接線）の aval は前半（主値）の aval と一致し、graph は
check-graph を満たす（compare は :i1 の主値に :i1 の接線が付く）。"
  (%do-ew-cases (case)
    (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                  (lambda (seed)
                    (let ((shape (%ew-shape seed)))
                      (every (lambda (dtype)
                               (let* ((jvp (nb::jvp-graph (%ew-graph case dtype shape)))
                                      (avals (mapcar #'nb:var-aval (nb:graph-outvars jvp)))
                                      (half (/ (length avals) 2)))
                                 (and (nb::check-graph jvp)
                                      (integerp half)
                                      (equalp (subseq avals 0 half) (subseq avals half)))))
                             (getf case :dtypes))))
                  :regression-id jvp-elementwise/tangent-aval-matches-primal-aval
                  :regression-file (regression-path "jvp-elementwise-aval"))
        "~S" (%ew-case-name case))))

(defun %jvp-tangent-of-binary (fn x y tx ty)
  "FN（2入力）の jvp の接線出力（1つ）を、f64 の配列 X Y TX TY について返す。"
  (let* ((aval (nb:make-aval (array-dimensions x) :f64))
         (graph (nb:trace-to-graph fn (list aval aval))))
    (second (%ew-eval (nb::jvp-graph graph) (list x y tx ty)))))

(defun %f64-vector (&rest numbers)
  (make-array (length numbers) :element-type 'double-float
                               :initial-contents (mapcar (lambda (n) (coerce n 'double-float)) numbers)))

(test jvp-elementwise/max-min-ties-split-evenly
  "max / min で2入力が等しい要素では、接線は各側に 0.5 ずつ（JAX の _balanced_eq と
同じ）。等しくない要素では大きい（小さい）方の接線だけが通る。"
  (let ((x (%f64-vector 1 2 3 4))
        (y (%f64-vector 1 5 3 0))
        (tx (%f64-vector 10 20 30 40))
        (ty (%f64-vector 100 200 300 400)))
    (is (equalp (%f64-vector 55 200 165 40)
                (%jvp-tangent-of-binary (nb:with-tracing (x y) (max x y)) x y tx ty)))
    (is (equalp (%f64-vector 55 20 165 400)
                (%jvp-tangent-of-binary (nb:with-tracing (x y) (min x y)) x y tx ty)))))

(test jvp-elementwise/compare-tangent-is-all-false
  "compare の接線は :i1 で全 false（symbolic zero を instantiate したもの）。"
  (let* ((aval (nb:make-aval '(3) :f64))
         (graph (nb:trace-to-graph (nb:with-tracing (x y) (< x y)) (list aval aval)))
         (jvp (nb::jvp-graph graph))
         (result (%ew-eval jvp (list (%f64-vector 1 2 3) (%f64-vector 3 2 1)
                                     (%f64-vector 1 1 1) (%f64-vector 1 1 1)))))
    (is (equalp #*100 (first result)))
    (is (equalp #*000 (second result)))
    (is (eq :i1 (nb:aval-dtype (nb:var-aval (second (nb:graph-outvars jvp))))))))

(test jvp-elementwise/select-passes-tangent-of-chosen-branch-and-ignores-pred
  "select の接線は選ばれた側の接線。片方の枝の接線がゼロでも動く（ゼロの枝は 0 で
埋める）。pred を作る入力の接線は無視される。"
  (let* ((aval (nb:make-aval '(3) :f64))
         (x (%f64-vector 1 5 3))
         (y (%f64-vector 2 2 2))
         (ones (%f64-vector 1 1 1))
         ;; 接線を渡すのは x だけ（y の接線はゼロ）: out = where(x < y, x, 7)。
         (graph (nb:trace-to-graph (nb:with-tracing (x y) (nb:where (< x y) x 7d0)) (list aval aval)))
         (jvp (nb::jvp-graph graph :nonzero '(t nil)))
         (result (%ew-eval jvp (list x y (%f64-vector 10 20 30)))))
    (declare (ignore ones))
    (is (equalp (%f64-vector 10 0 0) (second result)))))

(test jvp-elementwise/convert-tangent-is-converted-tangent
  "convert の接線は接線を同じ dtype へ convert したもの（float → float）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (let* ((shape (%ew-shape seed))
                         (case (find :convert-down *elementwise-jvp-cases* :key #'%ew-case-name))
                         (graph (%ew-graph case :f64 shape))
                         (primals (%ew-inputs case shape :f64 seed))
                         (tangents (%ew-inputs case shape :f64 seed :tangent t))
                         (result (%ew-eval (nb::jvp-graph graph) (append primals tangents))))
                    (equalp (second result) (nb:convert (first tangents) :f32))))
                :regression-id jvp-elementwise/convert-tangent-is-converted-tangent
                :regression-file (regression-path "jvp-elementwise-convert"))))

(test jvp-elementwise/every-elementwise-primitive-has-a-jvp-rule
  "sub mul div exp log tanh max min convert compare select neg add に jvp ルールがある。"
  (dolist (name '(:add :neg :sub :mul :div :exp :log :tanh :max :min :convert :compare :select))
    (is (nb::primitive-jvp (nb::find-primitive name)) "~S" name)))
