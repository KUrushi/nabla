;;;; 整数 dtype（:i32 / :u32 / :u64）の性質（issue #126、small）。
;;;;
;;;; 整数は *dtypes* に入れず *integer-dtypes*（tests/support/dtypes.lisp）で
;;;; 別に生成する。生成器 MAKE-RANDOM-INTEGER-ARRAY は端の値（最小・最大・
;;;; その隣）を半分混ぜるので、加減乗は高い確率でオーバーフローする。
;;;; IREE との一致は tests/iree/integer-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %int-bits (dtype) (ecase dtype ((:i32 :u32) 32) (:u64 64)))

(defun %ref-wrap (dtype exact)
  "数学的に厳密な整数 EXACT を DTYPE の範囲に2の補数で折り返した値。
オラクルとして src/dtype.lisp の WRAP-INTEGER とは別に書く（算術で書く）。"
  (let* ((modulus (expt 2 (%int-bits dtype)))
         (r (mod exact modulus)))
    (if (and (eq dtype :i32) (>= r (/ modulus 2))) (- r modulus) r)))

(defun %int-spec-seed-generator ()
  (generator (tuple (array-spec :dtypes *integer-dtypes*)
                    (uniform-integer :lo 0 :hi (1- (expt 2 31))))))

(defun %trace-eval (fn avals &rest arrays)
  "FN（WITH-TRACING の関数）を AVALS でトレースし、ARRAYS で eager 評価した結果。"
  (apply #'nb:eval-graph (nb:trace-to-graph fn avals) arrays))

;;; --- dtype の基本 ---

(test integer-dtype/basics
  "整数 dtype の Lisp 要素型・バイト幅・MLIR の綴りが決まった値になる。"
  (is (equal '(signed-byte 32) (nb:dtype-element-type :i32)))
  (is (equal '(unsigned-byte 32) (nb:dtype-element-type :u32)))
  (is (equal '(unsigned-byte 64) (nb:dtype-element-type :u64)))
  (is (equal '(4 4 8) (mapcar #'nb:dtype-byte-width '(:i32 :u32 :u64))))
  (is (equal '("i32" "ui32" "ui64") (mapcar #'nb::dtype-mlir-name '(:i32 :u32 :u64)))))

(test integer-dtype/array-dtype-infers-without-argument
  "整数配列は DTYPE を渡さなくても array-dtype が推論でき、食い違う DTYPE は
dtype-mismatch になる。"
  (is (check-it (%int-spec-seed-generator)
                (lambda (spec-and-seed)
                  (destructuring-bind (spec seed) spec-and-seed
                    (let ((array (make-random-array spec :seed seed))
                          (dtype (array-spec-dtype spec)))
                      (and (eq dtype (nb:array-dtype array))
                           (eq dtype (nb:array-dtype array dtype))
                           (handler-case (progn (nb:array-dtype array :f32) nil)
                             (nb:dtype-mismatch () t))))))
                :regression-id integer-dtype/array-dtype-infers-without-argument
                :regression-file (regression-path "integer-dtype-array-dtype"))))

(test integer-dtype/generator-hits-edges
  "整数の生成器は端の値（最小・最大）を実際に出す（オーバーフローを試すため）。"
  (let ((seen (make-hash-table)))
    (dotimes (seed 20)
      (let ((a (make-random-array (make-array-spec '(8) :i32) :seed seed)))
        (dotimes (i 8) (setf (gethash (aref a i) seen) t))))
    (is (gethash -2147483648 seen))
    (is (gethash 2147483647 seen))))

;;; --- eager の算術はオーバーフローで折り返す ---

(defmacro %def-binary-wrap-test (name op exact-form)
  `(test ,name
     ,(format nil "整数の (~(~A~) a b) の eager 結果は、厳密な整数演算の結果を
dtype の範囲に折り返した値と全要素で一致する（オーバーフローを含む）。" op)
     (is (check-it (%int-spec-seed-generator)
                   (lambda (spec-and-seed)
                     (destructuring-bind (spec seed) spec-and-seed
                       (let* ((dtype (array-spec-dtype spec))
                              (aval (nb:make-aval (array-spec-shape spec) dtype))
                              (a (make-random-array spec :seed seed))
                              (b (make-random-array spec :seed (1+ seed)))
                              (result (%trace-eval (nb:with-tracing (x y) (,op x y))
                                                   (list aval aval) a b)))
                         (and (equalp (nb:array-aval result dtype) aval)
                              (dotimes (i (array-total-size a) t)
                                (let ((x (row-major-aref a i)) (y (row-major-aref b i)))
                                  (declare (ignorable x y))
                                  (unless (= (row-major-aref result i)
                                             (%ref-wrap dtype ,exact-form))
                                    (return nil))))))))
                   :regression-id ,name
                   :regression-file (regression-path ,(format nil "~(~A~)" (substitute #\- #\/ (string name))))))))

(%def-binary-wrap-test integer-dtype/add-wraps + (+ x y))
(%def-binary-wrap-test integer-dtype/sub-wraps - (- x y))
(%def-binary-wrap-test integer-dtype/mul-wraps * (* x y))
(%def-binary-wrap-test integer-dtype/max-is-exact max (max x y))
(%def-binary-wrap-test integer-dtype/min-is-exact min (min x y))

(test integer-dtype/neg-wraps
  "(- x) は 2 の補数の否定（-2^31 は -2^31 のまま、符号なしは 2^n - x）。"
  (is (check-it (%int-spec-seed-generator)
                (lambda (spec-and-seed)
                  (destructuring-bind (spec seed) spec-and-seed
                    (let* ((dtype (array-spec-dtype spec))
                           (aval (nb:make-aval (array-spec-shape spec) dtype))
                           (a (make-random-array spec :seed seed))
                           (result (%trace-eval (nb:with-tracing (x) (- x)) (list aval) a)))
                      (dotimes (i (array-total-size a) t)
                        (unless (= (row-major-aref result i) (%ref-wrap dtype (- (row-major-aref a i))))
                          (return nil))))))
                :regression-id integer-dtype/neg-wraps
                :regression-file (regression-path "integer-dtype-neg"))))

(test integer-dtype/overflow-examples
  "代表的なオーバーフローの具体例（StableHLO と同じ折り返し）。"
  (flet ((i32 (&rest xs) (make-array (length xs) :element-type '(signed-byte 32) :initial-contents xs))
         (u32 (&rest xs) (make-array (length xs) :element-type '(unsigned-byte 32) :initial-contents xs))
         (u64 (&rest xs) (make-array (length xs) :element-type '(unsigned-byte 64) :initial-contents xs)))
    (let ((f (nb:with-tracing (x) (+ x 1))))
      (is (equalp (i32 -2147483648) (%trace-eval f (list (nb:make-aval '(1) :i32)) (i32 2147483647))))
      (is (equalp (u32 0) (%trace-eval f (list (nb:make-aval '(1) :u32)) (u32 4294967295))))
      (is (equalp (u64 0) (%trace-eval f (list (nb:make-aval '(1) :u64)) (u64 18446744073709551615)))))
    (is (equalp (i32 -2147483648)
                (%trace-eval (nb:with-tracing (x) (- x)) (list (nb:make-aval '(1) :i32))
                             (i32 -2147483648))))
    (is (equalp (u32 1)
                (%trace-eval (nb:with-tracing (x y) (* x y)) (list (nb:make-aval '(1) :u32) (nb:make-aval '(1) :u32))
                             (u32 4294967295) (u32 4294967295))))))

(test integer-dtype/literal-lifts-to-the-operand-dtype
  "整数リテラルは相手の整数 dtype にリフトされる。浮動小数点リテラルや範囲外の
整数は tracing-error。浮動小数点のトレーサ + 整数リテラルは従来どおり :f32。"
  (let ((graph (nb:trace-to-graph (nb:with-tracing (x) (+ x 1)) (list (nb:make-aval '(2) :u32)))))
    (is (eq :u32 (nb:aval-dtype (nb:var-aval (first (nb:graph-outvars graph)))))))
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (+ x 1.5)) (list (nb:make-aval '(2) :i32))))
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (+ x -1)) (list (nb:make-aval '(2) :u32))))
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (+ x 2147483648)) (list (nb:make-aval '(2) :i32))))
  (let ((graph (nb:trace-to-graph (nb:with-tracing (x) (+ x 1)) (list (nb:make-aval '(2) :f32)))))
    (is (eq :f32 (nb:aval-dtype (nb:var-aval (first (nb:graph-outvars graph))))))))

;;; --- compare / select / reduce / shape ---

(test integer-dtype/compare-and-select-are-exact
  "整数の比較は符号を考慮した厳密な比較（u64 の 2^63 以上も正しく大きい）で、
select は条件どおりに生の値を選ぶ。"
  (is (check-it (%int-spec-seed-generator)
                (lambda (spec-and-seed)
                  (destructuring-bind (spec seed) spec-and-seed
                    (let* ((dtype (array-spec-dtype spec))
                           (aval (nb:make-aval (array-spec-shape spec) dtype))
                           (a (make-random-array spec :seed seed))
                           (b (make-random-array spec :seed (1+ seed)))
                           (lt (%trace-eval (nb:with-tracing (x y) (< x y)) (list aval aval) a b))
                           (picked (%trace-eval (nb:with-tracing (x y) (if (< x y) x y))
                                                (list aval aval) a b)))
                      (and (eq :i1 (nb:array-dtype lt))
                           (dotimes (i (array-total-size a) t)
                             (let ((x (row-major-aref a i)) (y (row-major-aref b i)))
                               (unless (and (= (row-major-aref lt i) (if (< x y) 1 0))
                                            (= (row-major-aref picked i) (if (< x y) x y)))
                                 (return nil))))))))
                :regression-id integer-dtype/compare-and-select-are-exact
                :regression-file (regression-path "integer-dtype-compare-select"))))

(test integer-dtype/reduce-sum-wraps-and-reduce-max-is-exact
  "reduce-sum は折り返す総和、reduce-max は最大値（負の i32 だけの配列でも
-2^31 を初期値にして正しく返す）。"
  (is (check-it (generator (tuple (array-spec :dtypes *integer-dtypes* :max-rank 1 :max-dim 8)
                                  (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (spec-and-seed)
                  (destructuring-bind (spec seed) spec-and-seed
                    (let* ((dtype (array-spec-dtype spec))
                           (shape (array-spec-shape spec))
                           (aval (nb:make-aval shape dtype)))
                      (if (null shape)
                          t
                          (let* ((a (make-random-array spec :seed seed))
                                 (elements (loop for i below (array-total-size a) collect (row-major-aref a i)))
                                 (sum (%trace-eval (nb:with-tracing (x) (nb:reduce-sum x :axes '(0)))
                                                   (list aval) a))
                                 (mx (%trace-eval (nb:with-tracing (x) (nb:reduce-max x :axes '(0)))
                                                  (list aval) a)))
                            (and (= (aref sum) (%ref-wrap dtype (reduce #'+ elements)))
                                 (= (aref mx) (reduce #'max elements))))))))
                :regression-id integer-dtype/reduce-sum-wraps-and-reduce-max-is-exact
                :regression-file (regression-path "integer-dtype-reduce"))))

(test integer-dtype/shape-ops-preserve-values
  "reshape / transpose / broadcast-in-dim は整数の値をそのまま動かす。"
  (let* ((a (make-array '(2 3) :element-type '(signed-byte 32)
                              :initial-contents '((-2147483648 -1 0) (1 2 2147483647))))
         (aval (nb:make-aval '(2 3) :i32)))
    (is (equalp (make-array '(3 2) :element-type '(signed-byte 32)
                                   :initial-contents '((-2147483648 -1) (0 1) (2 2147483647)))
                (%trace-eval (nb:with-tracing (x) (nb:reshape x '(3 2))) (list aval) a)))
    (is (equalp (make-array '(3 2) :element-type '(signed-byte 32)
                                   :initial-contents '((-2147483648 1) (-1 2) (0 2147483647)))
                (%trace-eval (nb:with-tracing (x) (nb:transpose x '(1 0))) (list aval) a)))
    (is (equalp (make-array '(2 2 3) :element-type '(signed-byte 32)
                                     :initial-contents '(((-2147483648 -1 0) (1 2 2147483647))
                                                         ((-2147483648 -1 0) (1 2 2147483647))))
                (%trace-eval (nb:with-tracing (x) (nb:broadcast-in-dim x '(2 2 3) '(1 2)))
                             (list aval) a)))))

;;; --- convert ---

(test integer-dtype/convert-int-float-i1
  "convert: 整数 → 浮動小数点は値を保ち、浮動小数点 → 整数は 0 方向への丸め
（NaN は 0）、整数どうしは折り返し、→ :i1 は 0 以外が 1、:i1 → 整数 / 浮動小数点は 0 / 1。"
  (flet ((conv (array from to)
           (%trace-eval (nb:with-tracing (x) (nb:convert x to))
                        (list (nb:make-aval (array-dimensions array) from)) array))
         (vec (type &rest xs) (make-array (length xs) :element-type type :initial-contents xs)))
    (is (equalp (vec 'single-float -3.0 0.0 7.0)
                (conv (vec '(signed-byte 32) -3 0 7) :i32 :f32)))
    (is (equalp (vec 'double-float 4294967295d0)
                (conv (vec '(unsigned-byte 32) 4294967295) :u32 :f64)))
    (is (equalp (vec '(signed-byte 32) -2 0 3 0)
                (conv (vec 'single-float -2.7 0.5 3.9 -0.5) :f32 :i32)))
    (is (equalp (vec '(unsigned-byte 32) 4294967295 0 1)
                (conv (vec '(signed-byte 32) -1 0 1) :i32 :u32)))
    (is (equalp (vec '(signed-byte 32) -1)
                (conv (vec '(unsigned-byte 64) 18446744073709551615) :u64 :i32)))
    ;; 範囲外・無限大は端に飽和、NaN は 0（どのバックエンドでも同じ）
    (is (equalp (vec '(signed-byte 32) -2147483648 2147483647 0 2147483647 -2147483648)
                (conv (vec 'single-float -1f10 1f10 (nb::%quiet-nan 'single-float)
                           sb-ext:single-float-positive-infinity
                           sb-ext:single-float-negative-infinity)
                      :f32 :i32)))
    (is (equalp (vec '(unsigned-byte 32) 0 4294967295 4294967295)
                (conv (vec 'double-float -5d0 1d12 4294967296d0) :f64 :u32)))
    (is (equalp (vec '(unsigned-byte 64) 0 18446744073709551615 0)
                (conv (vec 'single-float -3f0 1f30 -0.9f0) :f32 :u64)))
    (is (equalp (vec 'bit 1 0 1)
                (conv (vec '(signed-byte 32) -5 0 9) :i32 :i1)))
    (is (equalp (vec 'bit 0 1)
                (conv (vec 'single-float 0.0 0.25) :f32 :i1)))
    (is (equalp (vec '(signed-byte 32) 0 1)
                (conv (vec 'bit 0 1) :i1 :i32)))
    (is (equalp (vec 'single-float 0.0 1.0)
                (conv (vec 'bit 0 1) :i1 :f32)))))

;;; --- 拒否 ---

(test integer-dtype/rejects-float-only-primitives-at-trace-time
  "div / exp / log / tanh / dot-general は整数を拒否し、トレース時に
primitive-error を signal する。"
  (dolist (dtype *integer-dtypes*)
    (let ((v (nb:make-aval '(2) dtype))
          (m (nb:make-aval '(2 2) dtype)))
      (signals nb:primitive-error (nb:trace-to-graph (nb:with-tracing (x y) (/ x y)) (list v v)))
      (signals nb:primitive-error (nb:trace-to-graph (nb:with-tracing (x) (exp x)) (list v)))
      (signals nb:primitive-error (nb:trace-to-graph (nb:with-tracing (x) (log x)) (list v)))
      (signals nb:primitive-error (nb:trace-to-graph (nb:with-tracing (x) (tanh x)) (list v)))
      (signals nb:primitive-error (nb:trace-to-graph (nb:with-tracing (x y) (nb:dot x y)) (list m m))))))

(test integer-dtype/rejects-mixed-dtypes
  "整数と浮動小数点を直接足すと primitive-error（暗黙の型昇格はしない）。"
  (signals nb:primitive-error
    (nb:trace-to-graph (nb:with-tracing (x y) (+ x y))
                       (list (nb:make-aval '(2) :i32) (nb:make-aval '(2) :f32))))
  (signals nb:primitive-error
    (nb:trace-to-graph (nb:with-tracing (x y) (+ x y))
                       (list (nb:make-aval '(2) :i32) (nb:make-aval '(2) :u32)))))

;;; --- グラフの印字・読み込み ---

(test integer-dtype/print-graph-round-trips
  "整数の定数と入力を持つ graph が print-graph / read-graph で往復する
（ui32 / ui64 は dtype タグ :u32 / :u64 に戻る）。"
  (dolist (dtype *integer-dtypes*)
    (let* ((graph (nb:trace-to-graph (nb:with-tracing (x) (+ x 1)) (list (nb:make-aval '(2) dtype))))
           (text (nb:print-graph graph))
           (again (nb::read-graph text)))
      (is (equal text (nb:print-graph again))))))

;;; --- 自動微分 ---

(test integer-dtype/grad-wrt-integer-input-is-autodiff-error
  "整数の入力に対する grad は autodiff-error。"
  (let ((f (nb:with-tracing (x) (nb:reduce-sum (nb:convert x :f32))))
        (x (make-array 3 :element-type '(signed-byte 32) :initial-contents '(1 2 3))))
    (signals nb:autodiff-error (funcall (nb:grad f) x))))

(test integer-dtype/gradient-through-integer-path-is-zero
  "浮動小数点 → 整数 → 浮動小数点を通る計算の勾配は全部ゼロ（整数の接線は
常に symbolic zero）。整数の経路と浮動小数点の経路を足した関数の勾配は、
浮動小数点の経路の分だけ。"
  (let* ((x (make-array 3 :element-type 'single-float :initial-contents '(1.5 -2.5 3.5)))
         (through-int (nb:with-tracing (x)
                        (nb:reduce-sum (* (nb:convert (nb:convert x :i32) :f32) 2.0))))
         (mixed (nb:with-tracing (x)
                  (nb:reduce-sum (+ (nb:convert (nb:convert x :i32) :f32) (* x 3.0))))))
    (is (equalp (make-array 3 :element-type 'single-float :initial-element 0.0)
                (funcall (nb:grad through-int) x)))
    (is (equalp (make-array 3 :element-type 'single-float :initial-element 3.0)
                (funcall (nb:grad mixed) x)))))

(test integer-dtype/instantiate-zero-of-integer-tangent
  "整数の symbolic zero は、その dtype・shape の 0 の配列に実体化される。"
  (dolist (dtype *integer-dtypes*)
    (let* ((aval (nb:make-aval '(2 3) dtype))
           (graph (nb::%call-with-fresh-trace
                   '()
                   (lambda () (nb::instantiate-zero (nb::make-symbolic-zero aval))))))
      (let ((result (nb:eval-graph graph)))
        (is (equalp (nb:array-aval result dtype) aval))
        (is (every #'zerop (loop for i below 6 collect (row-major-aref result i))))))))

(test integer-dtype/clamp-upper-bound-is-the-largest-float-not-above-the-maximum
  "StableHLO の clamp の上限（%FLOAT-CLAMP-UPPER-BOUND）は、整数の最大値以下で
表せる最大の浮動小数点数: 最大値を超えず、1つ上の浮動小数点数は最大値を超える
（f32 の i32 では 2^31 - 128、f64 の i32 では最大値そのもの）。"
  (dolist (dtype *integer-dtypes*)
    (dolist (float-type '(single-float double-float))
      (multiple-value-bind (lo hi) (nb::%integer-range dtype)
        (declare (ignore lo))
        (let* ((bound (nb::%float-clamp-upper-bound hi float-type))
               ;; 1つ上の浮動小数点数（仮数部を1進める）を有理数で
               (next (multiple-value-bind (mantissa exponent) (integer-decode-float bound)
                       (* (1+ mantissa) (expt 2 exponent)))))
          (is (<= (rational bound) hi) "~A ~A" dtype float-type)
          (is (> next hi) "~A ~A" dtype float-type)))))
  (is (= 2147483520 (nb::%float-clamp-upper-bound 2147483647 'single-float)))
  (is (= 2147483647 (nb::%float-clamp-upper-bound 2147483647 'double-float))))

(test integer-dtype/integer-range-and-wrap-agree
  "整数の範囲の端 ±1 は WRAP-INTEGER で反対の端に折り返す。"
  (dolist (dtype *integer-dtypes*)
    (multiple-value-bind (lo hi) (nb::%integer-range dtype)
      (is (= lo (nb::wrap-integer lo dtype)))
      (is (= hi (nb::wrap-integer hi dtype)))
      (is (= lo (nb::wrap-integer (1+ hi) dtype)))
      (is (= hi (nb::wrap-integer (1- lo) dtype))))))

(test integer-dtype/reduce-max-init-and-literals-are-the-dtype-minimum
  "reduce-max の初期値（eager・StableHLO のリテラルとも）は整数 dtype の最小値。
最小値だけの配列でも最大値が最小値のまま返る。"
  (dolist (dtype *integer-dtypes*)
    (multiple-value-bind (lo hi) (nb::%integer-range dtype)
      (declare (ignore hi))
      (is (= lo (nb::%reduce-integer-init :max dtype)))
      (is (= 0 (nb::%reduce-integer-init :add dtype)))
      (is (equal (format nil "~D" lo) (nb::%reduce-init-literal :max dtype)))
      (is (equal "0" (nb::%reduce-init-literal :add dtype)))
      (let* ((array (make-array 3 :element-type (nb:dtype-element-type dtype) :initial-element lo))
             (result (%trace-eval (nb:with-tracing (x) (nb:reduce-max x :axes '(0)))
                                  (list (nb:make-aval '(3) dtype)) array)))
        (is (= lo (aref result)))))))

;;; --- StableHLO の出力（IREE の medium テストで実行して確かめたものの、
;;; 構造を small で固定する。mutation testing は small だけで走るため） ---

(defun %convert-emit-lines (from to)
  "FROM → TO の convert の emit を、(shape (4)) で行のリストにして返す。"
  (let ((text (nb::%convert-emit '("%a") (list (nb:make-aval '(4) from)) "%7" (nb:make-aval '(4) to))))
    (with-input-from-string (in text)
      (loop for line = (read-line in nil) while line collect (string-trim " " line)))))

(defun %defined-names (lines)
  (mapcar (lambda (line) (subseq line 0 (position #\Space line))) lines))

(test integer-dtype/convert-emit/int-to-bf16-goes-through-f32-with-a-barrier
  "整数 → bf16 は f32 への convert、optimization_barrier、bf16 への convert の3行で、
各行が前の行の結果を使い、最後の行が出力名を定義する。"
  (dolist (from *integer-dtypes*)
    (let ((lines (%convert-emit-lines from :bf16)))
      (is (= 3 (length lines)))
      (is (= 3 (length (remove-duplicates (%defined-names lines) :test #'string=))))
      (is (search "stablehlo.convert %a" (first lines)))
      (is (search "-> tensor<4xf32>" (first lines)))
      (is (search "stablehlo.optimization_barrier" (second lines)))
      (is (search (first (%defined-names lines)) (second lines)))
      (is (search (second (%defined-names lines)) (third lines)))
      (is (string= "%7" (third (%defined-names lines))))
      (is (search "-> tensor<4xbf16>" (third lines))))))

(test integer-dtype/convert-emit/other-int-conversions-are-one-line
  "整数 → f32 / f16、整数どうし、:i1 との convert は1行のまま。"
  (is (= 1 (length (%convert-emit-lines :i32 :f32))))
  (is (= 1 (length (%convert-emit-lines :u64 :f16))))
  (is (= 1 (length (%convert-emit-lines :i32 :u32))))
  (is (= 1 (length (%convert-emit-lines :i1 :i32))))
  (is (= 1 (length (%convert-emit-lines :f32 :i1)))))

(test integer-dtype/convert-emit/float-to-int-saturates-in-stablehlo
  "浮動小数点 → 整数は、NaN 判定・clamp・convert・上端の select・NaN → 0 の select を
含み、f32 / f64 は入力型のまま、f16 / bf16 は先頭で f32 に convert してから進める。
各行の定義名はすべて違い、最後の行が出力名を定義する。"
  (dolist (from '(:f32 :f64 :f16 :bf16))
    (let* ((lines (%convert-emit-lines from :i32))
           (names (%defined-names lines))
           (float-type (if (eq from :f64) "tensor<4xf64>" "tensor<4xf32>"))
           (joined (format nil "~{~A~^~%~}" lines)))
      (is (= (length names) (length (remove-duplicates names :test #'string=))))
      (is (string= "%7" (car (last names))))
      (is (search "stablehlo.compare NE" joined))
      (is (search "stablehlo.clamp" joined))
      (is (search "stablehlo.compare GE" joined))
      (is (= 2 (count-if (lambda (l) (search "stablehlo.select" l)) lines)))
      (is (search (format nil "~A" float-type) joined))
      (if (member from '(:f16 :bf16))
          (progn
            (is (search "stablehlo.convert %a" (first lines)))
            (is (search "-> tensor<4xf32>" (first lines)))
            (is (search "stablehlo.compare NE" (second lines))))
          (progn
            (is (search "stablehlo.compare NE" (first lines)))
            (is (not (search "stablehlo.convert %a" joined)))
            (is (search "stablehlo.convert" (find-if (lambda (l) (search "stablehlo.convert" l)) lines)))))
      ;; 上端: f32 では 2^31 の1つ下の 2147483520（最短の10進で 2.1474835e9）、f64 では 2147483647 そのもの
      (is (search (if (eq from :f64) "dense<2.147483647e9> : tensor<4xf64>" "dense<2.1474835e9> : tensor<4xf32>") joined))
      (is (search "dense<2147483647> : tensor<4xi32>" joined)))))

(test integer-dtype/read-graph-rejects-bad-dtypes
  "read-graph は、dtype として symbol でないもの・未知の symbol を graph-syntax-error にする。"
  (let ((text (nb:print-graph (nb:trace-to-graph (nb:with-tracing (x) (+ x 1))
                                                 (list (nb:make-aval '(2) :i32))))))
    (is (stringp text))
    (signals nb::graph-syntax-error (nb::read-graph (substitute-string text "i32" "q99")))
    (signals nb::graph-syntax-error (nb::read-graph (substitute-string text "i32" "7")))))

(defun substitute-string (text old new)
  (let ((position (search old text)))
    (concatenate 'string (subseq text 0 position) new (subseq text (+ position (length old))))))

(test integer-dtype/lifting-a-number-to-i1-is-a-tracing-error
  "数を :i1 のトレーサにリフトしようとすると tracing-error（整数 dtype へのリフトの
追加後も、:i1 は数値との対応を持たないので拒否したまま）。"
  (signals nb:tracing-error
    (nb:trace-to-graph (nb:with-tracing (x) (+ x 1)) (list (nb:make-aval '(2) :i1)))))

(test integer-dtype/reduce-max-init-literals-for-floats
  "reduce-max の初期値のリテラルは、浮動小数点では -inf の16進ビット列。"
  (is (equal "0xFF800000" (nb::%reduce-init-literal :max :f32)))
  (is (equal "0xFFF0000000000000" (nb::%reduce-init-literal :max :f64)))
  (is (equal "0xFF80" (nb::%reduce-init-literal :max :bf16)))
  (is (equal "0xFC00" (nb::%reduce-init-literal :max :f16))))
