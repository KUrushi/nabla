;;;; shift-right-logical / bitwise-or / bitcast-convert プリミティブの性質（issue #136、small）。
;;;;
;;;; 期待値は Lisp の整数演算（ASH / LOGIOR）と、手で書いた IEEE 754 のビット列で作る
;;;; （実装の %element-to-bits と同じ関数は使わない）。IREE / PJRT との一致は
;;;; tests/iree/prng-test.lisp と tests/pjrt/prng-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %bits-run (name arrays &rest params)
  "プリミティブ NAME（キーワード）の eager 実装を ARRAYS に適用する。"
  (apply (nb::primitive-eager (nb::find-primitive name))
         arrays (mapcar #'nb:array-aval arrays) params))

(defun %bits-random-array (shape dtype seed)
  "SEED から決まる整数 DTYPE の配列。端の値（0、最大、最小）も混ぜる。"
  (let ((rs (sb-ext:seed-random-state seed))
        (array (make-array shape :element-type (nb:dtype-element-type dtype)))
        (bits (ecase dtype ((:i32 :u32) 32) (:u64 64))))
    (dotimes (i (array-total-size array) array)
      (let ((raw (case (random 6 rs)
                   (0 0)
                   (1 (1- (expt 2 bits)))
                   (t (random (expt 2 bits) rs)))))
        (setf (row-major-aref array i)
              (if (and (eq dtype :i32) (>= raw (expt 2 31))) (- raw (expt 2 32)) raw))))))

(defparameter *bits-case-generator*
  (generator (tuple (array-spec :dtypes '(:f32))
                    (uniform-integer :lo 0 :hi 2)
                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
  "(形状の仕様 dtype の番号 シード)。")

(defmacro def-bits-property (name docstring (&rest vars) &body body)
  `(test ,name ,docstring
     (is (check-it *bits-case-generator*
                   (lambda (value)
                     (destructuring-bind (spec dtype-index seed) value
                       (let ((shape (array-spec-shape spec))
                             (dtype (nth dtype-index '(:i32 :u32 :u64))))
                         (declare (ignorable shape dtype seed))
                         (destructuring-bind ,vars (list shape dtype seed) ,@body))))
                   :regression-id ,name
                   :regression-file (regression-path
                                     ,(format nil "primitives-bits-~(~A~)"
                                              (substitute #\- #\/ (subseq (string name) (length "primitives/bits/"))))))
         ,(format nil "~A の性質が成り立たなかった" name))))

(def-bits-property primitives/bits/shift-right-logical-matches-unsigned-division
  "論理右シフトは、dtype を符号なしと見た値を 2^n で割って切り捨てた値と一致する。n はビット幅以上
（その場合は 0）や、符号付きの負の値（最上位ビットが立つ）も含む。"
  (shape dtype seed)
  (let* ((bits (ecase dtype ((:i32 :u32) 32) (:u64 64)))
         (x (%bits-random-array shape dtype seed))
         (n (%bits-random-array shape dtype (1+ seed)))
         ;; 量を [0, bits + 3] に絞る（符号なしの値で）
         (amount (let ((a (make-array shape :element-type (nb:dtype-element-type dtype))))
                   (dotimes (i (array-total-size a) a)
                     (setf (row-major-aref a i) (mod (ldb (byte bits 0) (row-major-aref n i)) (+ bits 4))))))
         (result (%bits-run :shift-right-logical (list x amount))))
    (and (equal (array-dimensions result) shape)
         (eq dtype (nb:array-dtype result))
         (dotimes (i (array-total-size x) t)
           (let* ((unsigned (mod (row-major-aref x i) (expt 2 bits)))
                  (expected (floor unsigned (expt 2 (row-major-aref amount i))))
                  (expected (if (and (eq dtype :i32) (>= expected (expt 2 31))) (- expected (expt 2 32)) expected)))
             (unless (= expected (row-major-aref result i)) (return nil)))))))

(def-bits-property primitives/bits/bitwise-or-matches-logior-and-is-commutative
  "bitwise-or は各要素で Lisp の LOGIOR（2の補数）と一致し、可換で、0 との or は元の値になる。"
  (shape dtype seed)
  (let* ((a (%bits-random-array shape dtype seed))
         (b (%bits-random-array shape dtype (1+ seed)))
         (zero (make-array shape :element-type (nb:dtype-element-type dtype) :initial-element 0))
         (ab (%bits-run :bitwise-or (list a b))))
    (and (equalp ab (%bits-run :bitwise-or (list b a)))
         (equalp a (%bits-run :bitwise-or (list a zero)))
         (eq dtype (nb:array-dtype ab))
         (dotimes (i (array-total-size a) t)
           (unless (= (logior (row-major-aref a i) (row-major-aref b i)) (row-major-aref ab i))
             (return nil))))))

(def-bits-property primitives/bits/bitcast-same-width-roundtrips-and-keeps-the-bits
  "同じ幅の bitcast（u32 ⇄ f32 / i32）は往復で元に戻り、u32 → i32 は2の補数の再解釈になる。"
  (shape dtype seed)
  (declare (ignorable dtype))
  (let* ((u (%bits-random-array shape :u32 seed))
         (as-f32 (%bits-run :bitcast-convert (list u) :dtype :f32))
         (as-i32 (%bits-run :bitcast-convert (list u) :dtype :i32)))
    (and (equal (array-dimensions as-f32) shape)
         ;; NaN を含むので、要素の比較は整数側で行う
         (equalp u (%bits-run :bitcast-convert (list as-f32) :dtype :u32))
         (equalp u (%bits-run :bitcast-convert (list as-i32) :dtype :u32))
         (dotimes (i (array-total-size u) t)
           (let ((x (row-major-aref u i)))
             (unless (= (row-major-aref as-i32 i) (if (>= x (expt 2 31)) (- x (expt 2 32)) x))
               (return nil)))))))

(def-bits-property primitives/bits/bitcast-u32-pairs-make-little-endian-u64
  "u32 の末尾の次元（長さ2）は u64 の (下位, 上位) になり（形は末尾の次元が消える）、
u64 から u32 へ戻すと末尾に長さ2の次元が付いて元の配列になる。"
  (shape dtype seed)
  (declare (ignorable dtype))
  (let* ((pairs (%bits-random-array (append shape '(2)) :u32 seed))
         (wide (%bits-run :bitcast-convert (list pairs) :dtype :u64)))
    (and (equal (array-dimensions wide) shape)
         (eq :u64 (nb:array-dtype wide))
         (equalp pairs (%bits-run :bitcast-convert (list wide) :dtype :u32))
         (dotimes (i (array-total-size wide) t)
           (unless (= (row-major-aref wide i)
                      (+ (row-major-aref pairs (* 2 i)) (* (expt 2 32) (row-major-aref pairs (1+ (* 2 i))))))
             (return nil))))))

(test primitives/bits/bitcast-known-ieee-patterns
  "IEEE 754 の既知のビット列: 1.0f0 = 0x3F800000、-2.0f0 = 0xC0000000、1.0d0 = 0x3FF0000000000000、
1.5d0 は上位 32 ビット 0x3FF80000 / 下位 0。"
  (flet ((cast (value dtype out)
           (aref (%bits-run :bitcast-convert
                            (list (make-array 1 :element-type (nb:dtype-element-type dtype)
                                                :initial-element value))
                            :dtype out)
                 0)))
    (is (= #x3F800000 (cast 1f0 :f32 :u32)))
    (is (= #xC0000000 (cast -2f0 :f32 :u32)))
    (is (= #x3FF0000000000000 (cast 1d0 :f64 :u64)))
    (is (= 1f0 (cast #x3F800000 :u32 :f32)))
    (is (= 1.5d0 (cast #x3FF8000000000000 :u64 :f64)))))

(test primitives/bits/abstract-eval-shapes-and-errors
  "bitcast-convert の形状推論: 同じ幅は同じ形、広い → 狭いは末尾に次元が付き、狭い → 広いは
末尾の次元（幅の比と同じ長さ）が消える。それ以外（幅の比と違う末尾、対象外の dtype）と、
shift-right-logical / bitwise-or の浮動小数点・dtype 不一致は PRIMITIVE-ERROR。"
  (flet ((out (name avals &rest params)
           (apply (nb::primitive-abstract-eval (nb::find-primitive name)) avals params)))
    (is (equalp (nb:make-aval '(3 4) :f32) (out :bitcast-convert (list (nb:make-aval '(3 4) :u32)) :dtype :f32)))
    (is (equalp (nb:make-aval '(3 2) :u32) (out :bitcast-convert (list (nb:make-aval '(3) :u64)) :dtype :u32)))
    (is (equalp (nb:make-aval '(3) :u64) (out :bitcast-convert (list (nb:make-aval '(3 2) :u32)) :dtype :u64)))
    (is (equalp (nb:make-aval '() :u64) (out :bitcast-convert (list (nb:make-aval '(2) :f32)) :dtype :u64)))
    (signals nb:primitive-error (out :bitcast-convert (list (nb:make-aval '(3 3) :u32)) :dtype :u64))
    (signals nb:primitive-error (out :bitcast-convert (list (nb:make-aval '() :u32)) :dtype :u64))
    (signals nb:primitive-error (out :bitcast-convert (list (nb:make-aval '(2) :f32)) :dtype :bf16))
    (signals nb:primitive-error (out :bitcast-convert (list (nb:make-aval '(2) :i1)) :dtype :u32))
    (signals nb:primitive-error (out :shift-right-logical (list (nb:make-aval '(2) :f32) (nb:make-aval '(2) :f32))))
    (signals nb:primitive-error (out :bitwise-or (list (nb:make-aval '(2) :u32) (nb:make-aval '(2) :i32))))
    (signals nb:primitive-error (out :bitwise-or (list (nb:make-aval '(2) :u32) (nb:make-aval '(3) :u32))))))

(test primitives/bits/emit-stablehlo
  "StableHLO の出力に、対応する op が入る。"
  (let* ((graph (nb:trace-to-graph
                 (nb:with-tracing (x)
                   (nb::%trace-eqn :bitcast-convert
                                   (list (nb::%trace-eqn :bitwise-or
                                                         (list (nb::%trace-eqn :shift-right-logical (list x x)) x)))
                                   :dtype :f32))
                 (list (nb:make-aval '(4) :u32))))
         (text (nb:emit-stablehlo graph)))
    (dolist (needle '("stablehlo.shift_right_logical" "stablehlo.or " "stablehlo.bitcast_convert"
                      "tensor<4xui32>" "tensor<4xf32>"))
      (is (search needle text) "~S が出力に無い" needle))))

;;; --- vmap のバッチ化ルール（単一出力なので今の vmap で通る） ---

(defun %bits-check-vmap (name f arrays in-axes)
  "(vmap F) の結果が、各要素を F で単独に呼んだ結果と等しい（EQUALP。ビット単位）。"
  (flet ((bits-of (array)
           ;; 乱数のビット列は NaN になりうるので、浮動小数点は整数に戻して比べる
           (if (eq :f32 (nb:array-dtype array))
               (%bits-run :bitcast-convert (list array) :dtype :u32)
               array)))
    (is (equalp (bits-of (first (reference-vmap f arrays :in-axes in-axes)))
                (bits-of (apply (nb:vmap f :in-axes in-axes) arrays)))
        "~A（in-axes ~S）: vmap の結果が各要素を単独に呼んだ結果と一致しない" name in-axes)))

(test primitives/bits/batch-rules-match-per-element-calls
  "shift-right-logical / bitwise-or / bitcast-convert（同じ幅・広い → 狭い・狭い → 広い）の
vmap の結果は、各要素を単独に呼んだ結果と一致する。バッチ軸が先頭・中間・末尾、バッチされない引数との
組み合わせを含む。bitcast は末尾の次元を増減するので、バッチ軸が末尾にあるケースが要点。"
  (let ((shift (primitive-function :shift-right-logical '() 2))
        (or-fn (primitive-function :bitwise-or '() 2)))
    (dolist (dtype '(:u32 :i32 :u64))
      (let ((a (%bits-random-array '(3 4) dtype 1))
            (b (%bits-random-array '(3 4) dtype 2))
            (single (%bits-random-array '(4) dtype 3)))
        (%bits-check-vmap "shift-right-logical" shift (list a b) '(0 0))
        (%bits-check-vmap "shift-right-logical" shift (list a b) '(1 1))
        (%bits-check-vmap "shift-right-logical" shift
                          (list a (%bits-random-array '(4 3) dtype 8)) '(1 0))
        (%bits-check-vmap "shift-right-logical" shift (list a single) '(0 nil))
        (%bits-check-vmap "bitwise-or" or-fn (list a b) '(1 1))
        (%bits-check-vmap "bitwise-or" or-fn (list single a) '(nil 0))))
    (let ((u32 (%bits-random-array '(3 2 4) :u32 4))
          (u64 (%bits-random-array '(3 4) :u64 5)))
      (dolist (axis '(0 1 2))
        (%bits-check-vmap "bitcast u32→f32" (primitive-function :bitcast-convert '(:dtype :f32) 1)
                          (list u32) axis)
        (%bits-check-vmap "bitcast u32→i32（同じ幅）" (primitive-function :bitcast-convert '(:dtype :i32) 1)
                          (list u32) axis))
      ;; 狭い → 広い: 要素の形は (3 2)（バッチ軸 2）、(2)（バッチ軸 0）、(2)（バッチ軸 1）。末尾の 2 を消す
      (%bits-check-vmap "bitcast u32→u64" (primitive-function :bitcast-convert '(:dtype :u64) 1)
                        (list u32) 2)
      (%bits-check-vmap "bitcast u32→u64" (primitive-function :bitcast-convert '(:dtype :u64) 1)
                        (list (%bits-random-array '(5 2) :u32 6)) 0)
      (%bits-check-vmap "bitcast u32→u64" (primitive-function :bitcast-convert '(:dtype :u64) 1)
                        (list (%bits-random-array '(2 5) :u32 7)) 1)
      ;; 広い → 狭い: 末尾に長さ 2 の次元が付く
      (dolist (axis '(0 1))
        (%bits-check-vmap "bitcast u64→u32" (primitive-function :bitcast-convert '(:dtype :u32) 1)
                          (list u64) axis)))))
