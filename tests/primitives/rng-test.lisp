;;;; rng-bit-generator プリミティブの性質（issue #133、small）。
;;;;
;;;; 入力の状態は ui64[2]（[0] = 鍵の下位32 / 上位32ビット、[1] = カウンタ）、
;;;; 出力は (新しい状態, 乱数ビット)。eager 実装は IREE の lowering
;;;; （StablehloToLinalgRandom.cpp）の Threefry-2x32 を写したもので、IREE との
;;;; ビット単位の一致は tests/iree/rng-test.lisp（medium）で確かめる。ここでは
;;;; eager 単体で成り立つ性質（決定性、形状と dtype、状態の進み方、
;;;; 出力どうしの関係、エラー）を見る。
;;;; 呼び出しは内部シンボル nb::rng-bit-generator 経由（#136 まで非公開）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %rng-state (seed)
  "SEED から決まる ui64[2] の状態配列（鍵もカウンタも64ビット全域の乱数）。"
  (let ((rs (sb-ext:seed-random-state seed))
        (state (make-array 2 :element-type '(unsigned-byte 64))))
    (setf (aref state 0) (random (expt 2 64) rs)
          (aref state 1) (random (expt 2 64) rs))
    state))

(defun %rng-run (state shape dtype)
  "nb::rng-bit-generator を eager に呼び、(新しい状態 ビット) をリストで返す。"
  (multiple-value-list (nb::rng-bit-generator state :shape shape :dtype dtype)))

(defun %rng-flat (array)
  (coerce (make-array (array-total-size array) :element-type (array-element-type array)
                                               :displaced-to array)
          'list))

(defparameter *rng-seed-generator*
  (generator (uniform-integer :lo 0 :hi (1- (expt 2 31)))))

(defparameter *rng-shape-generator*
  (generator (tuple (array-spec :dtypes '(:f32)) (uniform-integer :lo 0 :hi (1- (expt 2 31))))))

(defmacro def-rng-property (name docstring (&rest vars) gen &body body)
  "GEN から VARS に分配束縛した値で BODY（真偽を返す）を PBT で確かめる。"
  `(test ,name ,docstring
     (is (check-it ,gen
                   (lambda (value) (destructuring-bind ,vars value ,@body))
                   :regression-id ,name
                   :regression-file (regression-path
                                     ,(format nil "primitives-rng-~(~A~)"
                                              (substitute #\- #\/ (subseq (string name) (length "primitives/rng/"))))))
         ,(format nil "~A の性質が成り立たなかった" name))))

(def-rng-property primitives/rng/shape-and-dtype-match-abstract-eval
  "eager の出力は、新しい状態が ui64[2]、ビットが指定した shape・dtype（:u32 / :u64）で、
abstract-eval の aval と一致する。"
  (spec seed) *rng-shape-generator*
  (let ((shape (array-spec-shape spec)))
    (dolist (dtype '(:u32 :u64) t)
      (destructuring-bind (new-state bits) (%rng-run (%rng-state seed) shape dtype)
        (unless (and (equalp (nb:array-aval new-state :u64) (nb:make-aval '(2) :u64))
                     (equalp (nb:array-aval bits dtype) (nb:make-aval shape dtype))
                     (equalp (list (nb:make-aval '(2) :u64) (nb:make-aval shape dtype))
                             (funcall (nb::primitive-abstract-eval (nb::find-primitive :rng-bit-generator))
                                      (list (nb:make-aval '(2) :u64)) :shape shape :dtype dtype)))
          (return nil))))))

(def-rng-property primitives/rng/same-state-gives-same-bits
  "同じ状態・shape・dtype からは常に同じ新しい状態とビットが出る（入力の状態は書き換えない）。"
  (spec seed) *rng-shape-generator*
  (let* ((shape (array-spec-shape spec))
         (state (%rng-state seed))
         (copy (copy-seq state)))
    (dolist (dtype '(:u32 :u64) t)
      (unless (and (equalp (%rng-run state shape dtype) (%rng-run copy shape dtype))
                   (equalp state copy))
        (return nil)))))

(def-rng-property primitives/rng/state-advances-and-key-is-preserved
  "新しい状態は、鍵（[0]）を保ち、カウンタ（[1]）を生成した64ビット単位の個数だけ（2^64 で折り返して）進める。
:u64 は要素数、:u32 は（偶数個なら）その半分。"
  (spec seed) *rng-shape-generator*
  (let* ((shape (array-spec-shape spec))
         (state (%rng-state seed))
         (numel (reduce #'* shape))
         (new64 (first (%rng-run state shape :u64)))
         (new32 (first (%rng-run state shape :u32))))
    (and (= (aref new64 0) (aref state 0))
         (= (aref new32 0) (aref state 0))
         (= (aref new64 1) (ldb (byte 64 0) (+ (aref state 1) numel)))
         ;; :u32 の進みは、要素数が偶数なら（偶数の次元が半分になって）ちょうど半分。奇数なら
         ;; 切り上げた半分以上、要素数以下（全て奇数の次元の rank 2 以上は、最大の次元だけを半分にする）
         (let ((advance (ldb (byte 64 0) (- (aref new32 1) (aref state 1)))))
           (if (evenp numel)
               (= advance (/ numel 2))
               (<= (ceiling numel 2) advance numel))))))

(def-rng-property primitives/rng/different-keys-give-different-bits
  "鍵だけが違う2つの状態からは、8個の :u32 のビットが（2^-256 を無視すれば）一致しない。"
  (seed delta) (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                                 (uniform-integer :lo 1 :hi (1- (expt 2 31)))))
  (let* ((a (%rng-state seed))
         (b (copy-seq a)))
    (setf (aref b 0) (ldb (byte 64 0) (+ (aref a 0) delta)))
    (not (equalp (second (%rng-run a '(8) :u32)) (second (%rng-run b '(8) :u32))))))

(def-rng-property primitives/rng/u64-is-prefix-stable-and-continues-through-the-state
  "u64 の出力は要素ごとに独立: (n+m) 個の出力は、n 個を出してから新しい状態で m 個を出した列と一致する。"
  (seed n m) (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                               (uniform-integer :lo 1 :hi 9)
                               (uniform-integer :lo 1 :hi 9)))
  (let* ((state (%rng-state seed))
         (whole (second (%rng-run state (list (+ n m)) :u64)))
         (first-part (%rng-run state (list n) :u64))
         (second-part (%rng-run (first first-part) (list m) :u64)))
    (equal (%rng-flat whole)
           (append (%rng-flat (second first-part)) (%rng-flat (second second-part))))))

(def-rng-property primitives/rng/u32-pairs-are-the-halves-of-the-u64-words
  "1次元の :u64（n 個）の各要素は、同じ状態からの :u32（2n 個）の [2i]（下位）と [2i+1]（上位）を
つないだものに等しい（Threefry-2x32 の2出力を、u32 は別々の要素に、u64 は1語にまとめる）。"
  (seed n) (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                             (uniform-integer :lo 1 :hi 9)))
  (let* ((state (%rng-state seed))
         (words (%rng-flat (second (%rng-run state (list n) :u64))))
         (halves (%rng-flat (second (%rng-run state (list (* 2 n)) :u32)))))
    (equal words
           (loop for (lo hi) on halves by #'cddr collect (logior lo (ash hi 32))))))

(def-rng-property primitives/rng/u32-layout-halves-the-first-even-dimension
  "shape (rows cols) で cols が偶数のとき、半分にする次元は最初の偶数の次元（rows が偶数なら rows、
奇数なら cols）。a_k / b_k を u64 の語 k の下位 / 上位とすると、rows が偶数なら
出力 [j][r] = (j が偶数なら a、奇数なら b)[(j/2)*cols + r]、rows が奇数なら
出力 [p][j] = (j が偶数なら a、奇数なら b)[p*(cols/2) + j/2]（IREE の concat + reshape の配置）。"
  (seed rows half) (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                                     (uniform-integer :lo 1 :hi 5)
                                     (uniform-integer :lo 1 :hi 5)))
  (let* ((state (%rng-state seed))
         (cols (* 2 half))
         (words (second (%rng-run state (list (floor (* rows cols) 2)) :u64)))
         (a (lambda (k) (ldb (byte 32 0) (row-major-aref words k))))
         (b (lambda (k) (ldb (byte 32 32) (row-major-aref words k))))
         (matrix (second (%rng-run state (list rows cols) :u32))))
    (if (evenp rows)
        (loop for j below rows
              always (loop for r below cols
                           always (= (aref matrix j r)
                                     (funcall (if (evenp j) a b) (+ (* (floor j 2) cols) r)))))
        (loop for p below rows
              always (loop for j below cols
                           always (= (aref matrix p j)
                                     (funcall (if (evenp j) a b) (+ (* p half) (floor j 2)))))))))

(test primitives/rng/known-layout-for-odd-and-rank-zero-shapes
  "rank 0 と要素数1は最初の出力1語だけ、(3) は (a0 b0 a1) の先頭3語になる（奇数個は末尾を切る）。"
  (let* ((state (%rng-state 7))
         (four (%rng-flat (second (%rng-run state '(4) :u32))))
         (three (%rng-flat (second (%rng-run state '(3) :u32))))
         (scalar (second (%rng-run state '() :u32)))
         (one (second (%rng-run state '(1) :u32))))
    (is (equal three (subseq four 0 3)))
    (is (equalp (nb:array-aval scalar :u32) (nb:make-aval '() :u32)))
    (is (= (aref one 0) (first four)))
    (is (= (aref scalar) (first four)))))

(test primitives/rng/emit-writes-multiple-output-form
  "StableHLO は \"%s2, %b = stablehlo.rng_bit_generator %s, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<…xui32>)\" の形。"
  (dolist (case '(((4) :u32 "tensor<4xui32>") (() :u32 "tensor<ui32>")
                  ((2 3) :u64 "tensor<2x3xui64>")))
    (destructuring-bind (shape dtype type) case
      (let* ((graph (nb:trace-to-graph
                     (nb:with-tracing (s) (nb::rng-bit-generator s :shape shape :dtype dtype))
                     (list (nb:make-aval '(2) :u64))))
             (text (nb:emit-stablehlo graph)))
        (is (search "stablehlo.rng_bit_generator" text))
        (is (search "algorithm = THREE_FRY" text))
        (is (search (format nil "(tensor<2xui64>) -> (tensor<2xui64>, ~A)" type) text)
            "~A" text)))))

(test primitives/rng/rejects-invalid-arguments
  "状態が ui64[2] でない、dtype が :u32 / :u64 でない、shape に0以下の次元がある場合は primitive-error。"
  (let ((abstract (nb::primitive-abstract-eval (nb::find-primitive :rng-bit-generator)))
        (state (nb:make-aval '(2) :u64)))
    (flet ((bad (in-avals &rest params)
             (signals nb:primitive-error (apply abstract in-avals params))))
      (bad (list (nb:make-aval '(3) :u64)) :shape '(4) :dtype :u32)
      (bad (list (nb:make-aval '(2) :u32)) :shape '(4) :dtype :u32)
      (bad (list state state) :shape '(4) :dtype :u32)
      (bad (list state) :shape '(4) :dtype :f32)
      (bad (list state) :shape '(0 4) :dtype :u32)
      (bad (list state) :shape '(-1) :dtype :u32)
      (bad (list state) :shape 4 :dtype :u32))
    ;; eager も同じ検査をする（abstract-eval を通らずに呼ばれても生のエラーにならない）
    (let ((array (%rng-state 1)))
      (signals nb:primitive-error (nb::rng-bit-generator array :shape '(4) :dtype :f32))
      (signals nb:primitive-error (nb::rng-bit-generator array :shape '(0) :dtype :u32)))))

(test primitives/rng/is-not-differentiated
  "状態もビットも整数なので接線はゼロ: ビットを float に変換して使う関数を、rng を通らない
float の入力で grad できる（jvp ルール無しで、勾配は変換したビットそのもの）。"
  (let* ((state (%rng-state 3))
         (x (make-array '(4) :element-type 'single-float :initial-element 1.0))
         (f (nb:with-tracing (x s)
              (multiple-value-bind (new-state bits) (nb::rng-bit-generator s :shape '(4) :dtype :u32)
                (declare (ignore new-state))
                (nb:reduce-sum (* x (nb:convert bits :f32)) :axes '(0)))))
         (g (funcall (nb:grad f :argnums 0) x state))
         (bits (second (%rng-run state '(4) :u32))))
    (is (equalp g (map 'vector (lambda (b) (coerce b 'single-float)) bits)))))

;;; --- 既知の答え（JAX の fixture） ---
;;;
;;; 期待値は JAX 0.10.2 の lax.rng_bit_generator（algorithm = RNG_THREE_FRY、x64 有効）で
;;; 生成した（python3 は fixture の生成にだけ使う）:
;;;   s = np.zeros(2, np.uint64)
;;;   lax.rng_bit_generator(s, shape, dtype=np.uint32, algorithm=lax.RandomAlgorithm.RNG_THREE_FRY)
;;; IREE / PJRT が無い環境でも、Threefry の実装の誤りをここで検出できる。

(defparameter *rng-jax-zero-state-u32-8*
  '(1797259609 2579123966 1351547692 3235790642 1688610540 4229293427 3098264785 87550854))

(defparameter *rng-jax-zero-state-u32-3-3-3*
  '(1797259609 1351547692 1688610540 3098264785 2892874427 1447157908 2772180201 645374554
    755508351 2579123966 3235790642 4229293427 87550854 2813178819 239777021 2882477498
    1993301756 2169798897 757048634 882350354 1642178052 441180599 3103801366 1075548667
    608542336 2015619565 2652030227))

(test primitives/rng/matches-jax-known-answers
  "状態 [0 0]（ui64）から、shape (8) の :u32 は新しい状態 [0 4] とビット *rng-jax-zero-state-u32-8*、
shape (3 3 3) の :u32 は新しい状態 [0 18]（カウンタが 18 進む）とビット *rng-jax-zero-state-u32-3-3-3*
（行優先）を返す。JAX 0.10.2 の rng_bit_generator（THREE_FRY）と同じ値。"
  (let ((zero (make-array 2 :element-type '(unsigned-byte 64) :initial-element 0)))
    (destructuring-bind (state bits) (%rng-run zero '(8) :u32)
      (is (equalp #(0 4) state))
      (is (equal *rng-jax-zero-state-u32-8* (%rng-flat bits))))
    (destructuring-bind (state bits) (%rng-run zero '(3 3 3) :u32)
      (is (equalp #(0 18) state))
      (is (equal '(3 3 3) (array-dimensions bits)))
      (is (equal *rng-jax-zero-state-u32-3-3-3* (%rng-flat bits))))))
