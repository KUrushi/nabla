;;;; primitives/rng: rng-bit-generator プリミティブ（issue #133）。
;;;;
;;;; stablehlo.rng_bit_generator（algorithm = THREE_FRY）に対応する、複数出力
;;;; （契約 C1）のプリミティブ。入力は状態 ui64[2]、params は出力の :shape と
;;;; :dtype（:u32 / :u64）、出力は (新しい状態 ui64[2], 乱数ビット)。
;;;;
;;;; eager 実装は StableHLO → Linalg の lowering（下記のファイル）を写したもの（実行系とビット単位で
;;;; 一致させる。契約 C7）。写した元は固定コミットの
;;;;   third_party/stablehlo/stablehlo/conversions/linalg/transforms/
;;;;   StablehloToLinalgRandom.cpp
;;;; の runThreeFry2xi32 / generateLinalgThreeFry32 / generateLinalgThreeFry64 /
;;;; threeFry32Shape / extractKey32 / extractState64 / setState64。
;;;; 要点:
;;;;   - 状態 s = (s0, s1)。鍵は s0 を (下位32, 上位32) に分けた (key0, key1)、
;;;;     カウンタは s1。新しい状態は (s0, s1 + count)（鍵は変わらない）。
;;;;   - 要素 i（0 <= i < count）は Threefry-2x32 に (s1 + i) を
;;;;     (下位32, 上位32) に分けて入れた出力 (x0, x1)。20ラウンド
;;;;     （回転 13 15 26 6 / 17 29 16 24 を繰り返す）、鍵スケジュールの
;;;;     key2 = 0x1BD11BDA xor key0 xor key1。
;;;;   - :u64: count = 要素数、要素 i = x0 | (x1 << 32)（行優先）。
;;;;   - :u32: x0 と x1 を別の要素にする。count = 半分にした shape の要素数
;;;;     （要素数1または rank 0 のときは1）。半分にする次元 h は「最初の偶数の次元、
;;;;     無ければ最大の次元（最初のもの）」で、その大きさを ceil(d/2) にする。
;;;;     出力 [..., j, ...] は（j が偶数なら x0、奇数なら x1）の列から
;;;;     [前の次元..., j/2, 後ろの次元...] 番目を取ったもの。lowering は x0 / x1 を
;;;;     それぞれ半分の shape に並べ、h+1 次元で concatenate して reshape するので、
;;;;     h より後ろの次元の積 B について、出力の flat 位置 (p*2h' + j)*B + r
;;;;     （h' = ceil(d/2)）には (j&1 ? x1 : x0)[(p*h' + j/2)*B + r] が入り、
;;;;     d が奇数のときは末尾（j = d）を切り捨てる。
;;;; XLA の実装ともビット単位で一致する（実行系ごとの実測結果は docs/stablehlo-ops.md。
;;;; core は実行系の名前を知らない）。
;;;;
;;;; 状態・ビットは整数なので微分しない（jvp ルールは要らない。全入力の接線が
;;;; ゼロのとき jvp-graph はルールを呼ばず主値を再発行する）。

(in-package #:nabla)

(defun %rng-check-params (in-avals shape dtype)
  "RNG-BIT-GENERATOR の入力と params を検査する（満たさなければ PRIMITIVE-ERROR）。"
  (%check-arity :rng-bit-generator in-avals 1)
  (unless (equalp (first in-avals) (make-aval '(2) :u64))
    (error 'primitive-error :name :rng-bit-generator :in-avals in-avals
           :format-control "状態は ui64[2]（shape (2)・dtype :u64）でなければならない"
           :format-arguments '()))
  (unless (member dtype '(:u32 :u64))
    (error 'primitive-error :name :rng-bit-generator :in-avals in-avals
           :format-control "dtype は :u32 か :u64 でなければならない: ~S"
           :format-arguments (list dtype)))
  (unless (and (sb-int:proper-list-p shape)
               (every (lambda (d) (and (integerp d) (plusp d))) shape))
    (error 'primitive-error :name :rng-bit-generator :in-avals in-avals
           :format-control "shape は正の整数のリストでなければならない: ~S"
           :format-arguments (list shape))))

(defun %threefry-rotl32 (x r)
  (declare (type (unsigned-byte 32) x) (type (integer 1 31) r))
  (logior (ldb (byte 32 0) (ash x r)) (ash x (- r 32))))

(defun %threefry-2x32 (key0 key1 counter)
  "Threefry-2x32（20ラウンド）。COUNTER（64ビット）を (下位32, 上位32) の2語として
暗号化し、(VALUES x0 x1)（どちらも32ビット）を返す。"
  (declare (type (unsigned-byte 32) key0 key1) (type (unsigned-byte 64) counter)
           (optimize (speed 1)))
  (let* ((key2 (logxor #x1BD11BDA key0 key1))
         (ks (vector key0 key1 key2))
         (rotations #(13 15 26 6 17 29 16 24))
         (x0 (ldb (byte 32 0) (+ (ldb (byte 32 0) counter) key0)))
         (x1 (ldb (byte 32 0) (+ (ash counter -32) key1))))
    (declare (type (unsigned-byte 32) x0 x1))
    (dotimes (i 5)
      (let ((rot (mod (* 4 i) 8))
            (k1 (svref ks (mod (+ i 1) 3)))
            (k2 (svref ks (mod (+ i 2) 3))))
        (dotimes (j 4)
          (setf x0 (ldb (byte 32 0) (+ x0 x1))
                x1 (%threefry-rotl32 x1 (svref rotations (+ rot j)))
                x1 (logxor x0 x1)))
        (setf x0 (ldb (byte 32 0) (+ x0 k1))
              x1 (ldb (byte 32 0) (+ x1 k2 (1+ i))))))
    (values x0 x1)))

(defun %rng-half-dimension (shape)
  "u32 の出力で半分にする次元の添字（lowering の threeFry32Shape）: 最初の偶数の次元、
無ければ最大の次元（最初のもの）。"
  (or (position-if #'evenp shape)
      (position (reduce #'max shape) shape)))

(defun %rng-bit-generator-eager (arrays shape dtype)
  "状態 ARRAYS（ui64[2] を1つ）から (新しい状態 ビット) を返す。配置は冒頭のコメント。"
  (let* ((state (first arrays))
         (s0 (aref state 0))
         (s1 (aref state 1))
         (key0 (ldb (byte 32 0) s0))
         (key1 (ash s0 -32))
         (numel (reduce #'* shape))
         (new-state (make-array 2 :element-type '(unsigned-byte 64)))
         (bits (make-array shape :element-type (dtype-element-type dtype))))
    (flet ((generate (i)
             (%threefry-2x32 key0 key1 (ldb (byte 64 0) (+ s1 i)))))
      (ecase dtype
        (:u64
         (dotimes (i numel)
           (multiple-value-bind (x0 x1) (generate i)
             (setf (row-major-aref bits i) (logior x0 (ash x1 32)))))
         (setf (aref new-state 1) (ldb (byte 64 0) (+ s1 numel))))
        (:u32
         (if (= numel 1)
             (progn
               (setf (row-major-aref bits 0) (generate 0))
               (setf (aref new-state 1) (ldb (byte 64 0) (+ s1 1))))
             (let* ((half (%rng-half-dimension shape))
                    (d (nth half shape))
                    (h (ceiling d 2))
                    (inner (reduce #'* (nthcdr (1+ half) shape)))
                    (outer (reduce #'* (subseq shape 0 half)))
                    (count (* outer h inner)))
               ;; 出力の flat 位置 (p*d + j)*inner + r（j < d）に
               ;; (j が偶数なら x0、奇数なら x1)[(p*h + j/2)*inner + r] を置く
               (dotimes (k count)
                 (multiple-value-bind (x0 x1) (generate k)
                   (multiple-value-bind (ph r) (floor k inner)
                     (multiple-value-bind (p i) (floor ph h)
                       (let ((j0 (* 2 i)) (j1 (1+ (* 2 i))))
                         (setf (row-major-aref bits (+ (* (+ (* p d) j0) inner) r)) x0)
                         (when (< j1 d)
                           (setf (row-major-aref bits (+ (* (+ (* p d) j1) inner) r)) x1)))))))
               (setf (aref new-state 1) (ldb (byte 64 0) (+ s1 count))))))))
    (setf (aref new-state 0) s0)
    (list new-state bits)))

(defprimitive rng-bit-generator (:shape :dtype)
  :multiple-outputs t
  :abstract-eval
  (lambda (in-avals &key shape dtype)
    (%rng-check-params in-avals shape dtype)
    (list (make-aval '(2) :u64) (make-aval shape dtype)))
  :emit
  (lambda (in-names in-avals out-names out-avals &key shape dtype)
    (declare (ignore shape dtype))
    (format nil "~A, ~A = stablehlo.rng_bit_generator ~A, algorithm = THREE_FRY : (~A) -> (~A, ~A)"
            (first out-names) (second out-names) (first in-names)
            (tensor-type-string (first in-avals))
            (tensor-type-string (first out-avals))
            (tensor-type-string (second out-avals))))
  :eager
  (lambda (arrays in-avals &key shape dtype)
    (%rng-check-params in-avals shape dtype)
    (%rng-bit-generator-eager arrays shape dtype)))

(defgeneric rng-bit-generator (state &key shape dtype)
  (:documentation
   "STATE（ui64[2] の状態）から乱数ビットを作り、(VALUES 新しい状態 ビット) を返す
内部関数（公開の PRNG API は issue #136）。ビットは SHAPE・DTYPE（:u32 / :u64）。
配列を渡すと eager に、トレーサを渡すと :RNG-BIT-GENERATOR の eqn を足す。"))

(defmethod rng-bit-generator ((state array) &key shape dtype)
  (let ((results (funcall (primitive-eager (find-primitive :rng-bit-generator))
                          (list state) (list (array-aval state :u64))
                          :shape shape :dtype dtype)))
    (values (first results) (second results))))

(defmethod rng-bit-generator ((state tracer) &key shape dtype)
  (values-list (%trace-eqn* :rng-bit-generator (list state) :shape shape :dtype dtype)))
