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
;;;; 状態の先頭に「バッチ次元」を付けられる（issue #136）。状態の shape が
;;;; (lead... 2) のとき、各行 [i..., :] が独立した状態で、出力の新しい状態は同じ shape、
;;;; ビットは (lead... shape...)。各行は、その行だけを ui64[2] として単独に呼んだ結果と
;;;; ビット単位で一致する（vmap のバッチ化ルールがこれに頼る。vmap の結果は「各要素を
;;;; 単独に呼んだ結果」と一致しなければならない）。StableHLO の rng_bit_generator は
;;;; ui64[2] しか受けないので、emit は行ごとに slice して rng_bit_generator を呼び、
;;;; concatenate で積み直す（コンパイルコストは %rng-emit-batched の docstring。行数だけ演算が増える。静的形状のため展開する。
;;;; スキャンによる圧縮は将来の課題）。
;;;;
;;;; 状態・ビットは整数なので微分しない（jvp ルールは要らない。全入力の接線が
;;;; ゼロのとき jvp-graph はルールを呼ばず主値を再発行する）。

(in-package #:nabla)

(defun %rng-check-params (in-avals shape dtype)
  "RNG-BIT-GENERATOR の入力と params を検査する（満たさなければ PRIMITIVE-ERROR）。
状態は dtype :u64 で、shape の末尾が 2（先頭の次元はバッチ次元。無くてもよい）。"
  (%check-arity :rng-bit-generator in-avals 1)
  (let ((state (first in-avals)))
    (unless (and (eq (aval-dtype state) :u64)
                 (plusp (aval-rank state))
                 (= 2 (car (last (aval-shape state)))))
      (error 'primitive-error :name :rng-bit-generator :in-avals in-avals
             :format-control "状態は ui64[..., 2]（shape の末尾が 2・dtype :u64）でなければならない"
             :format-arguments '())))
  (unless (member dtype '(:u32 :u64))
    (error 'primitive-error :name :rng-bit-generator :in-avals in-avals
           :format-control "dtype は :u32 か :u64 でなければならない: ~S"
           :format-arguments (list dtype)))
  (unless (and (sb-int:proper-list-p shape)
               (every (lambda (d) (and (integerp d) (plusp d))) shape))
    (error 'primitive-error :name :rng-bit-generator :in-avals in-avals
           :format-control "shape は正の整数のリストでなければならない: ~S"
           :format-arguments (list shape))))

(defun %rng-lead-shape (state-aval)
  "状態の aval の、バッチ次元（末尾の 2 を除いた先頭部分）の shape。"
  (butlast (aval-shape state-aval)))

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

(defun %rng-bit-generator-single-eager (arrays shape dtype)
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

(defun %rng-bit-generator-eager (arrays shape dtype)
  "状態 ARRAYS（ui64[..., 2] を1つ）から (新しい状態 ビット) を返す。バッチ次元があれば
各行を単独の状態として %RNG-BIT-GENERATOR-SINGLE-EAGER に渡し、結果を行ごとに並べる。"
  (let* ((state (first arrays))
         (lead (butlast (array-dimensions state))))
    (if (null lead)
        (%rng-bit-generator-single-eager arrays shape dtype)
        (let* ((rows (reduce #'* lead))
               (bit-size (reduce #'* shape))
               (new-state (make-array (array-dimensions state) :element-type '(unsigned-byte 64)))
               (bits (make-array (append lead shape) :element-type (dtype-element-type dtype)))
               (flat (make-array (array-total-size state) :element-type '(unsigned-byte 64)
                                                          :displaced-to state)))
          (dotimes (row rows)
            (let ((row-state (make-array 2 :element-type '(unsigned-byte 64)
                                           :initial-contents (list (aref flat (* 2 row))
                                                                   (aref flat (1+ (* 2 row)))))))
              (destructuring-bind (row-new row-bits)
                  (%rng-bit-generator-single-eager (list row-state) shape dtype)
                (setf (row-major-aref new-state (* 2 row)) (aref row-new 0)
                      (row-major-aref new-state (1+ (* 2 row))) (aref row-new 1))
                (dotimes (i bit-size)
                  (setf (row-major-aref bits (+ (* row bit-size) i)) (row-major-aref row-bits i))))))
          (list new-state bits)))))

(defun %rng-emit-single (in-name state-aval out-names out-avals)
  "ui64[2] の状態 IN-NAME（STATE-AVAL）に対する rng_bit_generator 1行。"
  (format nil "~A, ~A = stablehlo.rng_bit_generator ~A, algorithm = THREE_FRY : (~A) -> (~A, ~A)"
          (first out-names) (second out-names) in-name
          (tensor-type-string state-aval)
          (tensor-type-string (first out-avals))
          (tensor-type-string (second out-avals))))

(defun %rng-emit-batched (in-name state-aval out-names out-avals shape dtype)
  "バッチ次元のある状態の StableHLO。状態を (行数 2) にならし、行ごとに slice →
rng_bit_generator → 積み直し（concatenate）、最後に元の shape に戻す。

実測のコンパイルコスト（CPU の実行系（README の「PRNG」参照）、バッチ次元つきの rng-bit-generator の eqn 1つ）: 32 行 4.0 秒、64 行 6.9 秒、256 行 42.8 秒（MLIR 179 KB）。実用上の上限は 64 行程度（vmap するキーの数）で、vmap を入れ子にすると行数は段ごとの積になる（B1×B2 行）ので同じ上限が全段の積にかかる。将来の対策は、行ごとの展開をやめて scan（while）で1行ぶんの本体を回す形にすること。"
  (let* ((rows (reduce #'* (%rng-lead-shape state-aval)))
         (base (subseq (first out-names) 1))
         (flat-state-aval (make-aval (list rows 2) :u64))
         (flat-state (format nil "%rng_flat_~A" base))
         (row-state-aval (make-aval '(2) :u64))
         (row-bits-aval (make-aval shape dtype))
         (one-state-aval (make-aval '(1 2) :u64))
         (one-bits-aval (make-aval (cons 1 shape) dtype))
         (all-bits-aval (make-aval (cons rows shape) dtype))
         (lines (list (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                              flat-state in-name (tensor-type-string state-aval)
                              (tensor-type-string flat-state-aval))))
         (states '()) (bits '()))
    (flet ((name (tag row) (format nil "%rng_~A~D_~A" tag row base))
           (add (line) (push line lines)))
      (dotimes (row rows)
        (let ((slice (name "slice" row)) (row-state (name "state" row))
              (new-state (name "newstate" row)) (row-bits (name "bits" row))
              (one-state (name "onestate" row)) (one-bits (name "onebits" row)))
          (add (format nil "~A = stablehlo.slice ~A [~D:~D, 0:2] : (~A) -> ~A"
                       slice flat-state row (1+ row)
                       (tensor-type-string flat-state-aval) (tensor-type-string one-state-aval)))
          (add (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                       row-state slice (tensor-type-string one-state-aval)
                       (tensor-type-string row-state-aval)))
          (add (%rng-emit-single row-state row-state-aval (list new-state row-bits)
                                 (list row-state-aval row-bits-aval)))
          (add (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                       one-state new-state (tensor-type-string row-state-aval)
                       (tensor-type-string one-state-aval)))
          (add (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                       one-bits row-bits (tensor-type-string row-bits-aval)
                       (tensor-type-string one-bits-aval)))
          (push one-state states)
          (push one-bits bits)))
      (let ((all-state (format nil "%rng_allstate_~A" base))
            (all-bits (format nil "%rng_allbits_~A" base)))
        (add (format nil "~A = stablehlo.concatenate ~{~A~^, ~}, dim = 0 : (~{~A~^, ~}) -> ~A"
                     all-state (reverse states)
                     (make-list rows :initial-element (tensor-type-string one-state-aval))
                     (tensor-type-string flat-state-aval)))
        (add (format nil "~A = stablehlo.concatenate ~{~A~^, ~}, dim = 0 : (~{~A~^, ~}) -> ~A"
                     all-bits (reverse bits)
                     (make-list rows :initial-element (tensor-type-string one-bits-aval))
                     (tensor-type-string all-bits-aval)))
        (add (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                     (first out-names) all-state (tensor-type-string flat-state-aval)
                     (tensor-type-string (first out-avals))))
        (add (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                     (second out-names) all-bits (tensor-type-string all-bits-aval)
                     (tensor-type-string (second out-avals))))))
    (format nil "~{~A~^~%~}" (reverse lines))))

(defprimitive rng-bit-generator (:shape :dtype)
  :multiple-outputs t
  :abstract-eval
  (lambda (in-avals &key shape dtype)
    (%rng-check-params in-avals shape dtype)
    (list (first in-avals) (make-aval (append (%rng-lead-shape (first in-avals)) shape) dtype)))
  :emit
  (lambda (in-names in-avals out-names out-avals &key shape dtype)
    (if (= 1 (aval-rank (first in-avals)))
        (%rng-emit-single (first in-names) (first in-avals) out-names out-avals)
        (%rng-emit-batched (first in-names) (first in-avals) out-names out-avals shape dtype)))
  :eager
  (lambda (arrays in-avals &key shape dtype)
    (%rng-check-params in-avals shape dtype)
    (%rng-bit-generator-eager arrays shape dtype)))

(defgeneric rng-bit-generator (state &key shape dtype)
  (:documentation
   "STATE（ui64[2] の状態。先頭にバッチ次元を付けた ui64[..., 2] も可で、各行が独立）から乱数ビットを作り、(VALUES 新しい状態 ビット) を返す
内部関数（公開の PRNG API は issue #136）。ビットは SHAPE・DTYPE（:u32 / :u64）。
配列を渡すと eager に、トレーサを渡すと :RNG-BIT-GENERATOR の eqn を足す。"))

(defmethod rng-bit-generator ((state array) &key shape dtype)
  (let ((results (funcall (primitive-eager (find-primitive :rng-bit-generator))
                          (list state) (list (array-aval state :u64))
                          :shape shape :dtype dtype)))
    (values (first results) (second results))))

(defmethod rng-bit-generator ((state tracer) &key shape dtype)
  (values-list (%trace-eqn* :rng-bit-generator (list state) :shape shape :dtype dtype)))
