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
;;;; ui64[2] しか受けないので、emit は stablehlo.while の1回で K 行（*RNG-ROWS-PER-ITERATION*）を
;;;; dynamic_slice して K 回 rng_bit_generator を呼び、dynamic_update_slice で積み直す（issue #164 /
;;;; #178。全行を展開するとコンパイル時間が行数とともに伸び、1行ずつ回すとバックエンドでループ1回ごとの
;;;; 起動の費用が行数に比例する。実測は docs/phase3-report.md §4.2）。
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

(defparameter *rng-rows-per-iteration* 16
  "バッチされた rng-bit-generator の emit（%RNG-EMIT-BATCHED）が while の1回で処理する行数 K（内部。
export しない。issue #178）。行数が K 以下なら while を出さずに行ごとに展開し、K を超えると
while の1回で K 行ぶんの rng_bit_generator を展開して回す。CPU の実行系の実行時間にはループ1回ごとの
起動の費用（4コアの機械で約 0.16 ms）× ceil(行数 / K) が残り、コンパイル時間は K とともに伸びる。
測定（docs/phase3-report.md §4.2）では eqn 1つのコンパイルが K = 16 で約 1 秒、K = 32 で
2〜3 秒だったので、約 2 秒に収まる 16 にした。emit のときに読むので、jit のキャッシュのキーには
入らない（コンパイル済みの関数には効かない）。")

(defun %rng-emit-block (n block k shape dtype)
  "(K 2) の状態 BLOCK の各行に rng_bit_generator を呼ぶ行（展開）。N は補助の名前を作る関数。
(VALUES 行のリスト 新しい状態 (K 2) の名前 ビット (K . SHAPE) の名前) を返す。
K = 1 なら slice と concatenate を出さない。"
  (let* ((block-aval (make-aval (list k 2) :u64))
         (row-aval (make-aval '(2) :u64))
         (one-aval (make-aval '(1 2) :u64))
         (bits-aval (make-aval shape dtype))
         (one-bits-aval (make-aval (cons 1 shape) dtype))
         (lines '()) (states '()) (bits '()))
    (flet ((reshape (out in from to)
             (push (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                           out in (tensor-type-string from) (tensor-type-string to))
                   lines))
           (concat (out ins one all)
             (push (format nil "~A = stablehlo.concatenate ~{~A~^, ~}, dim = 0 : (~{~A~^, ~}) -> ~A"
                           out ins (make-list k :initial-element (tensor-type-string one))
                           (tensor-type-string all))
                   lines)))
      (dotimes (j k)
        (let ((slice (funcall n (format nil "slice~D" j))) (row (funcall n (format nil "row~D" j)))
              (new (funcall n (format nil "new~D" j))) (row-bits (funcall n (format nil "bits~D" j)))
              (one (funcall n (format nil "onestate~D" j))) (one-bits (funcall n (format nil "onebits~D" j))))
          (if (= k 1)
              (setf slice block)
              (push (format nil "~A = stablehlo.slice ~A [~D:~D, 0:2] : (~A) -> ~A"
                            slice block j (1+ j) (tensor-type-string block-aval) (tensor-type-string one-aval))
                    lines))
          (reshape row slice one-aval row-aval)
          (push (%rng-emit-single row row-aval (list new row-bits) (list row-aval bits-aval)) lines)
          (reshape one new row-aval one-aval)
          (reshape one-bits row-bits bits-aval one-bits-aval)
          (push one states)
          (push one-bits bits)))
      (if (= k 1)
          (values (reverse lines) (first states) (first bits))
          (progn
            (concat (funcall n "news") (reverse states) one-aval block-aval)
            (concat (funcall n "blockbits") (reverse bits) one-bits-aval (make-aval (cons k shape) dtype))
            (values (reverse lines) (funcall n "news") (funcall n "blockbits")))))))

(defun %rng-emit-batched (in-name state-aval out-names out-avals shape dtype)
  "バッチ次元のある状態の StableHLO。状態を (行数 2) にならし、行ごとに rng_bit_generator を
呼んで積み、最後に元の shape に戻す。行数が K（*RNG-ROWS-PER-ITERATION*）以下なら
行ごとに展開する（slice → rng_bit_generator → concatenate。%RNG-EMIT-BLOCK）。K を超えると
stablehlo.while の1回で K 行ずつ処理する（issue #164 / #178。StableHLO の大きさは行数に依らない）:
dynamic_slice で (K 2) の状態を取り、K 行を展開して、dynamic_update_slice で K 行を1回で書く。

行数が K の倍数でないときは、最後の1回の開始位置を dynamic_slice / dynamic_update_slice の
添字のクランプ（行数 - K に丸められる）に任せ、前の回と重なる行を計算し直す。各行は元の状態
（ループで書き換えない carry <p>_src）から計算するので、重なった行は同じ値で上書きされるだけ。
詰め物をして切り出すより、余分な配列も2つ目のループも要らず短い。

while の定数オペランド（カウンタとビットのバッファの 0 初期値）は optimization_barrier を
通す（特定の版のバックエンドのコンパイラが落ちる回避策。docs/stablehlo-ops.md の制御構造の節）。
ビットのバッファは scan の ys と同じく扱う（issue #159）。本体の先頭で optimization_barrier に
通してから dynamic_update_slice に渡し（通さないと、バックエンドのコンパイラがループの carry を
本体で使うたびに丸ごとコピーする）、カウンタとビットのバッファの初期値は %SCAN-YS-INIT-LINES
（カウンタの 0・スカラーの 0・モジュールの中で一意な整数を1つの barrier に通し、スカラーを
broadcast_in_dim で広げてもう一度 barrier）で作る。状態の carry は小さいので barrier に通さない。"
  (let* ((rows (reduce #'* (%rng-lead-shape state-aval)))
         (k (min rows *rng-rows-per-iteration*))
         (p (format nil "%rng_~A" (subseq (first out-names) 1)))
         (flat-state-aval (make-aval (list rows 2) :u64))
         (all-bits-aval (make-aval (cons rows shape) dtype))
         (block-aval (make-aval (list k 2) :u64))
         (block-bits-aval (make-aval (cons k shape) dtype))
         (i32 "tensor<i32>")
         (types (list i32 (tensor-type-string flat-state-aval) (tensor-type-string flat-state-aval)
                      (tensor-type-string all-bits-aval))))
    (flet ((n (suffix) (format nil "~A_~A" p suffix))
           (reshape (out in from to)
             (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                     out in (tensor-type-string from) (tensor-type-string to))))
      (format nil "~{~A~^~%~}"
              (cond
                ((zerop rows)
                 ;; 0 行: 状態はそのまま、ビットは空
                 (list (reshape (first out-names) in-name state-aval (first out-avals))
                       (%scan-zero-constant-line (second out-names) (second out-avals))))
                ((= k rows)
                  (multiple-value-bind (lines news bits) (%rng-emit-block #'n (n "flat") k shape dtype)
                    (append (list (reshape (n "flat") in-name state-aval flat-state-aval))
                            lines
                            (list (reshape (first out-names) news flat-state-aval (first out-avals))
                                  (reshape (second out-names) bits all-bits-aval (second out-avals))))))
                (t
                  (append
                   (list
                    (reshape (n "flat") in-name state-aval flat-state-aval))
                  (%scan-ys-init-lines (n "i0") (list (n "b0")) (list all-bits-aval))
                  (list
                   (format nil "~A, ~A, ~A, ~A = \"stablehlo.while\"(~A, ~A, ~A, ~A) ({"
                           (n "n") (n "src_n") (n "states") (n "allbits")
                           (n "i0") (n "flat") (n "flat") (n "b0"))
                   ;; cond: 開始位置 < 行数
                   (format nil "^bb0(~A: ~A, ~A: ~A, ~A: ~A, ~A: ~A):"
                           (n "ci") (first types) (n "csrc") (second types) (n "cs") (third types)
                           (n "cb") (fourth types))
                   (format nil "~A = stablehlo.constant dense<~D> : ~A" (n "len") rows i32)
                   (format nil "~A = stablehlo.compare LT, ~A, ~A : (~A, ~A) -> tensor<i1>"
                           (n "lt") (n "ci") (n "len") i32 i32)
                   (format nil "stablehlo.return ~A : tensor<i1>" (n "lt"))
                   "}, {"
                   ;; body: i 行目からの K 行の元の状態を読み、K 行ぶんの結果を i 行目から書く
                   (format nil "^bb0(~A: ~A, ~A: ~A, ~A: ~A, ~A: ~A):"
                           (n "i") (first types) (n "src") (second types) (n "s") (third types)
                           (n "b") (fourth types))
                   (format nil "~A = stablehlo.optimization_barrier ~A : ~A" (n "bk") (n "b") (fourth types))
                   (format nil "~A = stablehlo.constant dense<0> : ~A" (n "z") i32)
                   (format nil "~A = stablehlo.dynamic_slice ~A, ~A, ~A, sizes = [~D, 2] : (~A, ~A, ~A) -> ~A"
                           (n "block") (n "src") (n "i") (n "z") k (second types) i32 i32
                           (tensor-type-string block-aval)))
                  (multiple-value-bind (lines news bits) (%rng-emit-block #'n (n "block") k shape dtype)
                    (append
                     lines
                     (list
                      (format nil "~A = stablehlo.dynamic_update_slice ~A, ~A, ~A, ~A : (~A, ~A, ~A, ~A) -> ~A"
                              (n "s2") (n "s") news (n "i") (n "z")
                              (third types) (tensor-type-string block-aval) i32 i32 (third types))
                      (format nil "~A = stablehlo.dynamic_update_slice ~A, ~A, ~A~{, ~A~} : (~A, ~A, ~A) -> ~A"
                              (n "b2") (n "bk") bits (n "i")
                              (make-list (length shape) :initial-element (n "z"))
                              (fourth types) (tensor-type-string block-bits-aval)
                              (%scan-index-types (1+ (length shape))) (fourth types)))))
                  (list
                   (format nil "~A = stablehlo.constant dense<~D> : ~A" (n "step") k i32)
                   (format nil "~A = stablehlo.add ~A, ~A : ~A" (n "next") (n "i") (n "step") i32)
                   (format nil "stablehlo.return ~A, ~A, ~A, ~A : ~{~A~^, ~}"
                           (n "next") (n "src") (n "s2") (n "b2") types)
                   (format nil "}) : (~{~A~^, ~}) -> (~{~A~^, ~})" types types)
                   (reshape (first out-names) (n "states") flat-state-aval (first out-avals))
                   (reshape (second out-names) (n "allbits") all-bits-aval (second out-avals))))))))))

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
