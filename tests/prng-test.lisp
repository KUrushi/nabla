;;;; PRNG の公開 API（prng-key / split / fold-in / uniform / normal。issue #136、small）。
;;;;
;;;; 守らせる性質:
;;;; - 決定性: 同じキーからは同じ値、別のキー（split の結果・fold-in の結果）からは別の値。
;;;; - 値域・形・dtype、有限性。
;;;; - 統計: 平均・分散・Kolmogorov–Smirnov 検定。固定のシードで行い、有意水準は下の
;;;;   *PRNG-ALPHA*（シードを選び直したときに誤って落ちる確率）。
;;;; - トレースの中（with-tracing の graph を eager 評価）でも、eager と同じ値になる。
;;;; - バッチ化: rng-bit-generator のバッチ化ルールが、各要素を単独に呼んだ結果と一致する。
;;;;   vmap でキーをバッチした結果は、複数出力の eqn を vmap が歩けるようになる（#140）まで
;;;;   は、そのこと自体を確かめるテスト（prng/vmap-over-keys-matches-per-key-calls）が
;;;;   PENDING-140 の理由でスキップされる。
;;;; eager と IREE / PJRT の一致は tests/iree/prng-test.lisp / tests/pjrt/prng-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *prng-alpha* 1d-3
  "統計テストの有意水準（片側ではなく両側。固定のシードのもとで、シードを選び直したときに
誤って棄却する確率）。")

(defparameter *prng-z* 3.2905d0
  "両側の有意水準 *PRNG-ALPHA* = 1e-3 に対応する標準正規分布の臨界値。")

(defun %prng-seed-generator ()
  (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31))))))

(defun %prng-shape-seed-generator ()
  (generator (tuple (array-spec :dtypes '(:f32)) (uniform-integer :lo 0 :hi (1- (expt 2 31))))))

(defmacro def-prng-property (name docstring (&rest vars) gen &body body)
  "GEN の値を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test ,name ,docstring
     (is (check-it ,gen
                   (lambda (value) (destructuring-bind ,vars value ,@body))
                   :regression-id ,name
                   :regression-file (regression-path
                                     ,(format nil "prng-~(~A~)"
                                              (substitute #\- #\/ (subseq (string name) (length "prng/"))))))
         ,(format nil "~A の性質が成り立たなかった" name))))

(defun %prng-flat (array)
  (coerce (make-array (array-total-size array) :element-type (array-element-type array)
                                               :displaced-to array)
          'list))

;;; --- キー ---

(test prng/key-layout
  "prng-key は [シードの上位32ビット, 下位32ビット] の :u32 (2)。負のシードは64ビットの2の補数。"
  (is (equalp #(0 42) (nb:prng-key 42)))
  (is (equalp #(1 5) (nb:prng-key (+ (expt 2 32) 5))))
  (is (equalp #(#xFFFFFFFF #xFFFFFFFF) (nb:prng-key -1)))
  (is (eq :u32 (nb:array-dtype (nb:prng-key 7))))
  (is (equal '(2) (array-dimensions (nb:prng-key 7))))
  (signals nb:prng-error (nb:prng-key 1.5))
  (signals nb:prng-error (nb:prng-key (expt 2 64)))
  (signals nb:prng-error (nb:prng-key (- (expt 2 63) 1 (expt 2 64)))))

(test prng/arguments-are-validated
  "キー・shape・dtype・範囲・個数・fold-in のデータの不正は PRNG-ERROR。"
  (let ((key (nb:prng-key 0)))
    (signals nb:prng-error (nb:uniform #(1 2) '(3)))
    (signals nb:prng-error (nb:uniform (make-array 2 :element-type '(unsigned-byte 32) :initial-element 0) '(-1)))
    (signals nb:prng-error (nb:uniform (make-array 3 :element-type '(unsigned-byte 32) :initial-element 0) '(3)))
    (signals nb:prng-error (nb:uniform key '(0)))
    (signals nb:prng-error (nb:uniform key 3))
    (signals nb:prng-error (nb:uniform key '(3) :dtype :i32))
    (signals nb:prng-error (nb:uniform key '(3) :minval 1 :maxval 1))
    (signals nb:prng-error (nb:uniform key '(3) :minval 2 :maxval 1))
    (signals nb:prng-error (nb:normal key '(3) :dtype :bf16))
    (signals nb:prng-error (nb:split key 0))
    (signals nb:prng-error (nb:split #(1 2) 2))
    (signals nb:prng-error (nb:normal #(1 2) '(3)))
    (signals nb:prng-error (nb:normal key '(0)))
    (signals nb:prng-error (nb:normal key 3))
    (signals nb:prng-error (nb:split key 1.5))
    (signals nb:prng-error (nb:fold-in key -1))
    (signals nb:prng-error (nb:fold-in key (expt 2 32)))
    (signals nb:prng-error (nb:fold-in key 1.5))
    (signals nb:prng-error (nb:fold-in key (make-array 2 :element-type '(unsigned-byte 32) :initial-element 0)))))

;;; --- 決定性と独立性 ---

(def-prng-property prng/same-key-gives-same-values
  "同じキー・shape・dtype からは、uniform / normal / split / fold-in が常に同じ値を返す
（入力のキーは書き換えない）。"
  (spec seed) (%prng-shape-seed-generator)
  (let* ((shape (array-spec-shape spec))
         (key (nb:prng-key seed))
         (copy (copy-seq key)))
    (and (equalp (nb:uniform key shape) (nb:uniform copy shape))
         (equalp (nb:normal key shape) (nb:normal copy shape))
         (equalp (nb:uniform key shape :dtype :f64) (nb:uniform copy shape :dtype :f64))
         (equalp (nb:split key 3) (nb:split copy 3))
         (equalp (nb:fold-in key 5) (nb:fold-in copy 5))
         (equalp key copy))))

(def-prng-property prng/result-shape-dtype-and-range
  "uniform は shape・dtype どおりで、全要素が [minval, maxval] に入る（maxval は丸めで等しくなりうる）。
normal は shape・dtype どおりで全要素が有限。"
  (spec seed) (%prng-shape-seed-generator)
  (let* ((shape (array-spec-shape spec))
         (key (nb:prng-key seed)))
    (dolist (dtype '(:f32 :f64) t)
      (let ((u (nb:uniform key shape :dtype dtype))
            (v (nb:uniform key shape :dtype dtype :minval -3 :maxval 5))
            (n (nb:normal key shape :dtype dtype)))
        (unless (and (equalp (nb:make-aval shape dtype) (nb:array-aval u))
                     (equalp (nb:make-aval shape dtype) (nb:array-aval n))
                     (every (lambda (x) (<= 0 x 1)) (%prng-flat u))
                     (every (lambda (x) (<= -3 x 5)) (%prng-flat v))
                     (every (lambda (x) (< (abs x) 10)) (%prng-flat n)))
          (return nil))))))

(def-prng-property prng/split-keys-are-distinct-and-give-different-values
  "split の結果は (n 2) の :u32 で、n 個のキーは互いに異なり、元のキーとも異なる。
各キーから引いた uniform の値も互いに異なる。"
  (seed n) (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                             (uniform-integer :lo 2 :hi 6)))
  (let* ((key (nb:prng-key seed))
         (keys (nb:split key n))
         (rows (loop for i below n collect (list (aref keys i 0) (aref keys i 1))))
         (samples (loop for row in rows
                        collect (%prng-flat (nb:uniform (make-array 2 :element-type '(unsigned-byte 32)
                                                                      :initial-contents row)
                                                        '(4))))))
    (and (equal (list n 2) (array-dimensions keys))
         (eq :u32 (nb:array-dtype keys))
         (= n (length (remove-duplicates rows :test #'equal)))
         (not (member (%prng-flat key) rows :test #'equal))
         (= n (length (remove-duplicates samples :test #'equal))))))

(def-prng-property prng/split-default-is-two
  "split の個数を省くと 2 個になる（個数 1 は (1 2)）。"
  (seed) (%prng-seed-generator)
  (let ((key (nb:prng-key seed)))
    (and (equalp (nb:split key) (nb:split key 2))
         (equal '(1 2) (array-dimensions (nb:split key 1))))))

(def-prng-property prng/fold-in-depends-on-key-and-data
  "fold-in は、データが違えば別のキー、キーが違えば別のキーになり、同じ引数なら同じキーになる。
結果は :u32 の (2)。"
  (seed data) (generator (tuple (uniform-integer :lo 0 :hi (1- (expt 2 31)))
                                (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
  (let* ((key (nb:prng-key seed))
         (other (nb:prng-key (1+ seed)))
         (folded (nb:fold-in key data)))
    (and (equal '(2) (array-dimensions folded))
         (eq :u32 (nb:array-dtype folded))
         (equalp folded (nb:fold-in key data))
         (not (equalp folded (nb:fold-in key (1+ data))))
         (not (equalp folded (nb:fold-in other data)))
         (not (equalp folded key)))))

(test prng/fold-in-takes-traced-data
  "fold-in のデータはトレースされた :u32 / :i32 のスカラーでもよく、整数で呼んだ結果と一致する。"
  (let ((key (nb:prng-key 3)))
    (dolist (dtype '(:u32 :i32))
      (let* ((graph (nb:trace-to-graph (nb:with-tracing (k i) (nb:fold-in k i))
                                       (list (nb:make-aval '(2) :u32) (nb:make-aval '() dtype))))
             (data (make-array '() :element-type (nb:dtype-element-type dtype) :initial-element 12)))
        (is (equalp (nb:fold-in key 12) (nb:eval-graph graph key data)))))))

(test prng/fold-in-does-not-overlap-the-sampling-stream
  "fold-in(key, 0) は、同じキーから uniform / split で引く列の先頭（カウンタ 0 の2語）と一致しない
（fold-in はカウンタ 2^32 + data を使う）。"
  (let* ((key (nb:prng-key 11))
         (head (nb:split key 1))
         (folded (nb:fold-in key 0)))
    (is (not (and (= (aref head 0 0) (aref folded 0)) (= (aref head 0 1) (aref folded 1)))))))

(test prng/different-keys-give-different-uniform-values
  "別のシードのキーからは、別の値が出る（シード 0..19 の先頭 8 個の列がすべて異なる）。"
  (let ((rows (loop for seed below 20 collect (%prng-flat (nb:uniform (nb:prng-key seed) '(8))))))
    (is (= 20 (length (remove-duplicates rows :test #'equal))))))

;;; --- 統計 ---
;;;
;;; 固定のシード・n = 20000 で、平均・分散の z 検定と Kolmogorov–Smirnov 検定を行う。
;;; 有意水準は *PRNG-ALPHA*（両側 1e-3）。KS の臨界値は Dvoretzky–Kiefer–Massart の
;;; 不等式 P(D > e) <= 2 exp(-2 n e^2) から e = sqrt(ln(2 / alpha) / (2 n))（漸近ではなく
;;; 有限の n でも成り立つ上限）。シードを選び直しても、1 つの検定が誤って落ちる確率は
;;; alpha 以下。

(defparameter *prng-sample-size* 20000)

(defun %prng-samples (array)
  "ARRAY の要素の DOUBLE-FLOAT のリスト。"
  (map 'list (lambda (x) (coerce x 'double-float)) (%prng-flat array)))

(defun %mean-and-variance (samples)
  (let* ((n (length samples))
         (mean (/ (reduce #'+ samples) n))
         (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) samples)) (1- n))))
    (values mean variance)))

(defun %ks-statistic (samples cdf)
  "SAMPLES の経験分布と累積分布関数 CDF の最大の差 D。"
  (let* ((sorted (sort (copy-list samples) #'<))
         (n (length sorted))
         (d 0d0))
    (loop for x in sorted for i from 0
          do (let ((f (funcall cdf x)))
               (setf d (max d (- f (/ i n)) (- (/ (1+ i) n) f)))))
    d))

(defun %ks-critical (n)
  (sqrt (/ (log (/ 2 *prng-alpha*)) (* 2 n))))

(defun %std-normal-cdf (x)
  "標準正規分布の累積分布関数（級数 Φ(x) = 1/2 + φ(x) Σ x^(2k+1) / (2k+1)!!、|x| <= 8）。"
  (let ((sum 0d0) (term x))
    (loop for k from 0
          do (incf sum term)
             (setf term (/ (* term x x) (+ (* 2 k) 3)))
          until (< (abs term) 1d-18))
    (+ 0.5d0 (* (/ (exp (/ (* x x) -2)) (sqrt (* 2 pi))) sum))))

(defun %check-uniform-statistics (seed dtype minval maxval)
  (let* ((n *prng-sample-size*)
         (samples (%prng-samples (nb:uniform (nb:prng-key seed) (list n)
                                             :dtype dtype :minval minval :maxval maxval)))
         (width (- maxval minval))
         (unit (mapcar (lambda (x) (/ (- x minval) width)) samples)))
    (multiple-value-bind (mean variance) (%mean-and-variance unit)
      (is (<= (abs (- mean 0.5d0)) (* *prng-z* (sqrt (/ 1d0 (* 12 n)))))
          "uniform ~S seed ~D: 平均 ~F が 0.5 から離れすぎ" dtype seed mean)
      (is (<= (abs (- variance (/ 1d0 12))) (* *prng-z* (sqrt (/ 1d0 (* 180 n)))))
          "uniform ~S seed ~D: 分散 ~F が 1/12 から離れすぎ" dtype seed variance)
      (let ((d (%ks-statistic unit (lambda (x) x))))
        (is (<= d (%ks-critical n)) "uniform ~S seed ~D: KS 統計量 ~F が臨界値 ~F を超えた"
            dtype seed d (%ks-critical n))))))

(defun %check-normal-statistics (seed dtype)
  (let* ((n *prng-sample-size*)
         (samples (%prng-samples (nb:normal (nb:prng-key seed) (list n) :dtype dtype))))
    (multiple-value-bind (mean variance) (%mean-and-variance samples)
      (is (<= (abs mean) (* *prng-z* (sqrt (/ 1d0 n))))
          "normal ~S seed ~D: 平均 ~F が 0 から離れすぎ" dtype seed mean)
      (is (<= (abs (- variance 1)) (* *prng-z* (sqrt (/ 2d0 n))))
          "normal ~S seed ~D: 分散 ~F が 1 から離れすぎ" dtype seed variance)
      (let ((d (%ks-statistic samples #'%std-normal-cdf)))
        (is (<= d (%ks-critical n)) "normal ~S seed ~D: KS 統計量 ~F が臨界値 ~F を超えた"
            dtype seed d (%ks-critical n))))))

(test prng/uniform-passes-statistical-tests
  "uniform（f32 / f64、既定の範囲と [-3, 5)）が、平均・分散・KS 検定を通る（固定のシード、有意水準 1e-3）。"
  (dolist (seed '(2024 7 99))
    (%check-uniform-statistics seed :f32 0 1)
    (%check-uniform-statistics seed :f64 0 1))
  (%check-uniform-statistics 31 :f32 -3 5)
  (%check-uniform-statistics 31 :f64 -3 5))

(test prng/normal-passes-statistical-tests
  "normal（f32 / f64）が、平均・分散・KS 検定を通る（固定のシード、有意水準 1e-3）。"
  (dolist (seed '(2024 7 99))
    (%check-normal-statistics seed :f32)
    (%check-normal-statistics seed :f64)))

(test prng/normal-tails-match-the-gaussian
  "正規乱数の裾: |x| > 2 と |x| > 3 の割合が、理論値（0.0455 / 0.0027）と二項分布の 3.29 標準偏差以内。
n = 200000。"
  (let* ((n 200000)
         (samples (%prng-samples (nb:normal (nb:prng-key 5) (list n)))))
    (dolist (cell '((2 0.04550026d0) (3 0.0026997961d0)))
      (destructuring-bind (threshold p) cell
        (let ((fraction (/ (count-if (lambda (x) (> (abs x) threshold)) samples) n)))
          (is (<= (abs (- fraction p)) (* *prng-z* (sqrt (/ (* p (- 1 p)) n))))
              "|x| > ~D の割合 ~F が理論値 ~F から離れすぎ" threshold fraction p))))))

(test prng/split-keys-are-uncorrelated
  "split した2つのキーから引いた一様乱数の標本相関が、0 から 3.29 / sqrt(n) 以内（独立なら標準偏差 1/sqrt(n)）。
親のキーの列とも無相関。"
  (let* ((n *prng-sample-size*)
         (key (nb:prng-key 77))
         (keys (nb:split key 2))
         (draws (loop for i below 2
                      collect (%prng-samples
                               (nb:uniform (make-array 2 :element-type '(unsigned-byte 32)
                                                         :initial-contents (list (aref keys i 0) (aref keys i 1)))
                                           (list n)))))
         (parent (%prng-samples (nb:uniform key (list n)))))
    (flet ((correlation (a b)
             (multiple-value-bind (ma va) (%mean-and-variance a)
               (multiple-value-bind (mb vb) (%mean-and-variance b)
                 (/ (/ (reduce #'+ (mapcar (lambda (x y) (* (- x ma) (- y mb))) a b)) (1- n))
                    (sqrt (* va vb)))))))
      (is (<= (abs (correlation (first draws) (second draws))) (/ *prng-z* (sqrt n))))
      (is (<= (abs (correlation parent (first draws))) (/ *prng-z* (sqrt n))))
      (is (<= (abs (correlation parent (second draws))) (/ *prng-z* (sqrt n)))))))

;;; --- トレースの中でも同じ値 ---

(def-prng-property prng/traced-graph-matches-eager
  "with-tracing の関数を graph にして eager 評価した結果が、直接呼んだ結果と一致する
（uniform f32 / f64、normal、split、fold-in）。"
  (spec seed) (%prng-shape-seed-generator)
  (let* ((shape (array-spec-shape spec))
         (key (nb:prng-key seed))
         (key-aval (nb:make-aval '(2) :u32)))
    (flet ((traced (fn) (nb:eval-graph (nb:trace-to-graph fn (list key-aval)) key)))
      (and (equalp (nb:uniform key shape)
                   (traced (nb:with-tracing (k) (nb:uniform k shape))))
           (equalp (nb:uniform key shape :dtype :f64 :minval -1 :maxval 4)
                   (traced (nb:with-tracing (k) (nb:uniform k shape :dtype :f64 :minval -1 :maxval 4))))
           (equalp (nb:normal key shape)
                   (traced (nb:with-tracing (k) (nb:normal k shape))))
           (equalp (nb:normal key shape :dtype :f64)
                   (traced (nb:with-tracing (k) (nb:normal k shape :dtype :f64))))
           (equalp (nb:split key 4) (traced (nb:with-tracing (k) (nb:split k 4))))
           (equalp (nb:fold-in key 9) (traced (nb:with-tracing (k) (nb:fold-in k 9))))))))

(test prng/constant-key-inside-tracing
  "トレースの中で prng-key を作って使っても（キーは定数として取り込まれる）、eager と同じ値になる。"
  (let* ((fn (nb:with-tracing (x) (+ x (nb:uniform (nb:prng-key 5) '(3)))))
         (graph (nb:trace-to-graph fn (list (nb:make-aval '(3) :f32))))
         (x (make-array 3 :element-type 'single-float :initial-element 10f0))
         (u (nb:uniform (nb:prng-key 5) '(3))))
    (is (allclose (nb:eval-graph graph x)
                  (make-array 3 :element-type 'single-float
                                :initial-contents (loop for i below 3 collect (+ 10f0 (aref u i))))
                  :dtype :f32))))

;;; --- バッチ化 ---

(defun %batch-rule-call (name args dims &rest params)
  "プリミティブ NAME のバッチ化ルールを直接呼び、出力のトレーサのリストと軸のリストを返す。"
  (multiple-value-list (apply (nb::primitive-batch (nb::find-primitive name)) args dims params)))

(defun %rng-batched-bits (states shape dtype)
  "バッチ化ルール（軸 0 でバッチされた状態）を通した (新しい状態 ビット)。graph にして eager 評価する。"
  (nb:eval-graph
   (nb:trace-to-graph
    (nb:with-tracing (s)
      (let ((result (%batch-rule-call :rng-bit-generator (list s) (list 0) :shape shape :dtype dtype)))
        (values-list (first result))))
    (list (nb:array-aval states :u64)))
   states))

(test prng/rng-batch-rule-output-axes-are-zero
  "ルールが返す出力の軸は、新しい状態もビットも 0（出力の個数 2 に対して (0 0)）。"
  (let ((axes nil))
    (flet ((call (s)
             ;; with-tracing の中では setq できないので、トレース対象の外の関数で記録する
             (let ((result (%batch-rule-call :rng-bit-generator (list s) (list 0) :shape '(3) :dtype :u32)))
               (setf axes (second result))
               (values-list (first result)))))
      (nb:trace-to-graph (nb:with-tracing (s) (call s)) (list (nb:make-aval '(4 2) :u64))))
    (is (equal '(0 0) axes))))

(test prng/rng-batched-emit-structure
  "バッチ次元つきの状態の StableHLO は、行数ぶんの slice と rng_bit_generator、2 回の concatenate、
行ごと 3 回 + 前後 3 回の reshape を持ち、SSA 名が不正（%%）にならない。バッチ次元の無い状態は
slice も concatenate も出さない。"
  (flet ((count-of (needle text)
           (loop with start = 0 for pos = (search needle text :start2 start)
                 while pos count t do (setf start (1+ pos))))
         (emit (state-shape)
           (nb:emit-stablehlo
            (nb:trace-to-graph (nb:with-tracing (s) (nb::rng-bit-generator s :shape '(4) :dtype :u32))
                               (list (nb:make-aval state-shape :u64))))))
    (let ((batched (emit '(3 2)))
          (single (emit '(2))))
      (is (= 3 (count-of "stablehlo.rng_bit_generator" batched)))
      (is (= 3 (count-of "stablehlo.slice" batched)))
      (is (= 2 (count-of "stablehlo.concatenate" batched)))
      (is (= 12 (count-of "stablehlo.reshape" batched)))
      (is (zerop (count-of "%%" batched)))
      (is (= 1 (count-of "stablehlo.rng_bit_generator" single)))
      (is (zerop (count-of "stablehlo.slice" single)))
      (is (zerop (count-of "stablehlo.concatenate" single))))))

(defun %prng-states (rows seed)
  (let ((rs (sb-ext:seed-random-state seed))
        (states (make-array (list rows 2) :element-type '(unsigned-byte 64))))
    (dotimes (i (* rows 2) states)
      (setf (row-major-aref states i) (random (expt 2 64) rs)))))

(def-prng-property prng/rng-batch-rule-matches-per-state-calls
  "rng-bit-generator のバッチ化ルール（軸 0）の出力は、各行の状態を単独に呼んだ結果（新しい状態・
ビット）を行ごとに積んだものとビット単位で一致し、出力の軸は (0 0)。:u32 / :u64 と複数の形で確かめる。"
  (spec rows seed) (generator (tuple (array-spec :dtypes '(:f32) :max-rank 2)
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
  (let ((shape (array-spec-shape spec))
        (states (%prng-states rows seed)))
    (dolist (dtype '(:u32 :u64) t)
      (multiple-value-bind (new-states bits) (%rng-batched-bits states shape dtype)
        (let ((singles (loop for r below rows
                             collect (multiple-value-list
                                      (nb::rng-bit-generator
                                       (make-array 2 :element-type '(unsigned-byte 64)
                                                     :initial-contents (list (aref states r 0) (aref states r 1)))
                                       :shape shape :dtype dtype)))))
          (unless (and (equal (list rows 2) (array-dimensions new-states))
                       (equal (cons rows shape) (array-dimensions bits))
                       (loop for r below rows
                             for (single-state single-bits) in singles
                             always (and (equalp single-state
                                                 (make-array 2 :element-type '(unsigned-byte 64)
                                                               :initial-contents (list (aref new-states r 0)
                                                                                       (aref new-states r 1))))
                                         (equalp (%prng-flat single-bits)
                                                 (subseq (%prng-flat bits) (* r (length (%prng-flat single-bits)))
                                                         (* (1+ r) (length (%prng-flat single-bits))))))))
            (return nil)))))))

(test prng/rng-batch-rule-axes-and-nested-batch
  "ルールは軸 0 以外のバッチ軸も受け、出力の軸は常に (0 0)。バッチ次元が2段（B1 B2 2）でも
各行が単独の呼び出しと一致する（vmap の入れ子の土台）。"
  (let* ((states (%prng-states 6 17))
         (nested (make-array '(2 3 2) :element-type '(unsigned-byte 64)
                                      :displaced-to (copy-seq (make-array 12 :element-type '(unsigned-byte 64)
                                                                             :initial-contents (%prng-flat states)))))
         (flat (multiple-value-list (nb::rng-bit-generator states :shape '(3) :dtype :u32)))
         (deep (multiple-value-list (nb::rng-bit-generator nested :shape '(3) :dtype :u32))))
    (is (equal '(6 2) (array-dimensions (first flat))))
    (is (equal '(6 3) (array-dimensions (second flat))))
    (is (equal '(2 3 2) (array-dimensions (first deep))))
    (is (equal '(2 3 3) (array-dimensions (second deep))))
    (is (equal (%prng-flat (second flat)) (%prng-flat (second deep))))
    ;; 軸 1 でバッチ（shape (2 B)）のとき、軸 0 へ動かしてから適用する
    (let* ((transposed (let ((a (make-array '(2 6) :element-type '(unsigned-byte 64))))
                         (dotimes (r 6 a)
                           (setf (aref a 0 r) (aref states r 0) (aref a 1 r) (aref states r 1)))))
           (result (nb:eval-graph
                    (nb:trace-to-graph
                     (nb:with-tracing (s)
                       (values-list (first (%batch-rule-call :rng-bit-generator (list s) (list 1)
                                                             :shape '(3) :dtype :u32))))
                     (list (nb:make-aval '(2 6) :u64)))
                    transposed)))
      (is (equalp result (first flat))))))

;;; vmap（複数出力の eqn を歩けるようになる #140 まではスキップ）

(defun %check-vmap-over-keys (name fn-of-key &key (rows 4))
  "(vmap FN-OF-KEY) をキー行列に適用した結果が、各行を単独に FN-OF-KEY した結果と一致する。"
  (let* ((keys (nb:split (nb:prng-key 123) rows))
         (batched (funcall (nb:vmap fn-of-key) keys))
         (expected (first (reference-vmap fn-of-key (list keys)))))
    (is (equalp expected batched) "~A: vmap の結果が各キーで単独に呼んだ結果と一致しない" name)))

(test prng/vmap-over-keys-matches-per-key-calls
  "vmap でキーをバッチすると、各要素はそのキーで単独に呼んだ結果と一致する
（uniform / normal / split / fold-in。ビット単位）。複数出力の eqn を vmap が歩けるようになる
（#140）までは PENDING-140 でスキップする（そのときのルール単体の性質は
prng/rng-batch-rule-matches-per-state-calls が確かめる）。"
  (if (not (vmap-walks-multiple-outputs-p))
      (skip "PENDING-140: vmap はまだ複数出力の eqn を歩けない")
      (progn
        (%check-vmap-over-keys "uniform f32" (nb:with-tracing (k) (nb:uniform k '(3 2))))
        (%check-vmap-over-keys "uniform f64 範囲つき"
                               (nb:with-tracing (k) (nb:uniform k '(5) :dtype :f64 :minval -2 :maxval 3)))
        (%check-vmap-over-keys "normal" (nb:with-tracing (k) (nb:normal k '(2 3))))
        (%check-vmap-over-keys "split" (nb:with-tracing (k) (nb:split k 3)))
        (%check-vmap-over-keys "fold-in" (nb:with-tracing (k) (nb:fold-in k 7)))
        ;; 入れ子（キーの行列の行列）と、キーをバッチしない引数との組み合わせ
        (let* ((keys (nb:split (nb:prng-key 5) 6))
               (grid (let ((a (make-array '(2 3 2) :element-type '(unsigned-byte 32))))
                       (dotimes (i 12 a) (setf (row-major-aref a i) (row-major-aref keys i)))))
               (nested (funcall (nb:vmap (nb:vmap (nb:with-tracing (k) (nb:uniform k '(2))))) grid)))
          (is (equalp (first (reference-vmap
                              (lambda (row) (first (reference-vmap (lambda (k) (nb:uniform k '(2))) (list row))))
                              (list grid)))
                      nested)))
        (let* ((keys (nb:split (nb:prng-key 6) 3))
               (x (make-array '(3 2) :element-type 'single-float :initial-element 1f0))
               (f (nb:with-tracing (k y) (+ y (nb:normal k '(2)))))
               (batched (funcall (nb:vmap f :in-axes '(0 0)) keys x)))
          (is (allclose batched
                        (first (reference-vmap f (list keys x)))
                        :dtype :f32))))))
