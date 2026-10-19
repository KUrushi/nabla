;;;; 基本レイヤー dense / layer-norm / embedding / dropout / sequential と活性化関数（issue #204、small）。
;;;;
;;;; 守らせる性質:
;;;; - Flax との数値一致: 同じパラメータ・入力・余接線で、順伝播の出力と、パラメータ・入力についての
;;;;   勾配が Flax（flax.linen）のフィクスチャ（tests/fixtures/layers/flax-layers.lisp、生成は同じ
;;;;   ディレクトリの generate.py）と f32 の許容誤差で一致する（eager。jit は medium、
;;;;   tests/nn/layers-iree-test.lisp）。
;;;; - init: 形・dtype・スロット名（Flax と同じ kernel / bias / scale / embedding）、同じキーで
;;;;   同じ値・別のキーで別の値、分布が Flax と同じ（lecun_normal の標準偏差と切断、embedding の
;;;;   標準偏差、bias = 0、scale = 1）。
;;;; - layer-norm: 出力は行ごとに平均 0・分散 1（scale = 1、bias = 0）、入力のずらしと正のスケールに
;;;;   不変、scale / bias がアフィン変換として効く。
;;;; - embedding: 出力は ids の指す行そのもの。範囲外の id は 0 の行（Flax の NaN と違う、README）。
;;;; - dropout: 推論時（deterministic）は入力そのもの、同じキーで同じマスク・別のキーで別のマスク、
;;;;   保持率が 1 - rate、保持した要素は 1 / (1 - rate) 倍、期待値は入力と同じ。
;;;; - sequential: 各層を順に適用したものと等しい。
;;;; - 活性化関数: relu / softmax / log-softmax の定義どおりの性質。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

;;; --- Flax フィクスチャ ---

(defvar *layers-fixture* nil)

(defun %layers-fixture ()
  (or *layers-fixture*
      (setf *layers-fixture*
            (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/layers/flax-layers.lisp"))
              (read stream)))))

(defun %fx-case (name)
  (or (find name (getf (%layers-fixture) :cases) :key (lambda (c) (getf c :name)) :test #'string=)
      (error "フィクスチャにケース ~S が無い" name)))

(defun %fx-entry (case name)
  (find name (getf case :arrays) :key #'first :test #'string=))

(defun %fx-array (case name &key ints)
  "ケース CASE の配列 NAME。INTS が真なら整数（:i32）、偽なら f32 のビットパターン。"
  (destructuring-bind (name shape values) (%fx-entry case name)
    (declare (ignore name))
    (let ((array (make-array shape :element-type (if ints '(signed-byte 32) 'single-float))))
      (loop for i from 0 for value in values
            do (setf (row-major-aref array i) (if ints value (nabla::%make-single-float value))))
      array)))

(defun %fx-param-names (case)
  "ケースのパラメータ名（grad.* のうち grad.x でないもの）。フィクスチャの並びで、葉の順と同じ。"
  (loop for (name) in (getf case :arrays)
        when (and (> (length name) 5) (string= "grad." name :end2 5) (string/= name "grad.x"))
          collect (subseq name 5)))

(defun %elements (array)
  "ARRAY の要素のリスト（row-major）。"
  (loop for i below (array-total-size array) collect (row-major-aref array i)))

(defun %slot-names (module)
  "MODULE-SLOTS を (小文字のスロット名 . 種類) にしたもの（チェックポイントのパス名に使う形）。"
  (mapcar (lambda (slot) (cons (string-downcase (symbol-name (car slot))) (cdr slot))) (module-slots module)))

(defun %case-tolerance (name)
  "ケース NAME の許容誤差（allclose のキーワード引数）。既定は f32 のまま（rtol 1e-5 / atol 1e-6）。
layer-norm だけ緩める: Flax と同じ速い分散 E[x^2] - E[x]^2 は、平均 3・標準偏差 2 の入力で f32 の桁を
1 桁ほど失い、XLA と nabla（IREE / eager）の丸めの違いがそのまま出る。実測の最大誤差は 6.9e-6
（既定の許容誤差 1e-5 + 1e-6 |期待値| を超える）。1e-4 / 1e-5 は実測の十数倍。"
  (if (and (>= (length name) 10) (string= "layer-norm" name :end2 10))
      '(:rtol 1d-4 :atol 1d-5)
      '()))

(defun %close-f32 (actual expected &rest tolerance)
  (apply #'allclose actual expected :dtype :f32 tolerance))

(defun %check-flax-case (name model forward &key ints)
  "ケース NAME の入力 x と余接線 ct で loss = sum(forward(model, x) * ct) を作り、出力と勾配を
Flax の値と比べる。MODEL が NIL（パラメータ無し）のときは x の勾配だけを見る。"
  (let* ((case (%fx-case name))
         (x (%fx-array case "x" :ints ints))
         (ct (%fx-array case "ct"))
         (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (funcall forward m x) ct))))
         (x-grad-p (%fx-entry case "grad.x"))
         (argnums (append (when model '(0)) (when x-grad-p '(1))))
         (grads (funcall (nabla:grad loss :argnums argnums) model x ct))
         (expected (append (mapcar (lambda (p) (%fx-array case (concatenate 'string "grad." p)))
                                   (%fx-param-names case))
                           (when x-grad-p (list (%fx-array case "grad.x"))))))
    (is (apply #'%close-f32 (funcall forward model x) (%fx-array case "out") (%case-tolerance name))
        "~A: 順伝播の出力が Flax と一致しない" name)
    (let ((actual (append (when model (nabla:tree-leaves (first grads)))
                          (when x-grad-p (last grads)))))
      (is (= (length expected) (length actual)) "~A: 勾配の個数が違う" name)
      (loop for a in actual for e in expected for i from 0
            do (is (apply #'%close-f32 a e (%case-tolerance name))
                     "~A: ~D 番目の勾配が Flax と一致しない" name i)))))

(defun %fx-dense (case prefix)
  (make-dense :use-bias t :kernel (%fx-array case (concatenate 'string prefix "kernel"))
              :bias (%fx-array case (concatenate 'string prefix "bias"))))

(defparameter *layers-case-table*
  (list
   (list "dense" (lambda (case) (%fx-dense case "")) (nabla:with-tracing (m x) (call m x)) nil)
   (list "dense-rank1" (lambda (case) (%fx-dense case "")) (nabla:with-tracing (m x) (call m x)) nil)
   (list "layer-norm"
         (lambda (case) (make-layer-norm :scale (%fx-array case "scale") :bias (%fx-array case "bias")
                                         :epsilon 1f-6))
         (nabla:with-tracing (m x) (call m x)) nil)
   (list "layer-norm-tiny-variance"
         (lambda (case) (make-layer-norm :scale (%fx-array case "scale") :bias (%fx-array case "bias")
                                         :epsilon 1f-6))
         (nabla:with-tracing (m x) (call m x)) nil)
   (list "embedding" (lambda (case) (make-embedding :embedding (%fx-array case "embedding")))
         (nabla:with-tracing (m x) (call m x)) t)
   (list "relu" (lambda (case) (declare (ignore case)) nil) (nabla:with-tracing (m x) m (relu x)) nil)
   (list "gelu" (lambda (case) (declare (ignore case)) nil) (nabla:with-tracing (m x) m (gelu x)) nil)
   (list "softmax" (lambda (case) (declare (ignore case)) nil) (nabla:with-tracing (m x) m (softmax x)) nil)
   (list "softmax-axis0" (lambda (case) (declare (ignore case)) nil)
         (nabla:with-tracing (m x) m (softmax x :axis 0)) nil)
   (list "log-softmax" (lambda (case) (declare (ignore case)) nil)
         (nabla:with-tracing (m x) m (log-softmax x)) nil)
   (list "mlp"
         (lambda (case)
           (make-sequential :layers (list (%fx-dense case "0.") (make-activation :fn :relu)
                                          (%fx-dense case "2."))))
         (nabla:with-tracing (m x) (call m x)) nil))
  "(ケース名 モデルを作る関数 順伝播（traceable-function (m x)） x が整数か)。medium からも使う。")

(defparameter *layers-extra-case-table*
  (list
   (list "dense-no-bias"
         (lambda (case) (make-dense :kernel (%fx-array case "kernel") :bias nil :use-bias nil))
         (nabla:with-tracing (m x) (call m x)) nil))
  "issue #241 で足した Flax との差のケース（*layers-case-table* と同じ形）。フィクスチャの再生成が要る。
medium からも使う。")

(test layers/extra-cases-match-flax
  "Flax の nn.Dense(use_bias=False) のフィクスチャで、eager の出力と kernel・入力の勾配が一致する
（bias の勾配は無い）。"
  (dolist (entry *layers-extra-case-table*)
    (destructuring-bind (name make forward ints) entry
      (%check-flax-case name (funcall make (%fx-case name)) forward :ints ints))))

(test layers/forward-and-grads-match-flax
  "dense（rank 3 と rank 1 の入力）、layer-norm（平均のずれた入力と分散がほぼ 0 の入力）、embedding、
relu / gelu / softmax（軸 -1 と 0）/ log-softmax、dense + relu + dense の sequential が、Flax のフィクスチャ
と同じ出力と勾配（パラメータと入力）を eager で返す。"
  (dolist (entry *layers-case-table*)
    (destructuring-bind (name make forward ints) entry
      (%check-flax-case name (funcall make (%fx-case name)) forward :ints ints))))

;;; --- init ---

(defun %stats (array)
  "(values 標準偏差 最大絶対値)。"
  (let* ((n (array-total-size array))
         (mean (/ (loop for i below n sum (coerce (row-major-aref array i) 'double-float)) n))
         (variance (/ (loop for i below n sum (expt (- (row-major-aref array i) mean) 2)) n)))
    (values (sqrt variance)
            (loop for i below n maximize (abs (coerce (row-major-aref array i) 'double-float))))))

(defun %init-stat (name)
  (rest (assoc name (getf (%layers-fixture) :init-stats))))

(test layers/dense-init-matches-flax-lecun-normal
  "init-dense (512 → 256) の kernel は Flax の lecun_normal と同じ分布: 標準偏差が 1/sqrt(fan_in)
（Flax のサンプルの標準偏差と 2% 以内）、[-2, 2] を stddev = 1/(0.8796 sqrt(fan_in)) 倍した範囲に切れていて
最大絶対値はその境界の 97% 以上。bias は 0。形 (in out)、dtype f32、スロットの順は kernel, bias。"
  (let* ((dense (init-dense (nabla:prng-key 204) 512 256))
         (kernel (dense-kernel dense))
         (bound (* 2 (/ (sqrt (/ 1d0 512)) 0.87962566103423978d0))))
    (destructuring-bind (shape flax-std flax-max) (%init-stat :dense-kernel)
      (declare (ignore shape))
      (multiple-value-bind (std max) (%stats kernel)
        (is (equal '(512 256) (array-dimensions kernel)))
        (is (eq 'single-float (array-element-type kernel)))
        (is (<= (abs (- std flax-std)) (* 0.02 flax-std)) "標準偏差 ~F と Flax の ~F" std flax-std)
        (is (<= (abs (- std (sqrt (/ 1d0 512)))) (* 0.02 std)) "標準偏差 ~F" std)
        (is (<= max (* bound (+ 1 1d-5))) "最大絶対値 ~F が境界 ~F を超えた" max bound)
        (is (>= max (* 0.97 bound)) "最大絶対値 ~F が境界 ~F に届かない（切断が違う）" max bound)
        (is (<= (abs (- max flax-max)) (* 0.03 flax-max)))))
    (is (equal '(256) (array-dimensions (dense-bias dense))))
    (is (every #'zerop (%elements (dense-bias dense))))
    (is (equal '(("kernel" . :param) ("bias" . :param) ("use-bias" . :static)) (%slot-names 'dense)))))

(test layers/embedding-init-matches-flax
  "init-embedding (1000 × 64) の embedding は N(0, 1/features)（Flax の embed 初期化）: 標準偏差が
Flax のサンプルと 3% 以内で 1/sqrt(64) に一致する。形 (num-embeddings features)。"
  (let ((embedding (embedding-embedding (init-embedding (nabla:prng-key 204) 1000 64))))
    (destructuring-bind (shape flax-std flax-max) (%init-stat :embedding)
      (declare (ignore shape flax-max))
      (is (equal '(1000 64) (array-dimensions embedding)))
      (let ((std (%stats embedding)))
        (is (<= (abs (- std flax-std)) (* 0.03 flax-std)) "標準偏差 ~F と Flax の ~F" std flax-std)
        (is (<= (abs (- std 0.125d0)) (* 0.03 0.125d0)))))))

(test layers/layer-norm-init-is-ones-and-zeros-with-flax-epsilon
  "init-layer-norm は scale = 1、bias = 0、既定の epsilon は Flax と同じ 1e-6。キーによらない。f64 も選べる。"
  (let ((norm (init-layer-norm (nabla:prng-key 1) 5)))
    (is (equalp (make-array 5 :element-type 'single-float :initial-element 1.0) (layer-norm-scale norm)))
    (is (equalp (make-array 5 :element-type 'single-float :initial-element 0.0) (layer-norm-bias norm)))
    (is (eql 1f-6 (layer-norm-epsilon norm)))
    (is (equalp (layer-norm-scale norm) (layer-norm-scale (init-layer-norm (nabla:prng-key 2) 5))))
    (is (eql 1f-5 (layer-norm-epsilon (init-layer-norm (nabla:prng-key 1) 5 :epsilon 1f-5))))
    (is (eq 'double-float (array-element-type
                           (layer-norm-scale (init-layer-norm (nabla:prng-key 1) 5 :dtype :f64)))))
    (is (equal '(("scale" . :param) ("bias" . :param) ("epsilon" . :static)) (%slot-names 'layer-norm)))
    (is (equal '(("embedding" . :param)) (%slot-names 'embedding)))))

(defmacro def-layer-property (name docstring (&rest vars) generator &body body)
  "GENERATOR の値を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test (,name :suite :nabla.small)
     ,docstring
     (is (check-it (generator ,generator)
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.NN.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

(defun %rand (shape seed &optional (dtype :f32))
  (make-random-array (make-array-spec shape dtype) :seed seed))

(defun %same-leaves-p (a b)
  (every #'equalp (nabla:tree-leaves a) (nabla:tree-leaves b)))

(def-layer-property layers/init-is-deterministic-per-key
  "dense / embedding / sequential の init は、同じキーなら同じパラメータ、違うキーなら違うパラメータ。
kernel と bias の形は (in out) と (out)。"
  (seed in out) (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 1 :hi 6)
                       (uniform-integer :lo 1 :hi 6))
  (let ((a (init-dense (nabla:prng-key seed) in out))
        (b (init-dense (nabla:prng-key seed) in out))
        (c (init-dense (nabla:prng-key (1+ seed)) in out))
        (ea (init-embedding (nabla:prng-key seed) in out))
        (eb (init-embedding (nabla:prng-key seed) in out))
        (ec (init-embedding (nabla:prng-key (1+ seed)) in out)))
    (and (%same-leaves-p a b) (not (%same-leaves-p a c))
         (equal (list in out) (array-dimensions (dense-kernel a)))
         (equal (list out) (array-dimensions (dense-bias a)))
         (%same-leaves-p ea eb) (not (%same-leaves-p ea ec))
         (equal (list in out) (array-dimensions (embedding-embedding ea))))))

;;; --- layer-norm の性質 ---

(def-layer-property layers/layer-norm-normalizes-each-row
  "scale = 1、bias = 0 の layer-norm の出力は、最後の軸ごとに平均 0・分散 1（epsilon の分だけ小さい）。"
  (seed rows features) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 4)
                              (uniform-integer :lo 2 :hi 8))
  (let* ((x (%rand (list rows features) seed))
         (y (call (init-layer-norm (nabla:prng-key 0) features :epsilon 1f-6) x)))
    (loop for r below rows
          always (let* ((row (loop for f below features collect (aref y r f)))
                        (mean (/ (reduce #'+ row) features))
                        (variance (/ (reduce #'+ (mapcar (lambda (v) (expt (- v mean) 2)) row)) features))
                        (x-variance (let* ((xs (loop for f below features collect (aref x r f)))
                                           (m (/ (reduce #'+ xs) features)))
                                      (/ (reduce #'+ (mapcar (lambda (v) (expt (- v m) 2)) xs)) features))))
                   ;; 分散が小さい行は Flax と同じ速い分散（E[x^2] - E[x]^2）の桁落ちで f32 の誤差が
                   ;; 相対的に大きくなる（Flax と同じ式なので仕様）。その行は分散の検査を飛ばす
                   (and (< (abs mean) 1d-4)
                        (or (< x-variance 0.05d0)
                            ;; y の分散 = v / (v + epsilon)
                            (< (abs (- variance (/ x-variance (+ x-variance 1d-6)))) 1d-4)))))))

(def-layer-property layers/layer-norm-is-invariant-to-shift-and-positive-scale
  "layer-norm の出力は、入力に定数を足しても、正の定数を掛けても（分散が epsilon より十分大きいとき）
変わらない。"
  (seed shift scale) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo -20 :hi 20)
                            (uniform-integer :lo 5 :hi 40))
  (let* ((features 6)
         (norm (init-layer-norm (nabla:prng-key 0) features))
         (x (%rand (list 3 features) seed))
         (a (/ scale 10d0))
         (moved (let ((y (make-array '(3 6) :element-type 'single-float)))
                  (dotimes (i 18 y)
                    (setf (row-major-aref y i)
                          (coerce (+ (* a (row-major-aref x i)) (/ shift 10d0)) 'single-float))))))
    (allclose (call norm moved) (call norm x) :dtype :f32 :rtol 1d-3 :atol 1d-3)))

(def-layer-property layers/layer-norm-scale-and-bias-are-affine
  "layer-norm の出力は、scale = 1・bias = 0 の出力に scale を掛けて bias を足したもの。"
  (seed) (tuple (uniform-integer :lo 0 :hi 100000))
  (let* ((features 5)
         (x (%rand (list 2 features) seed))
         (scale (%rand (list features) (+ seed 1)))
         (bias (%rand (list features) (+ seed 2)))
         (plain (call (init-layer-norm (nabla:prng-key 0) features) x))
         (full (call (make-layer-norm :scale scale :bias bias :epsilon 1f-6) x)))
    (loop for r below 2
          always (loop for f below features
                       always (< (abs (- (aref full r f)
                                         (+ (* (aref plain r f) (aref scale f)) (aref bias f))))
                                 1d-5)))))

;;; --- embedding / dense ---

(def-layer-property layers/embedding-looks-up-rows-and-zeroes-out-of-range-ids
  "embedding の出力 [b t] は embedding の行 ids[b t] そのもの。範囲外（負・語彙数以上）の id は 0 の行。"
  (seed vocab features) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 6)
                               (uniform-integer :lo 1 :hi 5))
  (let* ((module (init-embedding (nabla:prng-key seed) vocab features))
         (table (embedding-embedding module))
         (ids (make-array '(2 3) :element-type '(signed-byte 32)
                                 :initial-contents (list (list 0 (1- vocab) (mod seed vocab))
                                                         (list vocab -1 (mod (+ seed 1) vocab)))))
         (out (call module ids)))
    (and (equal (list 2 3 features) (array-dimensions out))
         (loop for b below 2
               always (loop for tt below 3
                            for id = (aref ids b tt)
                            always (loop for f below features
                                         always (= (aref out b tt f)
                                                   (if (< -1 id vocab) (aref table id f) 0.0))))))))

(def-layer-property layers/dense-is-an-affine-map-over-the-last-axis
  "dense の出力は x kernel + bias（行ごと）。入力は rank 1 から 3。"
  (seed rank in out) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 3)
                            (uniform-integer :lo 1 :hi 5) (uniform-integer :lo 1 :hi 5))
  (let* ((dense (make-dense :use-bias t :kernel (%rand (list in out) seed) :bias (%rand (list out) (+ seed 1))))
         (shape (append (make-list (1- rank) :initial-element 2) (list in)))
         (x (%rand shape seed))
         (y (call dense x))
         (kernel (dense-kernel dense))
         (bias (dense-bias dense)))
    (and (equal (append (butlast shape) (list out)) (array-dimensions y))
         (loop for i below (floor (array-total-size x) in)
               always (loop for o below out
                            always (< (abs (- (row-major-aref y (+ (* i out) o))
                                              (+ (aref bias o)
                                                 (loop for k below in
                                                       sum (* (row-major-aref x (+ (* i in) k))
                                                              (aref kernel k o))))))
                                      1d-5))))))

;;; --- dropout ---

(defun %ones (shape) (make-array shape :element-type 'single-float :initial-element 1.0))

(def-layer-property layers/dropout-deterministic-is-identity
  "deterministic が真なら、dropout は（rate や key によらず）入力そのものを返す。rate = 0 のときも。"
  (seed rate-percent) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 100))
  (let ((x (%rand '(3 4) seed))
        (module (make-dropout :rate (/ rate-percent 100f0))))
    (and (eq x (call module x (nabla:prng-key seed) t))
         (eq x (call (make-dropout :rate 0f0) x (nabla:prng-key seed) nil)))))

(def-layer-property layers/dropout-same-key-same-mask-different-key-different-mask
  "同じキーなら同じマスク（出力が等しい）、別のキーなら別のマスク。出力の要素は 0 か 1/(1-rate) 倍の入力。"
  (seed rate-percent) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 10 :hi 90))
  (let* ((rate (/ rate-percent 100f0))
         (module (make-dropout :rate rate))
         (x (%ones '(8 16)))
         (a (call module x (nabla:prng-key seed) nil))
         (b (call module x (nabla:prng-key seed) nil))
         (c (call module x (nabla:prng-key (1+ seed)) nil))
         (scaled (/ 1 (- 1 rate))))
    (and (equalp a b) (not (equalp a c))
         (loop for i below (array-total-size a)
               always (let ((v (row-major-aref a i)))
                        (or (zerop v) (< (abs (- v scaled)) (* 1d-5 scaled))))))))

(def-layer-property layers/dropout-keeps-the-expected-fraction
  "dropout の保持率は 1 - rate（20000 要素。二項分布の 4.5 標準偏差以内。誤って落ちる確率は 1e-5 未満）、
出力の平均は入力の平均に近い（inverted dropout）。"
  (seed rate-percent) (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 5 :hi 95))
  (let* ((rate (/ rate-percent 100d0))
         (n 20000)
         (out (call (make-dropout :rate (coerce rate 'single-float)) (%ones (list 200 100))
                    (nabla:prng-key seed) nil))
         (kept (loop for i below n count (plusp (row-major-aref out i))))
         (keep (- 1 rate))
         (sd (sqrt (/ (* keep rate) n)))
         (mean (/ (loop for i below n sum (row-major-aref out i)) n)))
    (and (< (abs (- (/ kept n) keep)) (* 4.5 sd))
         (< (abs (- mean 1)) (* 4.5 (/ sd keep))))))

(test layers/dropout-edge-rates-and-gradient
  "rate = 0 は入力そのもの、rate = 1 は全部 0。トレース中でも使え、勾配は保持した要素で 1/(1-rate)、
落とした要素で 0（マスクは入力によらない）。rate が 0 以上 1 以下の実数でなければ init-dropout が error。"
  (let ((x (%rand '(4 5) 3)))
    (is (eq x (call (make-dropout :rate 0f0) x (nabla:prng-key 1) nil)))
    (is (every #'zerop (%elements (call (make-dropout :rate 1f0) x (nabla:prng-key 1) nil)))))
  (let* ((module (make-dropout :rate 0.5f0))
         (key (nabla:prng-key 9))
         (x (%rand '(4 5) 4))
         (mask (call module (%ones '(4 5)) key nil))
         (loss (nabla:with-tracing (xx k) (nabla:reduce-sum (call module xx k nil))))
         (grad (funcall (nabla:grad loss) x key)))
    (is (equalp mask grad)))
  (signals error (init-dropout (nabla:prng-key 0) 1.5))
  (signals error (init-dropout (nabla:prng-key 0) -0.1))
  (signals error (init-dropout (nabla:prng-key 0) "0.1")))

;;; --- sequential / activation ---

(def-layer-property layers/sequential-is-composition
  "sequential の出力は、各層を順に適用したものと等しい（層の数 0 から 3。0 層は入力そのもの）。
init-sequential は各層に別のキーを渡す。"
  (seed depth) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 3))
  (let* ((width 4)
         (inits (loop repeat depth
                      collect (lambda (key) (init-dense key width width))
                      collect (lambda (key) (init-activation key :gelu))))
         (net (init-sequential (nabla:prng-key seed) inits))
         (x (%rand (list 2 width) seed))
         (expected (reduce (lambda (h layer) (call layer h)) (sequential-layers net) :initial-value x)))
    (and (= (length (sequential-layers net)) (* 2 depth))
         (equalp expected (call net x))
         (or (zerop depth) (eq x (call (init-sequential (nabla:prng-key seed) nil) x)))
         (let ((kernels (loop for layer in (sequential-layers net)
                              when (typep layer 'dense) collect (dense-kernel layer))))
           (= (length kernels) (length (remove-duplicates kernels :test #'equalp)))))))

(test layers/sequential-pytree-and-activation-slots
  "sequential の葉は層の葉を順に並べたもの（activation は葉を持たない）。:static の fn は treedef に入る。"
  (let ((net (init-sequential (nabla:prng-key 1)
                              (list (lambda (k) (init-dense k 3 4))
                                    (lambda (k) (init-activation k :relu))
                                    (lambda (k) (init-dense k 4 2))))))
    (is (= 4 (length (nabla:tree-leaves net))))
    (is (equal '(("layers" . :param)) (%slot-names 'sequential)))
    (is (equal '(("fn" . :static)) (%slot-names 'activation)))
    (is (not (nabla:treedef= (nabla:tree-structure (make-activation :fn :relu))
                             (nabla:tree-structure (make-activation :fn :gelu)))))
    (signals error (call (make-activation :fn :nonexistent) (%rand '(2) 1)))))

(def-layer-property layers/activations-satisfy-their-definitions
  "relu は max(x, 0)、softmax は正で和が 1 かつ定数の加算に不変、log-softmax は log(softmax)、
gelu(0) = 0 で大きい x では x に近く、非常に負の x では 0 に近い。軸は負の値でも指定できる。"
  (seed shift) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo -50 :hi 50))
  (let* ((x (%rand '(3 5) seed))
         (shifted (let ((y (make-array '(3 5) :element-type 'single-float)))
                    (dotimes (i 15 y) (setf (row-major-aref y i) (+ (row-major-aref x i) shift))))))
    (and (every #'= (%elements (relu x)) (mapcar (lambda (v) (max v 0.0)) (%elements x)))
         (let ((s (softmax x)))
           (and (loop for r below 3
                      always (< (abs (- 1 (loop for c below 5 sum (aref s r c)))) 1d-5))
                (every #'plusp (%elements s))
                (allclose (softmax shifted) s :dtype :f32 :rtol 1d-4 :atol 1d-5)
                (allclose (softmax x :axis 1) s :dtype :f32)
                (allclose (log-softmax x)
                          (let ((logs (make-array '(3 5) :element-type 'single-float)))
                            (dotimes (i 15 logs) (setf (row-major-aref logs i) (log (row-major-aref s i)))))
                          :dtype :f32 :rtol 1d-4 :atol 1d-5)))
         (let ((g (gelu (make-array 3 :element-type 'single-float :initial-contents '(0.0 10.0 -10.0)))))
           (and (zerop (aref g 0)) (< (abs (- (aref g 1) 10)) 1d-4) (< (abs (aref g 2)) 1d-4))))))

(test layers/activation-axis-errors
  "softmax / log-softmax の axis が範囲外（rank 以上、-rank より小さい）なら error。"
  (let ((x (%rand '(2 3) 1)))
    (signals error (softmax x :axis 2))
    (signals error (log-softmax x :axis -3))
    (is (allclose (softmax x :axis -2) (softmax x :axis 0) :dtype :f32))))

;;; --- 入力の検査・f64 ---

(test layers/misuse-signals-clear-errors
  "layer-norm は最後の軸の長さが scale と違う入力・rank 0 の入力、embedding は整数でない ids、
dropout は deterministic が偽でキーが NIL、layer-norm は epsilon が正の実数でないとき error。"
  (let ((norm (init-layer-norm (nabla:prng-key 0) 4)))
    (signals error (call norm (%rand '(2 5) 1)))
    (signals error (call norm (make-array '() :element-type 'single-float :initial-element 1.0))))
  (signals error (call (init-embedding (nabla:prng-key 0) 5 3) (%rand '(2) 1)))
  (signals error (call (make-dropout :rate 0.5f0) (%rand '(2 2) 1) nil nil))
  (signals error (init-layer-norm (nabla:prng-key 0) 4 :epsilon 0))
  (signals error (init-layer-norm (nabla:prng-key 0) 4 :epsilon -1f-6))
  (signals error (init-layer-norm (nabla:prng-key 0) 4 :dtype :bf16)))

(test layers/dropout-rate-one-gives-zeros-even-for-inf-and-nan
  "rate = 1 の dropout は、入力に inf や NaN があっても全部 0（Flax の zeros_like と同じ）で、勾配も 0（NaN にならない）。"
  (let ((x (make-array 3 :element-type 'single-float
                         :initial-contents (list sb-ext:single-float-positive-infinity 1.0
                                                 (nabla::%make-single-float #x7FC00000))))
        (module (make-dropout :rate 1f0))
        (key (nabla:prng-key 1)))
    (is (every #'zerop (%elements (call module x key nil))))
    (is (every #'zerop (%elements (funcall (nabla:grad (nabla:with-tracing (xx k) (nabla:reduce-sum (call module xx k nil))))
                                           (%rand '(3) 2) key))))))

(test layers/f64-uses-double-precision-constants
  "dtype :f64 の layer-norm の既定 epsilon は 1d-6（1f-6 を f64 に足さない）、gelu は f64 の精度で
tanh 近似の定義式（係数 0.044715d0）と一致する（相対誤差 1e-14 以内）。"
  (is (eql 1d-6 (layer-norm-epsilon (init-layer-norm (nabla:prng-key 1) 3 :dtype :f64))))
  (let* ((x (make-array 5 :element-type 'double-float :initial-contents '(-2d0 -0.5d0 0.3d0 1d0 3.7d0)))
         (y (gelu x)))
    (is (eq 'double-float (array-element-type y)))
    (dotimes (i 5)
      (let* ((v (aref x i))
             (expected (* v 0.5d0 (+ 1 (tanh (* (sqrt (/ 2 pi)) (+ v (* 0.044715d0 v v v))))))))
        (is (<= (abs (- (aref y i) expected)) (* 1d-14 (max 1d0 (abs expected)))))))))

(test layers/layer-norm-clamps-a-negative-fast-variance
  "速い分散 E[x^2] - E[x]^2 は定数の行でも f32 の丸めで負になることがある（たとえば定数 6.29 の 8 要素は
約 -7.6e-6）。Flax と同じく 0 で切るので、epsilon が極小でも NaN にならず、出力は有限（定数の行は bias のまま）。"
  (let ((norm (make-layer-norm :scale (%ones '(8)) :bias (make-array 8 :element-type 'single-float
                                                                       :initial-element 0.5)
                               :epsilon 1f-30)))
    (dolist (c '(1.85 3.33 4.81 6.29 6.66))
      (let ((y (call norm (make-array '(1 8) :element-type 'single-float
                                             :initial-element (coerce c 'single-float)))))
        (is (every (lambda (v) (and (not (sb-ext:float-nan-p v)) (< (abs v) 1e30))) (%elements y))
            "定数 ~F の行の出力が有限でない: ~S" c (%elements y))))))

(test layers/layer-norm-feature-mismatch-error-names-the-layer
  "入力の最後の軸の長さが scale と違うときのエラーは、低レベルの形状エラーでなく layer-norm の検査が出す
（メッセージに layer-norm と期待する長さが入っている）。"
  (let ((norm (init-layer-norm (nabla:prng-key 0) 4)))
    (handler-case (progn (call norm (%rand '(2 5) 1)) (fail "エラーにならなかった"))
      (error (condition)
        (let ((message (princ-to-string condition)))
          (is (search "layer-norm" message))
          (is (search "4" message)))))))

(defun %error-message (thunk)
  "THUNK を呼んで出た error のメッセージ（出なければ NIL）。"
  (handler-case (progn (funcall thunk) nil)
    (error (condition) (princ-to-string condition))))

(test layers/embedding-accepts-every-integer-dtype-and-rejects-floats-by-name
  "embedding の ids は :i32 / :u32 / :u64 のどれでも同じ出力になり、浮動小数点の ids は embedding の検査の
エラー（メッセージに embedding と dtype が入る）。"
  (let* ((module (init-embedding (nabla:prng-key 5) 6 3))
         (values '((0 1) (5 2)))
         (expected (call module (make-array '(2 2) :element-type '(signed-byte 32) :initial-contents values))))
    (dolist (type '((unsigned-byte 32) (unsigned-byte 64)))
      (is (equalp expected (call module (make-array '(2 2) :element-type type :initial-contents values)))
          "~S の ids の出力が :i32 と違う" type))
    (let ((message (%error-message (lambda () (call module (%rand '(2) 1))))))
      (is (and message (search "embedding" message) (search ":F32" message))
          "メッセージ ~S" message))))

(test layers/dropout-without-key-names-the-layer
  "deterministic が偽でキーが NIL のとき、エラーのメッセージは dropout の検査のもの（dropout とキーに言及する）。"
  (let ((message (%error-message (lambda () (call (make-dropout :rate 0.5f0) (%rand '(2 2) 1) nil nil)))))
    (is (and message (search "dropout" message) (search "PRNG" message)) "メッセージ ~S" message)))

(test layers/dense-feature-mismatch-error-names-the-layer
  "入力の最後の軸の長さが kernel の in と違うとき、eager でも grad のトレースの中でも、エラーは dense の検査が出す
（メッセージに dense と期待する長さが入っている）。"
  (let* ((dense (make-dense :use-bias t :kernel (%rand '(4 3) 1) :bias (%rand '(3) 2)))
         (bad (%rand '(2 5) 3))
         (loss (nabla:with-tracing (m x) (nabla:reduce-sum (call m x)))))
    (dolist (thunk (list (lambda () (call dense bad))
                         (lambda () (funcall (nabla:grad loss :argnums '(1)) dense bad))))
      (let ((message (%error-message thunk)))
        (is (and message (search "dense" message) (search "4" message)) "メッセージ ~S" message)))))

(test layers/dropout-mask-does-not-depend-on-the-input-dtype
  "同じキーなら、f32 の入力と f64 の入力で同じ位置が落ちる（乱数は常に f32 で引く。Flax と同じ）。"
  (let* ((module (make-dropout :rate 0.4f0))
         (key (nabla:prng-key 123))
         (out32 (call module (make-array '(10 10) :element-type 'single-float :initial-element 1.0)
                      key nil))
         (out64 (call module (make-array '(10 10) :element-type 'double-float :initial-element 1d0)
                      key nil)))
    (is (equal (mapcar #'zerop (%elements out32)) (mapcar #'zerop (%elements out64))))
    (is (some #'zerop (%elements out32)))
    (is (notevery #'zerop (%elements out32)))))

;;; --- use_bias（issue #241）---

(defun %dtype-keyword (array)
  (if (eq 'double-float (array-element-type array)) :f64 :f32))

(defun %zero-bias-twin (model)
  "bias が NIL の MODEL（dense / dense-general）と同じ kernel で、bias が 0 の（use-bias が真の）モデル。"
  (etypecase model
    (dense (make-dense :kernel (dense-kernel model)
                       :bias (nabla.nn::%filled (last (array-dimensions (dense-kernel model))) (%dtype-keyword (dense-kernel model)) 0)
                       :use-bias t))
    (dense-general (make-dense-general :kernel (dense-general-kernel model)
                                       :bias (nabla.nn::%filled (nthcdr (dense-general-contract-axes model)
                                                              (array-dimensions (dense-general-kernel model)))
                                                      (%dtype-keyword (dense-general-kernel model)) 0)
                                       :contract-axes (dense-general-contract-axes model) :use-bias t))))

(def-layer-property layers/dense-without-bias-is-dense-with-zero-bias
  "use-bias が偽の dense は、同じ kernel で bias が 0 の dense と同じ出力と kernel の勾配を返す。bias は NIL
（空の部分木）で葉は kernel の 1 つだけ、チェックポイントのパス名は kernel のまま。bias のある dense との差は
行ごとに bias そのもの（入力は rank 1 から 3）。"
  (seed rank in out) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 3)
                            (uniform-integer :lo 1 :hi 5) (uniform-integer :lo 1 :hi 5))
  (let* ((plain (init-dense (nabla:prng-key seed) in out :use-bias nil))
         (twin (%zero-bias-twin plain))
         (biased (make-dense :kernel (dense-kernel plain) :bias (%rand (list out) (+ seed 1)) :use-bias t))
         (x (%rand (append (make-list (1- rank) :initial-element 2) (list in)) seed))
         (ct (%rand (append (make-list (1- rank) :initial-element 2) (list out)) (+ seed 2)))
         (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (call m x) ct))))
         (grad (funcall (nabla:grad loss) plain x ct))
         (twin-grad (funcall (nabla:grad loss) twin x ct)))
    (and (null (dense-bias plain)) (not (dense-use-bias plain))
         (equalp (call twin x) (call plain x))
         (equal '("kernel") (nabla.nn:tree-paths plain))
         (= 1 (length (nabla:tree-leaves grad)))
         (null (dense-bias grad))
         (equalp (dense-kernel twin-grad) (dense-kernel grad))
         (let ((with (call biased x)) (without (call plain x)))
           (loop for i below (array-total-size with)
                 always (< (abs (- (row-major-aref with i) (row-major-aref without i)
                                   (aref (dense-bias biased) (mod i out))))
                           1d-5))))))

(test layers/dense-use-bias-init-slots-and-misuse
  "init-dense の use-bias の既定は真（bias は 0）。偽なら bias は NIL。use-bias は :static でスロットの最後。
真偽値でない use-bias、use-bias が真なのに bias が NIL、偽なのに bias がある dense は dense の名前入りの error。"
  (let ((with (init-dense (nabla:prng-key 1) 3 2))
        (without (init-dense (nabla:prng-key 1) 3 2 :use-bias nil)))
    (is (eq t (dense-use-bias with)))
    (is (equal '(2) (array-dimensions (dense-bias with))))
    (is (null (dense-bias without)))
    (is (null (dense-use-bias without)))
    (is (equalp (dense-kernel with) (dense-kernel without)) "use-bias は kernel の初期値に影響しない")
    (is (equal '("kernel" "bias") (nabla.nn:tree-paths with)))
    (is (equal '("kernel") (nabla.nn:tree-paths without)))
    (is (equal '(("kernel" . :param) ("bias" . :param) ("use-bias" . :static)) (%slot-names 'dense)))
    (is (not (nabla:treedef= (nabla:tree-structure with) (nabla:tree-structure without)))))
  (flet ((names (thunk) (search "dense" (or (%error-message thunk) ""))))
    (is (names (lambda () (init-dense (nabla:prng-key 1) 3 2 :use-bias 1))) "use-bias が真偽値でない")
    (is (names (lambda () (call (make-dense :kernel (%rand '(3 2) 1) :bias nil :use-bias t) (%rand '(3) 2))))
        "use-bias が真なのに bias が無い")
    (is (names (lambda () (call (make-dense :kernel (%rand '(3 2) 1) :bias (%rand '(2) 1) :use-bias nil)
                                (%rand '(3) 2))))
        "use-bias が偽なのに bias がある")))
