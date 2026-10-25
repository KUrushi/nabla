;;;; batch-norm と「状態を更新して返す call」の規約（issue #260、small）。
;;;;
;;;; 守らせる性質:
;;;; - Flax との数値一致（tests/fixtures/layers/flax-batch-norm.lisp、生成は generate-batch-norm.py）:
;;;;   学習モードを数ステップ続けたときの出力・更新後の移動平均・勾配と、推論モードの出力・勾配。
;;;;   学習ステップは (value-and-grad loss :has-aux t) で更新後のモジュールを aux として受け取る。
;;;; - 別実装（double のループ、2 パスの分散）との一致: 出力と更新後の mean / var（両モード、任意の axis）。
;;;; - 規約: call は常に (values 出力 新しいモジュール)。状態の treedef は更新の前後で treedef=、推論では
;;;;   状態の葉は元のまま（EQ）。:param の葉は学習モードでも元のまま（EQ）。
;;;; - :state に勾配が流れない: 学習モードでは mean / var の勾配はちょうど 0（順伝播に使わない）、推論モードでも
;;;;   map-state で stop-gradient すれば 0。
;;;; - 学習ステップ全体（has-aux + adamw + apply-updates）で、新しいモデルの :state は call の返した移動平均そのもの
;;;;   （weight decay がかからない）。
;;;; - vmap: モジュールをバッチせずデータだけをバッチすると、サンプルごとの call を並べたものと一致する。
;;;; - init: 形・dtype・初期値（scale 1、bias 0、mean 0、var 1）・スロットの種類（チェックポイントのパス名）。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

;;; --- Flax フィクスチャ ---

(defvar *batch-norm-fixture* nil)

(defun %bn-fixture ()
  (or *batch-norm-fixture*
      (setf *batch-norm-fixture*
            (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/layers/flax-batch-norm.lisp"))
              (let ((*package* (find-package '#:nabla.nn.tests))
                    (*read-default-float-format* 'double-float))
                (read stream))))))

(defun %bn-case (name)
  (or (find name (getf (%bn-fixture) :cases) :key (lambda (c) (getf c :name)) :test #'string=)
      (error "フィクスチャにケース ~S が無い" name)))

(defun %bn-array (case name)
  "ケース CASE の配列 NAME（f32 のビットパターン）。無ければ error（比べる相手が無いまま通らないように）。"
  (let ((entry (or (find name (getf case :arrays) :key #'first :test #'string=)
                   (error "フィクスチャのケース ~S に配列 ~S が無い" (getf case :name) name))))
    (progn
      (destructuring-bind (name shape values) entry
        (declare (ignore name))
        (let ((array (make-array shape :element-type 'single-float)))
          (loop for i from 0 for value in values
                do (setf (row-major-aref array i) (nabla::%make-single-float value)))
          array)))))

(defun %bn-step-name (base step)
  (format nil "~A.~D" base step))

(defun %bn-from-case (case)
  "フィクスチャのケースの scale / bias / 移動平均の初期値とハイパーパラメータで batch-norm を作る。"
  (let ((use-scale (getf case :use-scale)) (use-bias (getf case :use-bias)))
    (make-batch-norm :scale (and use-scale (%bn-array case "scale"))
                     :bias (and use-bias (%bn-array case "bias"))
                     :mean (%bn-array case "mean.init") :var (%bn-array case "var.init")
                     :momentum (getf case :momentum) :epsilon (coerce (getf case :epsilon) 'single-float)
                     :axis (getf case :axis) :use-scale use-scale :use-bias use-bias)))

(defparameter *bn-tolerance* '(:rtol 1d-4 :atol 1d-5)
  "Flax と同じ速い分散 E[x^2] - E[x]^2 は、平均 3・標準偏差 2 の入力で f32 の桁を 1 桁ほど失い、XLA と nabla の
丸めの違いがそのまま出る（layer-norm と同じ理由。layers-test.lisp の %case-tolerance）。")

(defun %bn-close (actual expected)
  (apply #'allclose actual expected :dtype :f32 *bn-tolerance*))

(defun %bn-loss (use-running-average)
  "loss(m, x, ct) = sum(call(m, x) * ct) を返し、aux に更新後のモジュールを返す関数。推論モードでは
:state を map-state で stop-gradient してから呼ぶ。"
  (if use-running-average
      ;; 推論モードは移動平均を順伝播に使うので、損失の中で map-state して :state を勾配から外す
      (nabla:with-tracing (m x ct)
        (multiple-value-bind (y new) (call (map-state #'nabla:stop-gradient m) x t)
          (values (nabla:reduce-sum (* y ct)) new)))
      (nabla:with-tracing (m x ct)
        (multiple-value-bind (y new) (call m x nil) (values (nabla:reduce-sum (* y ct)) new)))))

(defun %bn-param-grads (grad-model)
  "勾配のモジュールの scale / bias（NIL は除く）。"
  (remove nil (list (batch-norm-scale grad-model) (batch-norm-bias grad-model))))

(defun %bn-expected-param-grads (case prefix suffix)
  (remove nil (list (and (getf case :use-scale) (%bn-array case (format nil "~Agrad.scale~A" prefix suffix)))
                    (and (getf case :use-bias) (%bn-array case (format nil "~Agrad.bias~A" prefix suffix))))))

(defun %all-zero-p (array)
  (loop for i below (array-total-size array) always (zerop (row-major-aref array i))))

(defparameter *bn-case-names*
  '("batch-norm" "batch-norm-nhwc" "batch-norm-axis1" "batch-norm-two-feature-axes"
    "batch-norm-no-scale-no-bias" "batch-norm-tiny-variance"))

(defun %check-bn-flax-case (name &key (value-and-grad #'nabla:value-and-grad) (call-function #'call))
  "ケース NAME を Flax と比べる。学習モードを :steps 回続け（モデルは aux の更新後のモジュールで置き換える）、
最後の移動平均で推論モードを比べる。VALUE-AND-GRAD / CALL-FUNCTION は jit 版に差し替える（medium）。"
  (let* ((case (%bn-case name))
         (model (%bn-from-case case))
         (train (funcall value-and-grad (%bn-loss nil) :argnums '(0 1) :has-aux t))
         (evaluate (funcall value-and-grad (%bn-loss t) :argnums '(0 1) :has-aux t)))
    (dotimes (step (getf case :steps))
      (let ((x (%bn-array case (%bn-step-name "x" step)))
            (ct (%bn-array case (%bn-step-name "ct" step))))
        (multiple-value-bind (y new) (funcall call-function model x nil)
          (is (%bn-close y (%bn-array case (%bn-step-name "out" step))) "~A: ステップ ~D の出力" name step)
          (multiple-value-bind (loss grads aux) (funcall train model x ct)
            (declare (ignore loss))
            (is (%bn-close (batch-norm-mean aux) (%bn-array case (%bn-step-name "mean" step)))
                "~A: ステップ ~D の移動平均 mean" name step)
            (is (%bn-close (batch-norm-var aux) (%bn-array case (%bn-step-name "var" step)))
                "~A: ステップ ~D の移動平均 var" name step)
            (is (%bn-close (batch-norm-mean new) (batch-norm-mean aux)) "~A: call と aux の mean" name)
            (is (%bn-close (batch-norm-var new) (batch-norm-var aux)) "~A: call と aux の var" name)
            (let ((actual (%bn-param-grads (first grads)))
                  (expected (%bn-expected-param-grads case "" (format nil ".~D" step))))
              (is (and (= (length actual) (length expected)) (every #'%bn-close actual expected))
                  "~A: ステップ ~D の scale / bias の勾配" name step))
            (is (%bn-close (second grads) (%bn-array case (%bn-step-name "grad.x" step)))
                "~A: ステップ ~D の x の勾配" name step)
            (is (and (%all-zero-p (batch-norm-mean (first grads))) (%all-zero-p (batch-norm-var (first grads))))
                "~A: 学習モードの mean / var の勾配はちょうど 0" name)
            (setf model aux)))))
    (let ((x (%bn-array case "eval.x"))
          (ct (%bn-array case "eval.ct")))
      (multiple-value-bind (y same) (funcall call-function model x t)
        (is (%bn-close y (%bn-array case "eval.out")) "~A: 推論モードの出力" name)
        (is (%bn-close (batch-norm-var same) (batch-norm-var model)) "~A: 推論モードは状態を変えない" name))
      (multiple-value-bind (loss grads) (funcall evaluate model x ct)
        (declare (ignore loss))
        (let ((actual (%bn-param-grads (first grads)))
              (expected (%bn-expected-param-grads case "eval." "")))
          (is (and (= (length actual) (length expected)) (every #'%bn-close actual expected))
              "~A: 推論モードの scale / bias の勾配" name))
        (is (%bn-close (second grads) (%bn-array case "eval.grad.x")) "~A: 推論モードの x の勾配" name)
        (is (and (%all-zero-p (batch-norm-mean (first grads))) (%all-zero-p (batch-norm-var (first grads))))
            "~A: map-state で stop-gradient した mean / var の勾配はちょうど 0" name)))))

(test batch-norm/matches-flax-in-training-and-inference
  "フィクスチャの全ケースで、学習モードを数ステップ続けた出力・移動平均・勾配と、推論モードの出力・勾配が
Flax と一致する（eager）。"
  (dolist (name *bn-case-names*)
    (%check-bn-flax-case name)))

;;; --- 別実装との一致（PBT） ---

(defun %bn-feature-index (index dimensions feature-axes)
  "行優先の添字 INDEX の要素の、特徴の軸（FEATURE-AXES）だけで数えた行優先の添字。"
  (let ((multi (let ((rest index))
                 (reverse (loop for d in (reverse dimensions)
                                collect (prog1 (mod rest d) (setf rest (floor rest d))))))))
    (let ((result 0))
      (dolist (axis feature-axes result)
        (setf result (+ (* result (nth axis dimensions)) (nth axis multi)))))))

(defconstant +f32-unit-roundoff+ (scale-float 1d0 -24))

(defun %bn-reference (x feature-axes scale bias mean var epsilon momentum use-running-average)
  "double で計算した (values 出力 出力の許容誤差 新しい mean 新しい var 新しい var の許容誤差)。分散は 2 パス
（平均を引いてから二乗の平均）。

許容誤差（要素ごと）は f32 の既定 1e-5 + 1e-4 |期待値| に、Flax と同じ速い分散 E[x^2] - E[x]^2 の打ち消しの
誤差の上限を足したもの: f32 で計算した速い分散の誤差は 8 u E[x^2] 程度（u は f32 の丸めの単位）で、正規化した
値 z = (x - mean) / sqrt(var + eps) はその誤差で |z| dvar / (2 (var + eps)) だけずれる。学習モードだけに足す
（推論モードは統計を計算しない）。平均がずれた入力・標本の少ない軸でも、式が Flax と同じである限り通る。"
  (let* ((dimensions (array-dimensions x))
         (n-features (reduce #'* (mapcar (lambda (a) (nth a dimensions)) feature-axes)))
         (count (/ (array-total-size x) n-features))
         (sums (make-array n-features :initial-element 0d0))
         (squares (make-array n-features :initial-element 0d0))
         (raw-squares (make-array n-features :initial-element 0d0))
         (y (make-array dimensions :element-type 'single-float))
         (y-tolerance (make-array dimensions :element-type 'double-float))
         (feature (lambda (i) (%bn-feature-index i dimensions feature-axes))))
    (dotimes (i (array-total-size x))
      (let ((f (funcall feature i)) (v (row-major-aref x i)))
        (incf (aref sums f) v)
        (incf (aref raw-squares f) (* v v))))
    (let ((batch-mean (map 'vector (lambda (s) (/ s count)) sums))
          (var-error (map 'vector (lambda (s) (* 8 +f32-unit-roundoff+ (/ s count))) raw-squares)))
      (dotimes (i (array-total-size x))
        (let ((f (funcall feature i)))
          (incf (aref squares f) (expt (- (row-major-aref x i) (aref batch-mean f)) 2))))
      (let* ((batch-var (map 'vector (lambda (s) (/ s count)) squares))
             (flat (lambda (a f) (if a (coerce (row-major-aref a f) 'double-float) nil)))
             (use-mean (lambda (f) (if use-running-average (funcall flat mean f) (aref batch-mean f))))
             (use-var (lambda (f) (if use-running-average (funcall flat var f) (aref batch-var f)))))
        (dotimes (i (array-total-size x))
          (let* ((f (funcall feature i))
                 (z (/ (- (row-major-aref x i) (funcall use-mean f)) (sqrt (+ (funcall use-var f) epsilon))))
                 (s (or (funcall flat scale f) 1d0))
                 (expected (+ (* z s) (or (funcall flat bias f) 0d0))))
            (setf (row-major-aref y i) (coerce expected 'single-float)
                  (row-major-aref y-tolerance i)
                  (+ 1d-5 (* 1d-4 (abs expected))
                     (if use-running-average
                         0d0
                         (/ (* (abs (* z s)) (aref var-error f)) (* 2 (+ (aref batch-var f) epsilon))))))))
        (flet ((running (old batch)
                 (let ((new (make-array (array-dimensions old) :element-type 'single-float)))
                   (dotimes (f n-features new)
                     (setf (row-major-aref new f)
                           (coerce (if use-running-average
                                       (row-major-aref old f)
                                       (+ (* momentum (row-major-aref old f)) (* (- 1 momentum) (aref batch f))))
                                   'single-float))))))
          (let* ((new-var (running var batch-var))
                 (var-tolerance (make-array (array-dimensions new-var) :element-type 'double-float)))
            (dotimes (f n-features)
              (setf (row-major-aref var-tolerance f)
                    (+ 1d-5 (* 1d-4 (abs (row-major-aref new-var f)))
                       (if use-running-average 0d0 (* (- 1 momentum) (aref var-error f))))))
            (values y y-tolerance (running mean batch-mean) new-var var-tolerance)))))))

(defun %within-p (actual expected tolerance)
  "ACTUAL と EXPECTED の各要素の差が TOLERANCE の同じ位置の値以下（形も同じ）。外れたら最初の位置を表示する。"
  (and (equal (array-dimensions actual) (array-dimensions expected))
       (dotimes (i (array-total-size actual) t)
         (let ((diff (abs (- (row-major-aref actual i) (row-major-aref expected i)))))
           (unless (<= diff (row-major-aref tolerance i))
             (format t "~&%within-p: 添字 ~D で差 ~G > 許容 ~G（実際 ~G、期待 ~G）~%"
                     i diff (row-major-aref tolerance i) (row-major-aref actual i) (row-major-aref expected i))
             (return nil))))))

(defun %bn-random-model (seed feature-shape use-scale use-bias momentum axis)
  (make-batch-norm :scale (and use-scale (%rand feature-shape seed))
                   :bias (and use-bias (%rand feature-shape (+ seed 1)))
                   :mean (%rand feature-shape (+ seed 2))
                   :var (let ((v (%rand feature-shape (+ seed 3))))
                          (dotimes (i (array-total-size v) v)
                            (setf (row-major-aref v i) (+ 0.5 (abs (row-major-aref v i))))))
                   :momentum momentum :epsilon 1f-5 :axis axis :use-scale use-scale :use-bias use-bias))

(defun %bn-shifted (x shift)
  (let ((y (make-array (array-dimensions x) :element-type 'single-float)))
    (dotimes (i (array-total-size x) y)
      (setf (row-major-aref y i) (+ (* 2 (row-major-aref x i)) shift)))))

(def-layer-property batch-norm/matches-reference-in-both-modes
  "rank 2〜4 の入力・任意の特徴の軸（1つか2つ、負の値も）・use-scale / use-bias の組で、学習モードと推論モードの
出力と更新後の mean / var が double の別実装（2 パスの分散）と一致する（許容誤差は %bn-reference。平均がずれた入力で
Flax と同じ速い分散の打ち消しを含めて見る）。call は常に2値を返し、状態の treedef は
変わらず、推論モードでは状態の葉は元のまま（EQ）、:param の葉はどちらのモードでも元のまま（EQ）。"
  (seed rank axis-pick two-axes use-scale use-bias momentum-pick)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 4) (uniform-integer :lo 0 :hi 3)
         (uniform-integer :lo 0 :hi 1) (uniform-integer :lo 0 :hi 1) (uniform-integer :lo 0 :hi 1)
         (uniform-integer :lo 0 :hi 2))
  (let* ((dimensions (loop for i below rank collect (+ 2 (mod (+ seed (* 7 i)) 3))))
         (first-axis (- (mod axis-pick rank) (if (evenp seed) rank 0)))
         (axes (if (and (= two-axes 1) (> rank 2))
                   (list first-axis (if (= (mod first-axis rank) (1- rank)) 0 -1))
                   first-axis))
         (feature-axes (sort (remove-duplicates (mapcar (lambda (a) (mod a rank)) (if (listp axes) axes (list axes))))
                             #'<))
         (feature-shape (mapcar (lambda (a) (nth a dimensions)) feature-axes))
         (momentum (nth momentum-pick '(0.99d0 0.9d0 0d0)))
         (model (%bn-random-model seed feature-shape (= use-scale 1) (= use-bias 1) momentum axes))
         (x (%bn-shifted (%rand dimensions (+ seed 4)) 3)))
    (every (lambda (mode)
                  (multiple-value-bind (ey ey-tolerance emean evar evar-tolerance)
                      (%bn-reference x feature-axes (batch-norm-scale model) (batch-norm-bias model)
                                     (batch-norm-mean model) (batch-norm-var model) 1d-5 momentum mode)
                    (let ((results (multiple-value-list (call model x mode))))
                      (destructuring-bind (y new) results
                        (and (= 2 (length results))
                             (%within-p y ey ey-tolerance)
                             (%bn-close (batch-norm-mean new) emean)
                             (%within-p (batch-norm-var new) evar evar-tolerance)
                             (nabla:treedef= (nabla:tree-structure new) (nabla:tree-structure model))
                             (eq (batch-norm-scale new) (batch-norm-scale model))
                             (eq (batch-norm-bias new) (batch-norm-bias model))
                             (or (not mode)
                                 (and (eq (batch-norm-mean new) (batch-norm-mean model))
                                      (eq (batch-norm-var new) (batch-norm-var model)))))))))
                '(nil t))))

;;; --- 学習ステップ全体 ---

(test batch-norm/train-step-keeps-the-running-average-out-of-the-optimizer
  "has-aux の value-and-grad + adamw + (apply-updates new-model updates) の学習ステップで、新しいモデルの mean / var は
call が返した移動平均そのもの（EQ。weight decay がかからない）で、scale / bias は更新される。"
  (let* ((model (init-batch-norm (nabla:prng-key 0) 4))
         (optimizer (adamw 0.1d0 :weight-decay 0.5d0))
         (state (optimizer-init optimizer model))
         (x (%bn-shifted (%rand '(6 4) 3) 1))
         (ct (%rand '(6 4) 4))
         (vg (nabla:value-and-grad (%bn-loss nil) :has-aux t)))
    (multiple-value-bind (loss grads new-model) (funcall vg model x ct)
      (declare (ignore loss))
      (multiple-value-bind (updates new-state) (optimizer-update optimizer grads state model)
        (declare (ignore new-state))
        (let ((next (apply-updates new-model updates)))
          (is (eq (batch-norm-mean next) (batch-norm-mean new-model)))
          (is (eq (batch-norm-var next) (batch-norm-var new-model)))
          (is (not (equalp (batch-norm-mean next) (batch-norm-mean model))))
          (is (not (equalp (batch-norm-scale next) (batch-norm-scale model)))))))))

;;; --- vmap ---

(test batch-norm/vmap-over-data-matches-per-example-calls
  "モジュールをバッチせず（in-axes nil）データだけをバッチした vmap は、サンプルごとの call を並べたものと一致する
（出力も、サンプルごとに更新した移動平均も）。"
  (let* ((model (%bn-random-model 5 '(3) t t 0.9d0 -1))
         (xs (%bn-shifted (%rand '(4 5 3) 6) 2))
         (f (nabla:vmap (nabla:with-tracing (m x) (call m x nil)) :in-axes '(nil 0))))
    (multiple-value-bind (ys news) (funcall f model xs)
      (dotimes (b 4)
        (let ((x (make-array '(5 3) :element-type 'single-float)))
          (dotimes (i 15) (setf (row-major-aref x i) (row-major-aref xs (+ (* b 15) i))))
          (multiple-value-bind (y new) (call model x nil)
            (is (%bn-close (%bn-row ys b) y))
            (is (%bn-close (%bn-row (batch-norm-mean news) b) (batch-norm-mean new)))
            (is (%bn-close (%bn-row (batch-norm-var news) b) (batch-norm-var new)))))))))

(defun %bn-row (array b)
  "ARRAY の先頭の軸の B 番目。"
  (let* ((shape (rest (array-dimensions array)))
         (size (reduce #'* shape))
         (row (make-array shape :element-type (array-element-type array))))
    (dotimes (i size row)
      (setf (row-major-aref row i) (row-major-aref array (+ (* b size) i))))))

;;; --- init ---

(test batch-norm/init-shapes-values-and-slot-kinds
  "init の scale は 1、bias は 0、mean は 0、var は 1（Flax と同じ）で、形は features。スロットは scale / bias が
:param、mean / var が :state（チェックポイントのパス名は scale bias mean var）。use-scale / use-bias が偽なら
そのスロットは NIL。既定は momentum 0.99、epsilon 1e-5、axis -1。dtype :f64 なら全部 f64。"
  (let ((m (init-batch-norm (nabla:prng-key 0) 3)))
    (is (equalp (make-array 3 :element-type 'single-float :initial-element 1f0) (batch-norm-scale m)))
    (is (equalp (make-array 3 :element-type 'single-float :initial-element 0f0) (batch-norm-bias m)))
    (is (equalp (make-array 3 :element-type 'single-float :initial-element 0f0) (batch-norm-mean m)))
    (is (equalp (make-array 3 :element-type 'single-float :initial-element 1f0) (batch-norm-var m)))
    (is (equal '(0.99d0 1f-5 -1 t t)
               (list (batch-norm-momentum m) (batch-norm-epsilon m) (batch-norm-axis m)
                     (batch-norm-use-scale m) (batch-norm-use-bias m))))
    (is (equal '(("scale" . :param) ("bias" . :param) ("mean" . :state) ("var" . :state))
               (remove :static (%slot-names m) :key #'cdr)))
    (is (equal '("scale" "bias" "mean" "var") (tree-paths m))))
  (let ((m (init-batch-norm (nabla:prng-key 0) '(2 3) :axis '(1 2) :use-scale nil :use-bias nil :dtype :f64)))
    (is (null (batch-norm-scale m)))
    (is (null (batch-norm-bias m)))
    (is (equal '(2 3) (array-dimensions (batch-norm-mean m))))
    (is (eq 'double-float (array-element-type (batch-norm-var m))))
    (is (eql 1d-5 (batch-norm-epsilon m)))))

(test batch-norm/rejects-bad-arguments
  "init と call は不正な引数を error にする: features と axis の個数の不一致・正でない epsilon・[0, 1] の外の
momentum・有限でない epsilon、call の use-running-average が T / NIL でない、入力の dtype がモデルと違う、統計を取る軸が残らない、特徴の軸の大きさと scale / 状態の
形の不一致、軸が範囲外。"
  (signals error (init-batch-norm (nabla:prng-key 0) '(2 3)))
  (signals error (init-batch-norm (nabla:prng-key 0) 3 :epsilon 0))
  (signals error (init-batch-norm (nabla:prng-key 0) 3 :momentum 1.5d0))
  (signals error (init-batch-norm (nabla:prng-key 0) 3 :use-bias :yes))
  (signals error (init-batch-norm (nabla:prng-key 0) 3 :use-scale :yes))
  (signals error (init-batch-norm (nabla:prng-key 0) 0))
  (signals error (init-batch-norm (nabla:prng-key 0) 2.5))
  (signals error (init-batch-norm (nabla:prng-key 0) 3 :momentum -0.1d0))
  (signals error (init-batch-norm (nabla:prng-key 0) 3 :epsilon sb-ext:single-float-positive-infinity))
  (let ((m (init-batch-norm (nabla:prng-key 0) 3))
        (x (%rand '(4 3) 1)))
    (signals error (call m x :yes))
    ;; 入力の dtype がモデルと違う（f64 の入力に f32 のモデル）
    (handler-case (progn (call m (%rand '(4 3) 1 :f64) nil) (fail "dtype の違いが error にならなかった"))
      (error (c) (is (search "batch-norm" (princ-to-string c)))))
    (signals error (call m (%rand '(3) 1) nil))
    (signals error (call m (%rand '(4 5) 1) nil))
    (signals error (call (init-batch-norm (nabla:prng-key 0) 3 :axis 2) x nil))
    (signals error (call (init-batch-norm (nabla:prng-key 0) '(4 3) :axis '(0 1)) x nil))
    ;; 同じ軸の別の書き方（rank 2 の 1 と -1）は call で「重複」として error
    (handler-case (progn (call (init-batch-norm (nabla:prng-key 0) '(3 3) :axis '(1 -1)) x nil)
                         (fail "重複した axis が error にならなかった"))
      (error (c) (is (search "重複" (princ-to-string c)))))))

(test batch-norm/map-state-must-be-inside-the-loss
  "推論モードでは移動平均が順伝播に使われるので、損失の中で map-state #'stop-gradient しないと mean / var に勾配が
流れる。grad に渡す前に（損失の外で）かけても効かない（README の規約の根拠）。"
  (let* ((model (%bn-random-model 9 '(3) t t 0.9d0 -1))
         (x (%bn-shifted (%rand '(5 3) 10) 1))
         (ct (%rand '(5 3) 11))
         (outside (nabla:with-tracing (m x ct)
                    (multiple-value-bind (y new) (call m x t) (declare (ignore new)) (nabla:reduce-sum (* y ct)))))
         (grads (funcall (nabla:grad outside) (map-state #'nabla:stop-gradient model) x ct)))
    (is (not (%all-zero-p (batch-norm-mean grads))))
    (multiple-value-bind (loss grads-inside) (funcall (nabla:value-and-grad (%bn-loss t) :has-aux t) model x ct)
      (declare (ignore loss))
      (is (%all-zero-p (batch-norm-mean grads-inside)))
      (is (%all-zero-p (batch-norm-var grads-inside))))))
