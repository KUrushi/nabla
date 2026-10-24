;;;; オプティマイザ（Optax 相当、issue #205）のテスト。
;;;;
;;;; 守らせる性質:
;;;; - 各オプティマイザ・スケジュールの数値が Optax のフィクスチャと一致する
;;;;   （tests/fixtures/optimizer/optimizer.lisp、生成は generate.py）。更新量・更新後のパラメータ・
;;;;   最後の状態の葉（tree_leaves の順）を比べる。
;;;; - update の結果（更新量）はパラメータと同じ構造で、状態の構造は update の前後で変わらない
;;;;   （jit / scan に渡せる条件）。adam の mu / nu は params と同じ形。
;;;; - update は純粋: 入力を書き換えず、同じ入力から同じ結果を返す。
;;;; - chain は「順に適用したもの」と一致し、結合的で、空の chain は恒等。
;;;; - clip-by-global-norm の結果の大域ノルムは max-norm を超えず、ノルムが小さければ素通し。
;;;; - 閉じた形: sgd の更新は -lr g、adam の最初のステップは -lr g / (|g| + eps)。
;;;; - スケジュールの端点と単調性。count は update ごとに 1 増える。
;;;; - defoptimizer で書いた独自のオプティマイザも chain できる。

(in-package #:nabla.nn.tests)

;;; --- フィクスチャ ---

(defvar *optimizer-fixture* nil)

(defun %optimizer-fixture ()
  (or *optimizer-fixture*
      (setf *optimizer-fixture*
            (with-open-file (stream (asdf:system-relative-pathname
                                     "nabla" "tests/fixtures/optimizer/optimizer.lisp"))
              (let ((*package* (find-package '#:nabla.nn.tests)))
                (read stream))))))

(defun %opt-fx-array (shape bits)
  (let ((array (make-array shape :element-type 'single-float)))
    (loop for i from 0 for bit in bits
          do (setf (row-major-aref array i) (nabla::%make-single-float bit)))
    array))

(defun %opt-fx-tree (entries)
  "((w shape bits) (b shape bits)) → 配列のリスト。"
  (mapcar (lambda (entry) (%opt-fx-array (second entry) (third entry))) entries))

(defun %opt-fx-name (symbol)
  (string-downcase (symbol-name symbol)))

(defun %count-array (n)
  (make-array nil :element-type '(signed-byte 32) :initial-element n))

(defun %f32-close (a b)
  (nabla.tests.support:allclose a b :dtype :f32))

(defun %trees-close (a b)
  (and (= (length a) (length b)) (every #'%f32-close a b)))

(defparameter *optimizer-cases*
  (list (cons "sgd" (lambda () (sgd 0.1d0)))
        (cons "sgd-momentum" (lambda () (sgd 0.1d0 :momentum 0.9d0)))
        (cons "sgd-nesterov" (lambda () (sgd 0.1d0 :momentum 0.9d0 :nesterov t)))
        (cons "sgd-linear-schedule" (lambda () (sgd (linear-schedule 0.5d0 0.1d0 4))))
        (cons "sgd-warmup-cosine"
              (lambda () (sgd (warmup-cosine-decay-schedule 0d0 0.1d0 2 6 :end-value 0.01d0)
                              :momentum 0.9d0)))
        (cons "adam" (lambda () (adam 1d-3)))
        (cons "adam-custom" (lambda () (adam 0.01d0 :b1 0.8d0 :b2 0.9d0 :eps 1d-6 :eps-root 1d-9)))
        (cons "adam-cosine" (lambda () (adam (cosine-decay-schedule 0.01d0 5 :alpha 0.1d0))))
        (cons "adamw" (lambda () (adamw 0.01d0 :weight-decay 0.1d0)))
        (cons "clip-by-global-norm" (lambda () (clip-by-global-norm 1d0)))
        (cons "clip-adam" (lambda () (chain (clip-by-global-norm 1d0) (adam 0.01d0))))))

(defun %state-leaf-close (actual expected)
  "EXPECTED は (:f32 shape bits) か (:i32 shape ints)。"
  (destructuring-bind (kind shape data) expected
    (ecase kind
      (:f32 (%f32-close actual (%opt-fx-array shape data)))
      (:i32 (and (equal (array-dimensions actual) shape)
                 (equal (coerce (make-array (array-total-size actual)
                                            :displaced-to actual :element-type '(signed-byte 32))
                                'list)
                        data))))))

(test (optimizer/matches-optax-fixture :suite :nabla.small)
  "固定の勾配の列に対して、各オプティマイザの更新量・更新後のパラメータ・最後の状態の葉が
Optax と f32 の許容誤差で一致する（状態の葉の順は Optax の tree_leaves と同じ）。"
  (let* ((fixture (%optimizer-fixture))
         (grads (mapcar #'%opt-fx-tree (getf fixture :grads))))
    (dolist (case (getf fixture :cases))
      (let* ((name (%opt-fx-name (first case)))
             (make (cdr (assoc name *optimizer-cases* :test #'string=))))
        (is (not (null make)) "フィクスチャのケース ~A に対応するオプティマイザが無い" name)
        (when make
          (let* ((optimizer (funcall make))
                 (params (%opt-fx-tree (getf fixture :params)))
                 (state (optimizer-init optimizer params)))
            (loop for g in grads
                  for step from 0
                  for expected-updates in (getf (rest case) :updates)
                  for expected-params in (getf (rest case) :params)
                  do (multiple-value-bind (updates new-state) (optimizer-update optimizer g state params)
                       (is (%trees-close updates (%opt-fx-tree expected-updates))
                           "~A: ステップ ~D の更新量が Optax と一致しない" name step)
                       (setf params (apply-updates params updates)
                             state new-state)
                       (is (%trees-close params (%opt-fx-tree expected-params))
                           "~A: ステップ ~D の更新後のパラメータが Optax と一致しない" name step)))
            (let ((leaves (nabla:tree-leaves state))
                  (expected (getf (rest case) :state)))
              (is (= (length leaves) (length expected))
                  "~A: 状態の葉の個数が違う: ~D と ~D" name (length leaves) (length expected))
              (loop for leaf in leaves
                    for e in expected
                    for i from 0
                    do (is (%state-leaf-close leaf e) "~A: 状態の葉 ~D が Optax と一致しない" name i)))))))))

(defparameter *schedule-cases*
  (list (cons "constant" (lambda () (constant-schedule 0.3d0)))
        (cons "linear" (lambda () (linear-schedule 1d0 0.2d0 5 :transition-begin 2)))
        (cons "cosine-decay" (lambda () (cosine-decay-schedule 0.5d0 6 :alpha 0.1d0)))
        (cons "warmup-cosine-decay"
              (lambda () (warmup-cosine-decay-schedule 0d0 1d0 3 10 :end-value 0.1d0)))))

(defun %schedule-value (schedule step)
  (aref (funcall schedule (%count-array step))))

(test (optimizer/schedules-match-optax-fixture :suite :nabla.small)
  "constant / linear / cosine-decay / warmup-cosine-decay の step 0.. の値が Optax と一致する
（cosine は多項式近似なので許容誤差は f32 の rtol 1e-5）。"
  (dolist (entry (getf (%optimizer-fixture) :schedules))
    (let* ((name (%opt-fx-name (first entry)))
           (make (cdr (assoc name *schedule-cases* :test #'string=)))
           (expected (second entry)))
      (is (not (null make)) "スケジュール ~A が無い" name)
      (when make
        (let ((schedule (funcall make)))
          (loop for bit in expected
                for step from 0
                do (is (nabla.tests.support:approx= (%schedule-value schedule step)
                                                    (nabla::%make-single-float bit) :dtype :f32)
                       "~A: step ~D の値が Optax と一致しない: ~S と ~S"
                       name step (%schedule-value schedule step) (nabla::%make-single-float bit))))))))

;;; --- PBT の部品 ---

(defun %rand-array (shape seed &optional (scale 1.0))
  (let ((array (nabla.tests.support:make-random-array
                (nabla.tests.support:make-array-spec shape :f32) :seed seed)))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i) (* scale (row-major-aref array i))))))

(defun %rand-tree (seed &optional (scale 1.0))
  "形の違う3つの配列（形は固定、値は SEED から決まる）のリスト（PyTree）。"
  (list (%rand-array '(3 2) seed scale)
        (%rand-array '(3) (+ seed 1) scale)
        (%rand-array '(2 2) (+ seed 2) scale)))

(defun %leaves-equal (a b)
  (let ((la (nabla:tree-leaves a)) (lb (nabla:tree-leaves b)))
    (and (= (length la) (length lb)) (every #'equalp la lb))))

(defun %copy-leaves (tree)
  (mapcar (lambda (a)
            (let ((c (make-array (array-dimensions a) :element-type (array-element-type a))))
              (dotimes (i (array-total-size a) c)
                (setf (row-major-aref c i) (row-major-aref a i)))))
          (nabla:tree-leaves tree)))

(defun %sum-squares (leaves)
  (loop for a in leaves
        sum (loop for i below (array-total-size a)
                  sum (expt (coerce (row-major-aref a i) 'double-float) 2))))

(defparameter *generic-optimizers*
  (list (lambda () (sgd 0.1d0))
        (lambda () (sgd 0.05d0 :momentum 0.9d0 :nesterov t))
        (lambda () (adam 1d-2))
        (lambda () (adamw 1d-2 :weight-decay 0.1d0))
        (lambda () (clip-by-global-norm 0.5d0))
        (lambda () (chain (clip-by-global-norm 0.5d0) (adam 0.01d0)))
        (lambda () (sgd (linear-schedule 0.5d0 0.1d0 3) :momentum 0.5d0))))

(defun %generic-optimizer (index)
  (funcall (nth (mod index (length *generic-optimizers*)) *generic-optimizers*)))

(defmacro %pbt ((name vars generator) &body body)
  "NAME の回帰ファイルつきの check-it。GENERATOR の各タプルを VARS に分解して BODY を評価する。"
  `(is (check-it ,generator
                 (lambda (case) (destructuring-bind ,vars case ,@body))
                 :regression-id ,name
                 :regression-file (nabla.tests.support:regression-path
                                   ,(string-downcase (substitute #\- #\/ (symbol-name name)))
                                   :package "NABLA.NN.TESTS"))))

;;; --- 構造・純粋性 ---

(test (optimizer/state-and-updates-keep-the-tree-structure :suite :nabla.small)
  "updates は params と同じ構造で、update の前後で状態の構造が変わらない。adam の状態の葉は
count、mu（params と同じ形）、nu（同じ形）の順。"
  (%pbt (optimizer/state-keeps-structure (seed index) (generator (tuple (uniform-integer :lo 0 :hi 10000)
                                                                       (uniform-integer :lo 0 :hi 6))))
    (let* ((optimizer (%generic-optimizer index))
           (params (%rand-tree seed))
           (grads (%rand-tree (+ seed 7)))
           (state (optimizer-init optimizer params)))
      (multiple-value-bind (updates new-state) (optimizer-update optimizer grads state params)
        (and (nabla:treedef= (nabla:tree-structure updates) (nabla:tree-structure params))
             (nabla:treedef= (nabla:tree-structure state) (nabla:tree-structure new-state))
             (equal (mapcar #'array-dimensions updates) (mapcar #'array-dimensions params))))))
  (let* ((params (%rand-tree 5))
         (leaves (nabla:tree-leaves (optimizer-init (adam 1d-3) params)))
         (shapes (mapcar #'array-dimensions params)))
    (is (= (length leaves) (+ 1 (* 2 (length params)))))
    (is (equal '() (array-dimensions (first leaves))))
    (is (equal (append shapes shapes) (mapcar #'array-dimensions (rest leaves))))
    (is (every (lambda (a) (loop for i below (array-total-size a) always (zerop (row-major-aref a i))))
               (rest leaves))
        "mu / nu の初期値はゼロ")))

(test (optimizer/update-is-pure-and-deterministic :suite :nabla.small)
  "update は入力（勾配・状態・パラメータ）を書き換えず、同じ入力から同じ結果を返す。"
  (%pbt (optimizer/update-is-pure (seed index) (generator (tuple (uniform-integer :lo 0 :hi 10000)
                                                                (uniform-integer :lo 0 :hi 6))))
    (let* ((optimizer (%generic-optimizer index))
           (params (%rand-tree seed))
           (grads (%rand-tree (+ seed 7)))
           (state (optimizer-init optimizer params))
           (before (%copy-leaves (list grads state params))))
      (multiple-value-bind (u1 s1) (optimizer-update optimizer grads state params)
        (multiple-value-bind (u2 s2) (optimizer-update optimizer grads state params)
          (and (%leaves-equal u1 u2)
               (%leaves-equal s1 s2)
               (every #'equalp before (nabla:tree-leaves (list grads state params)))))))))

(test (optimizer/count-increments-once-per-update :suite :nabla.small)
  "adam と scale-by-schedule の count は update ごとに 1 増える（i32 の rank 0）。"
  (%pbt (optimizer/count-increments (steps) (generator (tuple (uniform-integer :lo 1 :hi 6))))
    (let* ((optimizer (adam (linear-schedule 0.1d0 0.01d0 3)))
           (params (%rand-tree 3))
           (state (optimizer-init optimizer params)))
      (dotimes (k steps)
        (setf state (nth-value 1 (optimizer-update optimizer (%rand-tree (+ 40 k)) state params))))
      (let ((counts (remove-if-not (lambda (a) (null (array-dimensions a)))
                                   (nabla:tree-leaves state))))
        (and (= 2 (length counts))
             (every (lambda (a) (= steps (aref a))) counts))))))

;;; --- 閉じた形 ---

(test (optimizer/sgd-update-is-minus-lr-times-gradient :suite :nabla.small)
  "momentum なしの sgd の更新量は -lr g。"
  (%pbt (optimizer/sgd-closed-form (seed lr-index) (generator (tuple (uniform-integer :lo 0 :hi 10000)
                                                                    (uniform-integer :lo 1 :hi 9))))
    (let* ((lr (* lr-index 0.05d0))
           (optimizer (sgd lr))
           (params (%rand-tree seed))
           (grads (%rand-tree (+ seed 7)))
           (updates (optimizer-update optimizer grads (optimizer-init optimizer params) params)))
      (every (lambda (u g)
               (let ((expected (make-array (array-dimensions g) :element-type 'single-float)))
                 (dotimes (i (array-total-size g))
                   (setf (row-major-aref expected i) (float (* (- lr) (row-major-aref g i)) 1.0)))
                 (%f32-close u expected)))
             updates grads))))

(test (optimizer/adam-first-step-is-normalized-gradient :suite :nabla.small)
  "adam の最初のステップ（count = 0）では mu_hat = g、nu_hat = g^2 なので、更新量は
-lr g / (|g| + eps)。"
  (%pbt (optimizer/adam-first-step (seed b1-index b2-index)
          (generator (tuple (uniform-integer :lo 0 :hi 10000)
                            (uniform-integer :lo 0 :hi 2)
                            (uniform-integer :lo 0 :hi 3))))
    ;; b2 が 1 に近いほどバイアス補正 1 - b2^1 が小さく、1 - exp(...) の桁落ちが出やすい
    (let* ((lr 1d0) (eps 1d-8)   ; lr 1 で更新量が 1 程度になり、f32 の誤差が許容誤差に出る
           (b1 (nth b1-index '(0.9d0 0.99d0 0.5d0)))
           (b2 (nth b2-index '(0.999d0 0.9999d0 0.99999d0 0.9d0)))
           (optimizer (adam lr :b1 b1 :b2 b2 :eps eps))
           (params (%rand-tree seed))
           (grads (%rand-tree (+ seed 7)))
           (updates (optimizer-update optimizer grads (optimizer-init optimizer params) params)))
      (every (lambda (u g)
               (let ((expected (make-array (array-dimensions g) :element-type 'single-float)))
                 (dotimes (i (array-total-size g))
                   (let ((x (coerce (row-major-aref g i) 'double-float)))
                     (setf (row-major-aref expected i) (float (/ (* (- lr) x) (+ (abs x) eps)) 1.0))))
                 (%f32-close u expected)))
             updates grads))))

(test (optimizer/adamw-with-zero-decay-is-adam :suite :nabla.small)
  "weight-decay 0 の adamw は adam と同じ。decay > 0 のときは、adam の更新量に -lr * wd * p を足したもの。"
  (%pbt (optimizer/adamw-decay (seed) (generator (tuple (uniform-integer :lo 0 :hi 10000))))
    (let* ((params (%rand-tree seed))
           (grads (%rand-tree (+ seed 7)))
           (adam (adam 0.02d0))
           (zero (adamw 0.02d0 :weight-decay 0d0))
           (decayed (adamw 0.02d0 :weight-decay 0.25d0))
           (ua (optimizer-update adam grads (optimizer-init adam params) params))
           (uz (optimizer-update zero grads (optimizer-init zero params) params))
           (ud (optimizer-update decayed grads (optimizer-init decayed params) params)))
      (and (%trees-close uz ua)
           (every (lambda (d a p)
                    (let ((expected (make-array (array-dimensions a) :element-type 'single-float)))
                      (dotimes (i (array-total-size a))
                        (setf (row-major-aref expected i)
                              (float (- (row-major-aref a i) (* 0.02d0 0.25d0 (row-major-aref p i))) 1.0)))
                      (%f32-close d expected)))
                  ud ua params)))))

;;; --- chain ---

(defun %run-steps (optimizer params grads-list)
  "GRADS-LIST の各勾配で update し、(更新量のリスト 最後の状態) を返す。パラメータは更新しない。"
  (let ((state (optimizer-init optimizer params)) (all '()))
    (dolist (g grads-list)
      (multiple-value-bind (u s) (optimizer-update optimizer g state params)
        (push u all)
        (setf state s)))
    (list (nreverse all) state)))

(defparameter *chain-parts*
  (list (lambda () (clip-by-global-norm 0.7d0))
        (lambda () (adam 0.01d0))
        (lambda () (scale -0.5d0))
        (lambda () (momentum-trace 0.9d0))
        (lambda () (scale-by-schedule (linear-schedule 1d0 0.2d0 3)))))

(defun %chain-part (index)
  (funcall (nth (mod index (length *chain-parts*)) *chain-parts*)))

(test (optimizer/chain-equals-applying-in-order :suite :nabla.small)
  "(chain a b) の update は、a の出力を b に渡したものと一致し、状態は a の状態と b の状態を順に並べたもの。"
  (%pbt (optimizer/chain-sequential (seed i j) (generator (tuple (uniform-integer :lo 0 :hi 10000)
                                                                (uniform-integer :lo 0 :hi 4)
                                                                (uniform-integer :lo 0 :hi 4))))
    (let* ((a (%chain-part i)) (b (%chain-part j))
           (both (chain a b))
           (params (%rand-tree seed))
           (state (optimizer-init both params))
           (sa (optimizer-init a params))
           (sb (optimizer-init b params)))
      (loop for k below 3
            always (let ((g (%rand-tree (+ seed 7 k))))
                     (multiple-value-bind (u new-state) (optimizer-update both g state params)
                       (multiple-value-bind (ua new-sa) (optimizer-update a g sa params)
                         (multiple-value-bind (ub new-sb) (optimizer-update b ua sb params)
                           (setf state new-state sa new-sa sb new-sb)
                           (and (%leaves-equal u ub)
                                (%leaves-equal state (list sa sb)))))))))))

(test (optimizer/chain-is-associative-and-has-identity :suite :nabla.small)
  "chain は結合的（入れ子の chain と平らな chain は同じ更新量と、同じ並びの状態の葉）で、
1つだけの chain は元のオプティマイザと、空の chain は恒等と一致する。"
  (%pbt (optimizer/chain-associative (seed i j k) (generator (tuple (uniform-integer :lo 0 :hi 10000)
                                                                   (uniform-integer :lo 0 :hi 4)
                                                                   (uniform-integer :lo 0 :hi 4)
                                                                   (uniform-integer :lo 0 :hi 4))))
    (let* ((params (%rand-tree seed))
           (grads (loop for n below 3 collect (%rand-tree (+ seed 7 n))))
           (left (chain (chain (%chain-part i) (%chain-part j)) (%chain-part k)))
           (right (chain (%chain-part i) (chain (%chain-part j) (%chain-part k))))
           (flat (chain (%chain-part i) (%chain-part j) (%chain-part k)))
           (single (%chain-part i))
           (wrapped (chain (%chain-part i)))
           (empty (chain))
           (r-left (%run-steps left params grads))
           (r-right (%run-steps right params grads))
           (r-flat (%run-steps flat params grads))
           (r-single (%run-steps single params grads))
           (r-wrapped (%run-steps wrapped params grads))
           (r-empty (%run-steps empty params grads)))
      (and (every #'%leaves-equal (first r-left) (first r-flat))
           (every #'%leaves-equal (first r-right) (first r-flat))
           (%leaves-equal (second r-left) (second r-flat))
           (%leaves-equal (second r-right) (second r-flat))
           (every #'%leaves-equal (first r-single) (first r-wrapped))
           (%leaves-equal (second r-single) (second r-wrapped))
           (every #'%leaves-equal (first r-empty) grads)
           (null (nabla:tree-leaves (second r-empty)))))))

;;; --- clip-by-global-norm ---

(test (optimizer/clip-by-global-norm-bounds-the-norm :suite :nabla.small)
  "clip-by-global-norm の結果の大域ノルムは max-norm を超えない（f32 の丸めの分だけ余裕を見る）。
入力のノルムが max-norm 未満なら素通しで、そうでなければ勾配と同じ向きに max-norm へ縮める。
期待する縮める係数は min(1, max-norm / in-norm)。in-norm が 0.999 max-norm 以上 max-norm 未満のときは
素通しなので、係数を 1 で頭打ちにする（頭打ちにしないと 1 より大きい係数を期待してしまう）。"
  (%pbt (optimizer/clip-norm (seed scale-index max-index)
          (generator (tuple (uniform-integer :lo 0 :hi 10000)
                            (uniform-integer :lo 1 :hi 40)
                            (uniform-integer :lo 1 :hi 20))))
    (let* ((max-norm (* max-index 0.1d0))
           (clipper (clip-by-global-norm max-norm))
           (grads (%rand-tree seed (* scale-index 0.1)))
           (params (%rand-tree (+ seed 3)))
           (updates (optimizer-update clipper grads (optimizer-init clipper params) params))
           (in-norm (sqrt (%sum-squares (nabla:tree-leaves grads))))
           (out-norm (sqrt (%sum-squares (nabla:tree-leaves updates)))))
      (and (<= out-norm (* max-norm (+ 1d0 1d-5)))
           (if (< in-norm (* max-norm 0.999d0))
               (%leaves-equal updates grads)
               (let ((factor (min 1d0 (/ max-norm in-norm))))
                 (every (lambda (u g)
                          (let ((expected (make-array (array-dimensions g) :element-type 'single-float)))
                            (dotimes (i (array-total-size g))
                              (setf (row-major-aref expected i)
                                    (float (* factor (row-major-aref g i)) 1.0)))
                            (%f32-close u expected)))
                        updates grads)))))))

;;; --- スケジュール ---

(test (optimizer/linear-schedule-endpoints-and-monotone :suite :nabla.small)
  "linear-schedule は transition-begin まで init、transition-steps 後は end で、間を単調に動く。"
  (%pbt (optimizer/linear-schedule (init-index end-index steps begin) (generator (tuple (uniform-integer :lo 0 :hi 10)
                                                                                    (uniform-integer :lo 0 :hi 10)
                                                                                    (uniform-integer :lo 1 :hi 8)
                                                                                    (uniform-integer :lo 0 :hi 5))))
    (let* ((init (* init-index 0.1d0)) (end (* end-index 0.1d0))
           (schedule (linear-schedule init end steps :transition-begin begin))
           (values (loop for s from 0 to (+ begin steps 2) collect (%schedule-value schedule s)))
           (tolerance (+ 1d-6 (* 1d-5 (max (abs init) (abs end))))))
      (and (<= (abs (- (nth begin values) init)) tolerance)
           (<= (abs (- (first values) init)) tolerance)
           (<= (abs (- (nth (+ begin steps) values) end)) tolerance)
           (<= (abs (- (car (last values)) end)) tolerance)
           (if (<= init end)
               (loop for (a b) on values while b always (<= a (+ b tolerance)))
               (loop for (a b) on values while b always (>= a (- b tolerance))))))))

(test (optimizer/cosine-decay-schedule-endpoints-and-monotone :suite :nabla.small)
  "cosine-decay-schedule は init から init*alpha へ単調に下がり、steps 以降は init*alpha のまま。"
  (%pbt (optimizer/cosine-schedule (init-index alpha-index steps) (generator (tuple (uniform-integer :lo 1 :hi 10)
                                                                                 (uniform-integer :lo 0 :hi 10)
                                                                                 (uniform-integer :lo 1 :hi 9))))
    (let* ((init (* init-index 0.1d0)) (alpha (* alpha-index 0.1d0))
           (schedule (cosine-decay-schedule init steps :alpha alpha))
           (values (loop for s from 0 to (+ steps 3) collect (%schedule-value schedule s)))
           (tolerance (+ 1d-6 (* 1d-5 init))))
      (and (<= (abs (- (first values) init)) tolerance)
           (<= (abs (- (nth steps values) (* init alpha))) tolerance)
           (<= (abs (- (car (last values)) (* init alpha))) tolerance)
           (loop for (a b) on values while b always (>= a (- b tolerance)))))))

(test (optimizer/warmup-cosine-decay-schedule-hits-peak :suite :nabla.small)
  "warmup-cosine-decay-schedule は init から始まり、warmup-steps でちょうど peak に着き、
decay-steps で end-value に着く。"
  (%pbt (optimizer/warmup-cosine (warmup extra) (generator (tuple (uniform-integer :lo 1 :hi 6)
                                                                  (uniform-integer :lo 1 :hi 8))))
    (let* ((decay (+ warmup extra))
           (schedule (warmup-cosine-decay-schedule 0.1d0 2d0 warmup decay :end-value 0.3d0)))
      (and (approx= (%schedule-value schedule 0) 0.1d0)
           (approx= (%schedule-value schedule warmup) 2d0)
           (approx= (%schedule-value schedule decay) 0.3d0)
           (approx= (%schedule-value schedule (+ decay 4)) 0.3d0)))))

;;; --- defoptimizer ---

(defoptimizer test-halve ()
  (:init (params) nil)
  (:update (updates state params) (values (nabla:tree-map (lambda (g) (* g 0.5)) updates) state)))

(defoptimizer test-step-counter (rate)
  (:init (params) (make-array nil :element-type '(signed-byte 32) :initial-element 0))
  (:update (updates state params)
   (values (nabla:tree-map (lambda (g) (* g rate)) updates) (+ state 1))))

(test (optimizer/defoptimizer-defines-composable-optimizers :suite :nabla.small)
  "defoptimizer で書いた init / update の組は、組み込みのオプティマイザと chain できる。
状態は init が返した木そのもの。"
  (let* ((params (%rand-tree 3))
         (grads (%rand-tree 9))
         (halve (test-halve))
         (counter (test-step-counter 4))
         (both (chain halve counter (sgd 1)))
         (state (optimizer-init both params)))
    (is (null (optimizer-init halve params)))
    (multiple-value-bind (updates new-state) (optimizer-update both grads state params)
      (is (every (lambda (u g)
                   (let ((expected (make-array (array-dimensions g) :element-type 'single-float)))
                     (dotimes (i (array-total-size g))
                       (setf (row-major-aref expected i) (* -2.0 (row-major-aref g i))))
                     (%f32-close u expected)))
                 updates grads))
      (is (= 1 (aref (second new-state)))))))

(test (optimizer/rejects-bad-arguments :suite :nabla.small)
  "不正な引数は update / 構築時にエラーにする: b1 / b2 が [0, 1) 外、chain に非オプティマイザ、
update の勾配と状態・パラメータで木の構造が違う。"
  (signals error (adam 0.1d0 :b1 1d0))
  (signals error (adam 0.1d0 :b2 -0.1d0))
  (signals error (adam 0.1d0 :eps -1d0))
  (signals error (sgd 0.1d0 :momentum 1.5d0))
  (signals error (sgd 0.1d0 :nesterov t))
  (signals error (chain (adam 0.1d0) 3))
  (signals error (clip-by-global-norm 0d0))
  (signals error (linear-schedule 1d0 0d0 0))
  (signals error (cosine-decay-schedule 1d0 0))
  (signals error (warmup-cosine-decay-schedule 0d0 1d0 5 5))
  (let* ((optimizer (adam 0.1d0))
         (params (%rand-tree 1))
         (state (optimizer-init optimizer params)))
    (signals nabla:pytree-structure-mismatch
      (optimizer-update optimizer (list (first params)) state params))))

;;; --- dtype・端の値 ---

(defun %f64-array (&rest values)
  (make-array (length values) :element-type 'double-float :initial-contents values))

(test (optimizer/keeps-the-dtype-of-each-leaf :suite :nabla.small)
  "f32 と f64 の葉が混ざった木でも、更新量は葉ごとの dtype のまま（スカラーを葉の dtype に
convert する）。f64 の葉の最初の adam ステップは f64 の精度で -lr g / (|g| + eps) に一致する。"
  (let* ((optimizer (chain (clip-by-global-norm 10d0)
                           (adam 0.1d0)))
         (params (list (%rand-array '(2) 3) (%f64-array 0.5d0 -0.25d0)))
         (grads (list (%rand-array '(2) 4) (%f64-array 0.3d0 -1.5d0)))
         (state (optimizer-init optimizer params)))
    (let ((updates (optimizer-update optimizer grads state params)))
      (is (equal '(single-float double-float) (mapcar #'array-element-type updates)))
      (let ((expected (mapcar (lambda (g) (/ (* -0.1d0 g) (+ (abs g) 1d-8))) (coerce (second grads) 'list))))
        (is (nabla.tests.support:allclose (second updates) (apply #'%f64-array expected) :dtype :f64
                                          :rtol 1d-12 :atol 1d-12))))))

(test (optimizer/zero-gradients-stay-finite :suite :nabla.small)
  "勾配がすべて 0 でも clip-by-global-norm / adam の更新量は有限（0 のノルムで割らない）で、0 のまま。"
  (let* ((params (%rand-tree 5))
         (grads (mapcar (lambda (p) (make-array (array-dimensions p) :element-type 'single-float
                                                                      :initial-element 0.0))
                        params)))
    (dolist (optimizer (list (clip-by-global-norm 0.5d0) (adam 0.1d0)
                             (chain (clip-by-global-norm 0.5d0) (adam 0.1d0))))
      (let ((updates (optimizer-update optimizer grads (optimizer-init optimizer params) params)))
        (is (every (lambda (u) (loop for i below (array-total-size u) always (zerop (row-major-aref u i))))
                   updates))))))

(test (optimizer/count-saturates-at-int32-max :suite :nabla.small)
  "count は :i32 の最大値で止まる（Optax の safe_increment）。オーバーフローして負になると
スケジュールとバイアス補正が壊れる。"
  (let* ((optimizer (scale-by-schedule (constant-schedule 1d0)))
         (params (%rand-tree 5))
         (state (optimizer-init optimizer params)))
    (setf (aref (first (nabla:tree-leaves state))) 2147483646)
    (dotimes (k 2)
      (setf state (nth-value 1 (optimizer-update optimizer params state params)))
      (is (= 2147483647 (aref (first (nabla:tree-leaves state))))))))

(test (optimizer/adam-bias-correction-handles-zero-decay-and-large-counts :suite :nabla.small)
  "b1 = 0（運動量なし）は mu_hat = g なので最初のステップは g / (sqrt(nu_hat) + eps)。count が大きくても
更新量は有限で、補正が 1 になった値 (1 - b1) g / (sqrt((1 - b2) g^2) + eps) に一致する。"
  (let* ((params (list (%f64-array 1d0)))
         (grads (list (%f64-array 2d0)))
         (plain (scale-by-adam :b1 0d0))
         (state (optimizer-init plain params)))
    (is (nabla.tests.support:allclose (first (optimizer-update plain grads state params))
                                      (%f64-array (/ 2d0 (+ 2d0 1d-8)))
                                      :dtype :f64 :rtol 1d-6 :atol 1d-12)))
  (let* ((adam (scale-by-adam))
         (params (list (%rand-array '(3) 1)))
         (grads (list (make-array 3 :element-type 'single-float :initial-element 0.5)))
         (state (optimizer-init adam params)))
    (setf (aref (first (nabla:tree-leaves state))) 1000000)
    ;; count が十分大きければ補正は 1。mu = 0.1 * 0.5、nu = 0.001 * 0.25 なので mu / (sqrt(nu) + eps)
    (let ((u (first (optimizer-update adam grads state params))))
      (is (nabla.tests.support:allclose u (make-array 3 :element-type 'single-float
                                                        :initial-element (float (/ 0.05d0 (+ (sqrt 0.00025d0) 1d-8)) 1.0))
                                        :dtype :f32)))))

(test (optimizer/rejects-non-finite-hyperparameters-and-bad-defoptimizer-forms :suite :nabla.small)
  "NaN / 無限大 / 実数でないハイパーパラメータは構築時にエラー。defoptimizer の節の形が違えば展開時にエラー。"
  (signals error (scale (sb-int:with-float-traps-masked (:invalid :divide-by-zero) (/ 0d0 0d0))))
  (signals error (sgd sb-ext:double-float-positive-infinity))
  (signals error (sgd "0.1"))
  (signals error (constant-schedule sb-ext:single-float-negative-infinity))
  (signals error (macroexpand-1 '(defoptimizer bad-one () (:init (p) nil))))
  (signals error (macroexpand-1 '(defoptimizer bad-two () (:init (p) nil) (:update (u s) (values u s)))))
  (signals error (macroexpand-1 '(defoptimizer bad-three () "doc" "extra" (:init (p) nil)
                                  (:update (u s p) (values u s)))))
  (signals error (macroexpand-1 '(defoptimizer bad-four () (:init (p) nil) (:init (p) nil)
                                  (:update (u s p) (values u s))))))

;;; --- 引数の検査（エラーメッセージが引数の名前を挙げる）・既定値・端 ---

(defmacro %signals-mentioning ((&rest needles) form)
  "FORM が error を signal し、そのメッセージ（小文字にしたもの）が NEEDLES をすべて含む。
検査が別の場所の偶然のエラーで通らないよう、どの引数を拒否したかをメッセージで確かめる。"
  `(let ((text (handler-case (progn ,form nil)
                 (error (condition) (string-downcase (princ-to-string condition))))))
     (is (and text (every (lambda (needle) (search needle text)) ',needles))
         "~S のエラーが ~S を挙げていない: ~S" ',form ',needles text)))

(test (optimizer/validation-errors-name-the-offending-argument :suite :nabla.small)
  "不正な引数は、関数名と引数名をメッセージに含むエラーにする。境界の値（0、1）は受け付けるか拒否するか
を区別する。"
  (%signals-mentioning ("scale" "factor") (scale sb-ext:double-float-positive-infinity))
  (%signals-mentioning ("add-decayed-weights" "weight-decay")
                       (add-decayed-weights sb-ext:double-float-positive-infinity))
  (%signals-mentioning ("clip-by-global-norm" "max-norm") (clip-by-global-norm 0))
  (%signals-mentioning ("clip-by-global-norm" "max-norm") (clip-by-global-norm -1))
  (dolist (bad '(1 -1/2 3/2))
    (%signals-mentioning ("momentum-trace" "decay") (momentum-trace bad))
    (%signals-mentioning ("scale-by-adam" "b1") (scale-by-adam :b1 bad))
    (%signals-mentioning ("scale-by-adam" "b2") (scale-by-adam :b2 bad)))
  (%signals-mentioning ("scale-by-adam" "eps") (scale-by-adam :eps -1/2))
  (%signals-mentioning ("scale-by-adam" "eps-root") (scale-by-adam :eps-root -1/2))
  (finishes (momentum-trace 0))
  (finishes (scale-by-adam :b1 0 :b2 0 :eps 0 :eps-root 0))
  (%signals-mentioning ("sgd") (sgd "0.1"))
  (%signals-mentioning ("adam") (adam nil))
  (%signals-mentioning ("adamw") (adamw 'x))
  (%signals-mentioning ("learning-rate") (sgd sb-ext:double-float-positive-infinity))
  (%signals-mentioning ("scale-by-schedule") (scale-by-schedule 0.1))
  (%signals-mentioning ("constant-schedule" "value") (constant-schedule sb-ext:double-float-positive-infinity))
  (%signals-mentioning ("linear-schedule" "init-value")
                       (linear-schedule sb-ext:double-float-positive-infinity 0 1))
  (%signals-mentioning ("linear-schedule" "end-value")
                       (linear-schedule 0 sb-ext:double-float-positive-infinity 1))
  (%signals-mentioning ("linear-schedule" "transition-steps") (linear-schedule 1 0 0))
  (%signals-mentioning ("linear-schedule" "transition-begin") (linear-schedule 1 0 3 :transition-begin -1))
  (finishes (linear-schedule 1 0 3 :transition-begin 0))
  (%signals-mentioning ("cosine-decay-schedule" "init-value")
                       (cosine-decay-schedule sb-ext:double-float-positive-infinity 3))
  (%signals-mentioning ("cosine-decay-schedule" "alpha")
                       (cosine-decay-schedule 1 3 :alpha sb-ext:double-float-positive-infinity))
  (%signals-mentioning ("cosine-decay-schedule" "decay-steps") (cosine-decay-schedule 1 0))
  (%signals-mentioning ("warmup-cosine-decay-schedule" "init-value")
                       (warmup-cosine-decay-schedule sb-ext:double-float-positive-infinity 1 1 3))
  (%signals-mentioning ("warmup-cosine-decay-schedule" "peak-value")
                       (warmup-cosine-decay-schedule 0 sb-ext:double-float-positive-infinity 1 3))
  (%signals-mentioning ("warmup-cosine-decay-schedule" "end-value")
                       (warmup-cosine-decay-schedule 0 1 1 3 :end-value sb-ext:double-float-positive-infinity))
  (%signals-mentioning ("warmup-cosine-decay-schedule" "warmup-steps") (warmup-cosine-decay-schedule 0 1 -1 3))
  (%signals-mentioning ("warmup-cosine-decay-schedule" "decay-steps") (warmup-cosine-decay-schedule 0 1 0 0))
  (%signals-mentioning ("warmup-cosine-decay-schedule" "decay-steps") (warmup-cosine-decay-schedule 0 1 3 3))
  (let* ((both (chain (adam 0.1d0) (scale 2d0)))
         (params (%rand-tree 1)))
    (%signals-mentioning ("chain")
                         (optimizer-update both params (list (optimizer-init (adam 0.1d0) params)) params))))

(test (optimizer/defoptimizer-accepts-well-formed-clauses :suite :nabla.small)
  "正しい形の defoptimizer（docstring あり・なし）は展開できる。"
  (finishes (macroexpand-1 '(defoptimizer ok-one () (:init (p) nil) (:update (u s p) (values u s)))))
  (finishes (macroexpand-1 '(defoptimizer ok-two (a b) "doc" (:update (u s p) (values u s)) (:init (p) nil))))
  (signals error (macroexpand-1 '(defoptimizer bad-five () (:init (p q) nil) (:update (u s p) (values u s))))))

(test (optimizer/defaults-match-optax :suite :nabla.small)
  "既定値は Optax と同じ: adam の eps は 1e-8（g = 1e-9 の最初のステップは -lr g / (|g| + 1e-8)）、
adamw の weight-decay は 1e-4。count は :i32 の rank 0。"
  (let* ((params (list (%f64-array 1d0)))
         (grads (list (%f64-array 1d-9)))
         (adam (adam 1d0))
         (state (optimizer-init adam params)))
    (is (nabla.tests.support:allclose (first (optimizer-update adam grads state params))
                                      (%f64-array (/ -1d-9 (+ 1d-9 1d-8)))
                                      :dtype :f64 :rtol 1d-9 :atol 1d-12))
    (let ((count (first (nabla:tree-leaves (first state)))))
      (is (null (array-dimensions count)))
      (is (equal '(signed-byte 32) (array-element-type count)))))
  ;; scale-by-adam は学習率を掛けず（+g/(|g|+eps)）、adamw（weight-decay 0）は -lr 倍する
  (loop for (optimizer sign) in (list (list (scale-by-adam) 1d0) (list (adamw 1d0 :weight-decay 0d0) -1d0))
        do (let* ((params (list (%f64-array 1d0)))
                  (grads (list (%f64-array 1d-9)))
                  (state (optimizer-init optimizer params)))
             (is (nabla.tests.support:allclose (first (optimizer-update optimizer grads state params))
                                               (%f64-array (* sign (/ 1d-9 (+ 1d-9 1d-8))))
                                               :dtype :f64 :rtol 1d-9 :atol 1d-12)
                 "既定の eps が 1e-8 でない")))
  (let* ((params (%rand-tree 4))
         (grads (%rand-tree 11))
         (default (adamw 1d0))
         (explicit (adamw 1d0 :weight-decay 1d-4))
         (plain (adam 1d0))
         (ud (optimizer-update default grads (optimizer-init default params) params))
         (ue (optimizer-update explicit grads (optimizer-init explicit params) params))
         (up (optimizer-update plain grads (optimizer-init plain params) params)))
    (is (%trees-close ud ue))
    ;; allclose は不一致を標準出力に書くので、ここでは equalp で静かに比べる
    (is (notevery #'equalp ud up) "既定の weight-decay は 0 でない")))

(test (optimizer/eps-root-is-added-inside-the-square-root :suite :nabla.small)
  "adam の最初のステップは -lr g / (sqrt(g^2 + eps_root) + eps)。eps-root が大きいと更新量が小さくなる。"
  (let* ((params (list (%f64-array 1d0 -2d0)))
         (grads (list (%f64-array 0.5d0 -3d0)))
         (adam (adam 1d0 :eps-root 4d0 :eps 0.5d0))
         (state (optimizer-init adam params)))
    (is (nabla.tests.support:allclose
         (first (optimizer-update adam grads state params))
         (apply #'%f64-array (mapcar (lambda (g) (/ (- g) (+ (sqrt (+ (* g g) 4d0)) 0.5d0))) '(0.5d0 -3d0)))
         :dtype :f64 :rtol 1d-9 :atol 1d-12))))

(test (optimizer/warmup-of-zero-steps-starts-at-the-peak :suite :nabla.small)
  "warmup-steps が 0 の warmup-cosine-decay-schedule は、step 0 から peak で、decay-steps で end-value。"
  (let ((schedule (warmup-cosine-decay-schedule 0d0 2d0 0 4 :end-value 0.5d0)))
    (is (approx= (%schedule-value schedule 0) 2d0))
    (is (approx= (%schedule-value schedule 4) 0.5d0))
    (is (approx= (%schedule-value schedule 2) 1.25d0 :dtype :f32))))

(test (optimizer/count-is-a-rank-0-i32 :suite :nabla.small)
  "schedule-state の count も :i32 の rank 0。"
  (let ((count (first (nabla:tree-leaves (optimizer-init (scale-by-schedule (constant-schedule 1d0))
                                                         (%rand-tree 1))))))
    (is (null (array-dimensions count)))
    (is (equal '(signed-byte 32) (array-element-type count)))))

;;; --- 微分・params の省略・内部の数値部品 ---

(test (optimizer/clip-by-global-norm-gradient-is-finite-at-zero :suite :nabla.small)
  "全勾配が 0（大域ノルム 0）の点で、clip-by-global-norm の更新量の和を勾配について微分しても NaN / 無限大に
ならない（sqrt(0) の微分と 0 割りを選ばれない側にも出さない）。ノルムが max-norm 未満なので更新量は
勾配そのもの（恒等）で、微分は全要素 1。"
  (let* ((clipper (clip-by-global-norm 0.5d0))
         (sum (nabla:with-tracing (grads)
                (let ((updates (optimizer-update clipper grads nil grads)))
                  (+ (nabla:reduce-sum (first updates)) (nabla:reduce-sum (second updates))))))
         (zeros (list (make-array '(2 2) :element-type 'single-float :initial-element 0.0)
                      (make-array 3 :element-type 'single-float :initial-element 0.0)))
         (gradient (funcall (nabla:grad sum) zeros)))
    (dolist (g gradient)
      (is (loop for i below (array-total-size g) always (= 1.0 (row-major-aref g i)))
          "勾配が 1 でない: ~S" g))))

(test (optimizer/weight-decay-without-params-names-the-argument :suite :nabla.small)
  "add-decayed-weights / adamw に params を渡さないと、木の構造の不一致ではなく、
add-decayed-weights と params を挙げたエラーにする。"
  (let ((params (%rand-tree 2)) (grads (%rand-tree 3)))
    (dolist (optimizer (list (add-decayed-weights 0.1d0) (adamw 0.1d0)))
      (let ((state (optimizer-init optimizer params)))
        (%signals-mentioning ("add-decayed-weights" "params") (optimizer-update optimizer grads state))))
    ;; 更新量が空の木なら params は要らない
    (finishes (optimizer-update (add-decayed-weights 0.1d0) nil nil))))

(test (optimizer/bias-correction-matches-the-power-in-double :suite :nabla.small)
  "1 - b^t の -expm1(t log b) による計算（%one-minus-pow）は、double の 1 - b^t と rtol 1e-6 で一致する。
b は 1 に近い値まで、t は 2^31 - 1 までの格子。f32 と f64 の両方。"
  (dolist (dtype '(:f32 :f64))
    (dolist (base '(0.5d0 0.9d0 0.99d0 0.999d0 0.999999d0))
      (dolist (steps '(1 2 10 1000 1048576 2147483647))
        (let ((expected (sb-int:with-float-traps-masked (:underflow :inexact)
                          (- 1d0 (expt base steps))))
              (actual (aref (nabla.nn::%one-minus-pow base (%count-array steps) dtype))))
          (is (approx= actual expected :rtol 1d-6 :atol 0d0)
              "dtype ~S: 1 - ~S^~D は ~S のはずが ~S" dtype base steps expected actual))))))

(test (optimizer/cosine-helper-matches-cos :suite :nabla.small)
  "cosine 値（%cosine-value。cos プリミティブによる 0.5 (1 + cos(pi t))）は 0.5 (1 + cos(pi t)) と f32 の許容誤差で一致する（t は 0..1 を
1000 分割した格子と、その外側の clamp）。"
  (dotimes (count 1003)
    (let ((expected (* 0.5d0 (+ 1d0 (cos (* pi (/ (min count 1000) 1000d0))))))
          (actual (aref (nabla.nn::%cosine-value (%count-array count) 1d0 0d0 1000))))
      (is (approx= actual expected :rtol 1d-5 :atol 2d-6)
          "count ~D: ~S のはずが ~S" count expected actual))))

(test (optimizer/global-norm-value-and-gradient-at-zero-leaves :suite :nabla.small)
  "大域ノルム（%global-norm）そのものを直接確かめる: 全葉が 0 なら値は 0（負や 1 ではない）で勾配は有限の 0、
一部の葉だけ 0 なら値は残りの葉のノルムで、勾配は g / norm（0 の葉は 0）。
これは clip の出力（ノルムが max-norm 未満なら恒等）からは見えない、ノルム自身の振る舞い。
\(sqrt の内側の where の定数 1 は、nabla の select の転置がマスクで勾配を落とすので、値も勾配も
変えない。二重 where の本体は where そのもので、sqrt を直接使うと上のテストが NaN で落ちる。)"
  (let* ((norm-fn (nabla:with-tracing (leaves) (nabla.nn::%global-norm leaves :f32)))
         (zero (list (make-array 2 :element-type 'single-float :initial-element 0.0)
                     (make-array 3 :element-type 'single-float :initial-element 0.0)))
         (partial (list (make-array 2 :element-type 'single-float :initial-element 0.0)
                        (make-array 3 :element-type 'single-float :initial-contents '(3.0 4.0 0.0)))))
    (is (= 0.0 (aref (funcall norm-fn zero))))
    (is (= 5.0 (aref (funcall norm-fn partial))))
    (let ((gz (funcall (nabla:grad norm-fn) zero))
          (gp (funcall (nabla:grad norm-fn) partial)))
      (is (every (lambda (g) (loop for i below (array-total-size g) always (= 0.0 (row-major-aref g i)))) gz))
      (is (%trees-close gp (list (make-array 2 :element-type 'single-float :initial-element 0.0)
                                 (make-array 3 :element-type 'single-float :initial-contents '(0.6 0.8 0.0))))))))
