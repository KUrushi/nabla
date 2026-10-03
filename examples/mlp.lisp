(require :asdf)
(asdf:load-system "nabla/iree")
(defpackage #:nabla-example-mlp (:use #:cl))
(in-package #:nabla-example-mlp)

;;; 2層 MLP（dense -> tanh -> dense）の softmax 交差エントロピー。
;;; ラベル y は one-hot の f32 配列 (N C)。logsumexp は最大値を引いて安定にする
;;; （最大値は stop-gradient で定数扱い。引いても微分が変わらない）。
(defun make-mlp-loss (n h c)
  (let ((scale (/ -1.0 n)))
    (nb:with-tracing (w1 b1 w2 b2 x y)
      (let* ((hidden (tanh (+ (nb:dot x w1) (nb:broadcast-in-dim b1 (list n h) '(1)))))
             (logits (+ (nb:dot hidden w2) (nb:broadcast-in-dim b2 (list n c) '(1))))
             (m (nb:stop-gradient (nb:reduce-max logits :axes '(1))))
             (shifted (- logits (nb:broadcast-in-dim m (list n c) '(0))))
             (lse (log (nb:reduce-sum (exp shifted) :axes '(1))))
             (logp (- shifted (nb:broadcast-in-dim lse (list n c) '(0)))))
        (* (nb:reduce-sum (* y logp)) scale)))))

(defun make-mlp-example-loss (c)
  "1サンプル（x: (D)、y: (C) の one-hot）の損失 (w1 b1 w2 b2 x y) → スカラー。
MAKE-MLP-LOSS の N = 1 のとき（バッチ軸なし）と同じ式で、(/ -1.0 n) の係数は掛けない
（サンプルごとの損失の平均がバッチ全体の損失になる）。per-example 勾配（VMAP の対象）に使う。"
  (nb:with-tracing (w1 b1 w2 b2 x y)
    (let* ((hidden (tanh (+ (nb:dot x w1) b1)))
           (logits (+ (nb:dot hidden w2) b2))
           (m (nb:stop-gradient (nb:reduce-max logits :axes '(0))))
           (shifted (- logits (nb:broadcast-in-dim m (list c) '())))
           (lse (log (nb:reduce-sum (exp shifted) :axes '(0))))
           (logp (- shifted (nb:broadcast-in-dim lse (list c) '()))))
      (- (nb:reduce-sum (* y logp))))))

(defun make-per-example-grad (&key (c 2))
  "サンプルごとの勾配を求める関数 (w1 b1 w2 b2 x y) → (w1の勾配 b1の勾配 w2の勾配 b2の勾配)
の多値を返す。x (N D) と y (N C) だけを軸 0 でバッチし（in-axes は (nil nil nil nil 0 0)）、
パラメータはバッチしない。各勾配の形は (N ...パラメータの形)。JAX の
(jax.vmap (jax.grad loss) :in_axes (None, 0, 0)) に相当する。
合成は (JIT ...)・VMAP の中の VMAP・GRAD の中の VMAP でも使える
（tests/iree/per-example-test.lisp）。"
  (let ((g (nb:grad (make-mlp-example-loss c) :argnums '(0 1 2 3))))
    ;; 勾配のリストは vmap の出力にできない（jit と同じ）ので多値に直す
    (nb:vmap (nb:with-tracing (w1 b1 w2 b2 x y)
               (values-list (funcall g w1 b1 w2 b2 x y)))
             :in-axes '(nil nil nil nil 0 0))))

(defun sgd-update (param grad lr)
  "PARAM - LR * GRAD（f32 の配列）を新しい配列で返す。"
  (let ((new (make-array (array-dimensions param) :element-type 'single-float)))
    (dotimes (i (array-total-size param) new)
      (setf (row-major-aref new i)
            (- (row-major-aref param i) (* lr (row-major-aref grad i)))))))

(defun make-mlp-train-step (&key (lr 0.5) (n 16) (h 8) (c 2) backend)
  "学習ステップ (lambda (params x y)) を返す。PARAMS は (w1 b1 w2 b2) の f32 配列のリスト。
戻り値は2つ: その関数と、jit した関数そのもの（(w1 b1 w2 b2 x y) を取り、
(損失 w1の勾配 b1の勾配 w2の勾配 b2の勾配) を多値で返す。コンパイル時間と
ステップ時間を分けて測るときに使う）。関数は多値の (更新前の損失 SGD で更新した PARAMS) を返す。
BACKEND は jit の :backend（NIL なら既定）。(jit (value-and-grad loss)) は
ここで1回だけ作り、呼び出しごとにはコンパイルしない（2回目以降はキャッシュを使う）。
argnums がリストのとき勾配のリストは jit の出力にできないので、
multiple-value-bind で受けて (値 勾配...) の多値に直す。"
  (let* ((vg (nb:value-and-grad (make-mlp-loss n h c) :argnums '(0 1 2 3)))
         (jitted (nb:jit (nb:with-tracing (w1 b1 w2 b2 x y)
                           (multiple-value-bind (loss grads) (funcall vg w1 b1 w2 b2 x y)
                             (values-list (cons loss grads))))
                         :backend backend)))
    (values
     (lambda (params x y)
       (destructuring-bind (loss &rest grads) (multiple-value-list (apply jitted (append params (list x y))))
         (values (aref loss)
                 (mapcar (lambda (p g) (sgd-update p g lr)) params grads))))
     jitted)))

(defun make-blobs (n seed)
  "XOR 風の2次元2クラス分類データ。(±1, ±1) を中心にノイズを足した N 点と、
x0*x1 > 0 かどうかの one-hot ラベル (N 2)。SEED で再現できる。"
  (let ((state (sb-ext:seed-random-state seed))
        (x (make-array (list n 2) :element-type 'single-float))
        (y (make-array (list n 2) :element-type 'single-float :initial-element 0.0)))
    (dotimes (i n (values x y))
      (let ((c0 (if (zerop (random 2 state)) -1.0 1.0))
            (c1 (if (zerop (random 2 state)) -1.0 1.0)))
        (setf (aref x i 0) (+ c0 (- (random 0.6 state) 0.3))
              (aref x i 1) (+ c1 (- (random 0.6 state) 0.3))
              (aref y i (if (= c0 c1) 1 0)) 1.0)))))

(defun init-params (seed &key (d 2) (h 8) (c 2))
  "w1 (D H)、b1 (H)、w2 (H C)、b2 (C) を [-0.5, 0.5) の一様乱数で初期化する。"
  (let ((state (sb-ext:seed-random-state (+ seed 1000003))))
    (flet ((uniform (&rest dims)
             (let ((a (make-array dims :element-type 'single-float)))
               (dotimes (i (array-total-size a) a)
                 (setf (row-major-aref a i) (- (random 1.0 state) 0.5))))))
      (list (uniform d h) (uniform h) (uniform h c) (uniform c)))))

(defun train (&key (steps 100) (seed 0) (lr 0.5))
  "STEPS 回の SGD を回し、各ステップの損失のリストを返す。"
  (let ((step (make-mlp-train-step :lr lr))
        (params (init-params seed)))
    (multiple-value-bind (x y) (make-blobs 16 seed)
      (loop repeat steps
            collect (multiple-value-bind (loss new-params) (funcall step params x y)
                      (setf params new-params)
                      loss)))))

(let ((losses (train)))
  (format t "~&loss[0] = ~,4F~%" (first losses))
  (format t "~&final loss = ~,4F~%" (car (last losses))))
