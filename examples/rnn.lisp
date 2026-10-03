(require :asdf)
(asdf:load-system "nabla/iree")
(defpackage #:nabla-example-rnn (:use #:cl))
(in-package #:nabla-example-rnn)

;;; Elman RNN を scan で書き、手書きのループで学習する例（issue #141）。
;;;
;;;   h' = tanh(h W_h + x_t W_x + b)      h: (B H)  x_t: (B D)   ※ バッチは行列積の行方向で扱う
;;;   out = h_T W_o + b_o                 最後の隠れ状態から線形層 (B O)
;;;   loss = mean((out - y)^2)
;;;
;;; 系列 xs は (T B D) で、scan が先頭の軸（時間）に沿って回す。バッチ軸は VMAP ではなく
;;; dot の行方向で持つ（JAX フィクスチャ tests/fixtures/rnn/generate.py と同じ式）。
;;; パラメータ (wh wx b wo bo) は scan 本体が閉包で捕まえる（ループ不変な consts になる）。

(defparameter +t+ 8)
(defparameter +b+ 4)
(defparameter +d+ 4)
(defparameter +h+ 8)
(defparameter +o+ 2)

(defun make-rnn-loss (&key (steps +t+) (batch +b+) (hidden +h+) (out +o+))
  "損失 (wh wx b wo bo xs y) → スカラー。xs: (STEPS BATCH D)、y: (BATCH OUT)。"
  (let ((scale (/ 1.0 (* batch out))))
    (nb:with-tracing (wh wx b wo bo xs y)
      (let* ((bias (nb:broadcast-in-dim b (list batch hidden) '(1)))
             (h0 (make-array (list batch hidden) :element-type 'single-float :initial-element 0.0))
             (final (first (nb:scan (nb:with-tracing (carry x)
                                      (values (list (tanh (+ (+ (nb:dot (first carry) wh)
                                                                (nb:dot (first x) wx))
                                                             bias)))
                                              '()))
                                    (list h0) (list xs) :length steps)))
             (pred (+ (nb:dot final wo) (nb:broadcast-in-dim bo (list batch out) '(1))))
             (diff (- pred y)))
        (* (nb:reduce-sum (* diff diff)) scale)))))

(defun sgd-update (param grad lr)
  "PARAM - LR * GRAD（f32 の配列）を新しい配列で返す。"
  (let ((new (make-array (array-dimensions param) :element-type 'single-float)))
    (dotimes (i (array-total-size param) new)
      (setf (row-major-aref new i)
            (- (row-major-aref param i) (* lr (row-major-aref grad i)))))))

(defun make-rnn-grad-fn (&key backend)
  "jit した (wh wx b wo bo xs y) → (損失 勾配...) の多値（勾配は wh wx b wo bo の順）。"
  (let ((vg (nb:value-and-grad (make-rnn-loss) :argnums '(0 1 2 3 4))))
    (nb:jit (nb:with-tracing (wh wx b wo bo xs y)
              (multiple-value-bind (loss grads) (funcall vg wh wx b wo bo xs y)
                (values-list (cons loss grads))))
            :backend backend)))

(defun make-rnn-train-step (&key (lr 0.1) backend)
  "学習ステップ (lambda (params xs y)) を返す。PARAMS は (wh wx b wo bo) の f32 配列のリスト。
戻り値は2つ: その関数と、jit した関数そのもの。関数は多値の (更新前の損失 更新後の PARAMS)。
jit は呼び出しごとではなくここで1回だけ作る（2回目以降はキャッシュを使う）。"
  (let ((jitted (make-rnn-grad-fn :backend backend)))
    (values
     (lambda (params xs y)
       (destructuring-bind (loss &rest grads) (multiple-value-list (apply jitted (append params (list xs y))))
         (values (aref loss)
                 (mapcar (lambda (p g) (sgd-update p g lr)) params grads))))
     jitted)))

(defun uniform-array (state scale &rest dims)
  (let ((a (make-array dims :element-type 'single-float)))
    (dotimes (i (array-total-size a) a)
      (setf (row-major-aref a i) (* scale (- (random 1.0 state) 0.5))))))

(defun init-params (seed)
  "wh (H H)、wx (D H)、b (H)、wo (H O)、bo (O) を一様乱数で初期化する。"
  (let ((state (sb-ext:seed-random-state (+ seed 1000003))))
    (list (uniform-array state 0.6 +h+ +h+) (uniform-array state 1.0 +d+ +h+) (uniform-array state 0.2 +h+)
          (uniform-array state 1.0 +h+ +o+) (uniform-array state 0.2 +o+))))

(defun make-sequences (seed)
  "系列 xs (T B D) と目標 y (B O)。y は系列の時間平均の第0・第1成分。"
  (let* ((state (sb-ext:seed-random-state seed))
         (xs (uniform-array state 2.0 +t+ +b+ +d+))
         (y (make-array (list +b+ +o+) :element-type 'single-float :initial-element 0.0)))
    (dotimes (i +b+)
      (dotimes (k +o+)
        (setf (aref y i k) (/ (loop for tt below +t+ sum (aref xs tt i k)) +t+))))
    (values xs y)))

(defun train (&key (steps 60) (seed 0) (lr 0.3))
  "STEPS 回の SGD を回し、各ステップの損失のリストを返す。"
  (let ((step (make-rnn-train-step :lr lr))
        (params (init-params seed)))
    (multiple-value-bind (xs y) (make-sequences seed)
      (loop repeat steps
            collect (multiple-value-bind (loss new-params) (funcall step params xs y)
                      (setf params new-params)
                      loss)))))

(let ((losses (train)))
  (format t "~&loss[0] = ~,4F~%" (first losses))
  (format t "~&final loss = ~,3E~%" (car (last losses))))
