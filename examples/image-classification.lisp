(require :asdf)
(asdf:load-system "nabla/nn")
(asdf:load-system "nabla/data")
(asdf:load-system "nabla/iree")
(defpackage #:nabla-example-image-classification
  (:use #:cl #:nabla.nn))
(in-package #:nabla-example-image-classification)

;;;; 小さな CNN で画像分類を学習する（issue #287、フェーズ5の完了条件）。データは nabla/data の
;;;; ローダから取る:
;;;;
;;;;   合成画像を write-npy で .npy に書く → npy-source → data-loader（シャッフル、u8 を f32 に直す
;;;;   map、左右反転とパディング付きの切り抜きの random-map）→ バッチを jit した学習ステップへ
;;;;
;;;; データセット: 16x16 のグレースケール（u8、(N 16 16 1)）の 4 クラス。0 縦線、1 横線、2 斜線
;;;; （右下がりか右上がり）、3 十字（縦線と横線）。背景は 0..63 の雑音、線は 192..255。左右の反転と
;;;; 2 画素までの平行移動ではクラスが変わらないように作る（斜線は両方の向きを同じクラスにする）。
;;;; ネットワークに頼らないように、実行時に決定的に作る（ラベルは i32 の (N)）。
;;;;
;;;; モデル: conv 3x3 stride 2（1 → 8）+ relu、conv 3x3 stride 2（8 → 16）+ relu（16x16 → 8x8 → 4x4）、
;;;; 空間の平均、dense（16 → 4）。損失は softmax 交差エントロピー、オプティマイザは Adam。
;;;; jit する学習ステップの入力の葉は パラメータ 6 + mu 6 + nu 6 + count 1 + バッチ 2 = 21 個で、
;;;; IREE の local-task の上限（160）に収まる。
;;;;
;;;; 再開: 「モデル + オプティマイザの状態 + ローダの状態（iterator-state）」を1つのリストにして
;;;; save-checkpoint し、load-checkpoint で復元した続きは、中断しなかった学習とビット単位で同じになる
;;;; （バッチの列はローダのシードと位置だけで決まり、ワーカー数にも依らない）。

(defparameter *image-size* 16 "画像の一辺の画素数。")
(defparameter *classes* 4 "クラスの数。")
(defparameter *padding* 2 "切り抜きの拡張でまわりに足す 0 の画素数（平行移動は ±*PADDING*）。")

;;; --- データ ---

(defun make-images (count seed)
  "COUNT 枚の合成画像を (values 画像 ラベル) で返す。画像は (COUNT 16 16 1) の (unsigned-byte 8)、
ラベルは (COUNT) の (signed-byte 32) で、i 枚目のクラスは i mod 4（クラスの数を揃える）。SEED で再現できる。"
  (let* ((state (sb-ext:seed-random-state seed))
         (n *image-size*)
         (images (make-array (list count n n 1) :element-type '(unsigned-byte 8)))
         (classes (make-array count :element-type '(signed-byte 32))))
    (dotimes (i count (values images classes))
      (let ((class (mod i *classes*))
            ;; 線の位置は端から 3 画素以上離す（±2 の平行移動で線が画像の外に出ない）
            (a (+ 3 (random (- n 6) state)))
            (b (+ 3 (random (- n 6) state)))
            (offset (- (random 7 state) 3))
            (anti (zerop (random 2 state))))
        (setf (aref classes i) class)
        (dotimes (y n)
          (dotimes (x n)
            (setf (aref images i y x 0) (random 64 state))))
        (flet ((ink (y x)
                 (when (and (< -1 y n) (< -1 x n))
                   (setf (aref images i y x 0) (+ 192 (random 64 state))))))
          (dotimes (k n)
            (ecase class
              (0 (ink k a))
              (1 (ink a k))
              (2 (ink k (if anti (- n 1 (+ k offset)) (+ k offset))))
              (3 (ink k a) (ink b k)))))))))

(defun write-dataset (directory &key (count 256) (seed 0))
  "COUNT 枚の合成画像（MAKE-IMAGES）を DIRECTORY に images.npy・labels.npy で書き、NPY-SOURCE に渡す
パス名のリスト (画像 ラベル) を返す。"
  (multiple-value-bind (images classes) (make-images count seed)
    (list (nabla.data:write-npy (merge-pathnames "images.npy" directory) images)
          (nabla.data:write-npy (merge-pathnames "labels.npy" directory) classes))))

(defun to-float-record (record)
  "レコード (画像 ラベル)（u8 の (16 16 1) と rank 0 の i32）を (画像 正解) にする。画像は 255 で割った
f32、正解はクラスの one-hot（f32 の (4)）。u8 は nabla の dtype に無いので jit の前に直す。"
  (destructuring-bind (image class) record
    (let ((x (make-array (array-dimensions image) :element-type 'single-float))
          (y (make-array *classes* :element-type 'single-float :initial-element 0.0)))
      (dotimes (i (array-total-size image))
        (setf (row-major-aref x i) (/ (row-major-aref image i) 255.0)))
      (setf (aref y (aref class)) 1.0)
      (list x y))))

(defun %flip-left-right (image)
  "(H W C) の IMAGE を左右に反転した新しい配列。"
  (destructuring-bind (h w c) (array-dimensions image)
    (let ((out (make-array (list h w c) :element-type (array-element-type image))))
      (dotimes (i h out)
        (dotimes (j w)
          (dotimes (k c)
            (setf (aref out i j k) (aref image i (- w 1 j) k))))))))

(defun %crop-padded (image padding dy dx)
  "(H W C) の IMAGE のまわりに PADDING 個の 0 を足した画像から、(DY, DX) を左上とする (H W C) の窓。"
  (destructuring-bind (h w c) (array-dimensions image)
    (let ((out (make-array (list h w c) :element-type (array-element-type image)
                                           :initial-element (coerce 0 (array-element-type image)))))
      (dotimes (i h out)
        (dotimes (j w)
          (let ((si (- (+ i dy) padding))
                (sj (- (+ j dx) padding)))
            (when (and (< -1 si h) (< -1 sj w))
              (dotimes (k c)
                (setf (aref out i j k) (aref image si sj k))))))))))

(defun augment (record key)
  "RANDOM-MAP-TRANSFORM の関数: レコード (画像 正解) の画像を確率 1/2 で左右に反転し、*PADDING* 個の
0 で囲んでランダムな位置で元の大きさに切り抜く。乱数は位置ごとに決まるキー KEY から NABLA:UNIFORM で
引くので、ワーカー数や再開に依らない。"
  (destructuring-bind (image target) record
    (let* ((u (nabla:uniform key '(3) :dtype :f64))
           (span (1+ (* 2 *padding*)))
           ;; uniform は丸めで 1 ちょうどを返しうるので、オフセットは 2 * *PADDING* で止める
           (dy (min (* 2 *padding*) (floor (* (aref u 1) span))))
           (dx (min (* 2 *padding*) (floor (* (aref u 2) span)))))
      (list (%crop-padded (if (< (aref u 0) 0.5d0) (%flip-left-right image) image) *padding* dy dx)
            target))))

(defun make-loader (paths &key (batch-size 32) (seed 0) num-epochs (augment t))
  "PATHS（WRITE-DATASET の戻り値）の学習用のローダ。毎エポックをシード SEED でシャッフルし、
TO-FLOAT-RECORD と（AUGMENT が真なら）AUGMENT をかけ、BATCH-SIZE 枚ずつ (画像 (B 16 16 1) 正解 (B 4))
に積む。端数は捨てる（バッチの形を固定して jit の再コンパイルを避ける）。NUM-EPOCHS が nil なら終わらない。"
  (nabla.data:data-loader (nabla.data:npy-source paths)
                          :batch-size batch-size :shuffle t :seed (nabla:prng-key seed)
                          :num-epochs num-epochs
                          :transforms (list* (nabla.data:map-transform #'to-float-record)
                                             (and augment
                                                  (list (nabla.data:random-map-transform #'augment))))))

;;; --- モデル ---

(declaim (ftype function %cnn-apply)) ; 下で定義（defmodule の :call から呼ぶ）

(defmodule cnn (conv1 conv2 head)
  (:init (key classes)
   (destructuring-bind (conv1-key conv2-key head-key) (split-keys key 3)
     (make-cnn :conv1 (init-conv conv1-key 1 8 '(3 3) :strides 2)
               :conv2 (init-conv conv2-key 8 16 '(3 3) :strides 2)
               :head (init-dense head-key 16 classes))))
  (:call (self x)
   (%cnn-apply self x)))

(defun %spatial-mean (h)
  "(B H W C) の H を空間の軸 (1 2) で平均した (B C)。"
  (destructuring-bind (b height width c) (nabla:aval-shape (nabla:value-aval h))
    (declare (ignore b c))
    (funcall (load-time-value
              (nabla:with-tracing (h count) (/ (nabla:reduce-sum h :axes '(1 2)) count)))
             h (* height width))))

(defun %cnn-apply (model x)
  (let* ((h (relu (call (cnn-conv1 model) x)))
         (h (relu (call (cnn-conv2 model) h))))
    (call (cnn-head model) (%spatial-mean h))))

(defun make-example-cnn (&key (seed 0))
  "例のモデル（パラメータの葉 6 個: conv1・conv2・head の kernel と bias）。"
  (init-cnn (nabla:prng-key seed) *classes*))

;;; --- 損失と学習 ---

(defun cnn-loss (model x y)
  "交差エントロピー: -sum(Y * log-softmax(logits)) / sum(Y)（Y は one-hot なのでバッチの平均）。"
  (funcall (load-time-value
            (nabla:with-tracing (model x y)
              (- (/ (nabla:reduce-sum (* y (log-softmax (call model x)))) (nabla:reduce-sum y)))))
           model x y))

(defun make-optimizer (&key (learning-rate 3d-2))
  (adam learning-rate))

(defun make-train-step (optimizer &key (jit t))
  "学習ステップ (model opt-state x y) → (values 損失 新しい model 新しい opt-state)。JIT が真（既定）なら
nabla:jit したもの（ループの外で1回だけ作る）、nil なら同じ計算を eager で行う関数。"
  (let* ((value-and-grad (nabla:value-and-grad
                          (nabla:with-tracing (model x y) (cnn-loss model x y))))
         (step (nabla:with-tracing (model opt-state x y)
                 (multiple-value-bind (loss grads) (funcall value-and-grad model x y)
                   (multiple-value-bind (updates new-state) (optimizer-update optimizer grads opt-state model)
                     (values loss (apply-updates model updates) new-state))))))
    (if jit (nabla:jit step) step)))

(defun run-steps (train-step loader model opt-state steps &key state (num-workers 0))
  "LOADER のバッチで STEPS 回（列が先に終わればそこまで）学習する。STATE（ITERATOR-STATE か nil）が
あればその位置から、無ければ先頭からバッチを取り、NUM-WORKERS 個のワーカーで読む。
\(values 損失（single-float）のリスト 最後の model 最後の opt-state 続きのローダの状態) を返す。"
  (nabla.data:with-data-iterator (iterator loader :state state :num-workers num-workers)
    (let ((losses '()))
      (loop repeat steps
            do (multiple-value-bind (batch more) (nabla.data:next-batch iterator)
                 (unless more (loop-finish))
                 (destructuring-bind (x y) batch
                   (multiple-value-bind (loss new-model new-state) (funcall train-step model opt-state x y)
                     (push (aref loss) losses)
                     (setf model new-model opt-state new-state)))))
      (values (nreverse losses) model opt-state (nabla.data:iterator-state iterator)))))

(defun save-training (path model opt-state loader-state)
  "モデル・オプティマイザの状態・ローダの状態を1つの木 (model opt-state loader-state) で PATH に保存する。"
  (save-checkpoint path (list model opt-state loader-state)))

(defun load-training (path model opt-state loader)
  "SAVE-TRAINING のファイルを読み、(values model opt-state ローダの状態) を返す。MODEL と OPT-STATE は
ひな形（形と dtype の照合に使う）、ローダの状態のひな形は LOADER の先頭の状態。"
  (let ((template-state (nabla.data:with-data-iterator (iterator loader)
                          (nabla.data:iterator-state iterator))))
    (destructuring-bind (model opt-state loader-state)
        (load-checkpoint path (list model opt-state template-state))
      (values model opt-state loader-state))))

(defun accuracy (model paths &key (batch-size 64))
  "PATHS のデータセット全体（拡張なし）での MODEL の正解率（0 から 1 の有理数）。eager で計算する。"
  (let ((loader (nabla.data:data-loader (nabla.data:npy-source paths)
                                        :batch-size batch-size :num-epochs 1 :drop-remainder nil
                                        :transforms (list (nabla.data:map-transform #'to-float-record))))
        (correct 0)
        (total 0))
    (dolist (batch (nabla.data:epoch loader 0) (/ correct total))
      (destructuring-bind (x y) batch
        (let ((logits (call model x)))
          (dotimes (i (array-dimension x 0))
            (flet ((argmax (array)
                     (let ((best 0))
                       (dotimes (k *classes* best)
                         (when (> (aref array i k) (aref array i best)) (setf best k))))))
              (incf total)
              (when (= (argmax logits) (argmax y)) (incf correct)))))))))

(defun call-with-temporary-directory (function)
  "プロセスごとに別の一時ディレクトリを作って FUNCTION に渡し、終わったら消す。名前が既にあるディレクトリと
重なったら別の名前にする（自分で作ったディレクトリだけを消す）。"
  (let ((directory
          (loop with state = (make-random-state t)
                for candidate = (uiop:ensure-directory-pathname
                                 (merge-pathnames (format nil "nabla-image-classification-~D/"
                                                          (random most-positive-fixnum state))
                                                  (uiop:temporary-directory)))
                when (nth-value 1 (ensure-directories-exist candidate))
                  return candidate)))
    (unwind-protect (funcall function directory)
      (uiop:delete-directory-tree directory :validate t :if-does-not-exist :ignore))))

(defun main (&key (steps 64) (num-workers 2))
  "データセットを作り、STEPS（2 以上）回の半分で保存・復元をはさんで学習して、損失と正解率を出す。
\(values 損失のリスト 学習後の正解率 最後の model 最後の opt-state) を返す。"
  (unless (and (integerp steps) (>= steps 2))
    (error "main: steps は 2 以上の整数でなければならない: ~S" steps))
  (call-with-temporary-directory
   (lambda (directory)
     (let* ((paths (write-dataset directory))
            (loader (make-loader paths))
            (optimizer (make-optimizer))
            (train-step (make-train-step optimizer))
            (model (make-example-cnn))
            (opt-state (optimizer-init optimizer model))
            (half (floor steps 2))
            (checkpoint (merge-pathnames "training.safetensors" directory)))
       (format t "~&accuracy before = ~,3F~%" (accuracy model paths))
       (multiple-value-bind (losses-1 model opt-state loader-state)
           (run-steps train-step loader model opt-state half :num-workers num-workers)
         (save-training checkpoint model opt-state loader-state)
         (multiple-value-bind (model opt-state loader-state) (load-training checkpoint model opt-state loader)
           (multiple-value-bind (losses-2 model opt-state)
               (run-steps train-step loader model opt-state (- steps half)
                          :state loader-state :num-workers num-workers)
             (format t "~&loss[0] = ~,4F~%" (first losses-1))
             (format t "~&loss[~D]（保存の直前）= ~,4F~%" (1- half) (car (last losses-1)))
             (format t "~&final loss = ~,4F~%" (car (last losses-2)))
             (let ((accuracy (accuracy model paths)))
               (format t "~&accuracy after = ~,3F~%" accuracy)
               (values (append losses-1 losses-2) accuracy model opt-state)))))))))

;; テストが関数だけを使うために、:nabla-no-demo を *features* に足して load すると学習は走らない。
(unless (member :nabla-no-demo *features*)
  (main))
