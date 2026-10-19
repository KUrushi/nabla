;;;; 学習ステップ全体（value-and-grad + optimizer-update + apply-updates）を jit する（issue #205）。
;;;;
;;;; 守らせる性質:
;;;; - IREE の local で jit した学習ステップは、数ステップにわたって eager と一致する
;;;;   （損失・パラメータ・オプティマイザの状態）。
;;;; - ステップ数（count）は状態の中の rank 0 の配列で、jit の入力として渡るので、
;;;;   ステップごとに再コンパイルしない（2回目以降の呼び出しで jit のキャッシュに当たる）。
;;;;   スケジュールで学習率が変わっていくので、count が定数に畳み込まれていれば eager と一致しない。
;;;; - 学習が進む（損失が下がる）。
;;;;
;;;; IREE 3.11 の local-task は jit の入力が 162 個以上だと落ちる（iree-too-many-inputs）。
;;;; adam の学習ステップの入力は パラメータ + mu + nu + count + バッチ なので、モデルは小さくする。

(in-package #:nabla.nn.tests)

(defmodule ot-lin ((w :param) (b :param) (batch :static) (width :static))
  (:call (self x)
    (+ (nabla:dot x (ot-lin-w self))
       (nabla:broadcast-in-dim (ot-lin-b self) (list (ot-lin-batch self) (ot-lin-width self)) '(1)))))

(defmodule ot-net ((l1 :param) (l2 :param))
  (:call (self x) (call (ot-net-l2 self) (tanh (call (ot-net-l1 self) x)))))

(defun %ot-net (seed)
  (make-ot-net :l1 (make-ot-lin :w (%rand-array '(2 4) seed 0.5) :b (%rand-array '(4) (+ seed 1) 0.5)
                                :batch 5 :width 4)
               :l2 (make-ot-lin :w (%rand-array '(4 1) (+ seed 2) 0.5) :b (%rand-array '(1) (+ seed 3) 0.5)
                                :batch 5 :width 1)))

(defun %ot-train-step (optimizer)
  "(model opt-state x y) → (values loss new-model new-opt-state)。jit しない（eager でも呼べる）。"
  (let ((loss (nabla:with-tracing (model x y)
                (let ((diff (- (call model x) y)))
                  (nabla:reduce-sum (* diff diff))))))
    (nabla:with-tracing (model opt-state x y)
      (multiple-value-bind (loss-value grads) (funcall (nabla:value-and-grad loss) model x y)
        (multiple-value-bind (updates new-state) (optimizer-update optimizer grads opt-state model)
          (values loss-value (apply-updates model updates) new-state))))))

(test (optimizer/jitted-train-step-matches-eager-on-iree-local :suite :nabla.medium)
  "adam + warmup-cosine-decay + clip-by-global-norm の学習ステップ全体を IREE local で jit すると、
5ステップにわたって eager と損失・パラメータ・状態が一致し、2回目以降はコンパイルしない。"
  (when (%require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (optimizer (chain (clip-by-global-norm 1d0)
                             (adam (warmup-cosine-decay-schedule 0d0 0.05d0 2 8 :end-value 0.01d0))))
           (x (%rand-array '(5 2) 31))
           (y (%rand-array '(5 1) 32))
           (eager-step (%ot-train-step optimizer))
           (jit-step (nabla:jit eager-step :backend backend))
           (eager-model (%ot-net 7))
           (eager-state (optimizer-init optimizer eager-model))
           (jit-model eager-model)
           (jit-state eager-state)
           (misses nil))
      (unwind-protect
           (dotimes (k 5)
             (multiple-value-bind (eager-loss new-eager-model new-eager-state)
                 (funcall eager-step eager-model eager-state x y)
               (multiple-value-bind (jit-loss new-jit-model new-jit-state)
                   (funcall jit-step jit-model jit-state x y)
                 (when (= k 0) (setf misses nabla::*jit-miss-count*))
                 (is (%f32-close jit-loss eager-loss) "ステップ ~D の損失が一致しない" k)
                 (is (typep new-jit-model 'ot-net))
                 (is (nabla:treedef= (nabla:tree-structure new-jit-state) (nabla:tree-structure new-eager-state)))
                 (is (%trees-close (nabla:tree-leaves new-jit-model) (nabla:tree-leaves new-eager-model))
                     "ステップ ~D のパラメータが一致しない" k)
                 (is (%trees-close (remove-if-not (lambda (a) (eq 'single-float (array-element-type a)))
                                                  (nabla:tree-leaves new-jit-state))
                                   (remove-if-not (lambda (a) (eq 'single-float (array-element-type a)))
                                                  (nabla:tree-leaves new-eager-state)))
                     "ステップ ~D の状態が一致しない" k)
                 (setf eager-model new-eager-model eager-state new-eager-state
                       jit-model new-jit-model jit-state new-jit-state))))
        (is (= misses nabla::*jit-miss-count*) "2回目以降のステップで再コンパイルしている")
        (nabla::%jit-cache-forget jit-step)))))

(test (optimizer/training-reduces-the-loss :suite :nabla.small)
  "eager の adam で 30 ステップ学習すると損失が最初の半分以下になる（更新の向きが正しい）。"
  (let* ((optimizer (adam 0.05d0))
         (step (%ot-train-step optimizer))
         (model (%ot-net 11))
         (state (optimizer-init optimizer model))
         (x (%rand-array '(5 2) 41))
         (y (%rand-array '(5 1) 42))
         (first-loss nil) (last-loss nil))
    (dotimes (k 30)
      (multiple-value-bind (loss new-model new-state) (funcall step model state x y)
        (when (= k 0) (setf first-loss (aref loss)))
        (setf last-loss (aref loss) model new-model state new-state)))
    (is (< last-loss (* 0.5 first-loss)) "損失 ~S → ~S" first-loss last-loss)))

(test (optimizer/checkpoint-round-trip-continues-training :suite :nabla.medium)
  "モデルとオプティマイザの状態を1つの木 (list model opt-state) にして save-checkpoint / load-checkpoint
すると、全葉（count の :i32 も）が一致し、復元した木から学習を続けても元の木から続けた場合と同じになる
\(count は続けて増える)。パス名は 0.l1.w（モデル）と 1.0.count / 1.0.mu.l1.w（adam の状態）。"
  (let* ((optimizer (adam 0.05d0))
         (step (%ot-train-step optimizer))
         (x (%rand-array '(5 2) 51))
         (y (%rand-array '(5 1) 52))
         (model (%ot-net 13))
         (state (optimizer-init optimizer model)))
    (dotimes (k 3)
      (multiple-value-bind (loss new-model new-state) (funcall step model state x y)
        (declare (ignore loss))
        (setf model new-model state new-state)))
    (let ((paths (tree-paths (list model state))))
      (is (member "0.l1.w" paths :test #'string=))
      (is (member "1.0.count" paths :test #'string=))
      (is (member "1.0.mu.l1.w" paths :test #'string=))
      (is (member "1.0.nu.l2.b" paths :test #'string=)))
    (nabla.tests.support:with-temporary-directory (dir)
      (let ((file (merge-pathnames "ckpt.safetensors" dir)))
        (save-checkpoint file (list model state))
        (destructuring-bind (loaded-model loaded-state) (load-checkpoint file (list model state))
          (is (every #'equalp (nabla:tree-leaves (list model state))
                     (nabla:tree-leaves (list loaded-model loaded-state))))
          (multiple-value-bind (loss1 model1 state1) (funcall step model state x y)
            (declare (ignore loss1))
            (multiple-value-bind (loss2 model2 state2) (funcall step loaded-model loaded-state x y)
              (declare (ignore loss2))
              (is (every #'equalp (nabla:tree-leaves (list model1 state1))
                         (nabla:tree-leaves (list model2 state2))))
              (is (= 4 (aref (first (nabla:tree-leaves (first state1))))))
              (is (= 4 (aref (first (nabla:tree-leaves (first state2)))))))))))))
