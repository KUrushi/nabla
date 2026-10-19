(require :asdf)
(asdf:load-system "nabla/nn")
(asdf:load-system "nabla/iree")
(defpackage #:nabla-example-tutorial
  (:use #:cl #:nabla.nn))
(in-package #:nabla-example-tutorial)

;;;; README のチュートリアル（issue #210）。defmodule でモデルを書き、init、jit した学習ステップ、
;;;; オプティマイザ、チェックポイントの保存と復元までを1本の流れで書く。
;;;;
;;;; README のコードブロックは、下の「;;;; [[tutorial:名前]]」と「;;;; [[/tutorial]]」で挟んだ行と
;;;; 一字一句同じでなければならない（tests/nn/tutorial-test.lisp が検査する）。README を直すときは
;;;; このファイルを先に直して、ブロックを貼り直す。

;;;; [[tutorial:model]]
;; 2層 MLP の分類器。dense は nabla.nn の層、(key in width classes) の key は PRNG キー。
(defmodule mlp (hidden out)
  (:init (key in width classes)
   (destructuring-bind (hidden-key out-key) (split-keys key 2)
     (make-mlp :hidden (init-dense hidden-key in width)
               :out (init-dense out-key width classes))))
  (:call (self x)
   (call (mlp-out self) (tanh (call (mlp-hidden self) x)))))
;;;; [[/tutorial]]

;;;; [[tutorial:data]]
;; XOR 風の2クラス分類。x は [-1, 1) の2次元の点で、x0 * x1 > 0 ならクラス 1、そうでなければクラス 0。
;; y はクラスの one-hot（f32 の (batch 2)）。ステップ番号 step から決まるので、何度呼んでも同じ。
(defun make-batch (step &key (batch-size 32))
  (let* ((key (nabla:fold-in (nabla:prng-key 100) step))
         (x (nabla:uniform key (list batch-size 2) :minval -1 :maxval 1))
         (y (make-array (list batch-size 2) :element-type 'single-float :initial-element 0.0)))
    (dotimes (i batch-size)
      (setf (aref y i (if (> (* (aref x i 0) (aref x i 1)) 0) 1 0)) 1.0))
    (values x y)))
;;;; [[/tutorial]]

;;;; [[tutorial:train-step]]
;; 損失は交差エントロピー。サンプル数は one-hot の総和 (reduce-sum y) で割る。
;; value-and-grad の引数 model が木（defmodule）でも、grad は同じ構造の木で返る。
;; 学習ステップ (model opt-state x y) → (values 損失 新しい model 新しい opt-state) は jit する。
;; jit はループの外で1回だけ作る（2回目以降の呼び出しはコンパイルしない）。
(defun make-optimizer ()
  (chain (clip-by-global-norm 1.0) (adam 0.05)))

(defun make-train-step (optimizer)
  (let ((value-and-grad
          (nabla:value-and-grad
           (nabla:with-tracing (model x y)
             (- (/ (nabla:reduce-sum (* y (log-softmax (call model x))))
                   (nabla:reduce-sum y)))))))
    (nabla:jit (nabla:with-tracing (model opt-state x y)
                 (multiple-value-bind (loss grads) (funcall value-and-grad model x y)
                   (multiple-value-bind (updates new-state)
                       (optimizer-update optimizer grads opt-state model)
                     (values loss (apply-updates model updates) new-state)))))))
;;;; [[/tutorial]]

;;;; [[tutorial:loop]]
;; 手書きのループ。FROM から TO の手前までのステップを回し、損失のリストと最後の model / opt-state を返す。
(defun run-steps (train-step model opt-state from to)
  (let ((losses '()))
    (loop for step from from below to
          do (multiple-value-bind (x y) (make-batch step)
               (multiple-value-bind (loss new-model new-state) (funcall train-step model opt-state x y)
                 (push (aref loss) losses)
                 (setf model new-model opt-state new-state))))
    (values (nreverse losses) model opt-state)))
;;;; [[/tutorial]]

;;;; [[tutorial:checkpoint]]
;; model と opt-state を1つのリストにして1ファイル（safetensors）に保存する。
;; 復元には同じ構造の「ひな形」が要る（葉の名前・形・dtype を照合する）。
(defun save-training (path model opt-state)
  (save-checkpoint path (list model opt-state)))

(defun load-training (path template-model template-opt-state)
  (destructuring-bind (model opt-state) (load-checkpoint path (list template-model template-opt-state))
    (values model opt-state)))
;;;; [[/tutorial]]

;;;; [[tutorial:per-example]]
;; vmap でサンプルごとの勾配。パラメータはバッチせず (in-axes が nil)、x と y だけを軸 0 でバッチする。
;; 戻り値は model と同じ構造の木で、各葉に先頭のバッチ軸がつく。
(defun per-example-grads (model x y)
  (funcall (nabla:vmap (nabla:grad (nabla:with-tracing (model x y)
                                      (- (nabla:reduce-sum (* y (log-softmax (call model x)))))))
                       :in-axes '(nil 0 0))
           model x y))
;;;; [[/tutorial]]

(defun main ()
  (let* ((optimizer (make-optimizer))
         (train-step (make-train-step optimizer))
         (model (init-mlp (nabla:prng-key 0) 2 16 2))
         (opt-state (optimizer-init optimizer model)))
    (multiple-value-bind (losses-1 model opt-state) (run-steps train-step model opt-state 0 30)
      (format t "~&loss[0] = ~,4F~%loss[29] = ~,4F~%" (first losses-1) (car (last losses-1)))
      (let ((path (merge-pathnames "nabla-tutorial.safetensors" (uiop:temporary-directory))))
        (save-training path model opt-state)
        (multiple-value-bind (model opt-state) (load-training path model opt-state)
          (let ((losses-2 (run-steps train-step model opt-state 30 60)))
            (format t "~&復元して続けた loss[59] = ~,4F~%" (car (last losses-2)))))
        (uiop:delete-file-if-exists path)))))

;; テストが関数だけを使うために、:nabla-no-demo を *features* に足して load すると学習は走らない。
(unless (member :nabla-no-demo *features*)
  (main))
