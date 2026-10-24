;;;; apply-updates は :state のスロットを素通しする（issue #260）。
;;;;
;;;; BatchNorm の移動平均のような :state の葉は学習しないので、オプティマイザの更新量（weight decay を含む）を
;;;; 足さない。学習ステップは更新後のモジュール（value-and-grad の :has-aux の aux）に apply-updates する:
;;;;   (multiple-value-bind (loss grads new-model) (funcall vg model x) ... (apply-updates new-model updates))
;;;;
;;;; 守らせる性質:
;;;; - apply-updates の結果の :state の葉は PARAMS の葉そのもの（EQ）で、:param の葉は p + u。
;;;;   木の構造（treedef）は変わらない。どのオプティマイザでも、勾配が :state にも非ゼロでも成り立つ。
;;;; - :state の無い木では、従来どおり全葉に足す。
;;;; - 構造の違う updates は pytree-structure-mismatch。

(in-package #:nabla.nn.tests)

(defmodule os-stateful ((w :param) (stats :state) (tag :static)))
(defmodule os-net ((inp :param) (blocks :param) (stats :state)))

(defun %os-model (seed)
  "os-net（:param と :state の葉と、入れ子の os-stateful）とプレーンな配列をリストに混ぜたモデル。"
  (list (make-os-net :inp (%rand-array '(3 2) seed)
                     :blocks (vector (make-os-stateful :w (%rand-array '(2) (+ seed 1))
                                                       :stats (list (%rand-array '(2) (+ seed 2))
                                                                    (%rand-array '(2) (+ seed 3)))
                                                       :tag :a)
                                     (make-os-stateful :w (%rand-array '(4) (+ seed 4))
                                                       :stats (%rand-array '(4) (+ seed 5))
                                                       :tag :b))
                     :stats (%rand-array '(3) (+ seed 6)))
        (%rand-array '(2 2) (+ seed 7))))

(defun %os-state-leaves (model)
  "MODEL の :state のスロットの葉（tree-leaves の順）。"
  (destructuring-bind (net plain) model
    (declare (ignore plain))
    (append (nabla:tree-leaves (os-stateful-stats (svref (os-net-blocks net) 0)))
            (nabla:tree-leaves (os-stateful-stats (svref (os-net-blocks net) 1)))
            (list (os-net-stats net)))))

(defun %os-param-leaves (model)
  (destructuring-bind (net plain) model
    (list (os-net-inp net)
          (os-stateful-w (svref (os-net-blocks net) 0))
          (os-stateful-w (svref (os-net-blocks net) 1))
          plain)))

(defun %os-sum (a b)
  (let ((c (make-array (array-dimensions a) :element-type 'single-float)))
    (dotimes (i (array-total-size a) c)
      (setf (row-major-aref c i) (+ (row-major-aref a i) (row-major-aref b i))))))

(test (optimizer-state/apply-updates-passes-state-through :suite :nabla.small)
  "どのオプティマイザでも、勾配が :state の葉にも非ゼロでも、apply-updates の結果の :state の葉は
PARAMS の葉そのもの（EQ）で、:param の葉は p + u（f32 の許容誤差）。treedef は変わらない。"
  (%pbt (optimizer-state/apply-updates-passes-state-through
         (seed index) (generator (tuple (uniform-integer :lo 0 :hi 10000) (uniform-integer :lo 0 :hi 6))))
    (let* ((optimizer (%generic-optimizer index))
           (model (%os-model seed))
           (grads (%os-model (+ seed 100)))
           (state (optimizer-init optimizer model)))
      (multiple-value-bind (updates new-state) (optimizer-update optimizer grads state model)
        (declare (ignore new-state))
        (let ((new-model (apply-updates model updates)))
          (and (nabla:treedef= (nabla:tree-structure new-model) (nabla:tree-structure model))
               (every #'eq (%os-state-leaves new-model) (%os-state-leaves model))
               (every #'%f32-close (%os-param-leaves new-model)
                      (mapcar #'%os-sum (%os-param-leaves model) (%os-param-leaves updates)))))))))

(test (optimizer-state/apply-updates-without-state-adds-every-leaf :suite :nabla.small)
  ":state の無い木（配列のリスト）では、全葉が p + u になる。"
  (%pbt (optimizer-state/apply-updates-without-state
         (seed) (generator (tuple (uniform-integer :lo 0 :hi 10000))))
    (let* ((params (%rand-tree seed))
           (updates (%rand-tree (+ seed 9))))
      (every #'%f32-close (apply-updates params updates) (mapcar #'%os-sum params updates)))))

(test (optimizer-state/apply-updates-rejects-mismatched-trees :suite :nabla.small)
  "updates の構造が params と違えば pytree-structure-mismatch（:state の部分木の違いも）。"
  (let ((model (%os-model 1)))
    (signals nabla:pytree-structure-mismatch (apply-updates model (%rand-tree 2)))
    (signals nabla:pytree-structure-mismatch
      (apply-updates model (list (first (%os-model 3)) (%rand-array '(2 2) 4) (%rand-array '(1) 5))))
    (signals nabla:pytree-structure-mismatch
      (apply-updates (first model)
                     (make-os-net :inp (%rand-array '(3 2) 1)
                                  :blocks (vector)
                                  :stats (%rand-array '(3) 2))))))
