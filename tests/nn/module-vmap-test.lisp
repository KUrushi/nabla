;;;; defmodule のモデルを vmap に渡す（issue #200 と #202 の完了条件）。
;;;;
;;;; 守らせる性質:
;;;; - モデルをバッチせず（in-axes nil）データだけバッチした (vmap call) は、サンプルごとに
;;;;   call を適用して積み直したものと一致する。モデルまるごとをバッチした場合
;;;;   （in-axes の接頭辞が 0）も、モデルと同じ構造で葉ごとに軸を書いた in-axes も同じ。
;;;; - per-example 勾配 (vmap (grad loss) :in-axes '(nil 0 0)) は、モデルと同じクラス・
;;;;   スロットのモジュールで、各スロットがサンプルごとの勾配を軸 0 に積んだもの。
;;;; - その per-example 勾配を IREE の local で jit した結果は eager と一致する（medium）。

(in-package #:nabla.nn.tests)

(defmodule vm-lin ((w :param) (b :param))
  (:call (self x) (+ (* (vm-lin-w self) x) (vm-lin-b self))))

(defmodule vm-net ((inp :param) (out :param))
  (:call (self x) (call (vm-net-out self) (call (vm-net-inp self) x))))

(defun %vm-array (shape seed)
  (nabla.tests.support:make-random-array (nabla.tests.support:make-array-spec shape :f64) :seed seed))

(defun %vm-net (seed)
  (make-vm-net :inp (make-vm-lin :w (%vm-array '(3) seed) :b (%vm-array '(3) (+ seed 1)))
               :out (make-vm-lin :w (%vm-array '(3) (+ seed 2)) :b (%vm-array '(3) (+ seed 3)))))

(defparameter *vm-call* (nabla:with-tracing (model x) (call model x)))

(defparameter *vm-loss*
  (nabla:with-tracing (model x y) (nabla:reduce-sum (* (call model x) y))))

(defun %vm-slices (array count)
  (loop for i below count collect (nabla.tests.support:slice-along-axis array 0 i)))

(defun %vm-stack (arrays)
  (nabla.tests.support:stack-along-axis arrays 0))

(defun %vm-close (a b)
  (nabla.tests.support:allclose a b :dtype :f64))

(defun %vm-trees-close (actual expected)
  (and (nabla:treedef= (nabla:tree-structure actual) (nabla:tree-structure expected))
       (every #'%vm-close (nabla:tree-leaves actual) (nabla:tree-leaves expected))))

(defmacro def-module-vmap-property (name docstring (&rest vars) &body body)
  "(seed batch) を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test (,name :suite :nabla.small)
     ,docstring
     (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000)
                                     (uniform-integer :lo 1 :hi 3)))
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (nabla.tests.support:regression-path
                                     ,(substitute #\- #\/ (string-downcase (string name)))
                                     :package "NABLA.NN.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

(def-module-vmap-property module-vmap/unbatched-model-and-batched-data-match-per-example-calls
  "in-axes (nil 0): モデルは全サンプルで共有、x だけをバッチした vmap の結果は、サンプルごとの
call を積み直したものと一致する。"
  (seed batch)
  (let* ((net (%vm-net seed))
         (xs (%vm-array (list batch 3) (+ seed 10)))
         (expected (%vm-stack (mapcar (lambda (x) (funcall *vm-call* net x)) (%vm-slices xs batch)))))
    (%vm-close (funcall (nabla:vmap *vm-call* :in-axes '(nil 0)) net xs) expected)))

(def-module-vmap-property module-vmap/batched-model-prefix-and-slotwise-axes-agree
  "モデルの葉がすべて先頭にバッチ軸を持つとき、in-axes (0 0)（モデルまるごとの接頭辞）と、モデルと
同じ構造で葉ごとに 0 を書いた in-axes は同じ結果で、サンプルごとに (モデルの切り出し, x) へ call
を適用して積み直したものと一致する。"
  (seed batch)
  (let* ((nets (loop for i below batch collect (%vm-net (+ seed (* 10 i)))))
         (batched-net (nabla:tree-unflatten
                       (nabla:tree-structure (first nets))
                       (apply #'mapcar (lambda (&rest leaves) (%vm-stack leaves))
                              (mapcar #'nabla:tree-leaves nets))))
         (xs (%vm-array (list batch 3) (+ seed 10)))
         (expected (%vm-stack (loop for net in nets for x in (%vm-slices xs batch)
                                    collect (funcall *vm-call* net x))))
         (slotwise (nabla:tree-map (constantly 0) batched-net)))
    (and (%vm-close (funcall (nabla:vmap *vm-call* :in-axes '(0 0)) batched-net xs) expected)
         (%vm-close (funcall (nabla:vmap *vm-call* :in-axes (list slotwise 0)) batched-net xs) expected))))

(def-module-vmap-property module-vmap/per-example-gradients-are-a-batched-module
  "(vmap (grad loss) :in-axes '(nil 0 0)): 結果はモデルと同じ treedef のモジュールで、各スロットが
サンプルごとの (grad loss) を軸 0 に積んだものと一致する。"
  (seed batch)
  (let* ((net (%vm-net seed))
         (xs (%vm-array (list batch 3) (+ seed 10)))
         (ys (%vm-array (list batch 3) (+ seed 20)))
         (g (nabla:grad *vm-loss*))
         (result (funcall (nabla:vmap g :in-axes '(nil 0 0)) net xs ys))
         (per-example (loop for x in (%vm-slices xs batch) for y in (%vm-slices ys batch)
                            collect (funcall g net x y)))
         (expected (nabla:tree-unflatten
                    (nabla:tree-structure net)
                    (apply #'mapcar (lambda (&rest leaves) (%vm-stack leaves))
                           (mapcar #'nabla:tree-leaves per-example)))))
    (and (typep result 'vm-net)
         (typep (vm-net-inp result) 'vm-lin)
         (%vm-trees-close result expected))))

(test (module-vmap/model-axes-must-match-the-module-structure :suite :nabla.small)
  "モデルの接頭辞に別のクラスや別の構造を渡すと vmap-error。"
  (let ((net (%vm-net 1)) (x (%vm-array '(2 3) 5)))
    (signals nabla:vmap-error
      (funcall (nabla:vmap *vm-call* :in-axes (list (make-vm-lin :w 0 :b 0) 0)) net x))
    (signals nabla:vmap-error
      (funcall (nabla:vmap *vm-call* :in-axes '((0 0) 0)) net x))))

(test (module-vmap/jit-of-per-example-gradients-matches-eager-on-iree-local :suite :nabla.medium)
  "(jit (vmap (grad loss) :in-axes '(nil 0 0))) を IREE の local で実行した結果は、jit しない結果と
同じ treedef のモジュールで、f32 の許容誤差内で葉ごとに一致する。モデルをバッチしたまま
(jit (vmap call)) も eager と一致する。"
  (when (%require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (counter 0)
           (net (nabla:tree-map (lambda (a)
                                  (declare (ignore a))
                                  (nabla.tests.support:make-random-array
                                   (nabla.tests.support:make-array-spec '(3) :f32)
                                   :seed (incf counter)))
                                (%vm-net 3)))
           (xs (nabla.tests.support:make-random-array (nabla.tests.support:make-array-spec '(2 3) :f32) :seed 31))
           (ys (nabla.tests.support:make-random-array (nabla.tests.support:make-array-spec '(2 3) :f32) :seed 32))
           (per-example (nabla:vmap (nabla:grad *vm-loss*) :in-axes '(nil 0 0)))
           (batched-call (nabla:vmap *vm-call* :in-axes '(nil 0)))
           (jp (nabla:jit per-example :backend backend))
           (jc (nabla:jit batched-call :backend backend)))
      (unwind-protect
           (let ((jit-result (funcall jp net xs ys))
                 (eager-result (funcall per-example net xs ys)))
             (is (typep jit-result 'vm-net))
             (is (nabla:treedef= (nabla:tree-structure jit-result) (nabla:tree-structure eager-result)))
             (is (every (lambda (a e) (nabla.tests.support:allclose a e :dtype :f32))
                        (nabla:tree-leaves jit-result) (nabla:tree-leaves eager-result)))
             (is (nabla.tests.support:allclose (funcall jc net xs) (funcall batched-call net xs) :dtype :f32)))
        (nabla::%jit-cache-forget (nabla::%jitted-function-fn jp))
        (nabla::%jit-cache-forget (nabla::%jitted-function-fn jc))))))
