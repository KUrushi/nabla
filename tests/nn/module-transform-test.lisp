;;;; defmodule のモデルを jit / grad に直接渡す（issue #199 と #202 の完了条件）。
;;;;
;;;; 守らせる性質:
;;;; - (grad loss) の結果は、モデルと同じクラス・同じスロット（入れ子のサブモジュールも）で、
;;;;   各パラメータの勾配になっている（手で微分した式と一致する）。
;;;; - モデルを引数に取る関数を IREE の local で jit した結果は eager と一致する（medium）。

(in-package #:nabla.nn.tests)

(defmodule gm-lin ((w :param) (b :param))
  (:call (self x) (+ (* (gm-lin-w self) x) (gm-lin-b self))))

(defmodule gm-net ((inp :param) (out :param))
  (:call (self x) (call (gm-net-out self) (call (gm-net-inp self) x))))

(defun %gm-vec (seed)
  (nabla.tests.support:make-random-array (nabla.tests.support:make-array-spec '(3) :f32) :seed seed))

(defun %gm-net (seed)
  (make-gm-net :inp (make-gm-lin :w (%gm-vec seed) :b (%gm-vec (+ seed 1)))
               :out (make-gm-lin :w (%gm-vec (+ seed 2)) :b (%gm-vec (+ seed 3)))))

(defun %gm-close (a b)
  (nabla.tests.support:allclose a b :dtype :f32))

(test (module/grad-returns-a-module-of-the-same-class-with-matching-slots :suite :nabla.small)
  "loss = sum(out(inp(x))) の勾配は gm-net で、inp / out は gm-lin、スロットは手で微分した式
（dw1 = w2 x、db1 = w2、dw2 = w1 x + b1、db2 = 1）と一致する。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 10000)))
                (lambda (case)
                  (let* ((seed (first case))
                         (net (%gm-net seed))
                         (x (%gm-vec (+ seed 4)))
                         (loss (nabla:with-tracing (model x) (nabla:reduce-sum (call model x))))
                         (g (funcall (nabla:grad loss) net x))
                         (w1 (gm-lin-w (gm-net-inp net)))
                         (b1 (gm-lin-b (gm-net-inp net)))
                         (w2 (gm-lin-w (gm-net-out net)))
                         (ones (make-array 3 :element-type 'single-float :initial-element 1.0)))
                    (and (typep g 'gm-net)
                         (typep (gm-net-inp g) 'gm-lin)
                         (typep (gm-net-out g) 'gm-lin)
                         (nabla:treedef= (nabla:tree-structure net) (nabla:tree-structure g))
                         (%gm-close (gm-lin-w (gm-net-inp g)) (map-into (make-array 3 :element-type 'single-float) #'* w2 x))
                         (%gm-close (gm-lin-b (gm-net-inp g)) w2)
                         (%gm-close (gm-lin-w (gm-net-out g))
                                    (map-into (make-array 3 :element-type 'single-float) #'+
                                              (map-into (make-array 3 :element-type 'single-float) #'* w1 x) b1))
                         (%gm-close (gm-lin-b (gm-net-out g)) ones))))
                :regression-id module/grad-returns-a-module-of-the-same-class-with-matching-slots
                :regression-file (nabla.tests.support:regression-path
                                  "module-grad-matches-slots" :package "NABLA.NN.TESTS"))))

(test (module/value-and-grad-and-argnums-list-return-modules :suite :nabla.small)
  "value-and-grad は (値 勾配のモジュール)。argnums に引数の位置のリストを渡すと、モデルの勾配
（モジュール）と x の勾配（配列）のリストになる。"
  (let* ((net (%gm-net 7))
         (x (%gm-vec 11))
         (loss (nabla:with-tracing (model x) (nabla:reduce-sum (call model x)))))
    (multiple-value-bind (value g) (funcall (nabla:value-and-grad loss) net x)
      (is (%gm-close value (funcall loss net x)))
      (is (typep g 'gm-net)))
    (destructuring-bind (gm gx) (funcall (nabla:grad loss :argnums '(0 1)) net x)
      (is (typep gm 'gm-net))
      (is (equal '(3) (array-dimensions gx))))))

(defun %require-iree-or-skip ()
  "IREE の共有ライブラリが無ければ、NABLA_REQUIRE_IREE があれば失敗、なければスキップして NIL。"
  (or (nabla.iree:iree-available-p :library :both)
      (progn (if (uiop:getenv "NABLA_REQUIRE_IREE")
                 (fail "IREE が見つからない（NABLA_REQUIRE_IREE が設定されている）")
                 (skip "IREE が見つからない"))
             nil)))

(test (module/jit-takes-a-module-argument-on-iree-local :suite :nabla.medium)
  "モデルを引数に取る関数を IREE の local で jit した結果は eager と一致し、jit (grad loss) の結果は
モジュールで、eager の grad と一致する。同じ treedef の2回目の呼び出しはコンパイルしない。"
  (when (%require-iree-or-skip)
    (let* ((backend (nabla:find-backend :iree))
           (nabla:*compile-cache-directory* nil)
           (net (%gm-net 21))
           (x (%gm-vec 25))
           (f (nabla:with-tracing (model x) (call model x)))
           (loss (nabla:with-tracing (model x) (nabla:reduce-sum (call model x))))
           (g (nabla:grad loss))
           (jf (nabla:jit f :backend backend))
           (jg (nabla:jit g :backend backend)))
      (unwind-protect
           (let ((jit-grad (funcall jg net x))
                 (eager-grad (funcall g net x)))
             (is (%gm-close (funcall jf net x) (funcall f net x)))
             (is (typep jit-grad 'gm-net))
             (is (every #'%gm-close (nabla:tree-leaves jit-grad) (nabla:tree-leaves eager-grad)))
             (let ((before nabla::*jit-miss-count*))
               (funcall jf (%gm-net 99) x)
               (is (= before nabla::*jit-miss-count*))))
        (nabla::%jit-cache-forget f)
        (nabla::%jit-cache-forget g)))))
