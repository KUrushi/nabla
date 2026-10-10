;;;; local と cuda の最大誤差を実測して表にする（issue #12）。
;;;;
;;;; tests/iree/cross-device-test.lisp の large テストは許容誤差の中に
;;;; 収まるかどうかだけを見て、一致したときの誤差は出さない。
;;;; docs/iree-build.md の結果表を埋めるため、同じフィクスチャ・同じ入力の
;;;; 作り方で、seed ごとの最大絶対誤差と最大相対誤差を測って表示する。
;;;;
;;;; scripts/colab/remote-gpu-check.sh から次のように呼ぶ:
;;;;   sbcl --non-interactive --load scripts/colab/measure-cross-device.lisp
;;;; 環境変数 NABLA_MEASURE_SEEDS（既定 100）で seed の数を変えられる。

(require :asdf)
(asdf:load-system "nabla/iree/tests")

(in-package #:nabla.iree.tests)

(defparameter *measure-cases*
  ;; (fixture dtype 入力の形のリスト)。cross-device-test.lisp と同じ。
  '(("add" :f32 ((4 8) (4 8)))
    ("add_bf16" :bf16 ((4 8) (4 8)))
    ("matmul" :f32 ((2 3) (3 2)))
    ("matmul_bf16" :bf16 ((2 3) (3 2)))
    ("reduce_sum" :f32 ((4 8)))
    ("reduce_sum_bf16" :bf16 ((4 8)))))

(defun %measure-seeds ()
  (let ((value (uiop:getenv "NABLA_MEASURE_SEEDS")))
    (if (and value (plusp (length value))) (parse-integer value) 100)))

(defun %measure-case (local cuda fixture dtype shapes seeds)
  "SEEDS 個の seed について local と cuda の結果を比べ、
(values 最大絶対誤差 最大相対誤差 許容誤差に収まらなかった seed の数) を返す。"
  (let ((text (stablehlo-fixture fixture))
        (max-abs 0d0)
        (max-rel 0d0)
        (failures 0))
    (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
      (dotimes (seed seeds)
        (let* ((arrays (loop for shape in shapes
                             for k from 0
                             collect (make-random-array (make-array-spec shape dtype)
                                                        :seed (+ seed k))))
               (l (%cross-device-run local text "main" arrays dtype))
               (c (%cross-device-run cuda text "main" arrays dtype)))
          (dotimes (i (array-total-size l))
            (let* ((a (row-major-aref l i))
                   (b (row-major-aref c i))
                   (abs-err (abs (- a b)))
                   (rel-err (if (zerop a) abs-err (/ abs-err (abs a)))))
              (setf max-abs (max max-abs abs-err)
                    max-rel (max max-rel rel-err))))
          (unless (allclose l c :rtol rtol :atol atol)
            (incf failures)))))
    (values max-abs max-rel failures)))

(defun %measure-main (&key (target :cuda))
  "local と TARGET の backend で全フィクスチャを測り、Markdown の表を出す。"
  (let* ((seeds (%measure-seeds))
         (local (nabla:find-backend :iree))
         (cuda (nabla:make-backend :iree :target target))
         (nabla:*compile-cache-directory* nil))
    (format t "~&| fixture | dtype | seeds | 最大絶対誤差 | 最大相対誤差 | 旧既定の許容誤差外 |~%")
    (format t "| --- | --- | --- | --- | --- | --- |~%")
    (dolist (case *measure-cases*)
      (destructuring-bind (fixture dtype shapes) case
        (multiple-value-bind (max-abs max-rel failures)
            (%measure-case local cuda fixture dtype shapes seeds)
          (format t "| ~A | ~(~A~) | ~D | ~,3,,,,,'eE | ~,3,,,,,'eE | ~D |~%"
                  fixture dtype seeds max-abs max-rel failures))))
    (finish-output)))

;; NABLA_MEASURE_TARGET=local にすると、GPU の無いマシンで local 同士を
;; 比べてこのスクリプト自体を試せる（誤差はすべて 0 になるはず）。
(%measure-main :target (if (equal (uiop:getenv "NABLA_MEASURE_TARGET") "local")
                           :local
                           :cuda))
