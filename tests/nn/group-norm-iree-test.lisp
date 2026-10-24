;;;; group-norm を IREE の local で jit する（issue #260、medium）。
;;;;
;;;; 守らせる性質: jit した順伝播と jit した grad が、Flax のフィクスチャ（group-norm-test.lisp と同じケース）と
;;;; 一致する（許容誤差は eager と同じ）。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

(test group-norm/jit-forward-and-grads-match-flax-on-iree-local
  "group-norm-test.lisp のフィクスチャの全ケースで、IREE の local で jit した順伝播と jit した grad が Flax の
出力・勾配と一致する。"
  (when (%layers-require-iree-or-skip)
    (let ((backend (nabla:find-backend :iree))
          (nabla:*compile-cache-directory* nil))
      (dolist (case (%group-norm-fixture-cases))
        (let ((name (getf case :name)))
          ;; コンパイルの失敗はケースごとに報告して、残りのケースも確かめる
          (multiple-value-bind (out grads)
              (handler-case (%gn-flax-results case (%gn-fx-model case) :backend backend)
                (nabla:jit-compile-error (e)
                  (fail "~A: jit のコンパイルに失敗した: ~A" name e)
                  (values nil nil)))
            (when out
              (is (apply #'%close-f32 out (%fx-array case "out") *gn-tolerance*)
                  "~A: jit した順伝播が Flax と一致しない" name)
              (let ((expected (%gn-flax-expected-grads case)))
                (is (= (length expected) (length grads)) "~A: 勾配の個数が違う" name)
                (loop for a in grads for e in expected for i from 0
                      do (is (apply #'%close-f32 a e *gn-tolerance*)
                             "~A: jit した ~D 番目の勾配が Flax と一致しない" name i))))))))))
