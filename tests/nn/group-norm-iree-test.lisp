;;;; group-norm を IREE の local で jit する（issue #260、medium）。
;;;;
;;;; 守らせる性質: jit した順伝播と jit した grad が、Flax のフィクスチャ（group-norm-test.lisp と同じケース）と
;;;; 一致する（許容誤差は eager と同じ）。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

;; 群が 1 つ（one-group）と群の大きさ 1（instance）の grad の jit は、順伝播で rsqrt(var + epsilon) * scale の倍率を
;; 先に作る形（Flax の _normalize の順）だと IREE 3.11 のコンパイルが "large vector sizes"（vector.contract）で確率的に
;; 失敗していた（iree-compile 単体の znver4 で instance 35/200、one-group 3/200。CI でも one-group が 2 回落ちた）。
;; 正規化してから scale をかける形（src/nn/normalization.lisp）で 0 回になった（CI の 6 ジョブでこのテストを 180 回流して
;; 失敗 0 回）ので、全ケースで grad も jit する。
;; 再発の見張りは scripts/check-iree-repros.sh（docs/iree-upstream-bugs.md の 9）。

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
