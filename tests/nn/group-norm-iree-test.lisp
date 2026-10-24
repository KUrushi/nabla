;;;; group-norm を IREE の local で jit する（issue #260、medium）。
;;;;
;;;; 守らせる性質: jit した順伝播と jit した grad が、Flax のフィクスチャ（group-norm-test.lisp と同じケース）と
;;;; 一致する（許容誤差は eager と同じ）。

(in-package #:nabla.nn.tests)

(in-suite :nabla.medium)

;; 群の大きさ 1（instance norm。統計を取る軸が入力の最後の軸を含まない）の jit した grad は、IREE 3.11 のコンパイルが
;; "large vector sizes"（vector.contract）で失敗する（CI の x86_64 で 3 回中 3 回）。最後の軸を含まない軸で正規化する
;; layer-norm の grad と同じバグ（docs/iree-upstream-bugs.md の 9）。統計を [バッチ 群 要素] に transpose して最後の軸で
;; 縮約しても、群の大きさが 1 だと transpose がチャンネルの軸を動かすので回避できない。順伝播の jit は通るので、この
;; ケースは grad だけ eager で比べる。上流で直れば（docs の 9 の再現が通れば）この除外を外す。
(defparameter *group-norm-iree-failing-grad-cases* '("instance"))

(test group-norm/jit-forward-and-grads-match-flax-on-iree-local
  "group-norm-test.lisp のフィクスチャの全ケースで、IREE の local で jit した順伝播と jit した grad が Flax の
出力・勾配と一致する（*group-norm-iree-failing-grad-cases* は grad だけ eager）。"
  (when (%layers-require-iree-or-skip)
    (let ((backend (nabla:find-backend :iree))
          (nabla:*compile-cache-directory* nil))
      (dolist (case (%group-norm-fixture-cases))
        (let ((name (getf case :name)))
          ;; コンパイルの失敗はケースごとに報告して、残りのケースも確かめる
          (multiple-value-bind (out grads)
              (handler-case (%gn-flax-results case (%gn-fx-model case) :backend backend
                                 :jit-grad (not (member name *group-norm-iree-failing-grad-cases* :test #'string=)))
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
