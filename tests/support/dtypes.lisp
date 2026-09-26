;;;; dtype の一覧と、dtype ごとの既定の許容誤差。

(in-package #:nabla.tests.support)

(defparameter *dtypes* '(:f32 :f64 :bf16 :f16)
  "nabla がサポートする dtype の一覧。")

(defun dtype-tolerance (dtype)
  "DTYPE の既定の許容誤差を (values rtol atol) で返す。

数値はいずれも SKILL.md の表のとおり:
f64 は 1e-12/1e-12、f32 は 1e-5/1e-6、bf16/f16 は 1e-2/1e-3。"
  (ecase dtype
    (:f64 (values 1d-12 1d-12))
    (:f32 (values 1d-5 1d-6))
    (:bf16 (values 1d-2 1d-3))
    (:f16 (values 1d-2 1d-3))))
