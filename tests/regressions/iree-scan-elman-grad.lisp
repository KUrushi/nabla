(in-package #:nabla.iree.tests)

;; seed 23260: 入力を縮める前の graph で、db[1] が -0.03（項は 1 程度の打ち消し合い）になり、誤差 1.46e-6 が
;; f32 の既定の atol 1e-6 を超えた（issue #220）。今のテストは入力を ±0.3 に縮めた graph で、同じ seed は縮めた graph でも通る。
(check-it:regression-case :name scan/iree-elman-grad :datum "23260" :timestamp 4001383850)
