(in-package #:nabla.iree.tests)

;; seed 23260: db[1] が -0.03 で項が 1 程度の打ち消し合いになり、誤差 1.46e-6 が f32 の既定の atol 1e-6 を超えた（issue #220）。
(check-it:regression-case :name scan/iree-elman-grad :datum "23260" :timestamp 4001383850)
