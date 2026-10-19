(in-package #:nabla.iree.tests)

;; seed 94169: IREE の tanh（f32 で絶対誤差 3u）の誤差が 8 ステップで 1.7e-6 になり、f32 の既定の atol 1e-6 を超えた（issue #220）。
(check-it:regression-case :name scan/iree-nested-forward-reverse :datum "94169" :timestamp 4001383850)
