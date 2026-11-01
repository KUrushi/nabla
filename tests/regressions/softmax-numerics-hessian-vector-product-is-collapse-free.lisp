(in-package #:nabla.nn.tests)

;; seed 18231、n 4、k 1、余裕 36: 最大値の位置の成分で項 1e-17 の和が 3.2e-20 になり（打ち消し約 500 倍）、
;; 実装の f32 の丸めによる誤差 5.1e-23 が相対 1e-3（3.2e-23）を超えた（CI の run 41212666425 でまれに落ちた）。
;; 今は %hvp-rounding-bounds の上界（2.9e-22）を床に足して通る。
(check-it:regression-case :name softmax-numerics/hessian-vector-product-is-collapse-free :datum "(18231 4 1 36)" :timestamp 4001383850)
