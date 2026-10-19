(in-package #:nabla.nn.tests)
;; 入力のノルム 1.29952 が max-norm 1.3 の 0.999 倍以上・1 倍未満で、素通しになる場合。
;; 期待する係数を 1 で頭打ちにしていなかったテストの誤りを CI が見つけた例（実装は正しい）。
(REGRESSION-CASE :NAME OPTIMIZER/CLIP-NORM :DATUM "(15 6 13)" :TIMESTAMP 3999600000)
