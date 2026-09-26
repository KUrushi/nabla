(in-package #:nabla.tests)
;; この例は、実装者が atol を落として意図的に allclose を壊していたときに
;; 保存されたもの（本物のバグから見つかったものではない）。現在の実装では
;; 通るが、regressions.lisp のローダーが実際に load してケースを再生する
;; ことを確かめる回帰テストとして、そのまま残す。
(REGRESSION-CASE :NAME SUPPORT/ALLCLOSE/BOUNDARY :DATUM
                 "(-3.8013983 0.17502546 0.24678195)" :TIMESTAMP 3999389961)
