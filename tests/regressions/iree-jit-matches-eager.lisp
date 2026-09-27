(in-package #:nabla.iree.tests)
;; PR #67 のレビューで見つかった、bf16 の reduce-sum(+ a b) が
;; jit/pbt-matches-eager を落とした4つの例（issue #34）。原因は eager
;; （EVAL-GRAPH。演算ごとに丸める）を期待値にしていたことで、IREE は
;; エレメントワイズの + を丸めずに f32 の reduce-sum 累積へ渡す形に融合
;; することがあり、2項が打ち消し合う入力では出力の絶対値に対する rtol
;; だけでは吸収できないずれになっていた。%JIT-PBT-F32-ORACLE（bf16 を
;; f32 に昇格して丸めをほぼ含まない参照値にする）に直してからは、
;; これらはすべて通る。回帰として固定する。
(REGRESSION-CASE :NAME JIT/PBT-MATCHES-EAGER :DATUM "(7 4 2 4 :BF16 858539031)" :TIMESTAMP 3810000001)
(REGRESSION-CASE :NAME JIT/PBT-MATCHES-EAGER :DATUM "(7 3 2 1 :BF16 858525931)" :TIMESTAMP 3810000002)
(REGRESSION-CASE :NAME JIT/PBT-MATCHES-EAGER :DATUM "(7 3 2 1 :BF16 1)" :TIMESTAMP 3810000003)
(REGRESSION-CASE :NAME JIT/PBT-MATCHES-EAGER :DATUM "(7 3 2 1 :BF16 3)" :TIMESTAMP 3810000004)
