(in-package #:nabla.tests)
(REGRESSION-CASE :NAME SUPPORT/PRIMITIVE-GRAPH-RECIPE/VARS-MATCH-REPLAYED-AVALS
                 :DATUM
                 "((:IN :F32 (1 2)) (:IN :BF16 NIL) (:CONVERT 1 :F32) (:DOT 0 854935047 2)
 (:OUT 3))"
                 :TIMESTAMP 3999496868)
