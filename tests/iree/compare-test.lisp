;;;; compare / select / convert の StableHLO 出力を IREE の local backend で
;;;; 実際にコンパイル・実行し、eager 実装と一致することを確かめる medium
;;;; テスト（issue #31 p3）。
;;;;
;;;; このファイルを書いた時点（issue #31）では :i1 が to-device で拒否されて
;;;; いた（issue #72 で対応済み。:i1 の入出力は tests/iree/jit-dtype-test.lisp
;;;; が確かめる）ため、compare + select を1つのモジュールにまとめ（select(compare(a, b, direction), a, b)）、f32/bf16
;;;; の入出力だけを device に渡す。direction が :LT のときはこれが
;;;; min(a, b) に等しい（tests/primitives/compare-test.lisp と同じ性質）。
;;;;
;;;; convert は、書いた時点で f64 が to-device を通らなかったため、
;;;; f32/bf16/f16 の組だけを対象にする（契約 §4 の pitfall #6。f64 の
;;;; to-device は issue #72 で対応済み）。

(in-package #:nabla.iree.tests)

(defmacro def-iree-compare-select-test (test-name dtype direction)
  "shape (4 8)・DTYPE（:F32/:BF16）・DIRECTION（:LT :LE :GT :GE :EQ :NE）で
select(compare(a, b, DIRECTION), a, b) を1つのモジュールにまとめ、IREE の
local backend でコンパイル・実行した結果が、compare の eager 実装を経て
select の eager 実装に渡した結果と allclose :dtype ~A で一致することを
確かめる DEFINE-IREE-TEST を作る。"
  `(define-iree-test ,test-name
       ,(format nil "select(compare(a, b, ~A), a, b)（dtype ~(~A~)）を
find-backend :iree のプロトコル経由で実行した結果は、compare + select の
eager 実装を通した結果と allclose :dtype ~(~A~) で一致する。" direction dtype dtype)
     (skip-unless-iree :library :both)
     (let* ((aval (nb:make-aval '(4 8) ,dtype))
            (pred-aval (nb:make-aval '(4 8) :i1))
            (body (list (format nil "%c = stablehlo.compare ~A, %a0, %a1 : (~A, ~A) -> ~A"
                                 ,(symbol-name direction)
                                 (nb::tensor-type-string aval) (nb::tensor-type-string aval)
                                 (nb::tensor-type-string pred-aval))
                        (format nil "%0 = stablehlo.select %c, %a0, %a1 : ~A, ~A"
                                (nb::tensor-type-string pred-aval) (nb::tensor-type-string aval)))))
       (with-one-op-module
           ((backend module) (list aval aval) aval body)
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let* ((spec (make-array-spec '(4 8) ,dtype))
                                (a (make-random-array spec :seed seed))
                                (b (make-random-array spec :seed (1+ seed)))
                                (in-avals (list aval aval))
                                (pred (funcall (nb::primitive-eager (nb::find-primitive :compare))
                                                (list a b) in-avals :direction ,direction))
                                (select-in-avals (list pred-aval aval aval))
                                (expected (funcall (nb::primitive-eager (nb::find-primitive :select))
                                                    (list pred a b) select-in-avals)))
                           (with-device-arrays ((da (to-device a backend :dtype ,dtype))
                                                (db (to-device b backend :dtype ,dtype)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                               (and (equalp (device-array-aval result) aval)
                                    (allclose (to-host result) expected :dtype ,dtype))))))
                       :regression-id ,test-name
                       :regression-file (regression-path
                                         ,(format nil "iree-primitives-compare-select-~(~A~)-~(~A~)" direction dtype)
                                         :package "NABLA.IREE.TESTS"))
             ,(format nil "compare(~A)+select/~A: IREE の実行結果が eager 実装と一致しなかった" direction dtype))))))

(def-iree-compare-select-test iree/compare-select/lt/f32/matches-eager :f32 :lt)
(def-iree-compare-select-test iree/compare-select/lt/bf16/matches-eager :bf16 :lt)
(def-iree-compare-select-test iree/compare-select/le/f32/matches-eager :f32 :le)
(def-iree-compare-select-test iree/compare-select/le/bf16/matches-eager :bf16 :le)
(def-iree-compare-select-test iree/compare-select/gt/f32/matches-eager :f32 :gt)
(def-iree-compare-select-test iree/compare-select/gt/bf16/matches-eager :bf16 :gt)
(def-iree-compare-select-test iree/compare-select/ge/f32/matches-eager :f32 :ge)
(def-iree-compare-select-test iree/compare-select/ge/bf16/matches-eager :bf16 :ge)
(def-iree-compare-select-test iree/compare-select/eq/f32/matches-eager :f32 :eq)
(def-iree-compare-select-test iree/compare-select/eq/bf16/matches-eager :bf16 :eq)
(def-iree-compare-select-test iree/compare-select/ne/f32/matches-eager :f32 :ne)
(def-iree-compare-select-test iree/compare-select/ne/bf16/matches-eager :bf16 :ne)

(defmacro def-iree-convert-test (test-name from-dtype to-dtype)
  "shape (4)・FROM-DTYPE → TO-DTYPE（:F32/:BF16/:F16 の組。f64 はファイル
冒頭のコメントの理由で対象外）の convert を1演算モジュールとして IREE の local
backend でコンパイル・実行し、convert の eager 実装の結果と allclose
:dtype ~A で一致することを確かめる DEFINE-IREE-TEST を作る。"
  `(define-iree-test ,test-name
       ,(format nil "stablehlo.convert（~(~A~) → ~(~A~)）を find-backend :iree
のプロトコル経由で実行した結果は、convert の eager 実装の結果と allclose
:dtype ~(~A~) で一致する。" from-dtype to-dtype to-dtype)
     (skip-unless-iree :library :both)
     (let ((in-aval (nb:make-aval '(4) ,from-dtype))
           (out-aval (nb:make-aval '(4) ,to-dtype)))
       (with-one-op-module
           ((backend module) (list in-aval) out-aval
            (list (format nil "%0 = stablehlo.convert %a0 : (~A) -> ~A"
                          (nb::tensor-type-string in-aval) (nb::tensor-type-string out-aval))))
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let* ((spec (make-array-spec '(4) ,from-dtype))
                                (a (make-random-array spec :seed seed))
                                (expected (funcall (nb::primitive-eager (nb::find-primitive :convert))
                                                    (list a) (list in-aval) :dtype ,to-dtype)))
                           (with-device-arrays ((da (to-device a backend :dtype ,from-dtype)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                               (and (equalp (device-array-aval result) out-aval)
                                    (allclose (to-host result) expected :dtype ,to-dtype))))))
                       :regression-id ,test-name
                       :regression-file (regression-path
                                         ,(format nil "iree-primitives-convert-~(~A~)-to-~(~A~)" from-dtype to-dtype)
                                         :package "NABLA.IREE.TESTS"))
             ,(format nil "convert ~A->~A: IREE の実行結果が eager 実装と一致しなかった" from-dtype to-dtype))))))

(def-iree-convert-test iree/convert/f32-to-bf16/matches-eager :f32 :bf16)
(def-iree-convert-test iree/convert/bf16-to-f32/matches-eager :bf16 :f32)
(def-iree-convert-test iree/convert/f32-to-f16/matches-eager :f32 :f16)
(def-iree-convert-test iree/convert/f16-to-f32/matches-eager :f16 :f32)
(def-iree-convert-test iree/convert/bf16-to-f16/matches-eager :bf16 :f16)
