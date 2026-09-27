;;;; add / sub / mul / div の StableHLO 出力を IREE の local backend で
;;;; 実際にコンパイル・実行し、eager 実装と一致することを確かめる
;;;; medium テスト（issue #31 p1）。
;;;;
;;;; コンパイルは (op, dtype) の組ごとに1回だけ行い（tests/iree/support.lisp
;;;; の DEFINE-IREE-TEST macro を使い、tests/iree/primitive-support.lisp の
;;;; WITH-ONE-OP-MODULE で1回だけコンパイル・ロードする）、check-it の各
;;;; 試行はロード済みの module を再利用する。

(in-package #:nabla.iree.tests)

(defmacro def-iree-arith-test (test-name prim-name dtype mlir-op &optional (shape '(4 8)))
  "PRIM-NAME（:ADD/:SUB/:MUL/:DIV）・DTYPE（:F32/:BF16）・MLIR-OP（\"add\" の
ような StableHLO の綴り）から、SHAPE（既定 (4 8)）の1演算モジュールを IREE
の local backend でコンパイル・実行し、PRIM-NAME の eager 実装の結果と
allclose :dtype ~A で一致することを確かめる DEFINE-IREE-TEST を作る。div は
0除算（allclose が NaN/inf を許さない）を避けるため :domain :positive で
生成する。SHAPE を SIMD レーン数の倍数でない要素数（例 (3 5)）にすると、
IREE のベクトル化されたカーネルがパディングレーンで浮動小数点例外を
起こしうる経路を通す（issue #31 p1 レビュー: stablehlo.divide がこの経路で
SIGFPE を起こしていた。%iree-backend-ensure-device / backend-invoke の
with-float-traps-masked で修正済み）。"
  `(define-iree-test ,test-name
       ,(format nil "stablehlo.~A（dtype ~(~A~)、shape ~A）を find-backend
:iree のプロトコル経由で実行した結果は、~(~A~) の eager 実装の結果と
allclose :dtype ~(~A~) で一致する。" mlir-op dtype shape prim-name dtype)
     (skip-unless-iree :library :both)
     (let ((aval (nb:make-aval ',shape ,dtype)))
       (with-one-op-module
           ((backend module) (list aval aval) aval
            (list (format nil "%0 = stablehlo.~A %a0, %a1 : ~A" ,mlir-op (nb::tensor-type-string aval))))
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let* ((spec (make-array-spec ',shape ,dtype))
                                (domain ,(if (eq prim-name :div) :positive :any))
                                (a (make-random-array spec :seed seed :domain domain))
                                (b (make-random-array spec :seed (1+ seed) :domain domain)))
                           (with-device-arrays ((da (to-device a backend :dtype ,dtype))
                                                (db (to-device b backend :dtype ,dtype)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                               (and (equalp (device-array-aval result) aval)
                                    (allclose (to-host result)
                                              (funcall (nb::primitive-eager (nb::find-primitive ,prim-name))
                                                       (list a b) (list aval aval))
                                              :dtype ,dtype))))))
                       :regression-id ,test-name
                       :regression-file (regression-path
                                         ,(format nil "iree-primitives-arith-~(~A~)-~(~A~)" prim-name dtype)
                                         :package "NABLA.IREE.TESTS"))
             ,(format nil "~A/~A: IREE の実行結果が eager 実装と一致しなかった" prim-name dtype))))))

(def-iree-arith-test iree/arith/add/f32/matches-eager :add :f32 "add")
(def-iree-arith-test iree/arith/add/bf16/matches-eager :add :bf16 "add")
(def-iree-arith-test iree/arith/sub/f32/matches-eager :sub :f32 "subtract")
(def-iree-arith-test iree/arith/sub/bf16/matches-eager :sub :bf16 "subtract")
(def-iree-arith-test iree/arith/mul/f32/matches-eager :mul :f32 "multiply")
(def-iree-arith-test iree/arith/mul/bf16/matches-eager :mul :bf16 "multiply")
(def-iree-arith-test iree/arith/div/f32/matches-eager :div :f32 "divide")
(def-iree-arith-test iree/arith/div/bf16/matches-eager :div :bf16 "divide")

;; issue #31 p1 レビュー: shape (4 8) は SIMD レーン数（f32 で8）の倍数なので
;; パディングレーンが無く、レーン数の倍数でない形状でのみ再現する SIGFPE
;; （%iree-backend-ensure-device 参照）を隠してしまっていた。(3 5) で守る。
(def-iree-arith-test iree/arith/div/f32/non-aligned-shape-matches-eager :div :f32 "divide" (3 5))
(def-iree-arith-test iree/arith/div/bf16/non-aligned-shape-matches-eager :div :bf16 "divide" (3 5))
