;;;; nabla:backend プロトコルを、find-backend :iree 経由で確かめるテスト
;;;; （issue #9）。issue #8 の execute-test.lisp にあった add / matmul /
;;;; reduce_sum の数値一致の3テストは、ここでプロトコル経由に書き直した
;;;; （元のテストは削除。エラー経路の4テストは execute-test.lisp に残す）。
;;;;
;;;; find-backend を使い、make-backend を繰り返さない（device をプロセス
;;;; 寿命で1つだけ持つ設計。契約 §8）。backend-compile / backend-load /
;;;; backend-invoke は FFI のオーケストレーション（既存の compile-stablehlo /
;;;; session-append-module / invoke に薄く委譲するだけ）なので、
;;;; mutation testing の対象外（このスキルの「CFFI の生バインディングの
;;;; 疎通確認は例ベースでよい」の考え方をそのまま当てはめている）。

(in-package #:nabla.iree.tests)

(define-iree-test backend/protocol/add-matches-reference
    "add.mlir（要素ごとの加算、shape 4x8）を find-backend :iree の
プロトコル（backend-compile → backend-load → backend-invoke）経由で
実行した結果は、reference-add の期待値と allclose :dtype :f32 で一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let* ((spec (make-array-spec '(4 8) :f32))
                                (a (make-random-array spec :seed seed))
                                (b (make-random-array spec :seed (1+ seed))))
                           (with-device-arrays ((da (to-device a backend))
                                                (db (to-device b backend)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                               (and (equalp (device-array-aval result) (nabla:array-aval a :f32))
                                    (allclose (to-host result) (reference-add a b) :dtype :f32))))))
                       :regression-id backend/protocol/add-matches-reference
                       :regression-file (regression-path "iree-backend-add" :package "NABLA.IREE.TESTS")))
      (nabla:backend-unload backend module))))

(define-iree-test backend/protocol/matmul-matches-reference
    "matmul.mlir（dot_general、2x3 · 3x2）を find-backend :iree のプロトコル
経由で実行した結果は、reference-matmul の期待値と allclose :dtype :f32 で
一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "matmul")))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let ((a (make-random-array (make-array-spec '(2 3) :f32) :seed seed))
                               (b (make-random-array (make-array-spec '(3 2) :f32) :seed (1+ seed))))
                           (with-device-arrays ((da (to-device a backend))
                                                (db (to-device b backend)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                               (and (equalp (device-array-aval result) (nabla:make-aval '(2 2) :f32))
                                    (allclose (to-host result) (reference-matmul a b) :dtype :f32))))))
                       :regression-id backend/protocol/matmul-matches-reference
                       :regression-file (regression-path "iree-backend-matmul" :package "NABLA.IREE.TESTS")))
      (nabla:backend-unload backend module))))

(define-iree-test backend/protocol/reduce-sum-matches-reference
    "reduce_sum.mlir（shape 4x8 を dimension 1 で総和、結果 shape 4）を
find-backend :iree のプロトコル経由で実行した結果は、reference-reduce-sum
の期待値と allclose :dtype :f32 で一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "reduce_sum")))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                       (lambda (seed)
                         (let ((a (make-random-array (make-array-spec '(4 8) :f32) :seed seed)))
                           (with-device-arrays ((da (to-device a backend)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
                               (and (equalp (device-array-aval result) (nabla:make-aval '(4) :f32))
                                    (allclose (to-host result) (reference-reduce-sum a 1) :dtype :f32))))))
                       :regression-id backend/protocol/reduce-sum-matches-reference
                       :regression-file (regression-path "iree-backend-reduce-sum" :package "NABLA.IREE.TESTS")))
      (nabla:backend-unload backend module))))

(define-iree-test backend/make-backend/iree-returns-iree-backend
    "(make-backend :iree) は iree-backend 型のインスタンスを返す。"
  (skip-unless-iree :library :both)
  (is (typep (nabla:make-backend :iree) 'iree-backend)))

(define-iree-test backend/backend-unload/is-idempotent
    "backend-unload は同じ module に対して2回呼んでもエラーにならない
（idempotent）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
    (nabla:backend-unload backend module)
    (finishes (nabla:backend-unload backend module))))

(define-iree-test backend/iree-error/is-a-backend-error
    "iree-error は nabla:backend-error の subtype（IREE の共有ライブラリの
有無に関係なく成り立つ、クラス階層だけの性質）。"
  (is (subtypep 'iree-error 'nabla:backend-error)))
