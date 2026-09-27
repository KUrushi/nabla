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

;;; issue #12: cuda ターゲットの backend は device を遅延生成するので
;;; （src/iree/backend.lisp のファイル先頭コメント参照）、GPU の無い
;;; このマシンでも make-backend / backend-compile はここまで成功する
;;; （実行だけが GPU を要る。契約 §0 事実3）。

(define-iree-test backend/cuda-target/compiles-all-fixtures-without-a-gpu
    "target :cuda・cuda-arch \"sm_80\" の IREE-BACKEND は、GPU の無い
このマシンでも、6つのフィクスチャ（add / matmul / reduce_sum の f32・bf16
版）すべてを非空の vmfb にコンパイルできる（device は使わないので不要）。"
  (skip-unless-iree :library :both)
  (let ((cuda (nabla:make-backend :iree :target :cuda :cuda-arch "sm_80")))
    (dolist (fixture '("add" "matmul" "reduce_sum" "add_bf16" "matmul_bf16" "reduce_sum_bf16"))
      (let ((vmfb (nabla:backend-compile cuda (stablehlo-fixture fixture))))
        (is (plusp (length vmfb)) "~A の cuda 向けコンパイル結果が空だった" fixture)))))

(define-iree-test backend/cuda-target/fingerprint-and-cache-differ-from-local
    "target :cuda の backend-fingerprint は target :local と異なり、同じ
テキストを local と cuda でそれぞれ backend-compile すると、vmfb ディスク
キャッシュ（issue #10）に別々の .module ファイルができる（GPU 不要。
device は使わない）。"
  (skip-unless-iree :library :both)
  (with-temporary-directory (dir)
    (let* ((nabla:*compile-cache-directory* dir)
           (local (nabla:find-backend :iree))
           (cuda (nabla:make-backend :iree :target :cuda :cuda-arch "sm_80"))
           (text (stablehlo-fixture "add")))
      (is (not (equal (nabla:backend-fingerprint local) (nabla:backend-fingerprint cuda))))
      (nabla:backend-compile local text)
      (nabla:backend-compile cuda text)
      (is (= 2 (length (directory (make-pathname :name :wild :type "module" :defaults dir))))))))

(define-iree-test backend/bf16/local-matches-reference
    "add_bf16 / matmul_bf16 / reduce_sum_bf16 の各フィクスチャを local
backend で実行した結果は、decode-array で double-float に戻した
reference-* の期待値と、bf16 の許容誤差（rtol 1e-2 / atol 1e-3）で一致する
（GPU 不要）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree)))
    (multiple-value-bind (rtol atol) (dtype-tolerance :bf16)
      (flet ((%run-bf16-fixture (fixture arrays)
               (let ((module (nabla:backend-load backend (nabla:backend-compile backend (stablehlo-fixture fixture)))))
                 (unwind-protect
                      (let ((das (mapcar (lambda (a) (to-device a backend :dtype :bf16)) arrays)))
                        (unwind-protect
                             (multiple-value-bind (result)
                                 (apply #'nabla:backend-invoke backend module "main" das)
                               (unwind-protect
                                    (decode-array (to-host result) :bf16)
                                 (release-device-array result)))
                          (dolist (da das) (release-device-array da))))
                   (nabla:backend-unload backend module)))))
        (let* ((a (make-random-array (make-array-spec '(4 8) :bf16) :seed 100))
               (b (make-random-array (make-array-spec '(4 8) :bf16) :seed 101)))
          (is (allclose (%run-bf16-fixture "add_bf16" (list a b))
                        (reference-add (decode-array a :bf16) (decode-array b :bf16))
                        :rtol rtol :atol atol)))
        (let* ((a (make-random-array (make-array-spec '(2 3) :bf16) :seed 102))
               (b (make-random-array (make-array-spec '(3 2) :bf16) :seed 103)))
          (is (allclose (%run-bf16-fixture "matmul_bf16" (list a b))
                        (reference-matmul (decode-array a :bf16) (decode-array b :bf16))
                        :rtol rtol :atol atol)))
        (let ((a (make-random-array (make-array-spec '(4 8) :bf16) :seed 104)))
          (is (allclose (%run-bf16-fixture "reduce_sum_bf16" (list a))
                        (reference-reduce-sum (decode-array a :bf16) 1)
                        :rtol rtol :atol atol)))))))
