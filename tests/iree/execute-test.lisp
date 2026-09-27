;;;; StableHLO のテキストをメモリ上でコンパイル・ロード・実行する invoke の
;;;; テスト（issue #8）。
;;;;
;;;; invoke そのものは FFI のオーケストレーション（IREE の C API 呼び出しを
;;;; 順に並べているだけ）なので、mutation testing の対象外（このスキルの
;;;; 「CFFI の生バインディングの疎通確認は例ベースでよい」の考え方を、
;;;; ここでは公開 API 経由の値の一致という形の性質にしている）。

(in-package #:nabla.iree.tests)

(define-iree-test execute/invoke/add-matches-reference
    "add.mlir（要素ごとの加算、shape 4x8）を実行した結果は、
reference-add の期待値と allclose :dtype :f32 で一致する。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "add")))
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((spec (make-array-spec '(4 8) :f32))
                             (a (make-random-array spec :seed seed))
                             (b (make-random-array spec :seed (1+ seed))))
                        (with-device-arrays ((da (to-device a device))
                                             (db (to-device b device)))
                          (with-device-arrays ((result (invoke session "module.main" da db)))
                            (and (equalp (device-array-aval result) (nabla:array-aval a :f32))
                                 (allclose (to-host result) (reference-add a b) :dtype :f32))))))
                    :regression-id execute/invoke/add-matches-reference
                    :regression-file (regression-path "iree-execute-add" :package "NABLA.IREE.TESTS"))))))

(define-iree-test execute/invoke/matmul-matches-reference
    "matmul.mlir（dot_general、2x3 · 3x2）を実行した結果は、
reference-matmul の期待値と allclose :dtype :f32 で一致する。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "matmul")))
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((a (make-random-array (make-array-spec '(2 3) :f32) :seed seed))
                             (b (make-random-array (make-array-spec '(3 2) :f32) :seed (1+ seed))))
                        (with-device-arrays ((da (to-device a device))
                                             (db (to-device b device)))
                          (with-device-arrays ((result (invoke session "module.main" da db)))
                            (and (equalp (device-array-aval result)
                                         (nabla:make-aval '(2 2) :f32))
                                 (allclose (to-host result) (reference-matmul a b) :dtype :f32))))))
                    :regression-id execute/invoke/matmul-matches-reference
                    :regression-file (regression-path "iree-execute-matmul" :package "NABLA.IREE.TESTS"))))))

(define-iree-test execute/invoke/reduce-sum-matches-reference
    "reduce_sum.mlir（shape 4x8 を dimension 1 で総和、結果 shape 4）を
実行した結果は、reference-reduce-sum の期待値と allclose :dtype :f32 で
一致する。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "reduce_sum")))
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let ((a (make-random-array (make-array-spec '(4 8) :f32) :seed seed)))
                        (with-device-arrays ((da (to-device a device)))
                          (with-device-arrays ((result (invoke session "module.main" da)))
                            (and (equalp (device-array-aval result) (nabla:make-aval '(4) :f32))
                                 (allclose (to-host result) (reference-reduce-sum a 1) :dtype :f32))))))
                    :regression-id execute/invoke/reduce-sum-matches-reference
                    :regression-file (regression-path "iree-execute-reduce-sum" :package "NABLA.IREE.TESTS"))))))

(define-iree-test execute/invoke/wrong-shape-signals-invalid-argument
    "add.mlir は 4x8 の入力を宣言している。3x2 の device-array を渡すと、
IREE が hal.buffer_view.assert で検出し IREE-STATUS-ERROR（code
:invalid-argument）が signal される（クラッシュしない）。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "add")))
      (with-device-arrays ((wrong (to-device (make-random-array (make-array-spec '(3 2) :f32)) device))
                           (right (to-device (make-random-array (make-array-spec '(4 8) :f32)) device)))
        (handler-case
            (progn
              (invoke session "module.main" wrong right)
              (fiveam:fail "wrong shape should have signalled iree-status-error"))
          (iree-status-error (condition)
            (is (eq :invalid-argument (iree-status-error-code condition)))))))))

(define-iree-test execute/invoke/wrong-dtype-signals-invalid-argument
    "add.mlir は f32 の入力を宣言している。bf16 の device-array を渡すと
IREE-STATUS-ERROR（code :invalid-argument）が signal される。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "add")))
      (with-device-arrays ((wrong (to-device (make-random-array (make-array-spec '(4 8) :bf16)) device :dtype :bf16))
                           (right (to-device (make-random-array (make-array-spec '(4 8) :f32)) device)))
        (handler-case
            (progn
              (invoke session "module.main" wrong right)
              (fiveam:fail "wrong dtype should have signalled iree-status-error"))
          (iree-status-error (condition)
            (is (eq :invalid-argument (iree-status-error-code condition)))))))))

(define-iree-test execute/invoke/wrong-arity-signals-invalid-argument
    "add.mlir は引数2つを宣言している。1つしか渡さないと IREE-STATUS-ERROR
（code :invalid-argument）が signal される。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "add")))
      (with-device-arrays ((only (to-device (make-random-array (make-array-spec '(4 8) :f32)) device)))
        (handler-case
            (progn
              (invoke session "module.main" only)
              (fiveam:fail "wrong arity should have signalled iree-status-error"))
          (iree-status-error (condition)
            (is (eq :invalid-argument (iree-status-error-code condition)))))))))

(define-iree-test execute/invoke/released-argument-signals-iree-object-released
    "解放済みの device-array を invoke に渡すと IREE-OBJECT-RELEASED
（kind :device-array）が signal される。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "add")))
      (with-device-arrays ((right (to-device (make-random-array (make-array-spec '(4 8) :f32)) device)))
        (let ((released (to-device (make-random-array (make-array-spec '(4 8) :f32)) device)))
          (release-device-array released)
          (handler-case
              (progn
                (invoke session "module.main" released right)
                (fiveam:fail "released device-array should have signalled iree-object-released"))
            (iree-object-released (condition)
              (is (eq :device-array (iree-object-released-kind condition))))))))))

(define-iree-test execute/invoke/argument-from-another-device-signals-error
    "SESSION の device とは別の device で作った device-array を invoke に
渡すと（IREE 側の shape/dtype チェックの前に）plain ERROR が signal される。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (stablehlo-fixture "add")))
      (with-device (other-device :local-task)
        (with-device-arrays ((right (to-device (make-random-array (make-array-spec '(4 8) :f32)) device))
                             (wrong (to-device (make-random-array (make-array-spec '(4 8) :f32)) other-device)))
          (signals error (invoke session "module.main" wrong right)))))))
