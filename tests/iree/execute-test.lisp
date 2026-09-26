;;;; StableHLO のテキストをメモリ上でコンパイル・ロード・実行する invoke の
;;;; テスト（issue #8）。
;;;;
;;;; invoke そのものは FFI のオーケストレーション（IREE の C API 呼び出しを
;;;; 順に並べているだけ）なので、mutation testing の対象外（このスキルの
;;;; 「CFFI の生バインディングの疎通確認は例ベースでよい」の考え方を、
;;;; ここでは公開 API 経由の値の一致という形の性質にしている）。
;;;;
;;;; add / matmul / reduce_sum の数値一致の3テストは、issue #9 で
;;;; tests/iree/backend-test.lisp（nabla:backend プロトコル経由）に
;;;; 書き直して、ここからは削除した。invoke 自体の壊れた入力に対する
;;;; エラー経路（下の4テスト）はここに残す。

(in-package #:nabla.iree.tests)

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
