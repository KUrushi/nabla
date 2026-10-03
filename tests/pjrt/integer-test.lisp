;;;; 整数 dtype（:i32 / :u32 / :u64）の PJRT（CPU プラグイン）での往復と実行
;;;; （issue #126、medium）。

(in-package #:nabla.pjrt.tests)

(define-pjrt-test backend/to-host/round-trips-integers-exactly
  "整数 dtype（:i32 / :u32 / :u64）の配列は、どの形状（rank 0..4）でも
to-device → to-host で要素型・値（端の値を含む）が変わらない。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (is (check-it (generator (tuple (array-spec :dtypes *integer-dtypes*)
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (spec-and-seed)
                    (destructuring-bind (spec seed) spec-and-seed
                      (let ((x (make-random-array spec :seed seed))
                            (dtype (array-spec-dtype spec)))
                        (with-pjrt-arrays ((y (nabla:to-device x backend :dtype dtype)))
                          (let ((roundtripped (nabla:to-host y)))
                            (and (equal (array-element-type roundtripped) (array-element-type x))
                                 (equalp roundtripped x)))))))
                  :regression-id backend/to-host/round-trips-integers-exactly
                  :regression-file (regression-path "pjrt-backend-roundtrip-integers"
                                                    :package "NABLA.PJRT.TESTS")))))

(define-pjrt-test backend/integer-add/matches-eager
  "整数の (+ x y)（折り返しを含む）を PJRT で実行した結果が eager とビット単位で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (dolist (dtype *integer-dtypes*)
      (let* ((aval (nb:make-aval '(3 5) dtype))
             (graph (nb:trace-to-graph (nb:with-tracing (x y) (+ x y)) (list aval aval)))
             (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
        (unwind-protect
             (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                           (lambda (seed)
                             (let* ((spec (make-array-spec '(3 5) dtype))
                                    (a (make-random-array spec :seed seed))
                                    (b (make-random-array spec :seed (1+ seed))))
                               (with-pjrt-arrays ((da (nabla:to-device a backend :dtype dtype))
                                                  (db (nabla:to-device b backend :dtype dtype)))
                                 (with-pjrt-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                                   (equalp (nabla:to-host result) (nb:eval-graph graph a b))))))
                           :regression-id backend/integer-add/matches-eager
                           :regression-file (regression-path "pjrt-integer-add"
                                                             :package "NABLA.PJRT.TESTS"))
                 "~A: PJRT の結果が eager と一致しなかった" dtype)
          (nabla:backend-unload backend module))))))
