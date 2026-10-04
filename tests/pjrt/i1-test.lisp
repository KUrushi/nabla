;;;; :i1（PJRT_Buffer_Type_PRED）の PJRT（CPU プラグイン）での往復と jit の
;;;; 入出力（issue #166 (b)、medium）。

(in-package #:nabla.pjrt.tests)

(define-pjrt-test backend/to-host/round-trips-i1-exactly
  ":i1 の BIT 配列は、どの形状（rank 0..4。8 や 64 の倍数でない要素数を含む）でも
to-device → to-host で要素型・値が変わらない（ホストのビット詰めと PRED の
1要素1バイトの詰め直しが要素の位置をずらさない）。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (%pjrt-backend)))
    (is (check-it (generator (tuple (array-spec :dtypes '(:i1))
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (spec-and-seed)
                    (destructuring-bind (spec seed) spec-and-seed
                      (let ((x (make-random-array spec :seed seed)))
                        (with-pjrt-arrays ((y (nabla:to-device x backend)))
                          (let ((roundtripped (nabla:to-host y)))
                            (and (equal (array-element-type roundtripped) (array-element-type x))
                                 (equalp roundtripped x)))))))
                  :regression-id backend/to-host/round-trips-i1-exactly
                  :regression-file (regression-path "pjrt-backend-roundtrip-i1"
                                                    :package "NABLA.PJRT.TESTS")))))

(define-pjrt-test backend/to-device/zero-size-i1-round-trips
  "要素数0の :i1 の配列（shape (0) と (2 0 3)）も to-device → to-host で同じ
shape・要素型の配列に戻る。"
  (skip-unless-pjrt :kind :cpu)
  (dolist (x (list (make-array '(0) :element-type 'bit)
                   (make-array '(2 0 3) :element-type 'bit)))
    (with-pjrt-arrays ((y (nabla:to-device x (%pjrt-backend))))
      (let ((roundtripped (nabla:to-host y)))
        (is (equal (array-dimensions x) (array-dimensions roundtripped)))
        (is (equal (array-element-type x) (array-element-type roundtripped)))))))

(defun %i1-jit-matches-eager-p (f args)
  "F を PJRT で jit した結果（多値すべて）が、F を eager に呼んだ結果と equalp で
一致し、要素型も同じなら真。jit のキャッシュは呼んだ後で忘れる。"
  (unwind-protect
       (let ((actual (multiple-value-list (apply (nb:jit f :backend :pjrt) args)))
             (expected (multiple-value-list (apply f args))))
         (and (= (length actual) (length expected))
              (every (lambda (a e)
                       (and (equal (array-element-type a) (array-element-type e))
                            (equalp a e)))
                     actual expected)))
    (nb::%jit-cache-forget f)))

(define-pjrt-test jit/pjrt-i1-input-and-output-match-eager
  ":i1 を入力（where の条件）にも出力（比較の結果）にもする関数を PJRT で jit
すると、eager と同じ BIT 配列・f32 配列が返る。比較の両辺は1回の IEEE 演算で
作るので、丸めの違いで比較結果が反転することはない。"
  (skip-unless-pjrt :kind :cpu)
  (let ((*num-trials* 8)
        (nb:*compile-cache-directory* nil)
        (f (nb:with-tracing (c a b)
             (values (nb:where c a b) (< (+ a b) (- a b)) (< (nb:where c a b) b)))))
    (is (check-it (generator (tuple (array-spec :dtypes '(:f32) :max-rank 3 :max-dim 5)
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (spec seed) args
                      (%i1-jit-matches-eager-p
                       f
                       (list (make-random-array (make-array-spec (array-spec-shape spec) :i1)
                                                :seed seed)
                             (make-random-array spec :seed (+ seed 1))
                             (make-random-array spec :seed (+ seed 2))))))
                  :regression-id jit/pjrt-i1-input-and-output-match-eager
                  :regression-file (regression-path "pjrt-jit-i1-input-output"
                                                    :package "NABLA.PJRT.TESTS")))
    (gc-and-run-finalizers)))

(define-pjrt-test jit/pjrt-i1-control-flow-matches-eager
  "rank 0 の :i1 を引数に取って cond* の pred にする関数と、比較由来の :i1 の
carry を戻り値にする while-loop（IREE 3.11 ではコンパイラが落ちる形）を PJRT で
jit すると、eager と同じ結果が返る。"
  (skip-unless-pjrt :kind :cpu)
  (let ((*num-trials* 8)
        (nb:*compile-cache-directory* nil)
        (branch (nb:with-tracing (p x)
                  (nb:cond* p (nb:with-tracing (y) (* y 2.0)) (nb:with-tracing (y) (- y))
                            x)))
        (loop-flag (nb:with-tracing (n)
                     (let ((result (nb:while-loop
                                    (nb:with-tracing (c) (second c))
                                    (nb:with-tracing (c)
                                      (list (+ (first c) 1.0) (< (+ (first c) 1.0) 5.0)))
                                    (list n (< n 5.0)))))
                       (values (first result) (second result))))))
    (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1)
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (bit seed) args
                      (and (%i1-jit-matches-eager-p
                            branch
                            (list (make-array '() :element-type 'bit :initial-element bit)
                                  (make-random-array (make-array-spec '(3) :f32) :seed seed)))
                           (%i1-jit-matches-eager-p
                            loop-flag
                            (list (make-array '() :element-type 'single-float
                                                  :initial-element (float (mod seed 9) 1.0)))))))
                  :regression-id jit/pjrt-i1-control-flow-matches-eager
                  :regression-file (regression-path "pjrt-jit-i1-control-flow"
                                                    :package "NABLA.PJRT.TESTS")))
    (gc-and-run-finalizers)))
