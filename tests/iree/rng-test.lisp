;;;; rng-bit-generator の IREE（local）での実行（issue #133、medium）。
;;;;
;;;; eager 実装（src/primitives/rng.lisp。IREE の lowering を写したもの）が、IREE が
;;;; 実際にコンパイルして実行した結果と、新しい状態・乱数ビットともにビット単位で
;;;; 一致すること（rank 0..4、奇数・偶数・1の次元、:u32 / :u64、64ビット全域の状態）。
;;;; 同じ状態から jit を2回呼んでも、ディスクキャッシュ経由でも同じ結果になること。

(in-package #:nabla.iree.tests)

(defparameter *rng-iree-shapes*
  '(() (1) (2) (3) (7) (8) (1 1) (2 3) (3 2) (3 3) (4 3) (3 4) (5 1) (1 5)
    (2 3 5) (3 5 2) (3 3 3) (7 1 3) (2 2 2 2) (3 3 3 3) (1 1 1 1) (5 4 3 2)
    (16 16)))

(defun %rng-iree-state (seed)
  "SEED から決まる ui64[2]（鍵・カウンタとも64ビット全域。カウンタの折り返しも出る）。"
  (let ((rs (sb-ext:seed-random-state seed))
        (state (make-array 2 :element-type '(unsigned-byte 64))))
    (setf (aref state 0) (random (expt 2 64) rs)
          (aref state 1) (if (zerop (mod seed 5))
                             (- (expt 2 64) 1 (random 20 rs))
                             (random (expt 2 64) rs)))
    state))

(defun %rng-iree-run (backend module state)
  "MODULE（state → (新しい状態, ビット) の main）を STATE で実行し、(VALUES 新しい状態 ビット) を返す。"
  (with-device-arrays ((dstate (to-device state backend :dtype :u64)))
    (multiple-value-bind (new-state bits) (nabla:backend-invoke backend module "main" dstate)
      (unwind-protect (values (to-host new-state) (to-host bits))
        (release-device-array new-state)
        (release-device-array bits)))))

(defun %rng-graph (shape dtype)
  (nb:trace-to-graph (nb:with-tracing (s) (nb::rng-bit-generator s :shape shape :dtype dtype))
                     (list (nb:make-aval '(2) :u64))))

(define-iree-test iree/rng-bit-generator/matches-eager-bit-exactly
  "全ての形状・dtype について、IREE の出力（新しい状態とビット）が eager と要素型も値も EQUALP で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree)))
    (dolist (dtype '(:u32 :u64))
      (dolist (shape *rng-iree-shapes*)
        (let* ((graph (%rng-graph shape dtype))
               (module (nabla:backend-load
                        backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
          (unwind-protect
               (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                             (lambda (seed)
                               (let ((state (%rng-iree-state seed)))
                                 (multiple-value-bind (iree-state iree-bits) (%rng-iree-run backend module state)
                                   (multiple-value-bind (eager-state eager-bits) (nb:eval-graph graph state)
                                     (and (equal (array-element-type iree-bits) (array-element-type eager-bits))
                                          (equalp iree-state eager-state)
                                          (equalp iree-bits eager-bits))))))
                             :regression-id iree/rng-bit-generator/matches-eager-bit-exactly
                             :regression-file (regression-path "iree-rng-bit-generator-matches-eager"
                                                               :package "NABLA.IREE.TESTS"))
                   "shape ~S dtype ~S: IREE の結果が eager と一致しなかった" shape dtype)
            (nabla:backend-unload backend module)))))))

(define-iree-test iree/rng-bit-generator/jit-is-deterministic-and-cache-stable
  "同じ状態からは、jit を2回呼んでも、別の jit（ディスクキャッシュから vmfb を読む）でも、同じ
ビット・状態になり、eager とも一致する。"
  (skip-unless-iree :library :both)
  (with-temporary-directory (dir)
    (let* ((nabla:*compile-cache-directory* dir)
           (backend (nabla:find-backend :iree))
           (state (%rng-iree-state 11))
           (make (lambda () (nb:jit (nb:with-tracing (s) (nb::rng-bit-generator s :shape '(3 5) :dtype :u32))
                                    :backend backend)))
           (jitted (funcall make))
           (first (multiple-value-list (funcall jitted state)))
           (second (multiple-value-list (funcall jitted state)))
           ;; 新しい jit は新しいトレース・新しいメモリ上のモジュールだが、vmfb はディスクキャッシュにある
           (cached (multiple-value-list (funcall (funcall make) state)))
           (eager (multiple-value-list (nb::rng-bit-generator state :shape '(3 5) :dtype :u32))))
      (is (equalp first second))
      (is (equalp first cached))
      (is (equalp first eager))
      (is (plusp (length (directory (merge-pathnames "*.module" dir))))))))
