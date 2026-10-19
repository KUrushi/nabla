;;;; convolution プリミティブ（issue #197）の medium テスト。
;;;;
;;;; (1) 形・stride・padding・膨張・グループの異なる設定を、プリミティブの実際の :EMIT の
;;;; 出力で IREE (local) にコンパイルして実行し、eager 実装と一致することを PBT で確かめる
;;;; （f32・bf16・f16）。(2) JAX の conv_general_dilated のフィクスチャ（1〜3 次元、負の
;;;; padding、グループ）を jit で IREE に通して JAX の出力と比べる。

(in-package #:nabla.iree.tests)

(defparameter *conv-configs*
  ;; (lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation groups)
  '(((2 6 5 3) (3 2 3 4) (1 1) ((0 0) (0 0)) (1 1) (1 1) 1)
    ((1 7 6 2) (3 3 2 5) (2 3) ((1 2) (0 1)) (1 1) (1 1) 1)
    ((2 5 6 2) (2 3 2 3) (1 2) ((1 0) (2 1)) (2 1) (1 2) 1)
    ((2 5 5 4) (2 2 2 6) (1 1) ((1 1) (0 0)) (1 1) (1 1) 2)
    ((3 9 2) (4 2 3) (2) ((1 2)) (1) (1) 1)
    ((1 6 6 2) (3 3 2 2) (1 1) ((-1 2) (0 -1)) (1 1) (1 1) 1))
  "convolution の IREE 実行を確かめる設定。")

(define-iree-test convolution/iree-matches-eager
    "stride・padding（負も）・lhs / rhs の膨張・グループ・1〜2 次元の設定ごとに、convolution の
:EMIT の出力を IREE (local) で実行した結果は、eager 実装と f32・bf16・f16 それぞれの許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let ((nb:*compile-cache-directory* nil))
    (dolist (config *conv-configs*)
      (destructuring-bind (lhs-shape rhs-shape strides padding ld rd groups) config
        (dolist (dtype '(:f32 :bf16 :f16))
          (let* ((params (list :strides strides :padding padding :lhs-dilation ld
                               :rhs-dilation rd :feature-group-count groups))
                 (lhs-aval (nb:make-aval lhs-shape dtype))
                 (rhs-aval (nb:make-aval rhs-shape dtype))
                 (out-aval (apply (nb::primitive-abstract-eval (nb::find-primitive :convolution))
                                  (list lhs-aval rhs-aval) params))
                 (emit-text (apply (nb::primitive-emit (nb::find-primitive :convolution))
                                   '("%a0" "%a1") (list lhs-aval rhs-aval) "%0" out-aval params)))
            (with-one-op-module
                ((backend module) (list lhs-aval rhs-aval) out-aval
                 (uiop:split-string emit-text :separator '(#\Newline)))
              (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                            (lambda (seed)
                              (let ((a (make-random-array (make-array-spec lhs-shape dtype) :seed seed))
                                    (b (make-random-array (make-array-spec rhs-shape dtype) :seed (1+ seed))))
                                (with-device-arrays ((da (to-device a backend :dtype dtype))
                                                     (db (to-device b backend :dtype dtype)))
                                  (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
                                    (and (equalp (device-array-aval result) out-aval)
                                         (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                           (allclose (decode-array (to-host result) dtype)
                                                     (decode-array
                                                      (apply #'primitive-eager-oracle :convolution (list a b)
                                                             (list (nb:array-aval a dtype) (nb:array-aval b dtype))
                                                             params)
                                                      dtype)
                                                     :rtol rtol :atol atol)))))))
                            :regression-id convolution/iree-matches-eager
                            :regression-file (regression-path "iree-convolution" :package "NABLA.IREE.TESTS"))
                  "設定 ~S ~S で eager と一致しない" config dtype))))))))

(define-iree-test convolution/jit-matches-jax-fixture
    "JAX の conv_general_dilated のフィクスチャ（1〜3 次元、stride・padding（負も）・膨張・グループ）を
conv-general-dilated で書いて IREE (local) で jit した結果は、JAX の出力と f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (dolist (case (read-conv-fixture))
      (destructuring-bind (&key name strides padding lhs-dilation rhs-dilation groups lhs rhs out) case
        (let* ((f (nb:with-tracing (x w)
                    (nb:conv-general-dilated x w :strides strides :padding padding
                                                 :lhs-dilation lhs-dilation :rhs-dilation rhs-dilation
                                                 :feature-group-count groups)))
               (jf (nb:jit f :backend backend)))
          (unwind-protect
               (let ((actual (funcall jf lhs rhs)))
                 (is (equal (array-dimensions out) (array-dimensions actual)) "~A: 形が違う" name)
                 (is (allclose actual out :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: JAX と一致しない" name))
            (nb::%jit-cache-forget f)))))))
