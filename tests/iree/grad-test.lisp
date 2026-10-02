;;;; (jit (grad f)) の IREE 経由 end-to-end テスト（issue #86）。
;;;;
;;;; 期待値は、jit しない eager の (grad f)（grad のロジック自体は
;;;; tests/ad/grad-test.lisp の small テストが中心差分で確かめている）と、
;;;; f64 の中心差分。IREE はコンパイルが遅いので、PBT の試行回数は小さく抑える。
;;;; bf16 は eager が生の (unsigned-byte 16) 配列を扱えないので、f32 に上げた
;;;; 基準値と比べる（tests/iree/jit-test.lisp と同じ方式）。

(in-package #:nabla.iree.tests)

(defun %grad-values-function (g arity)
  "G（勾配のリストを返す grad した関数）を、リストを多値にして返す TRACEABLE-FUNCTION
にする（JIT の出力はリストにできないため）。"
  (nb::%make-traceable-function
   (loop repeat arity collect (gensym "X"))
   (lambda (&rest args) (values-list (apply g args)))))

(defun %grad-arrays-for (graph seed dtype)
  (loop for v in (nb:graph-invars graph) for i from 0
        collect (make-random-array (make-array-spec (nb:aval-shape (nb:var-aval v)) dtype)
                                   :seed (+ seed i))))

(defun %grad-finite-p (arrays)
  (every (lambda (a)
           (dotimes (i (array-total-size a) t)
             (let ((x (row-major-aref a i)))
               (unless (< (abs x) 1f15) (return nil)))))
         arrays))

(defun %grad-jit-matches-eager-p (backend recipe)
  (let* ((graph (build-primitive-graph recipe))
         (seed (mod (sxhash (format nil "~S" recipe)) 100000))
         (arrays (%grad-arrays-for graph seed :f32))
         (argnums (loop for i below (length arrays) collect i))
         (f (scalar-loss-function graph))
         (g (nb:grad f :argnums argnums))
         (eager (apply g arrays)))
    (or (not (%grad-finite-p eager))
        (let ((jitted (nb:jit (%grad-values-function g (length arrays)) :backend backend)))
          (unwind-protect
               (let ((result (multiple-value-list (apply jitted arrays))))
                 (and (= (length result) (length eager))
                      (every (lambda (r e) (allclose r e :dtype :f32)) result eager)))
            (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))))))

(define-iree-test grad/jit-matches-eager-on-random-graphs
    "ランダムな f32 の graph を reduce-sum でスカラーにした関数 f について、
(jit (grad f :argnums 全入力)) の結果（勾配）が、jit しない (grad f) の結果と f32 の
許容誤差で一致する（eager 側が非有限になる graph は対象外）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 8)
        (nb:*compile-cache-directory* nil)
        (*primitive-recipe-dtypes* '(:f32)))
    (is (check-it (generator (primitive-graph-recipe :max-ops 3))
                  (lambda (recipe) (%grad-jit-matches-eager-p backend recipe))
                  :regression-id grad/jit-matches-eager-on-random-graphs
                  :regression-file (regression-path "iree-grad-jit-matches-eager"
                                                    :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

;;; --- 2層 MLP の損失: 中心差分との一致と、2回目以降はコンパイルしない ---

(defun %mlp-loss ()
  "2層 MLP の損失 sum(tanh(x w1 + b1) w2 - y)^2。b1 は行ごとに broadcast する。"
  (nb:with-tracing (x w1 b1 w2 y)
    (let* ((h (tanh (+ (nb:dot x w1) (nb:broadcast-in-dim b1 '(4 3) '(1)))))
           (e (- (nb:dot h w2) y)))
      (nb:reduce-sum (* e e)))))

(defun %mlp-arrays (seed)
  (mapcar (lambda (shape i) (%grad-random-array shape :f64 (+ seed i)))
          '((4 2) (2 3) (3) (3 1) (4 1)) '(0 1 2 3 4)))

(defun %grad-random-array (shape dtype seed)
  (make-random-array (make-array-spec shape dtype) :seed seed))

(define-iree-test grad/jit-mlp-matches-central-difference
    "f64 の2層 MLP の損失 f について、(jit (grad f :argnums (1 2 3))) の勾配が f64 の中心差分と
一致し、同じ形の2回目の呼び出しは再コンパイルしない（*jit-miss-count* が増えない）。
value-and-grad の値は f の eager の結果と一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (loss (%mlp-loss))
         (g (nb:grad loss :argnums '(1 2 3)))
         (jitted (nb:jit (%grad-values-function g 5) :backend backend))
         (vg (nb:jit (nb:value-and-grad loss :argnums 1) :backend backend))
         (arrays (%mlp-arrays 7))
         (graph (nb:trace-to-graph loss (mapcar (lambda (a) (nb:array-aval a)) arrays)))
         (oracle (lambda (&rest xs) (apply #'nb:eval-graph graph xs))))
    (let* ((first-result (multiple-value-list (apply jitted arrays)))
           (misses nb::*jit-miss-count*)
           (second-result (multiple-value-list (apply jitted (%mlp-arrays 8))))
           (expected (central-difference-gradient oracle arrays)))
      (declare (ignore second-result))
      (is (= misses nb::*jit-miss-count*))
      (is (= 3 (length first-result)))
      (is (every (lambda (r e) (allclose r e :dtype :f64 :rtol 1d-4 :atol 1d-6))
                 first-result (subseq expected 1 4)))
      (multiple-value-bind (value grad) (apply vg arrays)
        (is (allclose value (apply oracle arrays) :dtype :f64))
        (is (allclose grad (second expected) :dtype :f64 :rtol 1d-4 :atol 1d-6))))
    (nb::%jit-cache-forget (nb::%jitted-function-fn jitted))
    (nb::%jit-cache-forget (nb::%jitted-function-fn vg))
    (gc-and-run-finalizers)))

;;; --- bf16: f32 に上げた eager の基準値との一致 ---

(define-iree-test grad/jit-bf16-matches-f32-promoted-eager
    "bf16 の f(x) = sum(tanh(x) * x) について、(jit (grad f)) の勾配（bf16）が、x を f32 に
上げた eager の (grad f) と bf16 の許容誤差で一致し、dtype は bf16 のまま。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (nb:with-tracing (x) (nb:reduce-sum (* (tanh x) x))))
         (x (%grad-random-array '(2 3) :bf16 11))
         (x32 (nb::decode-float16-array x :bf16))
         (jitted (nb:jit (nb:grad f) :backend backend))
         (device-x (to-device x backend :dtype :bf16)))
    (unwind-protect
         (let ((result (funcall jitted device-x))
               (expected (funcall (nb:grad f) x32)))
           (is (equal '(2 3) (array-dimensions result)))
           (multiple-value-bind (rtol atol) (dtype-tolerance :bf16)
             (is (allclose (decode-array result :bf16) (decode-array expected :f32) :rtol rtol :atol atol))))
      (release-device-array device-x)
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))

;;; --- grad の他の合成 ---

(nb:defjit %iree-grad-in-defjit (x)
  (funcall (nb:grad (nb:with-tracing (y) (nb:reduce-sum (* y y y)))) x))

(define-iree-test grad/composes-with-defjit-and-jitted-function
    "defjit の本体の中で grad を使える（3x^2）。jit した関数も grad の対象にできる。
(jit (grad (grad f))) は 6x（高階微分）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*default-backend* backend)
         (nb:*compile-cache-directory* nil)
         (x (make-array '(3) :element-type 'single-float :initial-contents '(1f0 -2f0 0.5f0)))
         (f (nb:with-tracing (x) (nb:reduce-sum (* x x x)))))
    (is (allclose (%iree-grad-in-defjit x)
                  (make-array '(3) :element-type 'single-float :initial-contents '(3f0 12f0 0.75f0))
                  :dtype :f32))
    (is (allclose (funcall (nb:grad (nb:jit f :backend backend)) x)
                  (make-array '(3) :element-type 'single-float :initial-contents '(3f0 12f0 0.75f0))
                  :dtype :f32))
    (let ((hessian-diag (funcall (nb:jit (nb:grad (nb:with-tracing (x) (nb:reduce-sum (funcall (nb:grad f) x))))
                                         :backend backend)
                                 x)))
      (is (allclose hessian-diag
                    (make-array '(3) :element-type 'single-float :initial-contents '(6f0 -12f0 3f0))
                    :dtype :f32)))
    (gc-and-run-finalizers)))

;;; --- with-tracing の中の multiple-value-bind を jit で（issue #115） ---

(define-iree-test grad/jit-multiple-value-bind-matches-eager
    "(jit (with-tracing (x y) (multiple-value-bind (v gs) (funcall vg x y) (values v (first gs) (second gs)))))
の結果が、値・勾配とも eager の value-and-grad と f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (vg (nb:value-and-grad (nb:with-tracing (x y) (nb:reduce-sum (+ (tanh (* x y)) x))) :argnums '(0 1)))
         (jitted (nb:jit (nb:with-tracing (x y)
                           (multiple-value-bind (v gs) (funcall vg x y)
                             (values v (first gs) (second gs))))
                         :backend backend))
         (x (%grad-random-array '(3) :f32 11))
         (y (%grad-random-array '(3) :f32 12)))
    (unwind-protect
         (let ((got (multiple-value-list (funcall jitted x y)))
               (expected (multiple-value-bind (v gs) (funcall vg x y) (cons v gs))))
           (is (= 3 (length got)))
           (is (every (lambda (r e) (allclose r e :dtype :f32)) got expected)))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))
