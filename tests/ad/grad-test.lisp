;;;; nb:grad / nb:value-and-grad の性質（issue #86）。
;;;;
;;;; 期待値は grad・jvp・transpose に依存しないオラクルにする:
;;;;   - ランダムな f64 の graph（実プリミティブ）を reduce-sum でスカラーに縮約した関数:
;;;;     central-difference-gradient（tests/support/autodiff.lisp）
;;;;   - 固定の例: 手で微分した式（x^3 -> 3x^2 -> 6x など）
;;;; IREE を通す (jit (grad f)) の一致は tests/iree/grad-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %f64-scalar (x)
  (make-array '() :element-type 'double-float :initial-element x))

(defun %vec (&rest xs)
  (make-array (list (length xs)) :element-type 'double-float :initial-contents xs))

;;; --- PBT: 中心差分との一致 ---

(test grad/matches-central-difference-on-random-graphs
  "実プリミティブのランダムな f64 の graph を reduce-sum でスカラーにした関数 f について、
全入力を argnums にした勾配が f64 の中心差分と許容誤差で一致し、形が入力と同じ。
value-and-grad の値は f の結果と一致する。f が非有限になる graph（exp の連鎖）は対象外。"
  (let ((*primitive-recipe-dtypes* '(:f64)))
    (is (check-it (generator (primitive-graph-recipe :max-ops 4))
                  (lambda (recipe)
                    (let* ((graph (build-primitive-graph recipe))
                           (arrays (%jvp-arrays graph :seed (%recipe-seed recipe)))
                           (oracle (scalar-loss-oracle graph))
                           (expected-value (apply oracle arrays))
                           (argnums (loop for i below (length arrays) collect i)))
                      (or (not (%finite-number-p (aref expected-value)))
                          (multiple-value-bind (value grads)
                              (apply (nb:value-and-grad (scalar-loss-function graph) :argnums argnums) arrays)
                            (and (= (length grads) (length arrays))
                                 (every (lambda (g a) (equal (array-dimensions g) (array-dimensions a)))
                                        grads arrays)
                                 (allclose value expected-value :dtype :f64)
                                 (%results-close-p grads (central-difference-gradient oracle arrays)
                                                   :rtol *autodiff-rtol* :atol *autodiff-atol*))))))
                  :regression-id grad/matches-central-difference-on-random-graphs
                  :regression-file (regression-path "grad-central-difference")))))

;;; --- 固定の例 ---

(test grad/cubic-and-higher-order
  "f(x) = x^3: grad は 3x^2、grad (grad f) は 6x（rank 0 は実数でも配列でも）、
3階微分は 6。配列版は reduce-sum で縮約した関数の勾配、その勾配の和の勾配。"
  (let* ((f (nb:with-tracing (x) (* x x x)))
         (g (nb:grad f))
         (gg (nb:grad g))
         (ggg (nb:grad gg)))
    (is (equalp (%f64-scalar 12d0) (funcall g 2d0)))
    (is (equalp (%f64-scalar 12d0) (funcall gg 2d0)))
    (is (equalp (%f64-scalar -18d0) (funcall gg -3d0)))
    (is (equalp (%f64-scalar 6d0) (funcall ggg 2d0)))
    (is (equalp (%f64-scalar 12d0) (funcall gg (%f64-scalar 2d0)))))
  (let* ((f (nb:with-tracing (x) (nb:reduce-sum (* x x x))))
         (g (nb:grad f))
         (sum-of-g (nb:with-tracing (x) (nb:reduce-sum (funcall g x))))
         (x (%vec 1d0 -2d0 0.5d0)))
    (is (equalp (%vec 3d0 12d0 0.75d0) (funcall g x)))
    (is (equalp (%vec 6d0 -12d0 3d0) (funcall (nb:grad sum-of-g) x)))))

(test grad/value-and-grad-returns-value-and-gradient
  "value-and-grad は多値の (値 勾配)。値は f の結果と一致する（f の出力は rank 0）。"
  (let ((f (nb:with-tracing (x) (nb:reduce-sum (* x x))))
        (x (%vec 1d0 2d0 3d0)))
    (multiple-value-bind (value grad) (funcall (nb:value-and-grad f) x)
      (is (equalp (%f64-scalar 14d0) value))
      (is (equalp (%vec 2d0 4d0 6d0) grad)))))

(test grad/argnums-selects-arguments-and-result-shape
  "argnums が整数なら勾配1つ、リストならリスト（argnums の順）。f(x, y) = sum(x * y) の
勾配は d/dx = y、d/dy = x。複数の引数・既定の argnums は 0。"
  (let ((f (nb:with-tracing (x y) (nb:reduce-sum (* x y))))
        (x (%vec 1d0 2d0))
        (y (%vec 3d0 5d0)))
    (is (equalp y (funcall (nb:grad f) x y)))
    (is (equalp y (funcall (nb:grad f :argnums 0) x y)))
    (is (equalp x (funcall (nb:grad f :argnums 1) x y)))
    (is (equalp (list y x) (funcall (nb:grad f :argnums '(0 1)) x y)))
    (is (equalp (list x y) (funcall (nb:grad f :argnums '(1 0)) x y)))
    (is (equalp (list x) (funcall (nb:grad f :argnums '(1)) x y)))
    (multiple-value-bind (value grads) (funcall (nb:value-and-grad f :argnums '(1 0)) x y)
      (is (equalp (%f64-scalar 13d0) value))
      (is (equalp (list x y) grads)))))

(test grad/invalid-argnums-signal-autodiff-error
  "範囲外・負・重複・整数でない・空の argnums は、(grad f) を作る時点で autodiff-error。"
  (let ((f (nb:with-tracing (x y) (+ x y))))
    (dolist (argnums '(2 -1 (0 2) (0 0) :x (0 :x) ()))
      (signals nb:autodiff-error (nb:grad f :argnums argnums))
      (signals nb:autodiff-error (nb:value-and-grad f :argnums argnums)))))

(test grad/non-traceable-function-signals-autodiff-error
  "WITH-TRACING / JIT が作った関数でないもの、静的引数のある jit した関数は autodiff-error。"
  (signals nb:autodiff-error (nb:grad (lambda (x) x)))
  (signals nb:autodiff-error
    (nb:grad (nb:jit (nb:with-tracing (x n) (* x n)) :static-args '(1)))))

(test grad/wrong-argument-count-signals-autodiff-error
  (signals nb:autodiff-error (funcall (nb:grad (nb:with-tracing (x y) (* x y))) 1d0)))

(test grad/non-scalar-output-signals-grad-requires-scalar-output
  "出力が rank 0 でない（配列）、浮動小数点でない（:i1）、1つでない場合は
grad-requires-scalar-output（autodiff-error の子）。出力の aval を読める。"
  (let ((x (%vec 1d0 2d0)))
    (handler-case (funcall (nb:grad (nb:with-tracing (x) (* x x))) x)
      (nb:grad-requires-scalar-output (c)
        (is (equalp (nb:make-aval '(2) :f64) (nb:grad-requires-scalar-output-aval c)))
        (is (typep c 'nb:autodiff-error)))
      (:no-error (&rest values) (declare (ignore values)) (fail "シグナルされなかった")))
    (signals nb:grad-requires-scalar-output
      (funcall (nb:value-and-grad (nb:with-tracing (x) (* x x))) x))
    (signals nb:grad-requires-scalar-output
      (funcall (nb:grad (nb:with-tracing (x) (< x 0d0))) 1d0))
    (signals nb:grad-requires-scalar-output
      (funcall (nb:grad (nb:with-tracing (x) (values (* x x) x))) 1d0))))

(test grad/non-float-argument-in-argnums-signals-autodiff-error
  "微分する引数が浮動小数点でない（:i1）ときは autodiff-error。そうでない引数は
:i1 でもよい（選ぶ側）。"
  (let ((f (nb:with-tracing (x p) (nb:reduce-sum (nb:where p x (* x 2d0)))))
        (x (%vec 1d0 2d0))
        (p (make-array '(2) :element-type 'bit :initial-contents '(1 0))))
    (signals nb:autodiff-error (funcall (nb:grad f :argnums 1) x p))
    (is (equalp (%vec 1d0 2d0) (funcall (nb:grad f) x p)))))

(test grad/gradient-has-the-dtype-of-the-argument
  "勾配の dtype は引数と同じ: f32 の配列には f32、f64 には f64。実数は rank 0 の配列として扱う。"
  (let ((f (nb:with-tracing (x) (nb:reduce-sum (* x x))))
        (x32 (make-array '(2) :element-type 'single-float :initial-contents '(1f0 2f0))))
    (let ((g (funcall (nb:grad f) x32)))
      (is (eq 'single-float (array-element-type g)))
      (is (equalp (make-array '(2) :element-type 'single-float :initial-contents '(2f0 4f0)) g)))
    (let ((g (funcall (nb:grad (nb:with-tracing (x) (* x x))) 3f0)))
      (is (eq 'single-float (array-element-type g)))
      (is (equal '() (array-dimensions g)))
      (is (= 6f0 (aref g))))))

;;; --- トレースの中での grad ---

(test grad/inside-with-tracing-body-and-trace-to-graph
  "with-tracing の本体の中で grad を使える。トレースすると graph になり（grad は eqn に展開される）、
eval-graph すると eager の grad と一致する。"
  (let* ((inner (nb:with-tracing (y) (* y y y)))
         (f (nb:with-tracing (x) (+ (* 2d0 (funcall (nb:grad inner) x)) x)))
         (graph (nb:trace-to-graph f (list (nb:make-aval '() :f64)))))
    (is (plusp (length (nb:graph-eqns graph))))
    (is (equalp (%f64-scalar 26d0) (nb:eval-graph graph (%f64-scalar 2d0))))
    (is (equalp (%f64-scalar 26d0) (funcall f 2d0)))))

(test grad/trace-to-graph-of-grad-matches-eager
  "(trace-to-graph (grad f) avals) の graph を eval-graph した結果が (grad f) の eager の結果と一致する。"
  (let* ((f (nb:with-tracing (x w) (nb:reduce-sum (tanh (nb:dot x w)))))
         (x (make-random-array (make-array-spec '(2 3) :f64) :seed 1))
         (w (make-random-array (make-array-spec '(3 4) :f64) :seed 2))
         (g (nb:grad f :argnums '(0 1)))
         (graph (nb:trace-to-graph (nb:with-tracing (x w) (values-list (funcall g x w)))
                                   (list (nb:make-aval '(2 3) :f64) (nb:make-aval '(3 4) :f64)))))
    (is (%results-close-p (multiple-value-list (nb:eval-graph graph x w)) (funcall g x w)
                          :rtol 1d-12 :atol 1d-12))))

(test grad/capturing-an-outer-tracer-in-a-closure-is-a-tracing-error
  "既知の制限: f が外側のトレースのトレーサを閉包で捕まえていると tracing-error。
外側の値を f の引数として渡せば動く。"
  (signals nb:tracing-error
    (nb:trace-to-graph
     (nb:with-tracing (x y) (funcall (nb:grad (nb:with-tracing (z) (* z y))) x))
     (list (nb:make-aval '() :f64) (nb:make-aval '() :f64))))
  (let ((graph (nb:trace-to-graph
                (nb:with-tracing (x y) (funcall (nb:grad (nb:with-tracing (z w) (* z w)) :argnums 0) x y))
                (list (nb:make-aval '() :f64) (nb:make-aval '() :f64)))))
    (is (equalp (%f64-scalar 5d0) (nb:eval-graph graph (%f64-scalar 2d0) (%f64-scalar 5d0))))))

(test grad/returns-a-traceable-function
  "(grad f) は traceable-function を返す。"
  (let ((f (nb:with-tracing (x) (* x x))))
    (is (typep (nb:grad f) 'nb:traceable-function))))

(test grad/string-argument-signals-autodiff-error
  (signals nb:autodiff-error (funcall (nb:grad (nb:with-tracing (x) x)) "abc")))

;;; --- with-tracing の中で multiple-value-bind が使える（issue #115） ---

(defparameter *mvb-avals* (list (nb:make-aval '(3) :f64) (nb:make-aval '(3) :f64)))

(defun %mvb-vg ()
  (nb:value-and-grad (nb:with-tracing (x y) (nb:reduce-sum (+ (tanh (* x y)) x))) :argnums '(0 1)))

(defun %mvb-graph (vg)
  "multiple-value-bind で (値 勾配リスト) を受ける with-tracing を graph にする。"
  (nb:trace-to-graph
   (nb:with-tracing (x y)
     (multiple-value-bind (v gs) (funcall vg x y)
       (values v (first gs) (second gs))))
   *mvb-avals*))

(defun %flat-value-and-grads (vg &rest args)
  "VG（argnums がリストの value-and-grad）を ARGS で呼び、(値 勾配...) を平らな多値にする
普通の関数（multiple-value-bind を使わずに with-tracing から呼ぶ従来の回避法）。"
  (let ((all (multiple-value-list (apply vg args))))
    (values-list (cons (first all) (second all)))))

(defun %values-list-graph (vg)
  "従来の回避法（普通の関数に切り出す）の graph。"
  (nb:trace-to-graph
   (nb:with-tracing (x y) (%flat-value-and-grads vg x y))
   *mvb-avals*))

(test grad/multiple-value-bind-of-value-and-grad-matches-values-list
  "with-tracing の本体で (multiple-value-bind (v g) (funcall (value-and-grad f) x y) ...) と
受けた graph を eval-graph した結果が、(values-list ...) で平らにした回避法の graph の
結果、および eager の value-and-grad の結果と一致する（ランダムな入力）。"
  (let* ((vg (%mvb-vg))
         (mvb (%mvb-graph vg))
         (old (%values-list-graph vg)))
    (is (check-it (generator (integer 0 100000))
                  (lambda (seed)
                    (let* ((arrays (list (make-random-array (make-array-spec '(3) :f64) :seed seed)
                                         (make-random-array (make-array-spec '(3) :f64) :seed (+ seed 1))))
                           (got (multiple-value-list (apply #'nb:eval-graph mvb arrays)))
                           (via-list (multiple-value-list (apply #'nb:eval-graph old arrays)))
                           (eager (multiple-value-bind (v gs) (apply vg arrays) (cons v gs))))
                      (and (= 3 (length got))
                           (%results-close-p got via-list :rtol 1d-12 :atol 1d-12)
                           (%results-close-p got eager :rtol 1d-12 :atol 1d-12))))
                  :regression-id grad/multiple-value-bind-of-value-and-grad-matches-values-list
                  :regression-file (regression-path "grad-multiple-value-bind")))))

(test grad/multiple-value-bind-with-fewer-or-more-variables
  "受ける変数が値の数より少なければ余りは捨て、多ければ NIL。"
  (let* ((f (nb:with-tracing (x) (* x x)))
         (vg (nb:value-and-grad f))
         (only-value (nb:with-tracing (x) (multiple-value-bind (v) (funcall vg x) v)))
         (third-is-nil (nb:with-tracing (x) (multiple-value-bind (v g z) (funcall vg x) (list v g z)))))
    (is (equalp (%f64-scalar 9d0) (funcall only-value 3d0)))
    (is (null (third (funcall third-is-nil 3d0))))))

(test grad/multiple-value-call-of-lambda-over-value-and-grad
  "multiple-value-call に (lambda (v g) ...) を直接渡す形も動く（graph も eager と一致）。"
  (let* ((vg (nb:value-and-grad (nb:with-tracing (x) (* x x))))
         (f (nb:with-tracing (x) (multiple-value-call (lambda (v g) (+ v g)) (funcall vg x))))
         (graph (nb:trace-to-graph f (list (nb:make-aval '() :f64)))))
    (is (equalp (%f64-scalar 15d0) (funcall f (%f64-scalar 3d0))))
    (is (equalp (%f64-scalar 15d0) (nb:eval-graph graph (%f64-scalar 3d0))))))

(test grad/multiple-value-bind-passes-through-non-tracer-values
  "トレーサ以外（実数・配列）の多値も、ふつうの Lisp の多値としてそのまま受け取れる。"
  (let ((f (nb:with-tracing (x) (multiple-value-bind (a b) (values 2d0 x) (* a b)))))
    (is (equalp (%f64-scalar 6d0) (funcall f (%f64-scalar 3d0))))
    (is (equalp (%f64-scalar 6d0)
                (nb:eval-graph (nb:trace-to-graph f (list (nb:make-aval '() :f64))) (%f64-scalar 3d0))))))
