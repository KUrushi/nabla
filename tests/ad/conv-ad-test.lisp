;;;; convolution の jvp / transpose / バッチ化ルールの性質（issue #203）。
;;;;
;;;; 守らせる性質（期待値は grad・jvp・transpose・vmap に依存しないオラクル）:
;;;;   - jvp: f64 の中心差分と一致する。接線について線形。片側の接線だけ非ゼロでもよい
;;;;   - transpose: 内積の恒等式 <conv(v, w), u> = <v, T_lhs(u)> と <conv(x, v), u> = <v, T_rhs(u)>
;;;;     を REFERENCE-CONV（窓を数え上げる別実装）で確かめる。stride・padding（負も）・
;;;;     膨張・グループ・depthwise に加え、batch-group-count > 1（2階微分で現れる畳み込み）も
;;;;   - vmap: バッチ軸で切り出した各要素に eager の convolution をかけて積み直したもの
;;;;     （REFERENCE-VMAP）と一致する。lhs だけ・rhs だけ・両方、feature / batch group あり
;;;;   - grad: 中心差分の勾配と一致する。2階微分（grad の grad）も中心差分と一致する
;;;; ケースの生成器は tests/primitives/conv-test.lisp のもの。jvp-shape-test.lisp /
;;;; transpose-rules-test.lisp / vmap-shape-test.lisp のヘルパーを使うのでその後にロードする。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %conv-ad-case ()
  "(lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation feature-groups batch-groups)。
3つに1つが batch-group-count > 1、残りは feature-group-count 1〜3（depthwise を含む）。"
  (if (zerop (random 3))
      (%conv-bgc-random-case)
      (append (%conv-random-case) (list 1))))

(defun %conv-ad-generator () (%ad-hoc-generator #'%conv-ad-case))

(defun %conv-ad-params (case)
  (destructuring-bind (lhs-shape rhs-shape strides padding ld rd fgc bgc) case
    (declare (ignore lhs-shape rhs-shape))
    (%conv-params-of strides padding ld rd fgc bgc)))

(defun %conv-ad-avals (case)
  (list (nb:make-aval (first case) :f64) (nb:make-aval (second case) :f64)))

(defun %conv-ad-graph (case)
  (apply #'%prim-graph :convolution (%conv-ad-avals case) (%conv-ad-params case)))

(defun %conv-ad-reference (case lhs rhs)
  (destructuring-bind (lhs-shape rhs-shape strides padding ld rd fgc bgc) case
    (declare (ignore lhs-shape rhs-shape))
    (reference-conv lhs rhs strides padding ld rd fgc bgc)))

;;; --- jvp ---

(test conv-ad/jvp-matches-central-difference
  "convolution の jvp は、ランダムな stride・padding・膨張・グループ・batch group で f64 の中心差分と一致する。"
  (let ((*num-trials* 30))
    (is (check-it (%conv-ad-generator)
                  (lambda (case) (%jvp-matches-central-difference-p (%conv-ad-graph case)))
                  :regression-id conv-ad/jvp-matches-central-difference
                  :regression-file (regression-path "conv-ad-jvp")))))

(test conv-ad/jvp-tangent-is-linear
  "convolution の jvp は接線について線形（スケール倍と加法）。"
  (let ((*num-trials* 20))
    (is (check-it (%conv-ad-generator)
                  (lambda (case) (%jvp-tangent-is-linear-p (%conv-ad-graph case)))
                  :regression-id conv-ad/jvp-tangent-is-linear
                  :regression-file (regression-path "conv-ad-jvp-linear")))))

(test conv-ad/jvp-one-sided-nonzero-drops-the-other-term
  "片側の接線だけ非ゼロにした jvp は、もう片側の接線に 0 を入れた完全な jvp と一致する。"
  (let ((*num-trials* 20))
    (is (check-it (%conv-ad-generator)
                  (lambda (case)
                    (let* ((graph (%conv-ad-graph case))
                           (full (nb::jvp-graph graph))
                           (primals (%jvp-arrays graph))
                           (tangents (%jvp-arrays graph :tangent t)))
                      (every (lambda (nonzero)
                               (let* ((partial (nb::jvp-graph graph :nonzero nonzero))
                                      (kept (loop for tg in tangents for f in nonzero when f collect tg))
                                      (zeroed (loop for tg in tangents for f in nonzero
                                                    collect (if f tg (%scale-array tg 0)))))
                                 (%results-close-p (%jvp-eval partial (append primals kept))
                                                   (%jvp-eval full (append primals zeroed)))))
                             '((t nil) (nil t)))))
                  :regression-id conv-ad/jvp-one-sided-nonzero
                  :regression-file (regression-path "conv-ad-jvp-one-sided")))))

;;; --- transpose ---

(defun %conv-linear-graph (case lhs-linear-p)
  "CASE の convolution の graph。入力は既知の側、線形な側の順。"
  (let ((params (%conv-ad-params case))
        (avals (%conv-ad-avals case)))
    (flet ((conv (x w) (apply #'nb::%trace-eqn :convolution (list x w) params)))
      (if lhs-linear-p
          (%tr-graph (list (second avals) (first avals)) (lambda (w x) (conv x w)))
          (%tr-graph avals (lambda (x w) (conv x w)))))))

(defun %conv-adjoint-case-p (case)
  "CASE について、lhs が線形・rhs が線形の両方で、転置の随伴性が REFERENCE-CONV と一致すれば真。"
  (let* ((seed (mod (sxhash case) 100000))
         (shapes (list (first case) (second case))))
    (flet ((rnd (shape s) (make-random-array (make-array-spec shape :f64) :seed s)))
      (let* ((x (rnd (first shapes) (+ seed 1))) (w (rnd (second shapes) (+ seed 2)))
             (vx (rnd (first shapes) (+ seed 3))) (vw (rnd (second shapes) (+ seed 4)))
             (out-shape (array-dimensions (%conv-ad-reference case x w)))
             (u (random-cotangent (nb:make-aval out-shape :f64) :seed (+ seed 5) :dtype :f64))
             (t-lhs (first (%jvp-eval (nb::transpose-graph (%conv-linear-graph case t) 1) (list w u))))
             (t-rhs (first (%jvp-eval (nb::transpose-graph (%conv-linear-graph case nil) 1) (list x u)))))
        (and (equal (array-dimensions t-lhs) (first shapes))
             (equal (array-dimensions t-rhs) (second shapes))
             (%tr-close-p (inner-product t-lhs vx) (inner-product u (%conv-ad-reference case vx w)) 1d-9)
             (%tr-close-p (inner-product t-rhs vw) (inner-product u (%conv-ad-reference case x vw)) 1d-9))))))

(test conv-ad/transpose-adjoint-against-reference
  "convolution の転置は随伴: lhs が線形のとき <T(u), v> = <u, conv(v, w)>、rhs が線形のとき
<T(u), v> = <u, conv(x, v)>（REFERENCE-CONV）。stride・padding（負も）・lhs / rhs の膨張・
feature group（depthwise を含む）・batch group をランダムにする。"
  (let ((*num-trials* 60))
    (is (check-it (%conv-ad-generator) #'%conv-adjoint-case-p
                  :regression-id conv-ad/transpose-adjoint-against-reference
                  :regression-file (regression-path "conv-ad-transpose-adjoint")))))

(test conv-ad/transpose-adjoint-covers-depthwise-and-strided-dilated
  "固定の例: depthwise（G = C = O）、stride 2 と lhs / rhs の膨張と負の padding、1 次元、3 次元、
batch group（B = 2）で転置の随伴性が成り立つ（PBT が通るべき形の下限）。"
  (dolist (case '(((1 6 6 3) (3 3 1 3) (1 1) ((1 1) (1 1)) (1 1) (1 1) 3 1)
                  ((2 7 6 2) (3 2 2 4) (2 3) ((1 -1) (2 0)) (2 1) (1 2) 1 1)
                  ((3 9 2) (4 2 3) (2) ((1 2)) (1) (1) 1 1)
                  ((1 4 4 4 2) (2 2 2 2 3) (1 2 1) ((0 0) (1 0) (0 1)) (1 1 1) (1 1 1) 1 1)
                  ((4 5 3) (2 3 6) (2) ((1 0)) (1) (1) 1 2)
                  ((2 5 5 4) (2 2 2 6) (1 1) ((0 0) (0 0)) (1 1) (1 1) 2 1)))
    (is (%conv-adjoint-case-p case) "case ~S" case)))

(defun %transposed-eqn-names (case lhs-linear-p)
  "CASE の convolution を転置した graph の eqn のプリミティブ名を、名前順に並べたリストで返す。"
  (sort (mapcar (lambda (e) (nb:primitive-name (nb:eqn-prim e)))
                (nb:graph-eqns (nb::transpose-graph (%conv-linear-graph case lhs-linear-p) 1)))
        #'string< :key #'symbol-name))

(test conv-ad/transpose-adds-no-redundant-reshapes-or-transposes
  "グループの無い畳み込みの転置は、必要な軸の並べ替えだけを足す（恒等の transpose / reshape を足さない）:
lhs の転置は rev・transpose（入出力チャネルの入れ替え）・convolution、rhs の転置は transpose 3つ
（lhs のバッチ軸とチャネル軸、ct のバッチ軸、結果の並び）・convolution。feature group があればカーネルの reshape が2つ増え、
batch group があれば lhs の転置に reshape がさらに2つと transpose が1つ増える。"
  (let ((plain '((2 5 5 3) (3 3 3 4) (1 1) ((1 1) (1 1)) (1 1) (1 1) 1 1)))
    (is (equal '(:convolution :rev :transpose) (%transposed-eqn-names plain t)))
    (is (equal '(:convolution :transpose :transpose :transpose) (%transposed-eqn-names plain nil))))
  (let ((grouped '((2 5 5 4) (2 2 2 6) (1 1) ((0 0) (0 0)) (1 1) (1 1) 2 1)))
    (is (equal '(:convolution :reshape :reshape :rev :transpose) (%transposed-eqn-names grouped t))))
  (let ((batch-grouped '((4 5 3) (2 3 6) (2) ((1 0)) (1) (1) 1 2)))
    (is (equal '(:convolution :reshape :reshape :reshape :reshape :rev :transpose :transpose)
               (%transposed-eqn-names batch-grouped t)))))

(test conv-ad/transpose-with-both-operands-linear-signals-autodiff-error
  "両方の入力が線形な convolution（接線どうしの積）の転置は autodiff-error。"
  (let* ((case '((1 5 2) (2 2 3) (1) ((0 0)) (1) (1) 1 1))
         (graph (%conv-ad-graph case)))
    (signals nb:autodiff-error (nb::transpose-graph graph 0))))

;;; --- grad ---

(defun %conv-loss-graph (case)
  "sum(conv(x, w)^2) のための graph（出力は conv(x, w)^2）。"
  (let ((params (%conv-ad-params case)))
    (%tr-graph (%conv-ad-avals case)
               (lambda (x w)
                 (let ((y (apply #'nb::%trace-eqn :convolution (list x w) params)))
                   (nb::%trace-eqn :mul (list y y)))))))

(test conv-ad/grad-matches-central-difference
  "sum(conv(x, w)^2) の grad（x と w の両方）は f64 の中心差分の勾配と一致する。
ランダムな stride・padding・膨張・グループ・batch group。"
  (let ((*num-trials* 20))
    (is (check-it (%conv-ad-generator)
                  (lambda (case)
                    (let* ((graph (%conv-loss-graph case))
                           (arrays (%jvp-arrays graph :seed (mod (sxhash case) 100000))))
                      (%results-close-p
                       (nth-value 1 (apply (nb:value-and-grad (scalar-loss-function graph) :argnums '(0 1))
                                           arrays))
                       (central-difference-gradient (scalar-loss-oracle graph) arrays)
                       :rtol *autodiff-rtol* :atol *autodiff-atol*)))
                  :regression-id conv-ad/grad-matches-central-difference
                  :regression-file (regression-path "conv-ad-grad")))))

(test conv-ad/second-order-grad-matches-central-difference
  "grad の grad: h(x, w) = sum(dL/dw(x, w)^2)（L = sum(conv(x, w)^2)）の勾配が、h を eager で
評価した中心差分と一致する。1階微分の転置は batch-group-count > 1 の畳み込み（feature group のとき）や
rev を含むので、それらの jvp / transpose も通る。"
  (dolist (case '(((1 4 4 2) (2 2 1 2) (1 1) ((0 0) (1 0)) (1 1) (1 1) 2 1)   ; depthwise
                  ((2 4 3) (2 3 3) (2) ((1 0)) (1) (2) 1 1)
                  ((2 3 3 2) (2 2 2 2) (1 1) ((0 0) (0 0)) (1 1) (1 1) 1 2)))
    (let* ((params (%conv-ad-params case))
           (loss (nb::%make-traceable-function
                  '(x w)
                  (lambda (x w)
                    (let ((y (apply #'nb::%trace-eqn :convolution (list x w) params)))
                      (nb:reduce-sum (nb::%trace-eqn :mul (list y y)))))))
           (grad-w (nb:grad loss :argnums 1))
           (h (nb::%make-traceable-function
               '(x w)
               (lambda (x w)
                 (let ((g (funcall grad-w x w)))
                   (nb:reduce-sum (nb::%trace-eqn :mul (list g g)))))))
           (arrays (list (make-random-array (make-array-spec (first case) :f64) :seed 11)
                         (make-random-array (make-array-spec (second case) :f64) :seed 12)))
           (oracle (lambda (x w)
                     (let ((g (funcall grad-w x w)))
                       (make-array '() :element-type 'double-float
                                       :initial-element (inner-product g g))))))
      (is (%results-close-p (nth-value 1 (apply (nb:value-and-grad h :argnums '(0 1)) arrays))
                            (central-difference-gradient oracle arrays)
                            :rtol 1d-5 :atol 1d-6)
          "case ~S" case))))

;;; --- JAX のフィクスチャ ---

(test conv-ad/vjp-matches-jax-fixture
  "jax.vjp の結果（余接線 ct に対する lhs と rhs の勾配。stride・padding（負も）・膨張・グループ・depthwise・
1〜3 次元・batch-group-count 2）と、nabla の vjp（転置ルール）の f32 の結果が許容誤差で一致する。
許容誤差を 1e-4 / 1e-5 にしているのは、総和の順序が JAX と違うため。"
  (let ((cases (read-conv-fixture "conv-grad")))
    (is (>= (length cases) 10))
    (dolist (case cases)
      (destructuring-bind (&key name strides padding lhs-dilation rhs-dilation groups batch-groups
                             lhs rhs out ct dlhs drhs)
          case
        (declare (ignore out))
        (let ((actual (%vjp-cotangents :convolution
                                       (list (nb:array-aval lhs :f32) (nb:array-aval rhs :f32))
                                       (%conv-params-of strides padding lhs-dilation rhs-dilation
                                                        groups batch-groups)
                                       (list lhs rhs ct))))
          (is (equal (array-dimensions dlhs) (array-dimensions (first actual))) "~A: dlhs の形が違う" name)
          (is (equal (array-dimensions drhs) (array-dimensions (second actual))) "~A: drhs の形が違う" name)
          (is (allclose (first actual) dlhs :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: dlhs が JAX と一致しない" name)
          (is (allclose (second actual) drhs :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: drhs が JAX と一致しない" name))))))

(test convolution/eager-batch-groups-matches-jax-fixture
  "batch-group-count 2 の eager の convolution は JAX の出力と一致する（フィクスチャ conv-grad の batch-groups-* ）。"
  (dolist (case (read-conv-fixture "conv-grad"))
    (destructuring-bind (&key name strides padding lhs-dilation rhs-dilation groups batch-groups lhs rhs out
                           &allow-other-keys)
        case
      (when (> batch-groups 1)
        (let ((actual (%eager-of :convolution (list lhs rhs)
                                 (list (nb:array-aval lhs :f32) (nb:array-aval rhs :f32))
                                 :strides strides :padding padding :lhs-dilation lhs-dilation
                                 :rhs-dilation rhs-dilation :feature-group-count groups
                                 :batch-group-count batch-groups)))
          (is (equal (array-dimensions out) (array-dimensions actual)) "~A: 形が違う" name)
          (is (allclose actual out :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: JAX と一致しない" name))))))

;;; --- vmap ---

(defun %conv-vmap-property (input)
  "INPUT = (case size mode code-l code-r code-out)。MODE は 0 lhs だけ / 1 rhs だけ / 2 両方 をバッチする。"
  (destructuring-bind (case size mode code-l code-r code-out) input
    (let* ((f (primitive-function :convolution (%conv-ad-params case) 2))
           (lhs-shape (first case))
           (rhs-shape (second case))
           (out-rank (length lhs-shape))
           (al (and (member mode '(0 2)) (mod code-l (1+ (length lhs-shape)))))
           (ar (and (member mode '(1 2)) (mod code-r (1+ (length rhs-shape)))))
           (out (mod code-out (1+ out-rank)))
           (seed (mod (sxhash case) 100000))
           (x (%vmap-batched-array lhs-shape size al seed))
           (y (%vmap-batched-array rhs-shape size ar (1+ seed)))
           (expected (reference-vmap f (list x y) :in-axes (list al ar) :out-axes out))
           (actual (funcall (nb:vmap f :in-axes (list al ar) :out-axes out) x y)))
      (allclose actual (first expected) :dtype :f64))))

(defun %conv-vmap-generator ()
  (generator (tuple (%conv-ad-generator)
                    (uniform-integer :lo 1 :hi 3)   ; バッチの長さ
                    (uniform-integer :lo 0 :hi 2)   ; 0 lhs だけ 1 rhs だけ 2 両方
                    (uniform-integer :lo 0 :hi 99)  ; lhs のバッチ軸
                    (uniform-integer :lo 0 :hi 99)  ; rhs のバッチ軸
                    (uniform-integer :lo 0 :hi 99)))) ; out-axes

(test conv-ad/vmap-matches-per-slice-reference
  "convolution の vmap は、lhs だけ・rhs だけ・両方をバッチし、バッチ軸の位置・out-axes・
feature group・batch group・stride・padding・膨張をランダムにしても、各要素に eager の
convolution をかけて積み直したものと一致する。"
  (let ((*num-trials* 60))
    (is (check-it (%conv-vmap-generator) #'%conv-vmap-property
                  :regression-id conv-ad/vmap-matches-per-slice-reference
                  :regression-file (regression-path "conv-ad-vmap")))))

(defun %vmapped-eqn-names (case in-axes)
  "CASE の convolution を IN-AXES（lhs, rhs）で vmap した graph の eqn のプリミティブ名を、名前順に並べたリストで返す。"
  (let* ((f (primitive-function :convolution (%conv-ad-params case) 2))
         (avals (loop for shape in (list (first case) (second case)) for axis in in-axes
                      collect (nb:make-aval (if axis (axis-at shape 3 axis) shape) :f64)))
         (graph (nb:trace-to-graph (nb:vmap f :in-axes in-axes :out-axes 0) avals)))
    (sort (mapcar (lambda (e) (nb:primitive-name (nb:eqn-prim e))) (nb:graph-eqns graph))
          #'string< :key #'symbol-name)))

(test conv-ad/vmap-folds-the-batch-into-one-convolution
  "vmap した畳み込みは、バッチ軸を既存の軸に併合した畳み込み1つに畳まれる（V 回の畳み込みにはならない）。
グループの無い lhs だけ（バッチ軸 0）のバッチ化は、reshape 2つ（併合と分割）と convolution だけ。
rhs だけ・両方のときは、併合する軸を動かす transpose と出力のバッチ軸を out-axes に動かす transpose が加わる。
グループがあるときは、グループを外側に保つための reshape / transpose が増える。"
  (let ((plain '((2 5 5 3) (3 3 3 4) (1 1) ((1 1) (1 1)) (1 1) (1 1) 1 1))
        (grouped '((2 5 5 4) (2 2 2 6) (1 1) ((0 0) (0 0)) (1 1) (1 1) 2 1)))
    (is (equal '(:convolution :reshape :reshape) (%vmapped-eqn-names plain '(0 nil))))
    (is (equal '(:convolution :reshape :reshape :transpose :transpose) (%vmapped-eqn-names plain '(nil 0))))
    (is (equal '(:convolution :reshape :reshape :reshape :transpose :transpose :transpose)
               (%vmapped-eqn-names plain '(0 0))))
    (is (= 1 (count :convolution (%vmapped-eqn-names grouped '(nil 0)))))
    (is (> (count :reshape (%vmapped-eqn-names grouped '(nil 0)))
           (count :reshape (%vmapped-eqn-names plain '(nil 0)))))))

(test conv-ad/vmap-covers-every-mode-and-group-kind
  "固定の例: {通常, feature group, batch group} × {lhs だけ, rhs だけ, 両方}（PBT が各モードを必ず通ることの下限）。"
  (dolist (case '(((2 5 5 3) (3 3 3 4) (1 1) ((1 1) (1 1)) (1 1) (1 1) 1 1)
                  ((2 5 5 4) (2 2 2 6) (1 1) ((0 0) (0 0)) (1 1) (1 1) 2 1)
                  ((1 6 4) (3 1 4) (2) ((1 1)) (1) (1) 4 1)
                  ((4 5 3) (2 3 6) (2) ((1 0)) (1) (1) 1 2)))
    (dolist (mode '(0 1 2))
      ;; (3 3 3) と (0 4 4) は、rhs のバッチ軸が出力チャネル軸より後ろにある場合を通す。
      (dolist (axes '((0 0 0) (1 2 1) (2 1 2) (3 3 3) (0 4 4)))
        (is (%conv-vmap-property (list case 2 mode (first axes) (second axes) (third axes)))
            "case ~S mode ~S axes ~S" case mode axes)))))
