;;;; conv レイヤー（Flax の nn.Conv。issue #207、small）。
;;;;
;;;; 守らせる性質:
;;;; - Flax との数値一致: 同じパラメータ・入力・余接線で、順伝播の出力と、パラメータ・入力についての
;;;;   勾配が Flax のフィクスチャ（tests/fixtures/layers/flax-conv.lisp、生成は同じディレクトリの
;;;;   generate-conv.py）と一致する（eager。jit は medium、tests/nn/conv-iree-test.lisp）。SAME・
;;;;   VALID・明示の padding、stride、偶数の kernel、グループ・depthwise、kernel の dilation、
;;;;   バッチ軸の無い入力、1次元を含む。
;;;; - SAME の padding は Flax（JAX の padtype_to_pads）と同じ: 出力の長さは ceil(入力 / stride)、
;;;;   padding の合計は max((出力 - 1) stride + 実効 kernel - 入力, 0)、左は合計の半分（切り捨て）、
;;;;   右が残り。
;;;; - 計算: 出力は、別実装の畳み込み（reference-conv）に bias を足したもの。1x1 の kernel は
;;;;   dense と同じ。バッチ軸の無い入力は、バッチ1の出力の先頭と同じ。
;;;; - init: 形 (kernel... in/G out)、dtype、スロット名（Flax と同じ kernel / bias）、lecun_normal
;;;;   （fan_in = kernel の面積 × in/G）の標準偏差と切断、bias = 0。不正な引数は error。
;;;; - vmap に通せる（各サンプルを単独に call した結果と一致）。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

;;; --- Flax フィクスチャ ---

(defvar *conv-fixture* nil)

(defun %conv-fixture ()
  (or *conv-fixture*
      (setf *conv-fixture*
            (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/layers/flax-conv.lisp"))
              (read stream)))))

(defun %cfx-case (name)
  (or (find name (getf (%conv-fixture) :cases) :key (lambda (c) (getf c :name)) :test #'string=)
      (error "フィクスチャにケース ~S が無い" name)))

(defun %cfx-model (case)
  "ケースの kernel・bias と :config から conv を作る（make-conv は N 次元でも作れる）。"
  (destructuring-bind (&key strides padding kernel-dilation groups) (getf case :config)
    (make-conv :kernel (%fx-array case "kernel") :bias (%fx-array case "bias")
                 :strides strides
                 :padding (if (stringp padding) (intern padding :keyword) padding)
                 :kernel-dilation kernel-dilation :feature-group-count groups)))

(defun %conv-case-names ()
  (mapcar (lambda (c) (getf c :name)) (getf (%conv-fixture) :cases)))

(defparameter *conv-tolerance* '()
  "畳み込みの f32 の許容誤差（allclose の既定、rtol 1e-5 / atol 1e-6。実測で Flax と一致する。bf16 だけ別）。")

(defun %check-conv-case (name forward)
  "ケース NAME で loss = sum(FORWARD(model, x) * ct) を作り、出力と勾配を Flax と比べる。"
  (let* ((case (%cfx-case name))
         (model (%cfx-model case))
         (x (%fx-array case "x"))
         (ct (%fx-array case "ct"))
         (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (funcall forward m x) ct))))
         (grads (funcall (nabla:grad loss :argnums '(0 1)) model x ct))
         (actual (append (nabla:tree-leaves (first grads)) (list (second grads))))
         (expected (list (%fx-array case "grad.kernel") (%fx-array case "grad.bias") (%fx-array case "grad.x"))))
    (is (apply #'%close-f32 (funcall forward model x) (%fx-array case "out") *conv-tolerance*)
        "~A: 順伝播の出力が Flax と一致しない" name)
    (loop for a in actual for e in expected for label in '("kernel" "bias" "x")
          do (is (apply #'%close-f32 a e *conv-tolerance*) "~A: ~A の勾配が Flax と一致しない" name label))))

(test conv/forward-and-grads-match-flax
  "Flax の nn.Conv のフィクスチャの全ケース（SAME・SAME + stride 2・偶数の kernel・VALID + 非対称の stride・
明示の padding・グループ・depthwise・kernel の dilation・バッチ軸の無い入力・1次元）で、eager の出力と、
kernel・bias・入力への勾配が一致する。"
  (dolist (name (%conv-case-names))
    (%check-conv-case name (nabla:with-tracing (m x) (call m x)))))

;;; --- SAME の padding ---

(defmacro def-conv-property (name docstring (&rest vars) generator &body body)
  "GENERATOR の値を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test (,name :suite :nabla.small)
     ,docstring
     (is (check-it (generator ,generator)
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.NN.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

(def-conv-property conv/same-padding-follows-the-jax-formula
  "SAME の padding (low high) は、出力の長さ ceil(n / s) に対して合計 max((out - 1) s + 実効 kernel - n, 0)、
low = 合計 / 2（切り捨て）、high = 合計 - low。high - low は 0 か 1。padding を入れた畳み込みの出力の長さが
ちょうど ceil(n / s)（実効 kernel = (k - 1) d + 1）。"
  (n k s d) (tuple (uniform-integer :lo 1 :hi 20) (uniform-integer :lo 1 :hi 6)
                   (uniform-integer :lo 1 :hi 5) (uniform-integer :lo 1 :hi 4))
  (destructuring-bind (low high) (nabla.nn::%same-padding n k s d)
    (let* ((out (ceiling n s))
           (effective (1+ (* (1- k) d)))
           (total (max (+ (* (1- out) s) (- effective n)) 0)))
      (and (= (+ low high) total)
           (= low (floor total 2))
           (<= 0 (- high low) 1)
           ;; 畳み込みの出力の長さ: floor((n + total - effective) / s) + 1
           (= out (1+ (floor (- (+ n low high) effective) s)))))))

(test conv/same-padding-known-values
  "SAME の padding の具体値（Flax と同じ）: n=6 k=3 s=2 は (0 1)、n=5 k=3 s=1 は (1 1)、n=5 k=4 s=1 は (1 2)、
n=7 k=3 s=2 は (1 1)、k=3 d=2 の実効 kernel 5 で n=8 s=1 は (2 2)、kernel が入力より大きいと (総計は
kernel 側に合わせる) n=2 k=5 s=1 は (2 2)。"
  (is (equal '(0 1) (nabla.nn::%same-padding 6 3 2 1)))
  (is (equal '(1 1) (nabla.nn::%same-padding 5 3 1 1)))
  (is (equal '(1 2) (nabla.nn::%same-padding 5 4 1 1)))
  (is (equal '(1 1) (nabla.nn::%same-padding 7 3 2 1)))
  (is (equal '(2 2) (nabla.nn::%same-padding 8 3 1 2)))
  (is (equal '(2 2) (nabla.nn::%same-padding 2 5 1 1))))

;;; --- 計算 ---

(defun %conv-property-case (seed n1 n2 k1 k2 s1 s2 d1 d2 mode pad groups batch)
  "PBT 用の入力を作る (values モジュール x 期待する padding)。MODE は 0 = SAME、1 = VALID、2 = 明示。
VALID と明示のときは、実効 kernel が（padding を入れた）入力に収まるように入力の長さを選ぶ。"
  (let* ((channels 4)
         (features 4)
         (kernel-size (list k1 k2))
         (strides (list s1 s2))
         (dilation (list d1 d2))
         (effective (mapcar (lambda (k d) (1+ (* (1- k) d))) kernel-size dilation))
         (spatial (list (max n1 (first effective)) (max n2 (second effective))))
         (padding (ecase mode
                    (0 :same)
                    (1 :valid)
                    (2 (list (list pad (mod (+ pad seed) 3)) (list (mod (+ pad 1) 3) pad)))))
         (module (make-conv :kernel (%rand (list k1 k2 (floor channels groups) features) seed)
                              :bias (%rand (list features) (+ seed 1))
                              :strides strides :padding padding :kernel-dilation dilation
                              :feature-group-count groups))
         (x (%rand (append (list batch) spatial (list channels)) (+ seed 2)))
         (explicit (ecase mode
                     (0 (mapcar (lambda (n k s d) (nabla.nn::%same-padding n k s d))
                                spatial kernel-size strides dilation))
                     (1 (list '(0 0) '(0 0)))
                     (2 padding))))
    (values module x explicit)))

(def-conv-property conv/matches-the-reference-convolution-plus-bias
  "任意の kernel・stride・dilation・padding（SAME / VALID / 明示）・グループ数・バッチで、conv の出力は
別実装の畳み込み reference-conv（padding は SAME を JAX の式で別に計算したもの）に bias を足したもの。"
  (seed n1 n2 k1 k2 s1 s2 d1 d2 mode pad group-exponent batch)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 7) (uniform-integer :lo 1 :hi 7)
         (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 3)
         (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 2) (uniform-integer :lo 1 :hi 2)
         (uniform-integer :lo 0 :hi 2) (uniform-integer :lo 0 :hi 2) (uniform-integer :lo 0 :hi 2)
         (uniform-integer :lo 1 :hi 2))
  (multiple-value-bind (module x explicit)
      (%conv-property-case seed n1 n2 k1 k2 s1 s2 d1 d2 mode pad (expt 2 group-exponent) batch)
    (let* ((groups (conv-feature-group-count module))
           (reference (reference-conv (decode-array x :f32) (decode-array (conv-kernel module) :f32)
                                      (conv-strides module) explicit '(1 1)
                                      (conv-kernel-dilation module) groups))
           (bias (conv-bias module))
           (expected (make-array (array-dimensions reference) :element-type 'double-float))
           (out (call module x)))
      (dotimes (i (array-total-size reference))
        (setf (row-major-aref expected i)
              (+ (row-major-aref reference i) (aref bias (mod i (length bias))))))
      (and (equal (array-dimensions reference) (array-dimensions out))
           (allclose out expected :dtype :f32)))))

(def-conv-property conv/same-output-length-is-ceil-of-input-over-stride
  "SAME の conv の出力の空間の長さは ceil(入力 / stride)（kernel と dilation によらない）。"
  (seed n1 n2 k s d) (tuple (uniform-integer :lo 0 :hi 1000) (uniform-integer :lo 1 :hi 9)
                            (uniform-integer :lo 1 :hi 9) (uniform-integer :lo 1 :hi 5)
                            (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 2))
  (let* ((module (make-conv :kernel (%rand (list k k 2 3) seed) :bias (%rand '(3) seed)
                              :strides (list s s) :padding :same :kernel-dilation (list d d)
                              :feature-group-count 1))
         (y (call module (%rand (list 1 n1 n2 2) seed))))
    (equal (list 1 (ceiling n1 s) (ceiling n2 s) 3) (array-dimensions y))))

(def-conv-property conv/one-by-one-kernel-is-a-dense-over-channels
  "1x1 の kernel の conv は、チャネルの軸への dense（kernel を (in out) に直したもの）と同じ。"
  (seed n1 n2 in out) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 5)
                             (uniform-integer :lo 1 :hi 5) (uniform-integer :lo 1 :hi 5)
                             (uniform-integer :lo 1 :hi 5))
  (let* ((kernel (%rand (list 1 1 in out) seed))
         (bias (%rand (list out) (1+ seed)))
         (flat (make-array (list in out) :element-type 'single-float
                                         :displaced-to kernel))
         (x (%rand (list 2 n1 n2 in) (+ seed 2)))
         (conv (make-conv :kernel kernel :bias bias :strides '(1 1) :padding :valid
                            :kernel-dilation '(1 1) :feature-group-count 1))
         (dense (make-dense :kernel (let ((copy (make-array (list in out) :element-type 'single-float)))
                                      (dotimes (i (* in out) copy) (setf (row-major-aref copy i) (row-major-aref flat i))))
                            :bias bias)))
    (allclose (call conv x) (call dense x) :dtype :f32 :rtol 1d-5 :atol 1d-5)))

(test conv/unbatched-input-equals-a-batch-of-one
  "バッチ軸の無い入力（rank = 空間 + 1）の出力は、バッチ1の出力からバッチ軸を除いたもの（Flax と同じ）。"
  (let* ((module (init-conv (nabla:prng-key 1) 3 4 '(3 3) :strides '(2 1)))
         (x (%rand '(6 7 3) 5))
         (batched (make-array '(1 6 7 3) :element-type 'single-float))
         (single (call module x)))
    (dotimes (i (array-total-size x)) (setf (row-major-aref batched i) (row-major-aref x i)))
    (let ((reference (call module batched)))
      (is (equal (rest (array-dimensions reference)) (array-dimensions single)))
      (is (equalp (%elements reference) (%elements single))))))

;;; --- init ---

(test conv/init-matches-flax-lecun-normal-with-groups
  "init-conv (in 16, out 16, kernel 3x3, 2 グループ) の kernel は形 (3 3 8 16)、Flax の lecun_normal と同じ
分布: fan_in = 3 * 3 * 8 = 72 で標準偏差が 1/sqrt(72)（Flax のサンプルと 5% 以内）、[-2, 2] を
1/(0.8796 sqrt(72)) 倍した範囲に切れていて最大絶対値は境界の 95% 以上。bias は 0 で形 (16)。f32。
スロットの順は kernel, bias に :static の strides padding kernel-dilation feature-group-count。"
  (let* ((module (init-conv (nabla:prng-key 207) 16 16 '(3 3) :feature-group-count 2))
         (kernel (conv-kernel module))
         (bound (* 2 (/ (sqrt (/ 1d0 72)) 0.87962566103423978d0))))
    (destructuring-bind (shape flax-std flax-max) (rest (assoc :conv-kernel (getf (%conv-fixture) :init-stats)))
      (is (equal shape (array-dimensions kernel)))
      (is (equal '(3 3 8 16) (array-dimensions kernel)))
      (is (eq 'single-float (array-element-type kernel)))
      (multiple-value-bind (std max) (%stats kernel)
        (is (<= (abs (- std flax-std)) (* 0.05 flax-std)) "標準偏差 ~F と Flax の ~F" std flax-std)
        (is (<= (abs (- std (sqrt (/ 1d0 72)))) (* 0.05 std)) "標準偏差 ~F" std)
        (is (<= max (* bound (+ 1 1d-5))) "最大絶対値 ~F が境界 ~F を超えた" max bound)
        (is (>= max (* 0.95 bound)) "最大絶対値 ~F が境界 ~F に届かない" max bound)
        (is (<= (abs (- max flax-max)) (* 0.1 flax-max)))))
    (is (equal '(16) (array-dimensions (conv-bias module))))
    (is (every #'zerop (%elements (conv-bias module))))
    (is (equal '(("kernel" . :param) ("bias" . :param) ("strides" . :static) ("padding" . :static)
                 ("kernel-dilation" . :static) ("feature-group-count" . :static))
               (%slot-names 'conv)))))

(test conv/init-defaults-and-normalized-options
  "init-conv の既定は Flax と同じ: stride 1、padding SAME、dilation 1、グループ 1。整数の stride /
dilation は空間の次元数に伸ばされる。padding はキーワード（:same / :valid）か明示の (low high) のリスト。
dtype :f64 を選べる。"
  (let ((module (init-conv (nabla:prng-key 1) 3 5 '(3 2))))
    (is (equal '(1 1) (conv-strides module)))
    (is (eq :same (conv-padding module)))
    (is (equal '(1 1) (conv-kernel-dilation module)))
    (is (= 1 (conv-feature-group-count module)))
    (is (equal '(3 2 3 5) (array-dimensions (conv-kernel module)))))
  (let ((module (init-conv (nabla:prng-key 1) 3 5 '(3 3) :strides 2 :kernel-dilation 2 :padding :valid)))
    (is (equal '(2 2) (conv-strides module)))
    (is (equal '(2 2) (conv-kernel-dilation module)))
    (is (eq :valid (conv-padding module))))
  (is (equal '((1 2) (0 1))
             (conv-padding (init-conv (nabla:prng-key 1) 3 5 '(3 3) :padding '((1 2) (0 1))))))
  (is (eq 'double-float (array-element-type
                         (conv-kernel (init-conv (nabla:prng-key 1) 3 5 '(3 3) :dtype :f64)))))
  (is (equalp (conv-kernel (init-conv (nabla:prng-key 5) 3 5 '(3 3)))
              (conv-kernel (init-conv (nabla:prng-key 5) 3 5 '(3 3)))))
  (is (not (equalp (conv-kernel (init-conv (nabla:prng-key 5) 3 5 '(3 3)))
                   (conv-kernel (init-conv (nabla:prng-key 6) 3 5 '(3 3)))))))

(test conv/init-supports-one-two-and-three-spatial-dims
  "init-conv の kernel-size の長さが空間の次元数（Flax の nn.Conv と同じ N 次元）: 1 / 2 / 3 次元で kernel の形が
(kernel-size... in out)、stride・dilation・padding がその次元数に伸び、SAME の出力の空間の大きさが
ceil(入力 / stride) になる。"
  (loop for (kernel-size input) in '(((3) (2 9 3)) ((3 2) (2 6 7 3)) ((3 2 2) (1 5 4 6 3)))
        for spatial = (length kernel-size)
        do (let* ((module (init-conv (nabla:prng-key 1) 3 4 kernel-size :strides 2 :kernel-dilation 2))
                  (y (call module (%rand input 7))))
             (is (equal (append kernel-size '(3 4)) (array-dimensions (conv-kernel module))))
             (is (equal (make-list spatial :initial-element 2) (conv-strides module)))
             (is (equal (make-list spatial :initial-element 2) (conv-kernel-dilation module)))
             (is (equal (append (list (first input))
                                (mapcar (lambda (n) (ceiling n 2)) (subseq input 1 (1+ spatial)))
                                '(4))
                        (array-dimensions y))))))

(test conv/hand-built-modules-are-normalized-at-call-time
  "make-conv で直接作ったモジュール（チェックポイントのひな形など）でも、整数の strides・kernel-dilation と
:same / :valid を init-conv と同じに扱う: 正規化済みの値で作ったモジュールと同じ出力を返す。"
  (let* ((kernel (%rand '(3 3 2 4) 1))
         (bias (%rand '(4) 2))
         (x (%rand '(2 7 6 2) 3)))
    (dolist (padding '(:same :valid))
      (let ((short (make-conv :kernel kernel :bias bias :strides 2 :padding padding :kernel-dilation 1
                              :feature-group-count 1))
            (full (make-conv :kernel kernel :bias bias :strides '(2 2) :padding padding
                             :kernel-dilation '(1 1) :feature-group-count 1)))
        (is (equalp (call full x) (call short x)))
        (is (equal (array-dimensions (call full x)) (array-dimensions (call short x))))))
    (is (equal '(2 4 3 4)
               (array-dimensions (call (make-conv :kernel kernel :bias bias :strides 2 :padding :same
                                                  :kernel-dilation 1 :feature-group-count 1)
                                       x))))))

(defun %error-text (thunk)
  "THUNK が error を signal したらそのメッセージ、signal しなければ NIL。"
  (handler-case (progn (funcall thunk) nil)
    (error (e) (princ-to-string e))))

(defmacro %is-conv-error (form)
  "FORM が conv の名前を含むメッセージの error を signal することを確かめる。"
  `(let ((text (%error-text (lambda () ,form))))
     (is (and text (search "conv:" text)) "~S が conv: のメッセージで error にならなかった（~S）" ',form text)))

(test conv/invalid-arguments-signal-errors
  "init の不正な引数（kernel-size が空・正の整数でない・ドット対、stride / dilation が正でない・長さが違う・
ドット対、padding が不正（文字列・:circular・長さ・負・ドット対）、グループ数が in / out を割り切らない・正でない）と
call の不正な入力（rank、チャネル数、bias の形、kernel の rank）は、メッセージに conv: を含む error。
dtype が不正でも error。VALID で kernel が入力より大きいときも error。"
  (flet ((init (&rest args) (apply #'init-conv (nabla:prng-key 0) 4 4 args)))
    (%is-conv-error (init nil))
    (%is-conv-error (init '(3 0)))
    (%is-conv-error (init '(3 -1)))
    (%is-conv-error (init '(3 3.0)))
    (%is-conv-error (init '(3 . 3)))
    (%is-conv-error (init 3))
    (%is-conv-error (init '(3 3) :strides '(1)))
    (%is-conv-error (init '(3 3) :strides 0))
    (%is-conv-error (init '(3 3) :strides '(1 . 1)))
    (%is-conv-error (init '(3 3) :kernel-dilation '(1 0)))
    (%is-conv-error (init '(3 3) :padding :circular))
    (%is-conv-error (init '(3 3) :padding "SAME"))
    (%is-conv-error (init '(3 3) :padding '((1 1))))
    (%is-conv-error (init '(3 3) :padding '((1 -1) (0 0))))
    (%is-conv-error (init '(3 3) :padding '((1 1) . (0 0))))
    (%is-conv-error (init '(3 3) :padding '((1 . 1) (0 0))))
    (%is-conv-error (init '(3 3) :feature-group-count 3))
    (%is-conv-error (init '(3 3) :feature-group-count 0))
    (signals error (init '(3 3) :dtype :i32)))
  (%is-conv-error (init-conv (nabla:prng-key 0) 4 6 '(3 3) :feature-group-count 4))
  (let ((module (init-conv (nabla:prng-key 0) 3 4 '(3 3))))
    (%is-conv-error (call module (%rand '(2 5 5 2) 1)))
    (%is-conv-error (call module (%rand '(5 3) 1)))
    (%is-conv-error (call module (%rand '(2 2 5 5 3) 1)))
    ;; bias の長さが出力のチャネル数と違う
    (%is-conv-error (call (make-conv :kernel (conv-kernel module) :bias (%rand '(5) 1) :strides '(1 1)
                                     :padding :same :kernel-dilation '(1 1) :feature-group-count 1)
                          (%rand '(2 5 5 3) 1)))
    ;; kernel の rank が 3 未満
    (%is-conv-error (call (make-conv :kernel (%rand '(3 4) 1) :bias (%rand '(4) 1) :strides 1
                                     :padding :same :kernel-dilation 1 :feature-group-count 1)
                          (%rand '(5 3) 1)))
    ;; hand-built で正規化できない値
    (let ((bad-strides (%error-text (lambda ()
                                      (call (make-conv :kernel (conv-kernel module) :bias (conv-bias module)
                                                       :strides 0 :padding :same :kernel-dilation 1
                                                       :feature-group-count 1)
                                            (%rand '(2 5 5 3) 1)))))
          (bad-dilation (%error-text (lambda ()
                                       (call (make-conv :kernel (conv-kernel module) :bias (conv-bias module)
                                                        :strides 1 :padding :same :kernel-dilation 0
                                                        :feature-group-count 1)
                                             (%rand '(2 5 5 3) 1))))))
      ;; メッセージはどの引数が悪いかを名指しする
      (is (and bad-strides (search "strides" bad-strides)))
      (is (and bad-dilation (search "kernel-dilation" bad-dilation)))))
  (signals error (call (init-conv (nabla:prng-key 0) 3 4 '(5 5) :padding :valid) (%rand '(1 3 3 3) 1))))

;;; --- PyTree / vmap ---

(test conv/pytree-leaves-and-static-options
  "葉は kernel と bias の2つ。strides / padding / kernel-dilation / feature-group-count は treedef の補助データで、
違えば別の treedef になる。"
  (let ((a (init-conv (nabla:prng-key 1) 3 4 '(3 3)))
        (b (init-conv (nabla:prng-key 1) 3 4 '(3 3) :strides 2))
        (c (init-conv (nabla:prng-key 1) 3 4 '(3 3) :padding :valid)))
    (is (= 2 (length (nabla:tree-leaves a))))
    (is (nabla:treedef= (nabla:tree-structure a) (nabla:tree-structure (init-conv (nabla:prng-key 2) 3 4 '(3 3)))))
    (is (not (nabla:treedef= (nabla:tree-structure a) (nabla:tree-structure b))))
    (is (not (nabla:treedef= (nabla:tree-structure a) (nabla:tree-structure c))))))

(def-conv-property conv/vmap-over-images-matches-per-image-call
  "vmap した (call m image)（image は [H W C]）は、バッチ全体の各画像を単独に call した結果を積んだものと一致する
（SAME・VALID・stride・グループを含む）。"
  (seed batch mode stride group-exponent)
  (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 0 :hi 1)
         (uniform-integer :lo 1 :hi 2) (uniform-integer :lo 0 :hi 1))
  (let* ((groups (expt 2 group-exponent))
         (module (make-conv :kernel (%rand (list 3 3 (floor 4 groups) 4) seed) :bias (%rand '(4) (1+ seed))
                              :strides (list stride stride) :padding (if (zerop mode) :same :valid)
                              :kernel-dilation '(1 1) :feature-group-count groups))
         (images (%rand (list batch 6 5 4) (+ seed 2)))
         (batched (funcall (nabla:vmap (nabla:with-tracing (x) (call module x))) images))
         (each (loop for b below batch
                     collect (let ((image (make-array '(6 5 4) :element-type 'single-float)))
                               (dotimes (i (array-total-size image))
                                 (setf (row-major-aref image i) (row-major-aref images (+ (* b 120) i))))
                               (call module image)))))
    (and (equal (cons batch (array-dimensions (first each))) (array-dimensions batched))
         (loop for b below batch
               for expected in each
               always (let ((slice (make-array (array-dimensions expected) :element-type 'single-float)))
                        (dotimes (i (array-total-size slice))
                          (setf (row-major-aref slice i)
                                (row-major-aref batched (+ (* b (array-total-size slice)) i))))
                        (allclose slice expected :dtype :f32 :rtol 1d-5 :atol 1d-5))))))

(defun %row-slice (array index)
  "ARRAY の先頭軸の INDEX 番目を新しい配列にする。"
  (let* ((dims (rest (array-dimensions array)))
         (size (reduce #'* dims))
         (slice (make-array dims :element-type (array-element-type array))))
    (dotimes (i size slice) (setf (row-major-aref slice i) (row-major-aref array (+ (* index size) i))))))

(test conv/per-example-gradients-through-vmap-of-grad-match-per-image-grads
  "vmap (grad loss) :in-axes (nil 0 0)（モデルはバッチしない、画像と余接線はバッチする。#200）で得た、モデルと同じ
構造の勾配の葉（kernel と bias）の先頭にバッチ軸がつき、各行は画像を単独に grad した勾配と一致する。"
  (let* ((module (make-conv :kernel (%rand '(3 3 2 4) 1) :bias (%rand '(4) 2) :strides '(2 1)
                            :padding :same :kernel-dilation '(1 1) :feature-group-count 1))
         (images (%rand '(3 6 5 2) 3))
         (cts (%rand '(3 3 5 4) 4))
         (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (call m x) ct))))
         (grad (nabla:grad loss))
         (batched (funcall (nabla:vmap grad :in-axes '(nil 0 0)) module images cts)))
    (is (typep batched 'conv))
    (is (equal '(3 3 3 2 4) (array-dimensions (conv-kernel batched))))
    (is (equal '(3 4) (array-dimensions (conv-bias batched))))
    (dotimes (b 3)
      (let ((single (funcall grad module (%row-slice images b) (%row-slice cts b))))
        (is (every (lambda (batched-leaf leaf) (%close-f32 (%row-slice batched-leaf b) leaf))
                   (nabla:tree-leaves batched) (nabla:tree-leaves single)))))))
