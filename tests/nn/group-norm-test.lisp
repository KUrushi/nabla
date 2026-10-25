;;;; group-norm（Flax の nn.GroupNorm。issue #260、small）。
;;;;
;;;; 守らせる性質:
;;;; - Flax との数値一致: 同じパラメータ・入力・余接線で、順伝播の出力と、パラメータ・入力についての勾配が
;;;;   Flax のフィクスチャ（tests/fixtures/layers/flax-group-norm.lisp、生成は同じディレクトリの
;;;;   generate-group-norm.py）と一致する（eager。jit は medium、group-norm-iree-test.lisp）。
;;;; - 任意の rank・群の数・群の大きさで、群（先頭のバッチ軸の添字とチャンネルの群）ごとに 2 パスで正規化する
;;;;   参照実装と一致する。正規化した各群は平均 0・分散 var / (var + epsilon)。
;;;; - num-groups G と group-size C / G は同じ出力。scale / bias はチャンネルごとのアフィン変換。
;;;; - 入力の勾配は各群で和が 0（群の中の平行移動に不変）。vmap はスライスごとの呼び出しを積んだもの。
;;;; - init の形・値・既定の epsilon・スロット名、不正な引数は層名入りの error。
;;;; layers-test.lisp の部品（%fx-array・def-layer-property・%rand など）を使う。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

;;; --- Flax フィクスチャ ---

(defvar *group-norm-fixture* nil)

(defun %group-norm-fixture-cases ()
  (getf (or *group-norm-fixture*
            (setf *group-norm-fixture*
                  (with-open-file (stream (asdf:system-relative-pathname
                                           "nabla" "tests/fixtures/layers/flax-group-norm.lisp"))
                    (read stream))))
        :cases))

(defun %gn-fx-model (case)
  "ケースの :config とパラメータから group-norm を作る。"
  (destructuring-bind (&key num-groups group-size use-scale use-bias epsilon) (getf case :config)
    (make-group-norm :scale (and use-scale (%fx-array case "scale")) :bias (and use-bias (%fx-array case "bias"))
                     :num-groups num-groups :group-size group-size :epsilon epsilon
                     :use-scale use-scale :use-bias use-bias)))

(defparameter *gn-tolerance* '(:rtol 1d-4 :atol 1d-5)
  "Flax と比べる許容誤差。layer-norm と同じ理由で緩める（平均 3・標準偏差 2 の入力の速い分散
E[x^2] - E[x]^2 は f32 の桁を 1 桁ほど失い、XLA と nabla の丸めの違いがそのまま出る。layers-test.lisp の
%case-tolerance）。")

(defparameter *gn-forward* (nabla:with-tracing (m x) (call m x)))

(defun %gn-flax-results (case model &key backend (jit-grad t))
  "CASE の x と ct で loss = sum(call(model, x) * ct) を作り、(values 出力 勾配のリスト（パラメータの葉 ++ x）)
を返す。BACKEND を渡すと順伝播と grad（JIT-GRAD が偽なら順伝播だけ）を jit する（medium）。"
  (let* ((x (%fx-array case "x"))
         (ct (%fx-array case "ct"))
         (forward *gn-forward*)
         (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (call m x) ct))))
         (grad (nabla:grad loss :argnums '(0 1)))
         (grads (funcall (if (and backend jit-grad) (nabla:jit grad :backend backend) grad) model x ct)))
    (values (funcall (if backend (nabla:jit forward :backend backend) forward) model x)
            (append (nabla:tree-leaves (first grads)) (last grads)))))

(defun %gn-flax-expected-grads (case)
  (append (mapcar (lambda (p) (%fx-array case (concatenate 'string "grad." p))) (%fx-param-names case))
          (list (%fx-array case "grad.x"))))

(test group-norm/forward-and-grads-match-flax
  "Flax の nn.GroupNorm（num_groups / group_size、rank 4・3・2・1、群が 1 つ、群の大きさ 1、use_scale /
use_bias が偽）のフィクスチャと、eager の出力と、パラメータ（あるものだけ）・入力の勾配が一致する。"
  (dolist (case (%group-norm-fixture-cases))
    (let ((name (getf case :name)))
      (multiple-value-bind (out grads) (%gn-flax-results case (%gn-fx-model case))
        (is (apply #'%close-f32 out (%fx-array case "out") *gn-tolerance*) "~A: 出力が Flax と一致しない" name)
        (let ((expected (%gn-flax-expected-grads case)))
          (is (= (length expected) (length grads)) "~A: 勾配の個数が違う" name)
          (loop for a in grads for e in expected for i from 0
                do (is (apply #'%close-f32 a e *gn-tolerance*) "~A: ~D 番目の勾配が Flax と一致しない" name i)))))))

;;; --- init ---

(test group-norm/init-is-ones-and-zeros-with-flax-defaults
  "init-group-norm は scale = 1・bias = 0（形はチャンネル数）、既定は Flax と同じ num-groups 32・group-size NIL・
epsilon 1e-6（f64 は 1d-6）。use-scale / use-bias が偽なら対応するスロットは NIL。スロット名は Flax と同じ。"
  (let ((norm (init-group-norm (nabla:prng-key 1) 64)))
    (is (equalp (make-array 64 :element-type 'single-float :initial-element 1.0) (group-norm-scale norm)))
    (is (equalp (make-array 64 :element-type 'single-float :initial-element 0.0) (group-norm-bias norm)))
    (is (eql 32 (group-norm-num-groups norm)))
    (is (null (group-norm-group-size norm)))
    (is (eql 1f-6 (group-norm-epsilon norm))))
  (let ((norm (init-group-norm (nabla:prng-key 1) 6 :num-groups nil :group-size 2 :dtype :f64
                                                    :use-scale nil)))
    (is (null (group-norm-num-groups norm)))
    (is (eql 2 (group-norm-group-size norm)))
    (is (eql 1d-6 (group-norm-epsilon norm)))
    (is (null (group-norm-scale norm)))
    (is (null (group-norm-use-scale norm)))
    (is (eq 'double-float (array-element-type (group-norm-bias norm)))))
  (is (null (group-norm-bias (init-group-norm (nabla:prng-key 1) 6 :num-groups 3 :use-bias nil))))
  (is (eql 1f-5 (group-norm-epsilon (init-group-norm (nabla:prng-key 1) 6 :num-groups 3 :epsilon 1f-5))))
  (is (equal '(("scale" . :param) ("bias" . :param) ("num-groups" . :static) ("group-size" . :static)
               ("epsilon" . :static) ("use-scale" . :static) ("use-bias" . :static))
             (%slot-names 'group-norm))))

;;; --- 性質 ---

(defun %ref-group-norm (x group-size epsilon)
  "f64 の X（rank 1 以上、最後の軸がチャンネル）を、群（rank 2 以上は先頭の軸の添字とチャンネルの群、rank 1 は
チャンネルの群）ごとに 2 パスの平均と分散で (x - mean) / sqrt(var + EPSILON) にした配列（素朴な参照実装）と、
群の分散の最小値の 2 値。"
  (let* ((shape (array-dimensions x))
         (channels (car (last shape)))
         (per-batch (floor (array-total-size x) (if (rest shape) (first shape) 1)))
         (groups (make-hash-table :test 'equal))
         (y (make-array shape :element-type 'double-float))
         (min-variance nil))
    (flet ((group-of (i) (list (floor i per-batch) (floor (mod i channels) group-size))))
      (dotimes (i (array-total-size x)) (push (row-major-aref x i) (gethash (group-of i) groups)))
      (dotimes (i (array-total-size x) (values y min-variance))
        (let* ((values (gethash (group-of i) groups))
               (n (length values))
               (mean (/ (reduce #'+ values) n))
               (variance (/ (reduce #'+ (mapcar (lambda (v) (expt (- v mean) 2)) values)) n)))
          (setf min-variance (min variance (or min-variance variance)))
          (setf (row-major-aref y i) (/ (- (row-major-aref x i) mean) (sqrt (+ variance epsilon)))))))))

(defun %gn-shape (dims rank channels)
  "DIMS の先頭 RANK - 1 個の後ろにチャンネル数を付けた形。"
  (append (subseq dims 0 (1- rank)) (list channels)))

(defmacro def-group-norm-property (name docstring (&rest vars) &body body)
  "VARS = (シード rank 先頭の軸の大きさ 3 つ 群の数 群の大きさ) を生成して BODY を PBT で確かめる。"
  `(def-layer-property ,name ,docstring ,vars
     (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 4)
            (list (uniform-integer :lo 1 :hi 3) :length 3) (uniform-integer :lo 1 :hi 4)
            (uniform-integer :lo 1 :hi 4))
     ,@body))

(def-group-norm-property group-norm/matches-the-loop-reference
  "rank 1〜4 の入力、任意の群の数・群の大きさで、scale 1・bias 0 の group-norm は参照実装と一致する（f64 は
1e-9、f32 は layer-norm と同じ速い分散の桁落ちの分だけ緩めた 1e-4 / 1e-5）。f32 は、分散が 0 より大きく 1e-2
未満の群を含む入力では比べない（Flax と同じ速い分散 E[x^2] - E[x]^2 の分散の誤差は数 ulp × E[x^2] ≈ 3e-8 で、
出力の相対誤差は約 0.5 × 3e-8 / var。3000 試行の実測で、失敗は分散 1e-3 未満の群を含む入力だけ。分散 0 ちょうど
（要素が 1 つの群など）は出力が 0 で比べられる）。"
  (seed rank dims groups size)
  (let* ((shape (%gn-shape dims rank (* groups size)))
         (x64 (%rand shape seed :f64))
         (x32 (%rand shape seed :f32))
         (x32-as-64 (make-array shape :element-type 'double-float)))
    (dotimes (i (array-total-size x32)) (setf (row-major-aref x32-as-64 i) (coerce (row-major-aref x32 i) 'double-float)))
    (and (allclose (call (init-group-norm (nabla:prng-key 0) (* groups size) :num-groups groups :dtype :f64) x64)
                   (%ref-group-norm x64 size 1d-6) :dtype :f64 :rtol 1d-9 :atol 1d-9)
         (multiple-value-bind (expected min-variance) (%ref-group-norm x32-as-64 size 1d-6)
           (or (< 0 min-variance 1d-2)
               (allclose (call (init-group-norm (nabla:prng-key 0) (* groups size) :num-groups groups) x32)
                         expected :rtol 1d-4 :atol 1d-5))))))

(def-group-norm-property group-norm/each-group-has-zero-mean-and-unit-variance
  "scale 1・bias 0 の出力（f64）は、各群で平均 0、分散は入力の群の分散 v に対して v / (v + epsilon)。"
  (seed rank dims groups size)
  (let* ((shape (%gn-shape dims rank (* groups size)))
         (x (%rand shape seed :f64))
         (y (call (init-group-norm (nabla:prng-key 0) (* groups size) :num-groups groups :dtype :f64) x))
         (per-batch (floor (array-total-size x) (if (> rank 1) (first shape) 1)))
         (members (make-hash-table :test 'equal)))
    (dotimes (i (array-total-size x))
      (push i (gethash (list (floor i per-batch) (floor (mod i (* groups size)) size)) members)))
    (loop for indices being the hash-values of members
          always (flet ((moments (array)
                          (let* ((values (mapcar (lambda (i) (row-major-aref array i)) indices))
                                 (mean (/ (reduce #'+ values) (length values))))
                            (values mean (/ (reduce #'+ (mapcar (lambda (v) (expt (- v mean) 2)) values))
                                            (length values))))))
                   (multiple-value-bind (y-mean y-variance) (moments y)
                     (let ((x-variance (nth-value 1 (moments x))))
                       (and (< (abs y-mean) 1d-9)
                            (< (abs (- y-variance (/ x-variance (+ x-variance 1d-6)))) 1d-9))))))))

(def-group-norm-property group-norm/num-groups-and-group-size-agree-and-scale-bias-are-affine
  "num-groups G の group-norm と group-size C / G の group-norm は同じ出力。scale / bias を持つ出力は、scale 1・
bias 0 の出力にチャンネルごとに scale を掛けて bias を足したもの。"
  (seed rank dims groups size)
  (let* ((channels (* groups size))
         (shape (%gn-shape dims rank channels))
         (x (%rand shape seed :f64))
         (scale (%rand (list channels) (+ seed 1) :f64))
         (bias (%rand (list channels) (+ seed 2) :f64))
         (plain (call (init-group-norm (nabla:prng-key 0) channels :num-groups groups :dtype :f64) x))
         (by-size (call (init-group-norm (nabla:prng-key 0) channels :num-groups nil :group-size size :dtype :f64) x))
         (full (call (make-group-norm :scale scale :bias bias :num-groups groups :group-size nil :epsilon 1d-6
                                      :use-scale t :use-bias t)
                     x))
         (expected (make-array shape :element-type 'double-float)))
    (dotimes (i (array-total-size x))
      (let ((c (mod i channels)))
        (setf (row-major-aref expected i) (+ (* (row-major-aref plain i) (aref scale c)) (aref bias c)))))
    (and (equalp plain by-size)
         (allclose full expected :dtype :f64 :rtol 1d-12 :atol 1d-12))))

(def-group-norm-property group-norm/input-gradient-sums-to-zero-in-each-group
  "loss = sum(group-norm(x) * ct)（scale あり、f64）の x の勾配は、各群で和が 0（群の中の平行移動に不変）。
|Σ g| <= 1e-9 (Σ |g| + 1)。"
  (seed rank dims groups size)
  (let* ((channels (* groups size))
         (shape (%gn-shape dims rank channels))
         (x (%rand shape seed :f64))
         (ct (%rand shape (+ seed 1) :f64))
         (norm (make-group-norm :scale (%rand (list channels) (+ seed 2) :f64) :bias nil :num-groups nil
                                :group-size size :epsilon 1d-6 :use-scale t :use-bias nil))
         (loss (nabla:with-tracing (m x ct) (nabla:reduce-sum (* (call m x) ct))))
         (g (first (funcall (nabla:grad loss :argnums '(1)) norm x ct)))
         (per-batch (floor (array-total-size x) (if (> rank 1) (first shape) 1)))
         (sums (make-hash-table :test 'equal))
         (magnitudes (make-hash-table :test 'equal)))
    (dotimes (i (array-total-size g))
      (let ((group (list (floor i per-batch) (floor (mod i channels) size))))
        (incf (gethash group sums 0d0) (row-major-aref g i))
        (incf (gethash group magnitudes 0d0) (abs (row-major-aref g i)))))
    (loop for group being the hash-keys of sums using (hash-value sum)
          always (<= (abs sum) (* 1d-9 (+ (gethash group magnitudes) 1))))))

(def-group-norm-property group-norm/vmap-stacks-per-slice-calls
  "x の新しい先頭の軸（長さ 2）で vmap した group-norm（モデルは in-axes nil）は、各スライスに group-norm を
適用して積んだもの（内側の入力の先頭の軸がバッチになる。Flax と同じ）。f64。"
  (seed rank dims groups size)
  (let* ((channels (* groups size))
         (shape (%gn-shape dims rank channels))
         (a (%rand shape seed :f64))
         (b (%rand shape (+ seed 1) :f64))
         (stacked (make-array (cons 2 shape) :element-type 'double-float))
         (norm (make-group-norm :scale (%rand (list channels) (+ seed 2) :f64)
                                :bias (%rand (list channels) (+ seed 3) :f64)
                                :num-groups groups :group-size nil :epsilon 1d-6 :use-scale t :use-bias t))
         (expected (make-array (cons 2 shape) :element-type 'double-float))
         (n (array-total-size a)))
    (dotimes (i n)
      (setf (row-major-aref stacked i) (row-major-aref a i)
            (row-major-aref stacked (+ n i)) (row-major-aref b i)))
    (let ((ya (call norm a)) (yb (call norm b)))
      (dotimes (i n)
        (setf (row-major-aref expected i) (row-major-aref ya i)
              (row-major-aref expected (+ n i)) (row-major-aref yb i))))
    (allclose (funcall (nabla:vmap *gn-forward* :in-axes '(nil 0)) norm stacked) expected
              :dtype :f64 :rtol 1d-12 :atol 1d-12)))

;;; --- 不正な引数 ---

(defun %gn-error-message (thunk)
  "THUNK が signal した error のメッセージ（signal しなければ NIL）。"
  (handler-case (progn (funcall thunk) nil)
    (error (e) (princ-to-string e))))

(defmacro %is-group-norm-error (form &rest words)
  "FORM が、メッセージに group-norm と WORDS（原因の引数・スロットの名前）がすべて入った error を出すことを確かめる。"
  `(let ((message (%gn-error-message (lambda () ,form))))
     (is (and message (every (lambda (word) (search word message)) (list "group-norm" ,@words)))
         "~S が group-norm と ~S の入った error を出さない: ~S" ',form (list ,@words) message)))

(test group-norm/invalid-arguments-signal-errors-naming-the-layer
  "チャンネル数が群の数・群の大きさで割り切れない、num-groups と group-size の両方・どちらも無し、正の整数でない
群の指定、正でない epsilon（make-group-norm で作ったものは call で）、:f32 / :f64 でない dtype、rank 0 の入力、scale / bias の形がチャンネル数と違う、T / NIL でない use-scale /
use-bias、use-scale と scale・use-bias と bias の食い違いは、メッセージに group-norm（と原因の名前）が入った error。"
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 4))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups nil :group-size 4))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 3 :group-size 2))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups nil))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 0))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups nil :group-size -2))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 1.5))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 3 :epsilon 0))
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 0 :num-groups 1))
  ;; scale も bias も持たないときも dtype を確かめる。make-group-norm で直接作った epsilon 0 は call で
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 3 :dtype :bf16 :use-scale nil :use-bias nil)
                        "dtype")
  (%is-group-norm-error (call (make-group-norm :scale nil :bias nil :num-groups 3 :group-size nil :epsilon 0
                                               :use-scale nil :use-bias nil)
                              (%rand '(2 6) 1))
                        "epsilon")
  (let ((norm (init-group-norm (nabla:prng-key 0) 6 :num-groups 3)))
    ;; 入力のチャンネル数 4 は scale の形 (6) とも num-groups 3 とも合わない。8 は群の数 2 では割り切れるが scale・bias と違う
    (%is-group-norm-error (call norm (%rand '(2 4) 1)))
    (%is-group-norm-error (call (make-group-norm :scale (%rand '(6) 1) :bias nil :num-groups 2 :group-size nil
                                                 :epsilon 1f-6 :use-scale t :use-bias nil)
                                (%rand '(2 8) 1))
                          "scale の形")
    (%is-group-norm-error (call (make-group-norm :scale nil :bias (%rand '(6) 1) :num-groups 2 :group-size nil
                                                 :epsilon 1f-6 :use-scale nil :use-bias t)
                                (%rand '(2 8) 1))
                          "bias の形")
    (%is-group-norm-error (call norm (make-array '() :element-type 'single-float :initial-element 1.0))))
  ;; use-scale / use-bias は T か NIL、スロットとの食い違いはどちらの名前も出す
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 3 :use-scale 1) "use-scale")
  (%is-group-norm-error (init-group-norm (nabla:prng-key 0) 6 :num-groups 3 :use-bias 1) "use-bias")
  (flet ((module (scale bias use-scale use-bias)
           (make-group-norm :scale scale :bias bias :num-groups 3 :group-size nil :epsilon 1f-6
                            :use-scale use-scale :use-bias use-bias)))
    (%is-group-norm-error (call (module nil nil t nil) (%rand '(2 6) 1)) "use-scale" "scale が NIL")
    (%is-group-norm-error (call (module (%rand '(6) 1) nil nil nil) (%rand '(2 6) 1)) "use-scale" "scale がある")
    (%is-group-norm-error (call (module nil nil nil t) (%rand '(2 6) 1)) "use-bias" "bias が NIL")
    (%is-group-norm-error (call (module nil (%rand '(6) 1) nil nil) (%rand '(2 6) 1)) "use-bias" "bias がある")))
