;;;; convolution プリミティブ（issue #197）の small テスト。
;;;;
;;;; 形状推論は、tests/support/reference.lisp の REFERENCE-CONV（lhs を膨らませて
;;;; padding した配列を実際に作り、窓の開始位置を数え上げる別実装）の出力の形と、
;;;; eager の出力は REFERENCE-CONV の値と、JAX の conv_general_dilated（フィクスチャ
;;;; tests/fixtures/conv/conv.lisp）の値と比べる。IREE での実行は tests/iree/conv-test.lisp。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %conv-window-count (size kernel stride pad lhs-dilation rhs-dilation)
  "出力サイズの式を使わずに、膨らませて padding した入力に窓（膨らませたカーネル）が
置ける開始位置の数を数える（プリミティブの式とは独立な別実装）。"
  (let ((padded (+ (1+ (* (1- size) lhs-dilation)) (first pad) (second pad)))
        (window (1+ (* (1- kernel) rhs-dilation))))
    (loop for start from 0 by stride while (<= (+ start window) padded) count t)))

(defun %conv-random-case ()
  "(lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation groups) を返す。
空間次元は 1〜2 個。出力の空間サイズがすべて 1 以上になる形だけを返す。"
  (loop
    (let* ((spatial (1+ (random 2)))
           (groups (1+ (random 3)))
           (in-per-group (1+ (random 3)))
           (out-per-group (1+ (random 3)))
           (strides (loop repeat spatial collect (1+ (random 3))))
           (lhs-dilation (loop repeat spatial collect (1+ (random 2))))
           (rhs-dilation (loop repeat spatial collect (1+ (random 2))))
           (padding (loop repeat spatial collect (list (- (random 4) 1) (- (random 4) 1))))
           (kernel (loop repeat spatial collect (1+ (random 3))))
           (in-sizes (loop repeat spatial collect (1+ (random 6))))
           (lhs-shape (append (list (1+ (random 3))) in-sizes (list (* groups in-per-group))))
           (rhs-shape (append kernel (list in-per-group (* groups out-per-group)))))
      (when (every #'plusp
                   (loop for d in in-sizes for k in kernel for s in strides for p in padding
                         for ld in lhs-dilation for rd in rhs-dilation
                         collect (%conv-window-count d k s p ld rd)))
        (return (list lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation groups))))))

(defun %conv-generator () (%ad-hoc-generator #'%conv-random-case))

(defun %conv-bgc-random-case ()
  "%CONV-RANDOM-CASE と同じ形で、feature-group-count が 1 で batch-group-count B（2〜3）の
ケース (lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation 1 B)（issue #203）。
lhs のバッチ N は B の倍数、rhs の出力チャネル O も B の倍数。"
  (loop
    (let* ((spatial (1+ (random 2)))
           (bgc (+ 2 (random 2)))
           (channels (1+ (random 3)))
           (strides (loop repeat spatial collect (1+ (random 3))))
           (lhs-dilation (loop repeat spatial collect (1+ (random 2))))
           (rhs-dilation (loop repeat spatial collect (1+ (random 2))))
           (padding (loop repeat spatial collect (list (- (random 4) 1) (- (random 4) 1))))
           (kernel (loop repeat spatial collect (1+ (random 3))))
           (in-sizes (loop repeat spatial collect (1+ (random 5))))
           (lhs-shape (append (list (* bgc (1+ (random 2)))) in-sizes (list channels)))
           (rhs-shape (append kernel (list channels (* bgc (1+ (random 2)))))))
      (when (every #'plusp
                   (loop for d in in-sizes for k in kernel for s in strides for p in padding
                         for ld in lhs-dilation for rd in rhs-dilation
                         collect (%conv-window-count d k s p ld rd)))
        (return (list lhs-shape rhs-shape strides padding lhs-dilation rhs-dilation 1 bgc))))))

(defun %conv-params-of (strides padding lhs-dilation rhs-dilation groups &optional (batch-groups 1))
  (list :strides strides :padding padding :lhs-dilation lhs-dilation
        :rhs-dilation rhs-dilation :feature-group-count groups :batch-group-count batch-groups))

(defun %conv-eager-result (dtype case)
  "CASE の形のランダムな入力に eager の convolution をかけ、(values 結果 lhs rhs) を返す。"
  (destructuring-bind (lhs-shape rhs-shape strides padding ld rd groups &optional (bgc 1)) case
    (let* ((lhs (make-random-array (make-array-spec lhs-shape dtype)))
           (rhs (make-random-array (make-array-spec rhs-shape dtype) :seed 1)))
      (values (apply #'%eager-of :convolution (list lhs rhs)
                     (list (nb:array-aval lhs dtype) (nb:array-aval rhs dtype))
                     (%conv-params-of strides padding ld rd groups bgc))
              lhs rhs))))

(test convolution/aval-matches-reference-shape
  "abstract-eval の出力の形は、参照実装（窓の数え上げ）の出力の形と、eager の結果の形に一致する。"
  (is (check-it (%conv-generator)
                (lambda (case)
                  (destructuring-bind (lhs-shape rhs-shape strides padding ld rd groups) case
                    (let* ((aval (apply #'%abstract-eval-of :convolution
                                        (list (nb:make-aval lhs-shape :f32) (nb:make-aval rhs-shape :f32))
                                        (%conv-params-of strides padding ld rd groups)))
                           (reference (reference-conv (make-array lhs-shape :element-type 'double-float
                                                                            :initial-element 1.0d0)
                                                      (make-array rhs-shape :element-type 'double-float
                                                                            :initial-element 1.0d0)
                                                      strides padding ld rd groups)))
                      (and (equal (nb:aval-shape aval) (array-dimensions reference))
                           (eq :f32 (nb:aval-dtype aval))
                           (equal (nb:aval-shape aval)
                                  (array-dimensions (%conv-eager-result :f32 case)))))))
                :regression-id convolution/aval-matches-reference-shape
                :regression-file (regression-path "convolution-aval"))))

(test convolution/eager-matches-reference
  "eager の convolution は、すべての float dtype で REFERENCE-CONV と dtype ごとの許容誤差で一致する。"
  (is (check-it (%conv-generator)
                (lambda (case)
                  (every (lambda (dtype)
                           (multiple-value-bind (result lhs rhs) (%conv-eager-result dtype case)
                             (destructuring-bind (lhs-shape rhs-shape strides padding ld rd groups) case
                               (declare (ignore lhs-shape rhs-shape))
                               (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                 (allclose (decode-array result dtype)
                                           (reference-conv (decode-array lhs dtype) (decode-array rhs dtype)
                                                           strides padding ld rd groups)
                                           :rtol rtol :atol atol)))))
                         *dtypes*))
                :regression-id convolution/eager-matches-reference
                :regression-file (regression-path "convolution-eager"))))

(test convolution/batch-groups-aval-and-eager-match-reference
  "batch-group-count B > 1（issue #203）: abstract-eval の出力のバッチは N/B で、eager は
すべての float dtype で REFERENCE-CONV（バッチ軸と出力チャネルを B 個の塊に分ける別実装）と一致する。"
  (is (check-it (%ad-hoc-generator #'%conv-bgc-random-case)
                (lambda (case)
                  (destructuring-bind (lhs-shape rhs-shape strides padding ld rd groups bgc) case
                    (and (equal (nb:aval-shape
                                 (apply #'%abstract-eval-of :convolution
                                        (list (nb:make-aval lhs-shape :f32) (nb:make-aval rhs-shape :f32))
                                        (%conv-params-of strides padding ld rd groups bgc)))
                                (array-dimensions (%conv-eager-result :f32 case)))
                         (= (floor (first lhs-shape) bgc) (first (array-dimensions (%conv-eager-result :f32 case))))
                         (every (lambda (dtype)
                                  (multiple-value-bind (result lhs rhs) (%conv-eager-result dtype case)
                                    (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                      (allclose (decode-array result dtype)
                                                (reference-conv (decode-array lhs dtype) (decode-array rhs dtype)
                                                                strides padding ld rd groups bgc)
                                                :rtol rtol :atol atol))))
                                *dtypes*))))
                :regression-id convolution/batch-groups-aval-and-eager-match-reference
                :regression-file (regression-path "convolution-batch-groups"))))

;;; --- JAX のフィクスチャ ---

(test convolution/eager-matches-jax-fixture
  "eager の convolution は、JAX の conv_general_dilated（stride・padding（負も）・膨張・
グループ・1〜3 次元）の f32 の出力と一致し、conv-general-dilated も同じ値を返す。"
  (let ((cases (read-conv-fixture)))
    (is (>= (length cases) 8))
    (dolist (case cases)
      (destructuring-bind (&key name strides padding lhs-dilation rhs-dilation groups lhs rhs out) case
        (let* ((expected out)
               (actual (nb:conv-general-dilated lhs rhs :strides strides :padding padding
                                                         :lhs-dilation lhs-dilation :rhs-dilation rhs-dilation
                                                         :feature-group-count groups)))
          (is (equal (array-dimensions expected) (array-dimensions actual))
              "~A: 形が違う ~S と ~S" name (array-dimensions actual) (array-dimensions expected))
          (is (allclose actual expected :dtype :f32 :rtol 1e-4 :atol 1e-5) "~A: 値が JAX と一致しない" name))))))

;;; --- 不正なパラメタの拒否 ---

(defun %conv-error-p (lhs-shape rhs-shape &rest params)
  "指定のパラメタ（%conv-params-of の plist の一部を上書きする）で abstract-eval が
PRIMITIVE-ERROR を出すなら真。"
  (let ((all (append params (%conv-params-of '(1 1) '((0 0) (0 0)) '(1 1) '(1 1) 1))))
    (handler-case
        (progn (apply #'%abstract-eval-of :convolution
                      (list (nb:make-aval lhs-shape :f32) (nb:make-aval rhs-shape :f32)) all)
               nil)
      (nb:primitive-error () t))))

(test convolution/rejects-invalid-params
  "v1 で受けない値は PRIMITIVE-ERROR になる: 長さの違うパラメタ、0 以下の stride・dilation、
不正な padding、チャネルとグループの不整合、出力が空になる形、rank の食い違い、整数 dtype。"
  (let ((l '(1 5 5 4)) (r '(3 3 4 2)))
    (is (not (%conv-error-p l r)))
    (is (%conv-error-p l r :strides '(1)))
    (is (%conv-error-p l r :strides '(0 1)))
    (is (%conv-error-p l r :lhs-dilation '(1 0)))
    ;; 0 の膨張は、出力が空にならない大きな padding でも拒否される
    (is (%conv-error-p l r :lhs-dilation '(1 0) :padding '((2 2) (2 2))))
    (is (%conv-error-p l r :lhs-dilation '(1)))
    (is (%conv-error-p l r :strides 5))                       ; リストでない
    (is (%conv-error-p l r :strides '(a 1)))                  ; 整数でない
    (is (%conv-error-p '(5 4) '(4 2) :strides '() :padding '() :lhs-dilation '() :rhs-dilation '()))
    (is (%conv-error-p l r :rhs-dilation '(1 1 1)))
    (is (%conv-error-p l r :padding '((0 0))))
    (is (%conv-error-p l r :padding '((0 0) (0 a))))
    (is (%conv-error-p l r :padding '((0 0 0) (0 0))))
    (is (%conv-error-p l r :feature-group-count 0))
    (is (%conv-error-p l r :feature-group-count 2))        ; rhs の入力チャネルが C/G と違う
    (is (%conv-error-p l '(3 3 2 3) :feature-group-count 2)) ; O が G で割り切れない
    ;; batch-group-count（issue #203）
    (is (not (%conv-error-p '(2 5 5 4) '(3 3 4 2) :batch-group-count 2)))
    (is (%conv-error-p '(2 5 5 4) '(3 3 4 2) :batch-group-count 0))
    (is (%conv-error-p '(2 5 5 4) '(3 3 4 2) :batch-group-count 1.5))
    (is (%conv-error-p '(3 5 5 4) '(3 3 4 2) :batch-group-count 2))   ; N が B で割り切れない
    (is (%conv-error-p '(2 5 5 4) '(3 3 4 3) :batch-group-count 2))   ; O が B で割り切れない
    (is (%conv-error-p '(2 5 5 4) '(3 3 2 2) :batch-group-count 2 :feature-group-count 2)) ; 両方 > 1
    (is (%conv-error-p '(1 2 2 4) r))                        ; 窓より入力が小さい
    (is (%conv-error-p l '(3 3 4)))                          ; rank が違う
    (is (%conv-error-p '(5 4) '(4 2)))                       ; 空間次元が無い
    (is (%conv-error-p '(0 5 5 4) r))                        ; N=0 も拒否（JAX は受け付ける）
    (is (%conv-error-p '(1 5 5 0) '(3 3 0 2)))               ; サイズ 0 の次元
    (is (%conv-error-p l '(0 3 4 2)))                        ; サイズ 0 のカーネル
    (dolist (avals (list (list (nb:make-aval l :f32))
                         (list (nb:make-aval l :f32) (nb:make-aval r :f32) (nb:make-aval r :f32))))
      (signals nb:primitive-error
        (apply #'%abstract-eval-of :convolution avals (%conv-params-of '(1 1) '((0 0) (0 0)) '(1 1) '(1 1) 1))))
    (signals nb:primitive-error
      (%abstract-eval-of :convolution (list (nb:make-aval l :i32) (nb:make-aval r :i32))
                         :strides '(1 1) :padding '((0 0) (0 0)) :lhs-dilation '(1 1)
                         :rhs-dilation '(1 1) :feature-group-count 1))
    (signals nb:primitive-error
      (%abstract-eval-of :convolution (list (nb:make-aval l :f32) (nb:make-aval r :f64))
                         :strides '(1 1) :padding '((0 0) (0 0)) :lhs-dilation '(1 1)
                         :rhs-dilation '(1 1) :feature-group-count 1))))

;;; --- StableHLO の出力（golden） ---

(test convolution/emit-golden-batch-group-count
  "batch-group-count は StableHLO の batch_group_count にそのまま出る（既定の 1 のときは従来の出力と同じ）。"
  (is (string= "%0 = stablehlo.convolution(%a, %b) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 2 : i64, feature_group_count = 1 : i64} : (tensor<4x5x3xf32>, tensor<2x3x6xf32>) -> tensor<2x4x6xf32>"
               (%emit-of :convolution '("%a" "%b")
                         (list (nb:make-aval '(4 5 3) :f32) (nb:make-aval '(2 3 6) :f32))
                         "%0" (nb:make-aval '(2 4 6) :f32)
                         :strides '(1) :padding '((0 0)) :lhs-dilation '(1) :rhs-dilation '(1)
                         :feature-group-count 1 :batch-group-count 2))))

(test convolution/emit-golden
  "f32 は stablehlo.convolution 1行、bf16 は f32 に convert して畳み込み、結果を戻す4行になる。"
  (let ((params '(:strides (2 1) :padding ((1 2) (0 0)) :lhs-dilation (1 1) :rhs-dilation (1 2)
                  :feature-group-count 3)))
    (is (string= "%0 = stablehlo.convolution(%a, %b) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 1], pad = [[1, 2], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 2]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<2x7x6x3xf32>, tensor<3x2x1x6xf32>) -> tensor<2x5x4x6xf32>"
                 (apply #'%emit-of :convolution '("%a" "%b")
                        (list (nb:make-aval '(2 7 6 3) :f32) (nb:make-aval '(3 2 1 6) :f32))
                        "%0" (nb:make-aval '(2 5 4 6) :f32) params)))
    (is (string= "%lhs32_0 = stablehlo.convert %a : (tensor<2x7x6x3xbf16>) -> tensor<2x7x6x3xf32>
%rhs32_0 = stablehlo.convert %b : (tensor<3x2x1x6xbf16>) -> tensor<3x2x1x6xf32>
%acc_0 = stablehlo.convolution(%lhs32_0, %rhs32_0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 1], pad = [[1, 2], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 2]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<2x7x6x3xf32>, tensor<3x2x1x6xf32>) -> tensor<2x5x4x6xf32>
%0 = stablehlo.convert %acc_0 : (tensor<2x5x4x6xf32>) -> tensor<2x5x4x6xbf16>"
                 (apply #'%emit-of :convolution '("%a" "%b")
                        (list (nb:make-aval '(2 7 6 3) :bf16) (nb:make-aval '(3 2 1 6) :bf16))
                        "%0" (nb:make-aval '(2 5 4 6) :bf16) params)))
    (let ((text (%emit-of :convolution '("%a" "%b")
                          (list (nb:make-aval '(1 5 2) :f16) (nb:make-aval '(2 2 2) :f16))
                          "%0" (nb:make-aval '(1 4 2) :f16)
                          :strides '(1) :padding '((0 0)) :lhs-dilation '(1) :rhs-dilation '(1)
                          :feature-group-count 1)))
      (is (search "%lhs32_0 = stablehlo.convert %a : (tensor<1x5x2xf16>) -> tensor<1x5x2xf32>" text))
      (is (search "-> tensor<1x4x2xf16>" text :from-end t)))
    (is (search "dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f]"
                (%emit-of :convolution '("%a" "%b")
                          (list (nb:make-aval '(1 5 2) :f32) (nb:make-aval '(2 2 2) :f32))
                          "%0" (nb:make-aval '(1 4 2) :f32)
                          :strides '(1) :padding '((0 0)) :lhs-dilation '(1) :rhs-dilation '(1)
                          :feature-group-count 1)))))

;;; --- conv-general-dilated（配列・トレーサ） ---

(test convolution/defaults-and-array-api
  "conv-general-dilated の既定は stride・dilation 1、padding なし、グループ 1。トレースすると
convolution の eqn になり、配列とトレーサを混ぜても呼べる。"
  (is (string= "(graph
 (:in (%0 f32 (1 5 5 2)) (%1 f32 (3 3 2 4)))
 (:const)
 (:eqns
  (%2 f32 (1 3 3 4) := convolution (:strides (1 1) :padding ((0 0) (0 0)) :lhs-dilation (1 1) :rhs-dilation (1 1) :feature-group-count 1 :batch-group-count 1) %0 %1))
 (:out %2))"
               (%print-graph-string (nb:with-tracing (x w) (nb:conv-general-dilated x w))
                                    (list (nb:make-aval '(1 5 5 2) :f32) (nb:make-aval '(3 3 2 4) :f32)))))
  (let* ((w (make-random-array (make-array-spec '(3 3 2 4) :f32) :seed 7))
         (x (make-random-array (make-array-spec '(1 5 5 2) :f32) :seed 8))
         (graph (nb:trace-to-graph (nb:with-tracing (x) (nb:conv-general-dilated x w :strides '(2 2)))
                                   (list (nb:make-aval '(1 5 5 2) :f32)))))
    (is (allclose (nb::eval-graph graph x) (nb:conv-general-dilated x w :strides '(2 2)) :dtype :f32))
    (is (equal '(1 2 2 4) (array-dimensions (nb::eval-graph graph x)))))
  (signals nb:primitive-error
    (nb:conv-general-dilated (make-random-array (make-array-spec '(1 5 5 2) :f32))
                             (make-random-array (make-array-spec '(3 3 2 4) :f32))
                             :strides '(1)))
  ;; rank が足りない入力も、生の型エラーではなく primitive-error
  (signals nb:primitive-error
    (nb:conv-general-dilated (make-random-array (make-array-spec '(3) :f32))
                             (make-random-array (make-array-spec '(2) :f32)))))
