;;;; gather / scatter-add プリミティブ（issue #240）の small テスト。
;;;;
;;;; gather は operand [N, rest...] の先頭の軸を整数の添字の配列 [idx...] で引き、[idx..., rest...] を返す。
;;;; scatter-add はその転置で、updates [idx..., rest...] を形 [N, rest...] の 0 の配列の行へ足し込む。
;;;; 範囲外の添字（負・N 以上）は gather では 0 の行になり、scatter-add ではその更新を捨てる
;;;; （JAX の FILL_OR_DROP と同じ構造で、埋める値だけ NaN でなく 0）。
;;;;
;;;; 形状推論・eager・StableHLO の出力と、jvp / transpose / バッチ化ルールを、独立に書いた
;;;; 添字のオラクルと性質で確かめる。添字は範囲外（-2..-1、N..N+1）と重複を含むように作る。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %gather-case (seed)
  "SEED から (operand-shape index-shape) を作る。operand は rank 1〜3（先頭の軸 N は 1〜5、残りは 1〜3）、
添字は rank 0〜2（各次元 1〜4）。"
  (let* ((rank (1+ (mod seed 3)))
         (operand-shape (cons (1+ (mod (floor seed 3) 5))
                              (loop for i from 1 below rank collect (1+ (mod (floor seed (expt 7 i)) 3)))))
         (index-rank (mod (floor seed 11) 3))
         (index-shape (loop for i below index-rank collect (1+ (mod (floor seed (expt 13 (1+ i))) 4)))))
    (list operand-shape index-shape)))

(defun %random-indices (shape n dtype seed)
  "形 SHAPE の DTYPE（整数）の添字の配列。値は -2..N+1（unsigned では負の代わりに 2^32 - 1 などの大きな値）。"
  (let ((state (sb-ext:seed-random-state seed))
        (array (make-array shape :element-type (nb:dtype-element-type dtype))))
    (dotimes (k (array-total-size array) array)
      (let ((value (- (random (+ n 4) state) 2)))
        (setf (row-major-aref array k)
              (cond ((>= value 0) value)
                    ((eq dtype :i32) value)
                    ((eq dtype :u32) (+ value (expt 2 32)))
                    (t (+ value (expt 2 64)))))))))

(defun %row-size (shape)
  (reduce #'* (rest shape) :initial-value 1))

(defun %gather-oracle (operand indices)
  "OPERAND の先頭の軸を INDICES で引いた配列。範囲外の添字の行は 0（各要素を row-major の位置から直接引く）。"
  (let* ((shape (array-dimensions operand))
         (n (first shape))
         (row (%row-size shape))
         (type (array-element-type operand))
         (result (make-array (append (array-dimensions indices) (rest shape))
                             :element-type type :initial-element (coerce 0 type))))
    (dotimes (k (array-total-size indices) result)
      (let ((i (row-major-aref indices k)))
        (when (< -1 i n)
          (dotimes (j row)
            (setf (row-major-aref result (+ (* k row) j)) (row-major-aref operand (+ (* i row) j)))))))))

(defun %scatter-add-oracle (indices updates n)
  "UPDATES（f64）の各行を、範囲内の INDICES の行へ足した形 [N, rest...] の f64 配列。"
  (let* ((rest (nthcdr (array-rank indices) (array-dimensions updates)))
         (row (reduce #'* rest :initial-value 1))
         (result (make-array (cons n rest) :element-type 'double-float :initial-element 0d0)))
    (dotimes (k (array-total-size indices) result)
      (let ((i (row-major-aref indices k)))
        (when (< -1 i n)
          (dotimes (j row)
            (incf (row-major-aref result (+ (* i row) j)) (row-major-aref updates (+ (* k row) j)))))))))

(defun %index-dtype (seed)
  (nth (mod (floor seed 17) 3) *integer-dtypes*))

(test gather/eager-matches-oracle-for-every-dtype
  "eager の gather は、独立に書いた添字のオラクルと一致する（すべての float dtype と整数 dtype の operand、
:i32 / :u32 / :u64 の添字、範囲外と重複を含む）。形状推論は [idx..., rest...] と operand の dtype を返す。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (operand-shape index-shape) (%gather-case seed)
                    (let* ((index-dtype (%index-dtype seed))
                           (indices (%random-indices index-shape (first operand-shape) index-dtype seed))
                           (index-aval (nb:array-aval indices index-dtype)))
                      (every (lambda (dtype)
                               (let* ((x (make-random-array (make-array-spec operand-shape dtype) :seed seed))
                                      (avals (list (nb:array-aval x dtype) index-aval)))
                                 (and (equalp (nb:make-aval (append index-shape (rest operand-shape)) dtype)
                                              (%abstract-eval-of :gather avals))
                                      (equalp (%gather-oracle x indices)
                                              (%eager-of :gather (list x indices) avals)))))
                             (append *dtypes* *integer-dtypes*)))))
                :regression-id gather/eager-matches-oracle-for-every-dtype
                :regression-file (regression-path "gather-eager"))))

(test scatter-add/eager-matches-oracle
  "eager の scatter-add は、範囲内の添字の行へ updates を足したオラクルと一致する（範囲外の更新は捨てる。
重複する添字の行は足し合わさる）。bf16 / f16 は f32 で累積して最後に丸める。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (operand-shape index-shape) (%gather-case seed)
                    (let* ((n (first operand-shape))
                           (index-dtype (%index-dtype seed))
                           (indices (%random-indices index-shape n index-dtype seed))
                           (updates-shape (append index-shape (rest operand-shape))))
                      (every (lambda (dtype)
                               (let* ((u (make-random-array (make-array-spec updates-shape dtype) :seed seed))
                                      (avals (list (nb:array-aval indices index-dtype) (nb:array-aval u dtype)))
                                      (expected (%scatter-add-oracle indices (decode-array u dtype) n)))
                                 (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
                                   (and (equalp (nb:make-aval operand-shape dtype)
                                                (%abstract-eval-of :scatter-add avals :size n))
                                        (allclose (decode-array (%eager-of :scatter-add (list indices u) avals :size n)
                                                                dtype)
                                                  expected :rtol rtol :atol atol)))))
                             *dtypes*))))
                :regression-id scatter-add/eager-matches-oracle
                :regression-file (regression-path "scatter-add-eager"))))

(test gather/scatter-add-is-the-adjoint
  "scatter-add は gather の随伴: <gather(x, i), u> = <x, scatter-add(i, u)>（f64、範囲外と重複を含む添字）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (operand-shape index-shape) (%gather-case seed)
                    (let* ((n (first operand-shape))
                           (indices (%random-indices index-shape n :i32 seed))
                           (x (make-random-array (make-array-spec operand-shape :f64) :seed seed))
                           (u (make-random-array (make-array-spec (append index-shape (rest operand-shape)) :f64)
                                                 :seed (1+ seed)))
                           (i-aval (nb:array-aval indices :i32)))
                      (%tr-close-p (inner-product (%eager-of :gather (list x indices)
                                                             (list (nb:array-aval x :f64) i-aval))
                                                  u)
                                   (inner-product x (%eager-of :scatter-add (list indices u)
                                                               (list i-aval (nb:array-aval u :f64)) :size n))
                                   1d-9))))
                :regression-id gather/scatter-add-is-the-adjoint
                :regression-file (regression-path "gather-adjoint"))))

(test gather/rejects-invalid-inputs
  "gather: 入力は 2 個、operand は rank 1 以上で先頭の軸が 1 以上 2^31 未満（:i32 の定数・添字に収まる）、:i1 でない。
添字は整数。違えば PRIMITIVE-ERROR。"
  (let ((x (nb:make-aval '(4 3) :f32))
        (i (nb:make-aval '(2) :i32)))
    (is (equalp (nb:make-aval '(2 3) :f32) (%abstract-eval-of :gather (list x i))))
    (is (equalp (nb:make-aval '(3) :f32) (%abstract-eval-of :gather (list x (nb:make-aval '() :u64)))))
    (signals nb:primitive-error (%abstract-eval-of :gather (list x)))
    (signals nb:primitive-error (%abstract-eval-of :gather (list x i i)))
    (signals nb:primitive-error (%abstract-eval-of :gather (list x (nb:make-aval '(2) :f32))))
    (signals nb:primitive-error (%abstract-eval-of :gather (list x (nb:make-aval '(2) :i1))))
    (signals nb:primitive-error (%abstract-eval-of :gather (list (nb:make-aval '() :f32) i)))
    (signals nb:primitive-error (%abstract-eval-of :gather (list (nb:make-aval '(0 3) :f32) i)))
    (signals nb:primitive-error (%abstract-eval-of :gather (list (nb:make-aval '(4) :i1) i)))
    (is (equalp (nb:make-aval '(2) :f32)
                (%abstract-eval-of :gather (list (nb:make-aval (list (1- (expt 2 31))) :f32) i))))
    (signals nb:primitive-error (%abstract-eval-of :gather (list (nb:make-aval (list (expt 2 31)) :f32) i)))))

(test scatter-add/rejects-invalid-inputs
  "scatter-add: 入力は 2 個（添字・updates）、:size は 1 以上 2^31 未満の整数、添字は整数、updates は浮動小数点で
形の先頭が添字の形。違えば PRIMITIVE-ERROR。"
  (let ((i (nb:make-aval '(2) :i32))
        (u (nb:make-aval '(2 3) :f32)))
    (is (equalp (nb:make-aval '(5 3) :f32) (%abstract-eval-of :scatter-add (list i u) :size 5)))
    (is (equalp (nb:make-aval '(5) :f64)
                (%abstract-eval-of :scatter-add (list (nb:make-aval '() :u32) (nb:make-aval '() :f64)) :size 5)))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list i) :size 5))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list i u) :size 0))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list i u) :size 2.0))
    (is (equalp (nb:make-aval (list (1- (expt 2 31)) 3) :f32)
                (%abstract-eval-of :scatter-add (list i u) :size (1- (expt 2 31)))))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list i u) :size (expt 2 31)))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list (nb:make-aval '(2) :f32) u) :size 5))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list i (nb:make-aval '(2 3) :i32)) :size 5))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list i (nb:make-aval '(3 2) :f32)) :size 5))
    (signals nb:primitive-error (%abstract-eval-of :scatter-add (list (nb:make-aval '(2 4) :i32) u) :size 5))
    (is (equalp (nb:make-aval '(5) :f32)
                (%abstract-eval-of :scatter-add (list (nb:make-aval '(2 3) :i32) u) :size 5)))))

(test gather/emit-golden
  "gather の StableHLO: 範囲内の判定（:i32 は 0 以上かつ N 未満）、範囲外を 0 にした添字を [idx..., 1] に
reshape した stablehlo.gather、範囲外の行を 0 にする select。"
  (is (string=
       (format nil "~{~A~^~%~}"
               '("%lo_0 = stablehlo.constant dense<0> : tensor<2xi32>"
                 "%ge_0 = stablehlo.compare GE, %i, %lo_0 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>"
                 "%hi_0 = stablehlo.constant dense<4> : tensor<2xi32>"
                 "%lt_0 = stablehlo.compare LT, %i, %hi_0 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>"
                 "%ok_0 = stablehlo.and %ge_0, %lt_0 : tensor<2xi1>"
                 "%safe_0 = stablehlo.select %ok_0, %i, %lo_0 : tensor<2xi1>, tensor<2xi32>"
                 "%vec_0 = stablehlo.reshape %safe_0 : (tensor<2xi32>) -> tensor<2x1xi32>"
                 "%rows_0 = \"stablehlo.gather\"(%x, %vec_0) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3>, indices_are_sorted = false} : (tensor<4x3xf32>, tensor<2x1xi32>) -> tensor<2x3xf32>"
                 "%okb_0 = stablehlo.broadcast_in_dim %ok_0, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1>"
                 "%zero_0 = stablehlo.constant dense<0.0> : tensor<2x3xf32>"
                 "%0 = stablehlo.select %okb_0, %rows_0, %zero_0 : tensor<2x3xi1>, tensor<2x3xf32>"))
       (%emit-of :gather '("%x" "%i") (list (nb:make-aval '(4 3) :f32) (nb:make-aval '(2) :i32))
                 "%0" (nb:make-aval '(2 3) :f32)))))

(test scatter-add/emit-golden
  "scatter-add の StableHLO（:u32 の添字は UNSIGNED の比較で N 未満だけを判定し :i32 に直す。bf16 は f32 で足して戻す）。"
  (is (string=
       (format nil "~{~A~^~%~}"
               '("%hi_0 = stablehlo.constant dense<4> : tensor<2xui32>"
                 "%ok_0 = stablehlo.compare LT, %i, %hi_0, UNSIGNED : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xi1>"
                 "%lo_0 = stablehlo.constant dense<0> : tensor<2xui32>"
                 "%safe_0 = stablehlo.select %ok_0, %i, %lo_0 : tensor<2xi1>, tensor<2xui32>"
                 "%safe32_0 = stablehlo.convert %safe_0 : (tensor<2xui32>) -> tensor<2xi32>"
                 "%vec_0 = stablehlo.reshape %safe32_0 : (tensor<2xi32>) -> tensor<2x1xi32>"
                 "%okb_0 = stablehlo.broadcast_in_dim %ok_0, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1>"
                 "%uzero_0 = stablehlo.constant dense<0x0000> : tensor<2x3xbf16>"
                 "%masked_0 = stablehlo.select %okb_0, %u, %uzero_0 : tensor<2x3xi1>, tensor<2x3xbf16>"
                 "%acc_0 = stablehlo.convert %masked_0 : (tensor<2x3xbf16>) -> tensor<2x3xf32>"
                 "%init_0 = stablehlo.constant dense<0.0> : tensor<4x3xf32>"
                 "%sum_0 = \"stablehlo.scatter\"(%init_0, %vec_0, %acc_0) ({"
                 "  ^bb0(%sa_0: tensor<f32>, %sb_0: tensor<f32>):"
                 "    %sc_0 = stablehlo.add %sa_0, %sb_0 : tensor<f32>"
                 "    stablehlo.return %sc_0 : tensor<f32>"
                 "}) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>) -> tensor<4x3xf32>"
                 "%0 = stablehlo.convert %sum_0 : (tensor<4x3xf32>) -> tensor<4x3xbf16>"))
       (%emit-of :scatter-add '("%i" "%u") (list (nb:make-aval '(2) :u32) (nb:make-aval '(2 3) :bf16))
                 "%0" (nb:make-aval '(4 3) :bf16) :size 4))))

;; golden の dtype 以外の分岐: f64 の 0 は浮動小数点のリテラル、f16 も bf16 と同じく f32 で累積する、
;; 要素が 0 個の定数は dense<>。
(test gather/emit-follows-the-dtype
  "f64 の gather の 0 は dense<0.0>、整数は dense<0>、f16 の scatter-add は f32 に直して足し f16 に戻す（f16 の 0 は
ビット列）、:u64 の添字は UNSIGNED で比べる、要素が 0 個の添字の定数は dense<>。"
  (let ((f64 (%emit-of :gather '("%x" "%i") (list (nb:make-aval '(4) :f64) (nb:make-aval '(2) :u64))
                       "%0" (nb:make-aval '(2) :f64)))
        (f16 (%emit-of :scatter-add '("%i" "%u") (list (nb:make-aval '(2) :i32) (nb:make-aval '(2) :f16))
                       "%0" (nb:make-aval '(4) :f16) :size 4))
        (empty (%emit-of :gather '("%x" "%i") (list (nb:make-aval '(4) :f32) (nb:make-aval '(0) :i32))
                         "%0" (nb:make-aval '(0) :f32))))
    (is (search "%zero_0 = stablehlo.constant dense<0.0> : tensor<2xf64>" f64))
    (is (search "UNSIGNED : (tensor<2xui64>, tensor<2xui64>)" f64))
    (is (search "%acc_0 = stablehlo.convert %masked_0 : (tensor<2xf16>) -> tensor<2xf32>" f16))
    (is (search "%0 = stablehlo.convert %sum_0 : (tensor<4xf32>) -> tensor<4xf16>" f16))
    (is (search "%lo_0 = stablehlo.constant dense<> : tensor<0xi32>" empty))
    (is (search "%uzero_0 = stablehlo.constant dense<0x0000> : tensor<2xf16>" f16))
    (is (search "%zero_0 = stablehlo.constant dense<0> : tensor<2xi32>"
                (%emit-of :gather '("%x" "%i") (list (nb:make-aval '(4) :i32) (nb:make-aval '(2) :i32))
                          "%0" (nb:make-aval '(2) :i32))))))

;;; --- 自動微分 ---

(defun %gather-graph (operand-shape indices)
  "operand（f64、形 OPERAND-SHAPE）だけを入力にし、INDICES（:i32）を定数にした gather の graph。"
  (let* ((x (nb::make-var (nb:make-aval operand-shape :f64)))
         (i (nb::make-var (nb:array-aval indices :i32)))
         (eqn (nb::make-eqn :gather (list x i))))
    (nb::make-graph (list x) (list eqn) (nb:eqn-outvars eqn) (list (cons i indices)))))

(defun %scatter-add-graph (updates-shape indices n)
  "updates（f64）だけを入力にし、INDICES（:i32）を定数にした scatter-add の graph。"
  (let* ((i (nb::make-var (nb:array-aval indices :i32)))
         (u (nb::make-var (nb:make-aval updates-shape :f64)))
         (eqn (nb::make-eqn :scatter-add (list i u) :size n)))
    (nb::make-graph (list u) (list eqn) (nb:eqn-outvars eqn) (list (cons i indices)))))

(defun %gather-graphs (seed)
  "SEED から (gather の graph . scatter-add の graph) を作る（同じ添字）。"
  (destructuring-bind (operand-shape index-shape) (%gather-case seed)
    (let ((indices (%random-indices index-shape (first operand-shape) :i32 seed)))
      (cons (%gather-graph operand-shape indices)
            (%scatter-add-graph (append index-shape (rest operand-shape)) indices (first operand-shape))))))

(test gather/jvp-matches-central-difference
  "gather と scatter-add の jvp は f64 の中心差分と一致する（どちらも operand / updates について線形）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (gather . scatter) (%gather-graphs seed)
                    (and (%jvp-matches-central-difference-p gather seed)
                         (%jvp-matches-central-difference-p scatter seed))))
                :regression-id gather/jvp-matches-central-difference
                :regression-file (regression-path "gather-jvp"))))

(test gather/jvp-treats-the-indices-as-constant
  "添字を入力にした gather の jvp: 添字の接線は symbolic zero で、出力の接線は gather(operand の接線, 添字)。"
  (let* ((x (nb::make-var (nb:make-aval '(4 2) :f64)))
         (i (nb::make-var (nb:make-aval '(3) :i32)))
         (eqn (nb::make-eqn :gather (list x i)))
         (graph (nb::make-graph (list x i) (list eqn) (nb:eqn-outvars eqn) '()))
         (indices (make-array 3 :element-type '(signed-byte 32) :initial-contents '(3 -1 0)))
         (primal (make-random-array (make-array-spec '(4 2) :f64) :seed 1))
         (tangent (make-random-array (make-array-spec '(4 2) :f64) :seed 2))
         (result (%jvp-eval (nb::jvp-graph graph) (list primal indices tangent))))
    (is (equalp (%gather-oracle primal indices) (first result)))
    (is (equalp (%gather-oracle tangent indices) (second result)))))

(test gather/vjp-is-scatter-add-and-back
  "gather の vjp の余接線は scatter-add のオラクル、scatter-add の vjp の余接線は gather のオラクルと一致する
（どちらも <vjp(u), v> = <u, jvp(v)> の内積テストも満たす）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                (lambda (seed)
                  (destructuring-bind (operand-shape index-shape) (%gather-case seed)
                    (let* ((n (first operand-shape))
                           (indices (%random-indices index-shape n :i32 seed))
                           (updates-shape (append index-shape (rest operand-shape)))
                           (x (make-random-array (make-array-spec operand-shape :f64) :seed seed))
                           (u (make-random-array (make-array-spec updates-shape :f64) :seed (1+ seed)))
                           (gather-ct (second (%jvp-eval (nb::vjp-graph (%gather-graph operand-shape indices))
                                                         (list x u))))
                           (scatter-ct (second (%jvp-eval (nb::vjp-graph (%scatter-add-graph updates-shape indices n))
                                                          (list u x)))))
                      (and (allclose gather-ct (%scatter-add-oracle indices u n) :dtype :f64)
                           (allclose scatter-ct (%gather-oracle x indices) :dtype :f64)
                           (%tr-close-p (inner-product gather-ct x) (inner-product u (%gather-oracle x indices))
                                        1d-9)))))
                :regression-id gather/vjp-is-scatter-add-and-back
                :regression-file (regression-path "gather-vjp"))))

(test gather/transpose-rejects-linear-indices
  "gather / scatter-add の添字を線形な入力にした転置は AUTODIFF-ERROR（添字は既知の主値でなければならない）。"
  (let* ((x (nb::make-var (nb:make-aval '(4 2) :f64)))
         (i (nb::make-var (nb:make-aval '(3) :i32)))
         (u (nb::make-var (nb:make-aval '(3 2) :f64)))
         (gather (nb::make-eqn :gather (list x i)))
         (scatter (nb::make-eqn :scatter-add (list i u) :size 4)))
    (signals nb:autodiff-error
      (nb::transpose-graph (nb::make-graph (list x i) (list gather) (nb:eqn-outvars gather) '()) 0))
    (signals nb:autodiff-error
      (nb::transpose-graph (nb::make-graph (list i u) (list scatter) (nb:eqn-outvars scatter) '()) 0))))

;;; --- バッチ化 ---

(defun %gather-vmap-case-p (seed code-operand code-indices code-out primitive)
  "PRIMITIVE（:gather か :scatter-add）の vmap が REFERENCE-VMAP と一致するか。CODE-* からバッチ軸（NIL もある。
少なくとも片方はバッチする）と out-axes を選ぶ。添字は :i32 / :u32 / :u64 で範囲外と重複を含む。"
  (destructuring-bind (operand-shape index-shape) (%gather-case seed)
    (let* ((n (first operand-shape))
           (size 3)
           (index-dtype (%index-dtype seed))
           (data-shape (if (eq primitive :gather) operand-shape (append index-shape (rest operand-shape))))
           (data-axis (%vmap-axis-or-nil code-operand (length data-shape)))
           (index-axis (if data-axis
                           (%vmap-axis-or-nil code-indices (length index-shape))
                           (mod code-indices (1+ (length index-shape)))))
           (data (%vmap-batched-array data-shape size data-axis seed))
           (indices (%random-indices (if index-axis (axis-at index-shape size index-axis) index-shape)
                                     n index-dtype seed))
           (out-rank (1+ (length (if (eq primitive :gather)
                                     (append index-shape (rest operand-shape))
                                     operand-shape))))
           (out (mod code-out out-rank)))
      (multiple-value-bind (f args in-axes)
          (if (eq primitive :gather)
              (values (primitive-function :gather '() 2) (list data indices) (list data-axis index-axis))
              (values (primitive-function :scatter-add (list :size n) 2) (list indices data)
                      (list index-axis data-axis)))
        (allclose (apply (nb:vmap f :in-axes in-axes :out-axes out) args)
                  (first (reference-vmap f args :in-axes in-axes :out-axes out))
                  :dtype :f64)))))

(test gather/batch-rule-matches-per-slice-reference
  "gather の vmap は、バッチ軸で切り出した各要素に gather をかけて積み直したものと一致する
（operand だけ・添字だけ・両方のバッチ、バッチ軸と out-axes の位置はランダム）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 99)
                                  (uniform-integer :lo 0 :hi 99) (uniform-integer :lo 0 :hi 99)))
                (lambda (case)
                  (destructuring-bind (seed a b c) case (%gather-vmap-case-p seed a b c :gather)))
                :regression-id gather/batch-rule-matches-per-slice-reference
                :regression-file (regression-path "gather-batch"))))

(test scatter-add/batch-rule-matches-per-slice-reference
  "scatter-add の vmap は、各要素に scatter-add をかけて積み直したものと一致する
（updates だけ・添字だけ・両方のバッチ）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 99)
                                  (uniform-integer :lo 0 :hi 99) (uniform-integer :lo 0 :hi 99)))
                (lambda (case)
                  (destructuring-bind (seed a b c) case (%gather-vmap-case-p seed a b c :scatter-add)))
                :regression-id scatter-add/batch-rule-matches-per-slice-reference
                :regression-file (regression-path "scatter-add-batch"))))

;;; --- 公開 API ---

(test gather/public-function-works-eagerly-and-under-grad
  "nb:gather は配列どうしなら eager に引き（形状推論の検査つき）、grad の中では添字の配列を定数にして
トレースする。Lisp の整数の添字は TRACING-ERROR。operand の勾配は、範囲内の添字の行に余接線を足したもの（範囲外の添字は寄与しない）。"
  (let* ((table (make-random-array (make-array-spec '(4 3) :f64) :seed 3))
         (ids (make-array '(2 3) :element-type '(signed-byte 32) :initial-contents '((0 2 2) (5 -1 3))))
         (weights (make-random-array (make-array-spec '(2 3 3) :f64) :seed 4))
         (loss (nb:with-tracing (x) (nb:reduce-sum (* (nb:gather x ids) weights)))))
    (is (equalp (%gather-oracle table ids) (nb:gather table ids)))
    (signals nb:primitive-error (nb:gather table (make-array 2 :element-type 'double-float)))
    (signals nb:tracing-error (nb:gather table 2))
    (is (allclose (funcall (nb:grad loss) table) (%scatter-add-oracle ids weights 4) :dtype :f64))))

(test gather/public-function-traces-two-tracers
  "operand も添字もトレーサの nb:gather は eqn を1つ足す: 添字を引数にした関数の grad は、範囲内の添字の行に 1 を
足した勾配になり、vmap（両方バッチ）も各要素の gather と一致する。"
  (let* ((f (nb:with-tracing (x i) (nb:reduce-sum (nb:gather x i))))
         (table (make-random-array (make-array-spec '(3 2) :f64) :seed 5))
         (ids (make-array 4 :element-type '(signed-byte 32) :initial-contents '(1 1 -1 3)))
         (expected (make-array '(3 2) :element-type 'double-float
                                      :initial-contents '((0d0 0d0) (2d0 2d0) (0d0 0d0))))
         (g (nb:with-tracing (x i) (nb:gather x i)))
         (tables (make-random-array (make-array-spec '(2 3 2) :f64) :seed 6))
         (batch-ids (make-array '(2 2) :element-type '(signed-byte 32) :initial-contents '((2 0) (1 5)))))
    (is (allclose (funcall (nb:grad f) table ids) expected :dtype :f64))
    (is (allclose (funcall (nb:vmap g) tables batch-ids)
                  (first (reference-vmap g (list tables batch-ids)))
                  :dtype :f64))))

(test gather/vmap-of-both-rejects-flat-indices-beyond-i32
  "operand と添字の両方をバッチした gather は、平らにした operand の行の数 B*N が 2^31 以上なら VMAP-ERROR（aval だけで
トレースするので大きな配列は作らない）。B*N = 2^31 - 2 は通る。"
  (let ((f (nb:vmap (primitive-function :gather '() 2) :in-axes '(0 0))))
    (flet ((trace-with (n)
             (nb::%call-with-fresh-trace
              (list (nb:make-aval (list 2 n) :f32) (nb:make-aval '(2 3) :i32))
              (lambda (x i) (funcall f x i)))))
      (finishes (trace-with (1- (expt 2 30))))
      (signals nb:vmap-error (trace-with (expt 2 30))))))
