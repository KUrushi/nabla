;;;; primitives/gather: gather と scatter-add（issue #240）。
;;;;
;;;; gather (operand [N, rest...], indices [idx...]) → [idx..., rest...]: operand の先頭の軸を整数の
;;;; 添字の配列で引く（JAX の jnp.take(operand, indices, axis=0)。StableHLO の一般の gather のうち、
;;;; この形だけを持つ）。scatter-add (indices [idx...], updates [idx..., rest...]) :size N → [N, rest...]:
;;;; その転置で、0 の配列の行へ updates を足し込む（重複する添字の行は足し合わさる）。
;;;;
;;;; 範囲外の添字（負・N 以上）: gather はその行を 0 にし、scatter-add はその更新を捨てる。JAX の
;;;; FILL_OR_DROP（jnp.take の既定）と同じ構造で、gather と転置が整合する（捨てた更新 ⇔ 0 の行）。
;;;; 埋める値は JAX の NaN でなく 0（one-hot で引いていた embedding と同じ）。StableHLO の gather は
;;;; 範囲外をクランプし、scatter は実装定義なので、emit は範囲の判定を自分で書き、範囲外の添字を 0 に
;;;; してから引き（足し）、結果（更新）を select で 0 にする。実行系の範囲外の挙動には頼らない。
;;;; 判定は添字の元の dtype で行い（unsigned は N 未満だけ）、判定の後で :i32 に直すので、:u64 の
;;;; 大きな添字が折り返して範囲内に入ることはない。
;;;;
;;;; operand は :i1 以外の任意の dtype（gather は値を解釈しない）。scatter-add の updates は浮動小数点
;;;; だけで、bf16 / f16 は reduce-sum と同じく f32 で累積して最後に丸める（emit も eager も）。

(in-package #:nabla)

;;; --- 形状推論 ---

(defun %gather-error (name in-avals control &rest arguments)
  (error 'primitive-error :name name :in-avals in-avals
                          :format-control control :format-arguments arguments))

(defun %gather-check-indices (name in-avals indices)
  "INDICES（aval）の dtype が整数でなければ PRIMITIVE-ERROR。"
  (unless (integer-dtype-p (aval-dtype indices))
    (%gather-error name in-avals "添字の dtype は整数（:i32 / :u32 / :u64）でなければならない: ~S"
                   (aval-dtype indices))))

(defun %gather-abstract-eval (in-avals)
  (%shape-check-arity :gather in-avals 2)
  (destructuring-bind (operand indices) in-avals
    (%gather-check-indices :gather in-avals indices)
    (let ((shape (aval-shape operand)))
      (unless (and shape (plusp (first shape)))
        (%gather-error :gather in-avals "operand は rank 1 以上で先頭の軸が 1 以上でなければならない: ~S" shape))
      (when (eq (aval-dtype operand) :i1)
        (%gather-error :gather in-avals "operand の dtype に :I1 は使えない"))
      (make-aval (append (aval-shape indices) (rest shape)) (aval-dtype operand)))))

(defun %scatter-add-abstract-eval (in-avals size)
  (%shape-check-arity :scatter-add in-avals 2)
  (unless (typep size '(integer 1))
    (%gather-error :scatter-add in-avals ":size は 1 以上の整数でなければならない: ~S" size))
  (destructuring-bind (indices updates) in-avals
    (%gather-check-indices :scatter-add in-avals indices)
    (unless (%float-dtype-p (aval-dtype updates))
      (%gather-error :scatter-add in-avals "updates の dtype は浮動小数点でなければならない: ~S"
                     (aval-dtype updates)))
    (let ((index-shape (aval-shape indices))
          (shape (aval-shape updates)))
      (unless (equal index-shape (subseq shape 0 (min (length shape) (length index-shape))))
        (%gather-error :scatter-add in-avals "updates の形 ~S の先頭が添字の形 ~S でない" shape index-shape))
      (make-aval (cons size (nthcdr (length index-shape) shape)) (aval-dtype updates)))))

;;; --- eager ---

(defun %gather-row-size (shape)
  "先頭の軸を除いた要素数（1 行の大きさ）。"
  (reduce #'* (rest shape) :initial-value 1))

(defun %gather-eager (arrays in-avals)
  "範囲内の添字の行を row-major の格納表現のままコピーする（bf16 / f16 もデコードしない）。範囲外は 0。"
  (destructuring-bind (operand indices) arrays
    (let* ((out-aval (%gather-abstract-eval in-avals))
           (shape (aval-shape (first in-avals)))
           (n (first shape))
           (row (%gather-row-size shape))
           (type (dtype-element-type (aval-dtype out-aval)))
           (result (make-array (aval-shape out-aval) :element-type type :initial-element (coerce 0 type))))
      (dotimes (k (array-total-size indices) result)
        (let ((i (row-major-aref indices k)))
          (when (< -1 i n)
            (dotimes (j row)
              (setf (row-major-aref result (+ (* k row) j)) (row-major-aref operand (+ (* i row) j))))))))))

(defun %scatter-add-eager (arrays in-avals size)
  "updates を計算用の値（bf16 / f16 は single-float）にして、範囲内の添字の行へ足す。"
  (destructuring-bind (indices updates) arrays
    (let* ((out-aval (%scatter-add-abstract-eval in-avals size))
           (dtype (aval-dtype out-aval))
           (row (%gather-row-size (aval-shape out-aval)))
           (compute-type (if (eq dtype :f64) 'double-float 'single-float))
           (decoded (%reduce-decode updates dtype))
           (result (make-array (aval-shape out-aval) :element-type compute-type
                                                     :initial-element (coerce 0 compute-type))))
      (with-ieee-arithmetic
        (dotimes (k (array-total-size indices))
          (let ((i (row-major-aref indices k)))
            (when (< -1 i size)
              (dotimes (j row)
                (incf (row-major-aref result (+ (* i row) j)) (row-major-aref decoded (+ (* k row) j))))))))
      (if (member dtype '(:bf16 :f16)) (encode-float16-array result dtype) result))))

;;; --- StableHLO ---

(defun %gather-zero-literal (dtype)
  (cond ((member dtype '(:f32 :f64)) "0.0")
        ((member dtype '(:bf16 :f16)) "0x0000")
        (t "0")))

(defun %gather-index-lines (index-name index-aval n out-name)
  "添字の範囲の判定と、範囲外を 0 にして :i32 の [idx..., 1] にした添字を作る行。
\(values 行のリスト 判定（:i1）の名前 添字の名前）。"
  (let* ((dtype (aval-dtype index-aval))
         (shape (aval-shape index-aval))
         (type (tensor-type-string index-aval))
         (bool-type (tensor-type-string (make-aval shape :i1)))
         (i32-type (tensor-type-string (make-aval shape :i32)))
         (vec-type (tensor-type-string (make-aval (append shape '(1)) :i32)))
         (lo (%convert-aux-name "lo" out-name))
         (hi (%convert-aux-name "hi" out-name))
         (ok (%convert-aux-name "ok" out-name))
         (safe (%convert-aux-name "safe" out-name))
         (safe32 (if (eq dtype :i32) safe (%convert-aux-name "safe32" out-name)))
         (vec (%convert-aux-name "vec" out-name)))
    (flet ((constant (name value) (format nil "~A = stablehlo.constant dense<~D> : ~A" name value type))
           (compare (name direction bound)
             (format nil "~A = stablehlo.compare ~A, ~A, ~A : (~A, ~A) -> ~A"
                     name direction index-name bound type type bool-type)))
      (values
       (append
        (if (eq dtype :i32)
            (list (constant lo 0)
                  (compare (%convert-aux-name "ge" out-name) "GE" lo)
                  (constant hi n)
                  (compare (%convert-aux-name "lt" out-name) "LT" hi)
                  (format nil "~A = stablehlo.and ~A, ~A : ~A" ok
                          (%convert-aux-name "ge" out-name) (%convert-aux-name "lt" out-name) bool-type))
            ;; unsigned は負にならないので、N 未満だけで判定する。
            (list (constant hi n)
                  (compare ok "LT" hi)
                  (constant lo 0)))
        (list (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" safe ok index-name lo bool-type type))
        (unless (eq dtype :i32)
          (list (format nil "~A = stablehlo.convert ~A : (~A) -> ~A" safe32 safe type i32-type)))
        (list (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A" vec safe32 i32-type vec-type)))
       ok vec))))

(defun %gather-mask-line (name ok index-aval aval)
  "判定 OK（形は添字の形）を AVAL の形へ広げる行。"
  (format nil "~A = stablehlo.broadcast_in_dim ~A, dims = [~{~D~^, ~}] : (~A) -> ~A"
          name ok (loop for d below (aval-rank index-aval) collect d)
          (tensor-type-string (make-aval (aval-shape index-aval) :i1))
          (tensor-type-string (make-aval (aval-shape aval) :i1))))

(defun %gather-axes (from count)
  (loop for d from from below (+ from count) collect d))

(defun %gather-emit (in-names in-avals out-name out-aval)
  (destructuring-bind (operand-aval index-aval) in-avals
    (multiple-value-bind (lines ok vec)
        (%gather-index-lines (second in-names) index-aval (first (aval-shape operand-aval)) out-name)
      (let* ((rank (aval-rank index-aval))
             (rest (rest (aval-shape operand-aval)))
             (rows (%convert-aux-name "rows" out-name))
             (okb (%convert-aux-name "okb" out-name))
             (zero (%convert-aux-name "zero" out-name))
             (out-type (tensor-type-string out-aval))
             (offset-dims (%gather-axes rank (length rest))))
        (format nil "~{~A~^~%~}"
                (append
                 lines
                 (list
                  (format nil "~A = \"stablehlo.gather\"(~A, ~A) {dimension_numbers = #stablehlo.gather<~@[offset_dims = [~{~D~^, ~}], ~]collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = ~D>, slice_sizes = array<i64: ~{~D~^, ~}>, indices_are_sorted = false} : (~A, ~A) -> ~A"
                          rows (first in-names) vec offset-dims rank (cons 1 rest)
                          (tensor-type-string operand-aval)
                          (tensor-type-string (make-aval (append (aval-shape index-aval) '(1)) :i32))
                          out-type)
                  (%gather-mask-line okb ok index-aval out-aval)
                  (format nil "~A = stablehlo.constant dense<~A> : ~A" zero
                          (%gather-zero-literal (aval-dtype out-aval)) out-type)
                  (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" out-name okb rows zero
                          (tensor-type-string (make-aval (aval-shape out-aval) :i1)) out-type))))))))

(defun %scatter-add-emit (in-names in-avals out-name out-aval)
  (destructuring-bind (index-aval updates-aval) in-avals
    (multiple-value-bind (lines ok vec)
        (%gather-index-lines (first in-names) index-aval (first (aval-shape out-aval)) out-name)
      (let* ((dtype (aval-dtype out-aval))
             (accumulate-in-f32 (member dtype '(:bf16 :f16)))
             (acc-dtype (if accumulate-in-f32 :f32 dtype))
             (rank (aval-rank index-aval))
             (updates-type (tensor-type-string updates-aval))
             (acc-updates-type (tensor-type-string (make-aval (aval-shape updates-aval) acc-dtype)))
             (acc-out-type (tensor-type-string (make-aval (aval-shape out-aval) acc-dtype)))
             (scalar-type (format nil "tensor<~A>" (dtype-mlir-name acc-dtype)))
             (okb (%convert-aux-name "okb" out-name))
             (uzero (%convert-aux-name "uzero" out-name))
             (masked (%convert-aux-name "masked" out-name))
             (acc (if accumulate-in-f32 (%convert-aux-name "acc" out-name) masked))
             (init (%convert-aux-name "init" out-name))
             (sum (if accumulate-in-f32 (%convert-aux-name "sum" out-name) out-name))
             (a (%convert-aux-name "sa" out-name))
             (b (%convert-aux-name "sb" out-name))
             (c (%convert-aux-name "sc" out-name))
             (window-dims (%gather-axes rank (1- (aval-rank out-aval)))))
        (format nil "~{~A~^~%~}"
                (append
                 lines
                 (list
                  (%gather-mask-line okb ok index-aval updates-aval)
                  (format nil "~A = stablehlo.constant dense<~A> : ~A" uzero (%gather-zero-literal dtype) updates-type)
                  (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" masked okb (second in-names) uzero
                          (tensor-type-string (make-aval (aval-shape updates-aval) :i1)) updates-type))
                 (when accumulate-in-f32
                   (list (format nil "~A = stablehlo.convert ~A : (~A) -> ~A" acc masked updates-type acc-updates-type)))
                 (list
                  (format nil "~A = stablehlo.constant dense<~A> : ~A" init (%gather-zero-literal acc-dtype) acc-out-type)
                  (format nil "~A = \"stablehlo.scatter\"(~A, ~A, ~A) ({" sum init vec acc)
                  (format nil "  ^bb0(~A: ~A, ~A: ~A):" a scalar-type b scalar-type)
                  (format nil "    ~A = stablehlo.add ~A, ~A : ~A" c a b scalar-type)
                  (format nil "    stablehlo.return ~A : ~A" c scalar-type)
                  (format nil "}) {scatter_dimension_numbers = #stablehlo.scatter<~@[update_window_dims = [~{~D~^, ~}], ~]inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = ~D>, indices_are_sorted = false, unique_indices = false} : (~A, ~A, ~A) -> ~A"
                          window-dims rank acc-out-type
                          (tensor-type-string (make-aval (append (aval-shape index-aval) '(1)) :i32))
                          acc-updates-type acc-out-type))
                 (when accumulate-in-f32
                   (list (format nil "~A = stablehlo.convert ~A : (~A) -> ~A" out-name sum acc-out-type
                                 (tensor-type-string out-aval))))))))))

;;; --- プリミティブ ---

(defprimitive gather ()
  :abstract-eval (lambda (in-avals) (%gather-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval) (%gather-emit in-names in-avals out-name out-aval))
  :eager (lambda (arrays in-avals) (%gather-eager arrays in-avals)))

(defprimitive scatter-add (:size)
  :abstract-eval (lambda (in-avals &key size) (%scatter-add-abstract-eval in-avals size))
  :emit (lambda (in-names in-avals out-name out-aval &key size)
          (declare (ignore size))
          (%scatter-add-emit in-names in-avals out-name out-aval))
  :eager (lambda (arrays in-avals &key size) (%scatter-add-eager arrays in-avals size)))
