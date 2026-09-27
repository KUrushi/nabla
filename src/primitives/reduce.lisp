;;;; reduce: reduce-sum / reduce-max プリミティブ（issue #31 p6）。
;;;;
;;;; StableHLO の stablehlo.reduce に対応する。:AXES で指定した次元
;;;; （非空・昇順・重複無し・[0, rank) の範囲内）を総和／最大値で潰す。
;;;; 出力の shape は入力の shape から :AXES に含まれる次元を取り除いた
;;;; もの（全軸を潰すと rank 0）。
;;;;
;;;; 契約 §2 の逸脱: 空の :AXES は（等軸のトレース層の識別コピーではなく）
;;;; PRIMITIVE-ERROR にする。「reduce しない」場合の処理は、この
;;;; プリミティブより上のレイヤー（wave 3 の配列 API）が axes を空に
;;;; 正規化して、そもそも eqn を作らないことで扱う。
;;;;
;;;; チェーンB（p4〜p6）は src/primitives/common.lisp（チェーンA所有）を
;;;; 編集しない約束なので、shape-common.lisp（p4）の %SHAPE-STRIDES /
;;;; %SHAPE-SUBSCRIPTS / %SHAPE-ROW-MAJOR-INDEX をそのまま再利用する。
;;;; dtype が float かどうかの判定は、このファイル専用の
;;;; %REDUCE-FLOAT-DTYPE-P に持つ（p5 の %DOT-FLOAT-DTYPE-P と同じ内容だが、
;;;; ファイルをまたいだ暗黙の依存を避けるための重複。DAMP、wave 3 で
;;;; 重複を解消する予定）。

(in-package #:nabla)

(defun %reduce-float-dtype-p (dtype)
  "DTYPE が reduce の入力として許される4つの浮動小数点 dtype
（:f32 :f64 :bf16 :f16）のいずれかかどうかを返す。"
  (and (member dtype '(:f32 :f64 :bf16 :f16)) t))

(defun %reduce-check-axes (name in-avals axes rank)
  "AXES（[0, RANK) の範囲内の、非空・重複無し・昇順の整数のリスト）を
検証する。不正なら PRIMITIVE-ERROR を signal する。"
  (unless (and (listp axes) (every #'integerp axes))
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "axes は整数のリストでなければならない: ~S"
           :format-arguments (list axes)))
  (when (null axes)
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "axes は空であってはならない"
           :format-arguments nil))
  (unless (every (lambda (a) (typep a `(integer 0 (,rank)))) axes)
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "axes ~S が rank ~S の範囲外"
           :format-arguments (list axes rank)))
  (unless (apply #'< axes)
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "axes ~S は重複の無い昇順でなければならない"
           :format-arguments (list axes))))

(defun %reduce-out-shape (shape axes)
  "SHAPE から AXES に含まれる次元を取り除いた shape を返す（全軸を
取り除くと rank 0 = '()）。"
  (loop for d in shape for i from 0
        unless (member i axes)
          collect d))

(defun %reduce-check (name in-avals axes)
  "入力の個数・dtype・AXES を検証し、(VALUES DTYPE SHAPE) を返す。不正なら
PRIMITIVE-ERROR を signal する。"
  (%shape-check-arity name in-avals 1)
  (let* ((in-aval (first in-avals))
         (dtype (aval-dtype in-aval))
         (shape (aval-shape in-aval)))
    (unless (%reduce-float-dtype-p dtype)
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "reduce は float dtype の入力しか受け付けない: ~S"
             :format-arguments (list dtype)))
    (%reduce-check-axes name in-avals axes (length shape))
    (values dtype shape)))

(defun %reduce-aux-name (tag out-name)
  "OUT-NAME（\"%7\" のような SSA 名）の数字部分を使った補助 SSA 名
（\"%init_7\"）を返す。"
  (format nil "%~A_~A" tag (subseq out-name 1)))

(defun %reduce-init-literal (op dtype)
  "OP（:ADD または :MAX）と DTYPE から、init 定数の MLIR リテラルを返す。
sum は 0（bf16/f16 はビット列そのまま \"0x0000\"）、max は -inf を
16進ビット列で書く（\"dense<-inf>\" は書かない）。"
  (ecase op
    (:add (if (member dtype '(:bf16 :f16)) "0x0000" "0.0"))
    (:max (ecase dtype
            (:f32 "0xFF800000")
            (:f64 "0xFFF0000000000000")
            (:bf16 "0xFF80")
            (:f16 "0xFC00")))))

(defun %reduce-emit (op in-names in-avals out-name out-aval axes)
  "reduce-sum（OP = :ADD）／reduce-max（OP = :MAX）の :EMIT 本体。init 定数
の宣言と stablehlo.reduce の2行を返す。"
  (let* ((in-aval (first in-avals))
         (dtype (aval-dtype in-aval))
         (init-name (%reduce-aux-name "init" out-name))
         (scalar-type (format nil "tensor<~A>" (dtype-mlir-name dtype)))
         (applies (ecase op (:add "add") (:max "maximum"))))
    (format nil "~A = stablehlo.constant dense<~A> : ~A~%~A = stablehlo.reduce(~A init: ~A) applies stablehlo.~A across dimensions = [~{~D~^, ~}] : (~A, ~A) -> ~A"
            init-name (%reduce-init-literal op dtype) scalar-type
            out-name (first in-names) init-name applies axes
            (tensor-type-string in-aval) scalar-type (tensor-type-string out-aval))))

(defun %reduce-decode (array dtype)
  "ARRAY が :bf16 / :f16 のビット列表現なら SINGLE-FLOAT にデコードし、
:f32 / :f64 ならそのまま ARRAY を返す。"
  (if (member dtype '(:bf16 :f16)) (decode-float16-array array dtype) array))

(defun %reduce-max-element (a b)
  "A・B（同じ COMPUTE-TYPE の浮動小数点数）の NaN 伝播する最大値を返す。
CL:MAX は NaN を伝播しない（(max nan 1.0) => 1.0 のことがある）ので、
どちらかが NaN ならその NaN をそのまま返す。"
  (cond
    ((sb-ext:float-nan-p a) a)
    ((sb-ext:float-nan-p b) b)
    (t (max a b))))

(defun %reduce-init-value (op compute-type)
  "OP（:ADD または :MAX）の総和／最大値の初期値を COMPUTE-TYPE で返す。"
  (ecase op
    (:add (coerce 0 compute-type))
    (:max (if (eq compute-type 'double-float)
              sb-ext:double-float-negative-infinity
              sb-ext:single-float-negative-infinity))))

(defun %reduce-eager (op in-avals arrays axes)
  "reduce-sum（OP = :ADD）／reduce-max（OP = :MAX）の :EAGER 本体。入力の
row-major index を driver にして、AXES を落とした出力の添字へ足し込む
（gather ではなく accumulate なので、出力側ではなく入力側を driver に
する。契約のピットフォール(4)）。"
  (let* ((in-aval (first in-avals))
         (dtype (aval-dtype in-aval))
         (compute-type (if (eq dtype :f64) 'double-float 'single-float))
         (in-shape (aval-shape in-aval))
         (out-shape (%reduce-out-shape in-shape axes))
         (in-array (%reduce-decode (first arrays) dtype))
         (in-strides (%shape-strides in-shape))
         (out-strides (%shape-strides out-shape))
         (result (make-array out-shape :element-type compute-type
                              :initial-element (%reduce-init-value op compute-type)))
         (combine (ecase op (:add #'+) (:max #'%reduce-max-element))))
    (sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero)
      (dotimes (in-i (array-total-size in-array))
        (let* ((in-subs (%shape-subscripts in-i in-shape in-strides))
               (out-subs (loop for s in in-subs for d from 0
                                unless (member d axes)
                                  collect s))
               (out-i (%shape-row-major-index out-subs out-shape out-strides)))
          (setf (row-major-aref result out-i)
                (funcall combine (row-major-aref result out-i) (row-major-aref in-array in-i))))))
    (if (member dtype '(:bf16 :f16))
        (encode-float16-array result dtype)
        result)))

(defprimitive reduce-sum (:axes)
  :abstract-eval
  (lambda (in-avals &key axes)
    (multiple-value-bind (dtype shape) (%reduce-check :reduce-sum in-avals axes)
      (make-aval (%reduce-out-shape shape axes) dtype)))
  :emit
  (lambda (in-names in-avals out-name out-aval &key axes)
    (%reduce-emit :add in-names in-avals out-name out-aval axes))
  :eager
  (lambda (arrays in-avals &key axes)
    (%reduce-eager :add in-avals arrays axes)))

(defprimitive reduce-max (:axes)
  :abstract-eval
  (lambda (in-avals &key axes)
    (multiple-value-bind (dtype shape) (%reduce-check :reduce-max in-avals axes)
      (make-aval (%reduce-out-shape shape axes) dtype)))
  :emit
  (lambda (in-names in-avals out-name out-aval &key axes)
    (%reduce-emit :max in-names in-avals out-name out-aval axes))
  :eager
  (lambda (arrays in-avals &key axes)
    (%reduce-eager :max in-avals arrays axes)))
