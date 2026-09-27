;;;; primitives/common: 要素ごとの演算（elementwise）を実装するプリミティブ
;;;; が共有する eager 用ヘルパーと StableHLO 出力ヘルパー（issue #31 p1）。
;;;;
;;;; ここに置くのは「add / sub / mul / div」だけでなく、後続の p2（neg /
;;;; exp / log / tanh / max / min）も使う共通部品。dtype ごとの分岐
;;;; （bf16 / f16 のデコード・エンコード、f64 の計算型）を1か所に集め、
;;;; 各プリミティブの :eager 実装からループを追い出す。

(in-package #:nabla)

(defun %float-dtype-p (dtype)
  "DTYPE が浮動小数点の dtype（:F32 :F64 :BF16 :F16 のいずれか。:I1 は
含まない）なら真を返す。"
  (member dtype '(:f32 :f64 :bf16 :f16)))

(defun %check-arity (name in-avals n)
  "IN-AVALS の個数が N でなければ PRIMITIVE-ERROR を signal する。"
  (unless (= (length in-avals) n)
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "~D個の入力が必要（~D個渡された）"
           :format-arguments (list n (length in-avals)))))

(defun %check-float-dtype (name in-avals aval)
  "AVAL の dtype が浮動小数点でなければ PRIMITIVE-ERROR を signal する。"
  (unless (%float-dtype-p (aval-dtype aval))
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "dtype ~S は浮動小数点でなければならない"
           :format-arguments (list (aval-dtype aval)))))

(defun %check-same-avals (name in-avals)
  "IN-AVALS の shape がすべて EQUAL、dtype がすべて EQ であることを確かめ、
最初の AVAL を返す。1つでも食い違えば PRIMITIVE-ERROR を signal する。"
  (let ((first-aval (first in-avals)))
    (dolist (aval (rest in-avals) first-aval)
      (unless (equal (aval-shape aval) (aval-shape first-aval))
        (error 'primitive-error :name name :in-avals in-avals
               :format-control "shape が一致しない: ~S / ~S"
               :format-arguments (list (aval-shape first-aval) (aval-shape aval))))
      (unless (eq (aval-dtype aval) (aval-dtype first-aval))
        (error 'primitive-error :name name :in-avals in-avals
               :format-control "dtype が一致しない: ~S / ~S"
               :format-arguments (list (aval-dtype first-aval) (aval-dtype aval)))))))

(defun %binary-float-abstract-eval (name in-avals)
  "2入力・浮動小数点・shape/dtype が一致する演算の abstract-eval の共通部分。
入力チェックをすべて終えたあと、出力 AVAL（最初の入力の AVAL）を返す。"
  (%check-arity name in-avals 2)
  (let ((result (%check-same-avals name in-avals)))
    (%check-float-dtype name in-avals result)
    result))

(defun %compute-element-type (dtype)
  "DTYPE の計算に使う Common Lisp の浮動小数点型を返す。:F64 は
DOUBLE-FLOAT、それ以外（:F32 :BF16 :F16）は SINGLE-FLOAT
（bf16 / f16 は一度 single-float に変換してから計算する。CLAUDE.md）。"
  (if (eq dtype :f64) 'double-float 'single-float))

(defun %decode-array (array dtype)
  "ARRAY（DTYPE の格納表現を持つ配列）を計算用の浮動小数点配列にする。

bf16 / f16 はビット列から SINGLE-FLOAT の新しい配列にデコードする。
f32 / f64 はすでに計算に使える表現なので ARRAY をそのまま返す
（呼び出し側は返り値を破壊的に変更してはならない）。"
  (if (member dtype '(:bf16 :f16))
      (decode-float16-array array dtype)
      array))

(defun %encode-array (array dtype)
  "SINGLE-FLOAT の ARRAY を、DTYPE の格納表現に戻す。

bf16 / f16 は RNE でビット列にエンコードする。それ以外（:F32 :F64 :I1）
は ARRAY をそのまま返す（すでに格納表現そのもの）。"
  (if (member dtype '(:bf16 :f16))
      (encode-float16-array array dtype)
      array))

(defmacro with-ieee-arithmetic (&body body)
  "BODY を、SBCL の既定の浮動小数点トラップ（:OVERFLOW :INVALID
:DIVIDE-BY-ZERO）をすべてマスクした状態で評価する。

SBCL は既定でこれらのトラップを有効にしているため、たとえば
`(/ 1.0 0.0)` はコンディションを signal してしまう。プリミティブの eager
実装は StableHLO / IREE と同じ IEEE 754 の挙動（±inf・NaN を返す）に
揃えたいので、eager 呼び出し全体をこのマクロで包む（要素ごとにマスクを
掛け外しすると遅いので、必ず呼び出し全体を包む。用語集の「浮動小数点
トラップ」参照）。"
  `(sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero)
     ,@body))

(defun %elementwise-eager (fn arrays in-avals out-aval)
  "ARRAYS（IN-AVALS の dtype で格納された配列のリスト。すべて OUT-AVAL と
同じ shape）の各要素に FN を適用し、OUT-AVAL の格納表現を持つ新しい配列を
返す。

各入力は自分の IN-AVAL の dtype に応じて計算用の浮動小数点値にデコード
してから FN に渡す。OUT-AVAL の dtype が :I1 なら FN の返り値
（0 または 1）をそのまま格納し、それ以外なら計算結果を OUT-AVAL の dtype
の格納表現にエンコードする。rank 0 や要素数0の配列も扱える。FN の呼び出し
全体を浮動小数点トラップから守りたいときは、呼び出し側で
WITH-IEEE-ARITHMETIC に包むこと（ここでは包まない）。"
  (let* ((out-dtype (aval-dtype out-aval))
         (out-shape (aval-shape out-aval))
         (decoded (mapcar (lambda (array in-aval) (%decode-array array (aval-dtype in-aval)))
                           arrays in-avals))
         (compute-type (if (eq out-dtype :i1) 'bit (%compute-element-type out-dtype)))
         (result (make-array out-shape :element-type compute-type)))
    (dotimes (i (array-total-size result))
      (setf (row-major-aref result i)
            (apply fn (mapcar (lambda (a) (row-major-aref a i)) decoded))))
    (%encode-array result out-dtype)))

(defun %unary-float-abstract-eval (name in-avals)
  "1入力・浮動小数点の演算の abstract-eval の共通部分（issue #31 p2）。
入力チェックをすべて終えたあと、出力 AVAL（入力そのものの AVAL）を返す。"
  (%check-arity name in-avals 1)
  (let ((aval (first in-avals)))
    (%check-float-dtype name in-avals aval)
    aval))

(defun %quiet-nan (element-type)
  "ELEMENT-TYPE（'SINGLE-FLOAT または 'DOUBLE-FLOAT）の canonical quiet NaN
を返す（issue #31 p2）。ビット列から直接組み立てる（NaN を作るのに NaN を
生む浮動小数点演算は使わない）。"
  (ecase element-type
    (single-float (sb-kernel:make-single-float #x7FC00000))
    (double-float (sb-kernel:make-double-float #x7FF80000 0))))

(defun %ieee-max (a b)
  "A と B の大きい方を返す。CL の MAX と違い、どちらか一方でも NaN なら
NaN を返す（StableHLO の stablehlo.maximum / IREE / jnp.maximum に合わせる。
issue #31 p2 の pitfall: (max nan 1.0) => 1.0 だが (max 1.0 nan) => NaN、と
CL の MAX は引数の順序で挙動が変わり NaN を伝播しない）。"
  (cond
    ((sb-ext:float-nan-p a) a)
    ((sb-ext:float-nan-p b) b)
    (t (max a b))))

(defun %ieee-min (a b)
  "A と B の小さい方を返す。%IEEE-MAX と同じ理由で、どちらか一方でも NaN
なら NaN を返す。"
  (cond
    ((sb-ext:float-nan-p a) a)
    ((sb-ext:float-nan-p b) b)
    (t (min a b))))

(defun %emit-elementwise (op-name in-names out-name out-aval)
  "shape/dtype を変えない要素ごとの StableHLO 演算1行を組み立てる:
\"<out-name> = stablehlo.<op-name> <in-names, 区切り> : <out-avalの型>\"。
OUT-AVAL は（add のように）出力と入力すべてが同じ shape/dtype の演算にだけ
使える（型注釈が1つだけの pretty form）。"
  (format nil "~A = stablehlo.~A ~{~A~^, ~} : ~A"
          out-name op-name in-names (tensor-type-string out-aval)))
