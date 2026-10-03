;;;; primitives/compare: compare / select / convert プリミティブ
;;;; （issue #31 p3）。
;;;;
;;;; 3つとも「2入力・浮動小数点専用」という arith.lisp / unary.lisp の形とは
;;;; 少し違う（compare は出力が :i1、select は :i1 の pred を受け取り任意の
;;;; dtype を通す、convert は出力の dtype がパラメタ）ため、abstract-eval は
;;;; %BINARY-FLOAT-ABSTRACT-EVAL 等の既存ヘルパーをそのまま使えるところだけ
;;;; 使い、それ以外は自分でチェックを書く。

(in-package #:nabla)

;;; --- compare ---

(defun %compare-element (a b direction)
  "A と B（1要素、DIRECTION の入力と同じ計算用浮動小数点型）を DIRECTION
（:LT :LE :GT :GE :EQ :NE のいずれか）で比較した結果を BIT（0 または 1）で
返す。

NaN が絡む比較は :NE 以外すべて偽（0）、:NE だけ真（1）になる（IEEE 754 の
規則。CLAUDE.md / 契約の pitfall）。SBCL の < > = はトラップをマスクした
状態でも NaN に対して正しく NIL/T を返すが、可読性と mutation testing の
的にするため分岐を明示的に書く。"
  (if (or (and (floatp a) (sb-ext:float-nan-p a)) (and (floatp b) (sb-ext:float-nan-p b)))
      (if (eq direction :ne) 1 0)
      (if (ecase direction
            (:lt (< a b))
            (:le (<= a b))
            (:gt (> a b))
            (:ge (>= a b))
            (:eq (= a b))
            (:ne (/= a b)))
          1
          0)))

(defun %compare-abstract-eval (in-avals &key direction)
  "2入力・浮動小数点か整数・shape/dtype が一致する演算の共通チェックをしたあと、
DIRECTION が :LT :LE :GT :GE :EQ :NE のいずれかであることを確かめ、入力と
同じ shape・dtype :I1 の AVAL を返す。"
  (let ((in-aval (%binary-numeric-abstract-eval :compare in-avals)))
    (unless (member direction '(:lt :le :gt :ge :eq :ne))
      (error 'primitive-error :name :compare :in-avals in-avals
             :format-control "direction ~S は :LT :LE :GT :GE :EQ :NE のいずれかでなければならない"
             :format-arguments (list direction)))
    (make-aval (aval-shape in-aval) :i1)))

(defun %compare-emit (in-names in-avals out-name out-aval &key direction)
  "\"<out-name> = stablehlo.compare <DIRECTION>, <a>, <b> : (<Ta>, <Tb>) -> <Tout>\"。"
  (format nil "~A = stablehlo.compare ~A, ~{~A~^, ~} : (~A, ~A) -> ~A"
          out-name (symbol-name direction) in-names
          (tensor-type-string (first in-avals)) (tensor-type-string (second in-avals))
          (tensor-type-string out-aval)))

(defun %compare-eager (arrays in-avals &key direction)
  (let ((out-aval (%compare-abstract-eval in-avals :direction direction)))
    (with-ieee-arithmetic
      (%elementwise-eager (lambda (a b) (%compare-element a b direction)) arrays in-avals out-aval))))

(defprimitive compare (:direction)
  :abstract-eval (lambda (in-avals &key direction) (%compare-abstract-eval in-avals :direction direction))
  :emit (lambda (in-names in-avals out-name out-aval &key direction)
          (%compare-emit in-names in-avals out-name out-aval :direction direction))
  :eager (lambda (arrays in-avals &key direction) (%compare-eager arrays in-avals :direction direction)))

;;; --- select ---

(defun %select-abstract-eval (in-avals)
  "3入力（pred on-true on-false）。pred の dtype は :I1、3つの shape はすべて
EQUAL、on-true/on-false の AVAL は EQUALP（dtype は :I1 を含む任意の dtype
でよい）であることを確かめ、on-true の AVAL を返す。"
  (%check-arity :select in-avals 3)
  (let ((pred (first in-avals)) (on-true (second in-avals)) (on-false (third in-avals)))
    (unless (eq (aval-dtype pred) :i1)
      (error 'primitive-error :name :select :in-avals in-avals
             :format-control "pred の dtype は :I1 でなければならない: ~S"
             :format-arguments (list (aval-dtype pred))))
    (unless (equal (aval-shape pred) (aval-shape on-true))
      (error 'primitive-error :name :select :in-avals in-avals
             :format-control "pred と on-true の shape が一致しない: ~S / ~S"
             :format-arguments (list (aval-shape pred) (aval-shape on-true))))
    (unless (equal (aval-shape pred) (aval-shape on-false))
      (error 'primitive-error :name :select :in-avals in-avals
             :format-control "pred と on-false の shape が一致しない: ~S / ~S"
             :format-arguments (list (aval-shape pred) (aval-shape on-false))))
    (unless (equalp on-true on-false)
      (error 'primitive-error :name :select :in-avals in-avals
             :format-control "on-true と on-false の aval が一致しない: ~S / ~S"
             :format-arguments (list on-true on-false)))
    on-true))

(defun %select-emit (in-names in-avals out-name out-aval)
  "\"<out-name> = stablehlo.select <pred>, <a>, <b> : <Tpred>, <Tout>\"。"
  (format nil "~A = stablehlo.select ~{~A~^, ~} : ~A, ~A"
          out-name in-names (tensor-type-string (first in-avals)) (tensor-type-string out-aval)))

(defun %select-eager (arrays in-avals)
  "pred のビットに応じて on-true / on-false の raw storage をそのまま
コピーする（decode しない。bf16/f16 でも正確、pred/on-true/on-false が
:I1 でも動く。契約の pitfall #3）。"
  (let* ((out-aval (%select-abstract-eval in-avals))
         (pred (first arrays)) (on-true (second arrays)) (on-false (third arrays))
         (result (make-array (aval-shape out-aval) :element-type (dtype-element-type (aval-dtype out-aval)))))
    (dotimes (i (array-total-size result) result)
      (setf (row-major-aref result i)
            (if (= 1 (row-major-aref pred i))
                (row-major-aref on-true i)
                (row-major-aref on-false i))))))

(defprimitive select ()
  :abstract-eval (lambda (in-avals) (%select-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (%select-emit in-names in-avals out-name out-aval))
  :eager (lambda (arrays in-avals) (%select-eager arrays in-avals)))

;;; --- convert ---

(defun %convert-abstract-eval (in-avals &key dtype)
  "1入力。入力の dtype も DTYPE も、浮動小数点・整数・:I1 のどれでもよい
（整数 ⇔ 浮動小数点 ⇔ :I1 を変換できる。issue #126）。DTYPE が dtype として
不正なら PRIMITIVE-ERROR。入力と同じ shape・DTYPE の AVAL を返す。"
  (%check-arity :convert in-avals 1)
  (let ((in-aval (first in-avals)))
    (unless (typep dtype 'dtype)
      (error 'primitive-error :name :convert :in-avals in-avals
             :format-control "dtype ~S は nabla の dtype でなければならない"
             :format-arguments (list dtype)))
    (make-aval (aval-shape in-aval) dtype)))

(defun %convert-emit (in-names in-avals out-name out-aval)
  "\"<out-name> = stablehlo.convert <a> : (<Tin>) -> <Tout>\"。"
  (format nil "~A = stablehlo.convert ~A : (~A) -> ~A"
          out-name (first in-names) (tensor-type-string (first in-avals)) (tensor-type-string out-aval)))

(defun %convert-to-integer (a dtype)
  "A（浮動小数点・整数・BIT の1要素）を整数 DTYPE にする。浮動小数点は
0 に向かって丸め、NaN は 0、範囲外は DTYPE の範囲の端に飽和させる（XLA の
convert と同じ。バックエンドによっては範囲外が未定義なので、範囲外の入力は
バックエンドと一致を保証しない）。整数どうしは WRAP-INTEGER で折り返す。"
  (if (floatp a)
      (let ((lo (wrap-integer (ash 1 (1- (integer-dtype-bits dtype))) dtype))
            (hi (if (eq dtype :i32) (1- (ash 1 31)) (1- (ash 1 (integer-dtype-bits dtype))))))
        (cond ((sb-ext:float-nan-p a) 0)
              ((sb-ext:float-infinity-p a) (if (plusp a) hi (if (eq dtype :i32) lo 0)))
              (t (let ((lo (if (eq dtype :i32) lo 0)))
                   (max lo (min hi (truncate a)))))))
      (wrap-integer a dtype)))

(defun %convert-element (a dtype)
  "A（入力 dtype の計算用の値）を DTYPE の計算用の値にする。:I1 への変換は
0 以外が 1（StableHLO の convert と同じ。NaN も 1）、整数への変換は
%CONVERT-TO-INTEGER、浮動小数点への変換は CL:COERCE。"
  (cond ((eq dtype :i1) (if (zerop a) 0 1))
        ((integer-dtype-p dtype) (%convert-to-integer a dtype))
        (t (coerce a (%compute-element-type dtype)))))

(defun %convert-eager (arrays in-avals &key dtype)
  "入力を計算用の値にデコードし（bf16/f16 は SINGLE-FLOAT に、f32/f64/整数/
:I1 はそのまま）、%CONVERT-ELEMENT で出力 DTYPE の計算型にしてから、出力
DTYPE の格納表現にエンコードする。

f64 → bf16/f16 は DOUBLE-FLOAT → SINGLE-FLOAT の CL:COERCE を経由するため
2回丸めになる（JAX は1回で丸める）。フェーズ1の既知の制約として許容する
（契約 §2 / 補足参照。差は高々 bf16/f16 の1ulp 程度でテストの許容誤差
（1e-2）に収まる）。f64 → f32 のオーバーフローは呼び出し側の
WITH-IEEE-ARITHMETIC が浮動小数点トラップをマスクしているので、
signal せず +/-inf になる。"
  (let ((out-aval (%convert-abstract-eval in-avals :dtype dtype)))
    (with-ieee-arithmetic
      (%elementwise-eager (lambda (a) (%convert-element a dtype))
                           arrays in-avals out-aval))))

(defprimitive convert (:dtype)
  :abstract-eval (lambda (in-avals &key dtype) (%convert-abstract-eval in-avals :dtype dtype))
  :emit (lambda (in-names in-avals out-name out-aval &key dtype)
          (declare (ignore dtype))
          (%convert-emit in-names in-avals out-name out-aval))
  :eager (lambda (arrays in-avals &key dtype) (%convert-eager arrays in-avals :dtype dtype)))
