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

(defun %convert-aux-name (tag out-name)
  "OUT-NAME（\"%7\" のような SSA 名）の数字部分を使った補助 SSA 名（\"%tag_7\"）。"
  (format nil "%~A_~A" tag (subseq out-name 1)))

(defun %convert-line (out-name in-name in-aval out-aval)
  "stablehlo.convert 1行。"
  (format nil "~A = stablehlo.convert ~A : (~A) -> ~A"
          out-name in-name (tensor-type-string in-aval) (tensor-type-string out-aval)))

(defun %integer-range (dtype)
  "整数 DTYPE の (values 最小値 最大値)。"
  (if (eq dtype :i32)
      (values (- (ash 1 31)) (1- (ash 1 31)))
      (values 0 (1- (ash 1 (integer-dtype-bits dtype))))))

(defun %float-clamp-upper-bound (max-int float-type)
  "MAX-INT 以下で最大の、FLOAT-TYPE（SINGLE-FLOAT / DOUBLE-FLOAT）で正確に
表せる整数値の浮動小数点数を返す。MAX-INT がちょうど表せるならその値、
丸めで 2^n に繰り上がってしまう（f32 の 2^31 - 1 など）ならその1つ下の値。
範囲に入れてから整数に変換しても飽和しない（未定義にならない）上限にする。"
  (let ((candidate (coerce max-int float-type)))
    (if (<= (rational candidate) max-int)
        candidate
        (let ((digits (float-digits candidate)))
          ;; 2^k の1つ下 = (2^digits - 1) * 2^(k - digits)
          (scale-float (coerce (1- (ash 1 digits)) float-type)
                       (- (integer-length max-int) digits))))))

(defun %convert-emit-saturating (in-names in-avals out-name out-aval)
  "浮動小数点 → 整数の convert を、StableHLO の上でも eager（%CONVERT-TO-INTEGER）と
同じく 0 方向への丸め・NaN は 0・範囲外は飽和にする行を返す（fptosi は
範囲外・NaN が未定義で、バックエンドごとに値が違うため。issue #126）:

  nan = x != x
  r   = convert(clamp(lo, x, hi))        ; hi は整数の最大値以下で表せる最大の浮動小数点数
  r   = select(x >= 2^n, 整数の最大値, r) ; hi が最大値に届かない（f32 の i32 など）分
  out = select(nan, 0, r)

f16 / bf16 は f32 にしてから同じ手順（f32 は f16 / bf16 を正確に含む）。"
  (let* ((in-aval (first in-avals))
         (shape (aval-shape in-aval))
         (out-dtype (aval-dtype out-aval))
         (source-dtype (if (member (aval-dtype in-aval) '(:f16 :bf16)) :f32 (aval-dtype in-aval)))
         (float-type (if (eq source-dtype :f64) 'double-float 'single-float))
         (source-aval (make-aval shape source-dtype))
         (pred-aval (make-aval shape :i1))
         (ft (tensor-type-string source-aval))
         (it (tensor-type-string out-aval))
         (pt (tensor-type-string pred-aval))
         (x (if (eq source-dtype (aval-dtype in-aval)) (first in-names) (%convert-aux-name "src" out-name)))
         (nan (%convert-aux-name "nan" out-name))
         (lo (%convert-aux-name "lo" out-name))
         (hi (%convert-aux-name "hi" out-name))
         (clamped (%convert-aux-name "clamped" out-name))
         (converted (%convert-aux-name "converted" out-name))
         (threshold (%convert-aux-name "threshold" out-name))
         (big (%convert-aux-name "big" out-name))
         (max-int (%convert-aux-name "maxint" out-name))
         (saturated (%convert-aux-name "saturated" out-name))
         (zero (%convert-aux-name "zero" out-name)))
    (multiple-value-bind (min-value max-value) (%integer-range out-dtype)
      (flet ((float-constant (name value)
               (format nil "~A = stablehlo.constant dense<~A> : ~A" name
                       (%stablehlo-float-literal value source-dtype) ft))
             (int-constant (name value)
               (format nil "~A = stablehlo.constant dense<~D> : ~A" name value it)))
        (format nil "~{~A~^~%~}"
                (append
                 (unless (eq x (first in-names))
                   (list (%convert-line x (first in-names) in-aval source-aval)))
                 (list
                  (format nil "~A = stablehlo.compare NE, ~A, ~A : (~A, ~A) -> ~A" nan x x ft ft pt)
                  (float-constant lo (coerce min-value float-type))
                  (float-constant hi (%float-clamp-upper-bound max-value float-type))
                  (format nil "~A = stablehlo.clamp ~A, ~A, ~A : ~A" clamped lo x hi ft)
                  (%convert-line converted clamped source-aval out-aval)
                  (float-constant threshold (coerce (1+ max-value) float-type))
                  (format nil "~A = stablehlo.compare GE, ~A, ~A : (~A, ~A) -> ~A" big x threshold ft ft pt)
                  (int-constant max-int max-value)
                  (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" saturated big max-int converted pt it)
                  (int-constant zero 0)
                  (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" out-name nan zero saturated pt it))))))))

(defun %convert-emit (in-names in-avals out-name out-aval)
  "convert の StableHLO。通常は \"<out-name> = stablehlo.convert <a> : (<Tin>) -> <Tout>\"
の1行。次の2つだけ複数行になる（issue #126）:
- 整数 → :bf16: 整数 → f32 → bf16 の2段（間に optimization_barrier を置く。一部のバックエンド（CPU コード生成）は整数から bf16 への
  直接の変換を __truncsfbf2 に落とし、リンクに失敗する）。
- 浮動小数点 → 整数: 飽和と NaN → 0 つき（%CONVERT-EMIT-SATURATING）。"
  (let* ((in-aval (first in-avals))
         (in-dtype (aval-dtype in-aval))
         (out-dtype (aval-dtype out-aval)))
    (cond
      ((and (integer-dtype-p in-dtype) (eq out-dtype :bf16))
       (let* ((mid (%convert-aux-name "f32" out-name))
              (mid-aval (make-aval (aval-shape in-aval) :f32)))
         ;; optimization_barrier が無いと、2つの convert がバックエンドの最適化で
         ;; 1つの「整数 → bf16」に畳まれ、元の問題に戻ってしまう
         (let ((barrier (%convert-aux-name "barrier" out-name)))
           (format nil "~A~%~A = stablehlo.optimization_barrier ~A : ~A~%~A"
                   (%convert-line mid (first in-names) in-aval mid-aval)
                   barrier mid (tensor-type-string mid-aval)
                   (%convert-line out-name barrier mid-aval out-aval)))))
      ((and (%float-dtype-p in-dtype) (integer-dtype-p out-dtype))
       (%convert-emit-saturating in-names in-avals out-name out-aval))
      (t (%convert-line out-name (first in-names) in-aval out-aval)))))

(defun %convert-to-integer (a dtype)
  "A（浮動小数点・整数・BIT の1要素）を整数 DTYPE にする。浮動小数点は
0 に向かって丸め、NaN は 0、範囲外（±無限大を含む）は DTYPE の範囲の端に
飽和させる。StableHLO の出力（%CONVERT-EMIT-SATURATING）も同じ値になるように
してあるので、どのバックエンドでも一致する。整数どうしは WRAP-INTEGER で
折り返す。"
  (if (floatp a)
      (multiple-value-bind (lo hi) (%integer-range dtype)
        (cond ((sb-ext:float-nan-p a) 0)
              ((sb-ext:float-infinity-p a) (if (plusp a) hi lo))
              (t (max lo (min hi (truncate a))))))
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
