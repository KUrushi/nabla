;;;; primitives/unary: 単項プリミティブ neg / exp / log / tanh（issue #31 p2）と
;;;; sqrt / rsqrt（issue #196）、log1p / expm1（issue #223）。
;;;;
;;;; いずれも1入力・浮動小数点専用で、出力は入力そのものの AVAL
;;;; （%UNARY-FLOAT-ABSTRACT-EVAL、src/primitives/common.lisp）。arith.lisp
;;;; と同じ理由で、DEFPRIMITIVE 本体は薄いラムダにとどめ、実際の計算は
;;;; 演算ごとの小さな named defun に分ける。

(in-package #:nabla)

(defun %neg-element (a)
  "-A（1要素）。"
  (- a))

(defun %exp-element (x)
  "EXP(X)（1要素）。オーバーフローは（呼び出し側が WITH-IEEE-ARITHMETIC で
浮動小数点トラップをマスクしている前提で）IEEE 754 どおり +無限大になる。"
  (exp x))

(defun %log-element (x)
  "LOG(X)（1要素）。CL の (log -1.0) は複素数を返してしまうため
（issue #31 p2 の pitfall）、負の値は明示的に quiet NaN にする。NaN の
チェックを先に行う（(< nan 0) はトラップをマスクしていても NIL になり、
負の値のチェックだけでは NaN を見落としうるため）。X=0 は符号によらず
負の無限大にする（IEEE 754 の規則どおり）。ZEROP で判定するのが本質的に
必要（MINUSP ではなく）: with-ieee-arithmetic の下でも SBCL の
(log 0.0d0) は real の -infinity を返すが、(log -0.0d0) は
#C(-infinity, pi) という複素数を返す（負の実軸の分岐切断のため）。
MINUSP は -0.0 を偽にするので、ZEROP のこの分岐が無いと -0.0 の入力が
(< x 0) の分岐にも入らずそのまま (log x) に落ち、複素数が漏れてしまう。"
  (let ((double-p (typep x 'double-float)))
    (cond
      ((sb-ext:float-nan-p x) x)
      ((minusp x) (%quiet-nan (if double-p 'double-float 'single-float)))
      ((zerop x) (if double-p
                     sb-ext:double-float-negative-infinity
                     sb-ext:single-float-negative-infinity))
      (t (log x)))))

(defun %tanh-element (x)
  "TANH(X)（1要素）。巨大な絶対値の X は ±1.0 になる（オーバーフローしない）。"
  (tanh x))

(defun %sqrt-element (x)
  "SQRT(X)（1要素）。CL の (sqrt -1.0) は複素数を返すため（%LOG-ELEMENT と同じ
pitfall）、負の値は明示的に quiet NaN にする。NaN と ±0 はそのまま返す
（IEEE 754 の sqrt(-0) = -0。ZEROP で先に拾わないと -0.0 が複素数側に落ちうる）。"
  (cond
    ((sb-ext:float-nan-p x) x)
    ((zerop x) x)
    ((minusp x) (%quiet-nan (if (typep x 'double-float) 'double-float 'single-float)))
    (t (sqrt x))))

(defun %rsqrt-element (x)
  "RSQRT(X) = 1/SQRT(X)（1要素）。±0 は ±無限大、負の値は NaN、+無限大は 0。
NaN のチェックを先に行う（%LOG-ELEMENT と同じ理由）。"
  (let ((double-p (typep x 'double-float)))
    (cond
      ((sb-ext:float-nan-p x) x)
      ((zerop x) (if (minusp (float-sign x))
                     (if double-p sb-ext:double-float-negative-infinity sb-ext:single-float-negative-infinity)
                     (if double-p sb-ext:double-float-positive-infinity sb-ext:single-float-positive-infinity)))
      ((minusp x) (%quiet-nan (if double-p 'double-float 'single-float)))
      (t (/ 1 (sqrt x))))))

(defun %libm-expm1 (x)
  "libm の expm1（DOUBLE-FLOAT）。SBCL は SB-KERNEL:%LOG1P を持つが %EXPM1 は持たない。"
  (sb-alien:alien-funcall (sb-alien:extern-alien "expm1" (function double-float double-float)) x))

(defun %log1p-element (x)
  "LOG(1 + X)（1要素）。小さな X でも 1 + X の丸めで桁を失わない（softmax の正規化項
log(1 + 小さな和) のため。docs/numerics.md）。double で計算して X の型に丸める。
X = -1 は負の無限大、X < -1 は NaN（libm の log1p と同じ。複素数は返さない）。"
  (if (sb-ext:float-nan-p x)
      x
      (coerce (sb-kernel:%log1p (coerce x 'double-float)) (type-of x))))

(defun %expm1-element (x)
  "EXP(X) - 1（1要素）。小さな X でも桁を失わない。double で計算して X の型に丸める
（大きな X は +無限大、負の無限大は -1）。"
  (if (sb-ext:float-nan-p x)
      x
      (coerce (%libm-expm1 (coerce x 'double-float)) (type-of x))))

(defun %neg-abstract-eval (in-avals)
  (%unary-numeric-abstract-eval :neg in-avals))

(defun %exp-abstract-eval (in-avals)
  (%unary-float-abstract-eval :exp in-avals))

(defun %log-abstract-eval (in-avals)
  (%unary-float-abstract-eval :log in-avals))

(defun %tanh-abstract-eval (in-avals)
  (%unary-float-abstract-eval :tanh in-avals))

(defun %sqrt-abstract-eval (in-avals)
  (%unary-float-abstract-eval :sqrt in-avals))

(defun %rsqrt-abstract-eval (in-avals)
  (%unary-float-abstract-eval :rsqrt in-avals))

(defun %log1p-abstract-eval (in-avals)
  (%unary-float-abstract-eval :log1p in-avals))

(defun %expm1-abstract-eval (in-avals)
  (%unary-float-abstract-eval :expm1 in-avals))

(defun %neg-emit (in-names out-name out-aval)
  (%emit-elementwise "negate" in-names out-name out-aval))

(defun %exp-emit (in-names out-name out-aval)
  (%emit-elementwise "exponential" in-names out-name out-aval))

(defun %log-emit (in-names out-name out-aval)
  (%emit-elementwise "log" in-names out-name out-aval))

(defun %tanh-emit (in-names out-name out-aval)
  (%emit-elementwise "tanh" in-names out-name out-aval))

(defun %sqrt-emit (in-names out-name out-aval)
  (%emit-elementwise "sqrt" in-names out-name out-aval))

(defun %rsqrt-emit (in-names out-name out-aval)
  (%emit-elementwise "rsqrt" in-names out-name out-aval))

(defun %unary-aux-name (tag out-name)
  "OUT-NAME（\"%7\" のような SSA 名）の数字部分を使った補助 SSA 名（\"%l1p_u_7\"）。"
  (format nil "%~A_~A" tag (subseq out-name 1)))

(defun %f64-log1p-lines (x out-name out-aval)
  "f64 の log1p を、log と四則で正確に計算する StableHLO の行（Goldberg の方法）:
u = 1 + x、u = 1 なら x、u = +inf なら u、それ以外は log(u) x / (u - 1)。u - 1 は丸め
誤差を含まないので、log(u) / (u - 1) の比が 1 + x の丸めを打ち消す。u は
optimization_barrier に通す（XLA の代数的な簡約が (x + 1) - 1 を x に畳むのを止める）。

IREE（llvm-cpu）は f64 の log_plus_one を多項式近似せず、MLIR の lowering が
log(1 + x) をそのまま出すので、小さな x の桁が消える（f32 は正確に近似される）。"
  (let ((type (tensor-type-string out-aval))
        (bool-type (tensor-type-string (make-aval (aval-shape out-aval) :i1))))
    (flet ((n (tag) (%unary-aux-name (concatenate 'string "l1p_" tag) out-name)))
      (format nil "~@{~A~^~%~}"
              (format nil "~A = stablehlo.constant dense<1.0> : ~A" (n "one") type)
              (format nil "~A = stablehlo.constant dense<0x7FF0000000000000> : ~A" (n "inf") type)
              (format nil "~A = stablehlo.add ~A, ~A : ~A" (n "sum") x (n "one") type)
              ;; barrier が無いと XLA が (x + 1) - 1 を x に畳み、素朴な log(1 + x) に戻る
              (format nil "~A = stablehlo.optimization_barrier ~A : ~A" (n "u") (n "sum") type)
              (format nil "~A = stablehlo.subtract ~A, ~A : ~A" (n "d") (n "u") (n "one") type)
              (format nil "~A = stablehlo.log ~A : ~A" (n "log") (n "u") type)
              (format nil "~A = stablehlo.multiply ~A, ~A : ~A" (n "num") (n "log") x type)
              (format nil "~A = stablehlo.divide ~A, ~A : ~A" (n "q") (n "num") (n "d") type)
              (format nil "~A = stablehlo.compare EQ, ~A, ~A : (~A, ~A) -> ~A" (n "small") (n "u") (n "one") type type bool-type)
              (format nil "~A = stablehlo.compare EQ, ~A, ~A : (~A, ~A) -> ~A" (n "big") (n "u") (n "inf") type type bool-type)
              (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" (n "r") (n "small") x (n "q") bool-type type)
              (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" out-name (n "big") (n "u") (n "r") bool-type type)))))

(defun %f64-expm1-lines (x out-name out-aval)
  "f64 の expm1 を、exp・log と四則で正確に計算する StableHLO の行（Kahan の方法）:
u = exp(x)、u = 1 なら x、u - 1 = -1 なら -1、u = +inf なら u、それ以外は (u - 1) x / log(u)。
理由は %F64-LOG1P-LINES と同じ（IREE の f64 の exponential_minus_one は exp(x) - 1 になる）。"
  (let ((type (tensor-type-string out-aval))
        (bool-type (tensor-type-string (make-aval (aval-shape out-aval) :i1))))
    (flet ((n (tag) (%unary-aux-name (concatenate 'string "em1_" tag) out-name)))
      (format nil "~@{~A~^~%~}"
              (format nil "~A = stablehlo.constant dense<1.0> : ~A" (n "one") type)
              (format nil "~A = stablehlo.constant dense<-1.0> : ~A" (n "mone") type)
              (format nil "~A = stablehlo.constant dense<0x7FF0000000000000> : ~A" (n "inf") type)
              (format nil "~A = stablehlo.exponential ~A : ~A" (n "e") x type)
              (format nil "~A = stablehlo.optimization_barrier ~A : ~A" (n "u") (n "e") type)
              (format nil "~A = stablehlo.subtract ~A, ~A : ~A" (n "d") (n "u") (n "one") type)
              (format nil "~A = stablehlo.log ~A : ~A" (n "log") (n "u") type)
              (format nil "~A = stablehlo.multiply ~A, ~A : ~A" (n "num") (n "d") x type)
              (format nil "~A = stablehlo.divide ~A, ~A : ~A" (n "q") (n "num") (n "log") type)
              (format nil "~A = stablehlo.compare EQ, ~A, ~A : (~A, ~A) -> ~A" (n "small") (n "u") (n "one") type type bool-type)
              (format nil "~A = stablehlo.compare EQ, ~A, ~A : (~A, ~A) -> ~A" (n "low") (n "d") (n "mone") type type bool-type)
              (format nil "~A = stablehlo.compare EQ, ~A, ~A : (~A, ~A) -> ~A" (n "big") (n "u") (n "inf") type type bool-type)
              (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" (n "r1") (n "small") x (n "q") bool-type type)
              (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" (n "r2") (n "low") (n "mone") (n "r1") bool-type type)
              (format nil "~A = stablehlo.select ~A, ~A, ~A : ~A, ~A" out-name (n "big") (n "u") (n "r2") bool-type type)))))

(defun %log1p-emit (in-names out-name out-aval)
  (if (eq (aval-dtype out-aval) :f64)
      (%f64-log1p-lines (first in-names) out-name out-aval)
      (%emit-elementwise "log_plus_one" in-names out-name out-aval)))

(defun %expm1-emit (in-names out-name out-aval)
  (if (eq (aval-dtype out-aval) :f64)
      (%f64-expm1-lines (first in-names) out-name out-aval)
      (%emit-elementwise "exponential_minus_one" in-names out-name out-aval)))

(defun %neg-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%neg-element arrays in-avals (%neg-abstract-eval in-avals))))

(defun %exp-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%exp-element arrays in-avals (%exp-abstract-eval in-avals))))

(defun %log-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%log-element arrays in-avals (%log-abstract-eval in-avals))))

(defun %tanh-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%tanh-element arrays in-avals (%tanh-abstract-eval in-avals))))

(defun %sqrt-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%sqrt-element arrays in-avals (%sqrt-abstract-eval in-avals))))

(defun %rsqrt-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%rsqrt-element arrays in-avals (%rsqrt-abstract-eval in-avals))))

(defun %log1p-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%log1p-element arrays in-avals (%log1p-abstract-eval in-avals))))

(defun %expm1-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%expm1-element arrays in-avals (%expm1-abstract-eval in-avals))))

(defprimitive neg ()
  :abstract-eval (lambda (in-avals) (%neg-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%neg-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%neg-eager arrays in-avals)))

(defprimitive exp ()
  :abstract-eval (lambda (in-avals) (%exp-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%exp-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%exp-eager arrays in-avals)))

(defprimitive log ()
  :abstract-eval (lambda (in-avals) (%log-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%log-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%log-eager arrays in-avals)))

(defprimitive tanh ()
  :abstract-eval (lambda (in-avals) (%tanh-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%tanh-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%tanh-eager arrays in-avals)))

(defprimitive sqrt ()
  :abstract-eval (lambda (in-avals) (%sqrt-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%sqrt-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%sqrt-eager arrays in-avals)))

(defprimitive rsqrt ()
  :abstract-eval (lambda (in-avals) (%rsqrt-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%rsqrt-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%rsqrt-eager arrays in-avals)))

(defprimitive log1p ()
  :abstract-eval (lambda (in-avals) (%log1p-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%log1p-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%log1p-eager arrays in-avals)))

(defprimitive expm1 ()
  :abstract-eval (lambda (in-avals) (%expm1-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%expm1-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%expm1-eager arrays in-avals)))
