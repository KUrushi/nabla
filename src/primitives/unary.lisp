;;;; primitives/unary: 単項プリミティブ neg / exp / log / tanh（issue #31 p2）と
;;;; sqrt / rsqrt（issue #196）、expm1 / log1p / sin / cos（issue #223）。
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

;;; expm1 / log1p / sin / cos は、全 dtype とも「入力を double に直して計算し、入力の
;;; 型（single / double）に丸める」。f32 / bf16 / f16 の入力は single で渡ってくるが、
;;; double の結果を single に丸める（RNE）ので、単精度の式を使うより誤差が小さい。
;;; CL に expm1 / log1p は無いので Kahan の形で書く。

(defun %round-like (value x)
  "VALUE（double）を X と同じ型（single-float または double-float）に丸める。
VALUE が無限大・NaN のときはそのまま同じ型の無限大・NaN になる。"
  (if (typep x 'double-float) value (coerce value 'single-float)))

(defun %expm1-double (x)
  "EXPM1(X) = exp(X) - 1（X は double）。Kahan の形: u = exp(x) として、u = 1 なら x
（x が極小でも ±0 の符号を保って x を返す）、u - 1 = -1 なら -1、それ以外は (u - 1) x / log(u)。
特殊値は明示的に分岐する: NaN はそのまま、+無限大は +無限大、-無限大は -1、u が +無限大
（exp がオーバーフロー）なら +無限大（(u - 1) x / log(u) は inf / inf で NaN になる）。"
  (cond
    ((sb-ext:float-nan-p x) x)
    ((sb-ext:float-infinity-p x) (if (plusp x) x -1d0))
    (t (let ((u (exp x)))
         (cond
           ((= u 1d0) x)
           ((= (- u 1d0) -1d0) -1d0)
           ((sb-ext:float-infinity-p u) u)
           (t (/ (* (- u 1d0) x) (log u))))))))

(defun %log1p-double (x)
  "LOG1P(X) = log(1 + X)（X は double）。Kahan の形: u = 1 + x として、u = 1 なら x
（x が極小でも ±0 の符号を保つ）、それ以外は log(u) x / (u - 1)。特殊値は明示的に分岐する
（%LOG-ELEMENT と同じく CL の log は負で複素数を返す）: NaN はそのまま、X < -1 は quiet NaN、
X = -1 は -無限大、+無限大は +無限大。"
  (cond
    ((sb-ext:float-nan-p x) x)
    ((< x -1d0) (%quiet-nan 'double-float))
    ((= x -1d0) sb-ext:double-float-negative-infinity)
    ((sb-ext:float-infinity-p x) x)
    (t (let ((u (+ 1d0 x)))
         (if (= u 1d0)
             x
             (/ (* (log u) x) (- u 1d0)))))))

(defun %expm1-element (x)
  "EXPM1(X)（1要素。X は single-float または double-float）。"
  (%round-like (%expm1-double (coerce x 'double-float)) x))

(defun %log1p-element (x)
  "LOG1P(X)（1要素。X は single-float または double-float）。"
  (%round-like (%log1p-double (coerce x 'double-float)) x))

(defun %sin-element (x)
  "SIN(X)（1要素）。double で計算して丸める。±無限大は NaN（CL の (sin inf) の挙動に
頼らず明示する）、NaN はそのまま。sin(-0) = -0。"
  (let ((d (coerce x 'double-float)))
    (cond
      ((sb-ext:float-nan-p d) x)
      ((sb-ext:float-infinity-p d) (%quiet-nan (if (typep x 'double-float) 'double-float 'single-float)))
      (t (%round-like (sin d) x)))))

(defun %cos-element (x)
  "COS(X)（1要素）。double で計算して丸める。±無限大は NaN、NaN はそのまま。"
  (let ((d (coerce x 'double-float)))
    (cond
      ((sb-ext:float-nan-p d) x)
      ((sb-ext:float-infinity-p d) (%quiet-nan (if (typep x 'double-float) 'double-float 'single-float)))
      (t (%round-like (cos d) x)))))

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

(defun %expm1-abstract-eval (in-avals)
  (%unary-float-abstract-eval :expm1 in-avals))

(defun %log1p-abstract-eval (in-avals)
  (%unary-float-abstract-eval :log1p in-avals))

(defun %sin-abstract-eval (in-avals)
  (%unary-float-abstract-eval :sin in-avals))

(defun %cos-abstract-eval (in-avals)
  (%unary-float-abstract-eval :cos in-avals))

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

(defun %expm1-emit (in-names out-name out-aval)
  (%emit-elementwise "exponential_minus_one" in-names out-name out-aval))

(defun %log1p-emit (in-names out-name out-aval)
  (%emit-elementwise "log_plus_one" in-names out-name out-aval))

(defun %sin-emit (in-names out-name out-aval)
  (%emit-elementwise "sine" in-names out-name out-aval))

(defun %cos-emit (in-names out-name out-aval)
  (%emit-elementwise "cosine" in-names out-name out-aval))

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

(defun %expm1-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%expm1-element arrays in-avals (%expm1-abstract-eval in-avals))))

(defun %log1p-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%log1p-element arrays in-avals (%log1p-abstract-eval in-avals))))

(defun %sin-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%sin-element arrays in-avals (%sin-abstract-eval in-avals))))

(defun %cos-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%cos-element arrays in-avals (%cos-abstract-eval in-avals))))

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

(defprimitive expm1 ()
  :abstract-eval (lambda (in-avals) (%expm1-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%expm1-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%expm1-eager arrays in-avals)))

(defprimitive log1p ()
  :abstract-eval (lambda (in-avals) (%log1p-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%log1p-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%log1p-eager arrays in-avals)))

(defprimitive sin ()
  :abstract-eval (lambda (in-avals) (%sin-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%sin-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%sin-eager arrays in-avals)))

(defprimitive cos ()
  :abstract-eval (lambda (in-avals) (%cos-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%cos-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%cos-eager arrays in-avals)))
