;;;; primitives/unary: 単項プリミティブ neg / exp / log / tanh（issue #31 p2）。
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

(defun %neg-abstract-eval (in-avals)
  (%unary-float-abstract-eval :neg in-avals))

(defun %exp-abstract-eval (in-avals)
  (%unary-float-abstract-eval :exp in-avals))

(defun %log-abstract-eval (in-avals)
  (%unary-float-abstract-eval :log in-avals))

(defun %tanh-abstract-eval (in-avals)
  (%unary-float-abstract-eval :tanh in-avals))

(defun %neg-emit (in-names out-name out-aval)
  (%emit-elementwise "negate" in-names out-name out-aval))

(defun %exp-emit (in-names out-name out-aval)
  (%emit-elementwise "exponential" in-names out-name out-aval))

(defun %log-emit (in-names out-name out-aval)
  (%emit-elementwise "log" in-names out-name out-aval))

(defun %tanh-emit (in-names out-name out-aval)
  (%emit-elementwise "tanh" in-names out-name out-aval))

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
