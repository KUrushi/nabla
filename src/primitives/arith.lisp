;;;; primitives/arith: 二項算術プリミティブ add / sub / mul / div
;;;; （issue #31 p1）。
;;;;
;;;; いずれも2入力・浮動小数点専用・shape と dtype が一致していること・
;;;; 出力は最初の入力の AVAL、という共通の形（%BINARY-FLOAT-ABSTRACT-EVAL、
;;;; src/primitives/common.lisp）を持つ。DEFPRIMITIVE 本体は薄いラムダに
;;;; とどめ、実際の計算は演算ごとの小さな named defun に分ける
;;;; （mutation testing の runner はトップレベル定義1つにつき1変異体しか
;;;; 作らないため、大きな defprimitive フォーム1つにまとめると変異体の
;;;; 数が減り、生き残りやすくなる）。

(in-package #:nabla)

(defun %add-element (a b)
  "A + B（1要素）。"
  (+ a b))

(defun %sub-element (a b)
  "A - B（1要素）。"
  (- a b))

(defun %mul-element (a b)
  "A * B（1要素）。"
  (* a b))

(defun %div-element (a b)
  "A / B（1要素）。B が 0 のとき（呼び出し側が WITH-IEEE-ARITHMETIC で
浮動小数点トラップをマスクしている前提で）IEEE 754 どおり ±無限大か NaN
になる（0/0 は NaN、非0/0 は符号付き無限大）。SBCL の既定のトラップの
もとでは DIVISION-BY-ZERO を signal してしまうため、この関数を裸で
呼び出してはならない。"
  (/ a b))

(defun %add-abstract-eval (in-avals)
  (%binary-float-abstract-eval :add in-avals))

(defun %sub-abstract-eval (in-avals)
  (%binary-float-abstract-eval :sub in-avals))

(defun %mul-abstract-eval (in-avals)
  (%binary-float-abstract-eval :mul in-avals))

(defun %div-abstract-eval (in-avals)
  (%binary-float-abstract-eval :div in-avals))

(defun %add-emit (in-names out-name out-aval)
  (%emit-elementwise "add" in-names out-name out-aval))

(defun %sub-emit (in-names out-name out-aval)
  (%emit-elementwise "subtract" in-names out-name out-aval))

(defun %mul-emit (in-names out-name out-aval)
  (%emit-elementwise "multiply" in-names out-name out-aval))

(defun %div-emit (in-names out-name out-aval)
  (%emit-elementwise "divide" in-names out-name out-aval))

(defun %add-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%add-element arrays in-avals (%add-abstract-eval in-avals))))

(defun %sub-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%sub-element arrays in-avals (%sub-abstract-eval in-avals))))

(defun %mul-eager (arrays in-avals)
  (with-ieee-arithmetic
    (%elementwise-eager #'%mul-element arrays in-avals (%mul-abstract-eval in-avals))))

(defun %div-eager (arrays in-avals)
  "A / B を要素ごとに計算する。0除算は SBCL の浮動小数点トラップを
WITH-IEEE-ARITHMETIC でマスクしているので signal せず、IEEE 754 の規則
どおり無限大・NaN を出力に含む配列を返す（1/0 → +inf、-1/0 → -inf、
0/0 → NaN）。"
  (with-ieee-arithmetic
    (%elementwise-eager #'%div-element arrays in-avals (%div-abstract-eval in-avals))))

(defprimitive add ()
  :abstract-eval (lambda (in-avals) (%add-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%add-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%add-eager arrays in-avals)))

(defprimitive sub ()
  :abstract-eval (lambda (in-avals) (%sub-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%sub-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%sub-eager arrays in-avals)))

(defprimitive mul ()
  :abstract-eval (lambda (in-avals) (%mul-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%mul-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%mul-eager arrays in-avals)))

(defprimitive div ()
  :abstract-eval (lambda (in-avals) (%div-abstract-eval in-avals))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (%div-emit in-names out-name out-aval))
  :eager (lambda (arrays in-avals) (%div-eager arrays in-avals)))
