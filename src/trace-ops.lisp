;;;; trace-ops: with-tracing がターゲットにする内部ジェネリック
;;;; %T-ADD 等（issue #32、t1）。
;;;;
;;;; 各ジェネリックは REAL / ARRAY / TRACER の3つの型を任意に組み合わせた
;;;; 9通りの2引数メソッド（単項は3通り）を持つ。メソッド本体はすべて
;;;; 一行で、実際の分岐は %TRACE-OP-* という名前付き DEFUN に委ねる
;;;; （mutation testing の runner は DEFMETHOD／DEFUN 単位で変異するため、
;;;; 判断ロジックを小さい DEFUN に分けておくと変異体を検出しやすい
;;;; テストを書きやすい）。
;;;;
;;;; dtype ごとの分岐・NaN 処理などは各プリミティブ（src/primitives/*.lisp）
;;;; の :EAGER にすでにある。ここでは「REAL / ARRAY / TRACER のどの組み合わせ
;;;; かで、CL の関数を直接呼ぶか・:EAGER を呼ぶか・EQN を足すか」だけを
;;;; 振り分ける。

(in-package #:nabla)

;;; --- 2引数の演算が共有する振り分けヘルパー ---

(defun %trace-op-real-real (fn a b)
  "A・B が両方 REAL のときの共通実装: FN（対応する CL の関数）をそのまま
呼ぶ。"
  (funcall fn a b))

(defun %trace-op-array-array (prim-name a b)
  "A・B が両方配列のときの共通実装: PRIM-NAME の :EAGER を直接呼ぶ
（shape/dtype の不一致は abstract-eval が PRIMITIVE-ERROR として報告する）。"
  (apply (primitive-eager (find-primitive prim-name)) (list a b) (list (array-aval a) (array-aval b)) nil))

(defun %trace-op-real-array (prim-name a b)
  "A が REAL、B が配列のときの共通実装: A を B と同じ shape/dtype に埋めて
（%FILL-ARRAY）から、配列どうしの実装（%TRACE-OP-ARRAY-ARRAY）に委ねる。"
  (%trace-op-array-array prim-name (%fill-array a b) b))

(defun %trace-op-array-real (prim-name a b)
  "A が配列、B が REAL のときの共通実装（%TRACE-OP-REAL-ARRAY と対称）。"
  (%trace-op-array-array prim-name a (%fill-array b a)))

(defun %trace-op-tracer-tracer (prim-name a b)
  "A・B が両方トレーサのときの共通実装: PRIM-NAME の EQN を1つ足す。"
  (%trace-eqn prim-name (list a b)))

(defun %trace-op-real-tracer (prim-name a b)
  "A が REAL、B がトレーサのときの共通実装: A を B と同じ dtype/shape に
リフト（%LIFT-NUMBER）してから、トレーサどうしの実装に委ねる。"
  (%trace-op-tracer-tracer prim-name (%lift-number a b) b))

(defun %trace-op-tracer-real (prim-name a b)
  "A がトレーサ、B が REAL のときの共通実装（%TRACE-OP-REAL-TRACER と対称）。"
  (%trace-op-tracer-tracer prim-name a (%lift-number b a)))

(defun %trace-op-array-tracer (prim-name a b)
  "A が配列、B がトレーサのときの共通実装: A を B と同じ dtype でリフト
（%LIFT-ARRAY）してから、トレーサどうしの実装に委ねる。"
  (%trace-op-tracer-tracer prim-name (%lift-array a b) b))

(defun %trace-op-tracer-array (prim-name a b)
  "A がトレーサ、B が配列のときの共通実装（%TRACE-OP-ARRAY-TRACER と対称）。"
  (%trace-op-tracer-tracer prim-name a (%lift-array b a)))

;;; --- 1引数の演算が共有する振り分けヘルパー ---

(defun %trace-op-real (fn a)
  "A が REAL のときの共通実装: FN をそのまま呼ぶ。"
  (funcall fn a))

(defun %trace-op-array (prim-name a)
  "A が配列のときの共通実装: PRIM-NAME の :EAGER を直接呼ぶ。"
  (apply (primitive-eager (find-primitive prim-name)) (list a) (list (array-aval a)) nil))

(defun %trace-op-tracer (prim-name a)
  "A がトレーサのときの共通実装: PRIM-NAME の EQN を1つ足す。"
  (%trace-eqn prim-name (list a)))

;;; --- %t-add: A + B（トレース対象の (+ a b)）。 ---

(defgeneric %t-add (a b)
  (:documentation "A + B（トレース対象の (+ a b)）。"))

(defmethod %t-add ((a real) (b real))
  (%trace-op-real-real #'+ a b))

(defmethod %t-add ((a array) (b array))
  (%trace-op-array-array :add a b))

(defmethod %t-add ((a real) (b array))
  (%trace-op-real-array :add a b))

(defmethod %t-add ((a array) (b real))
  (%trace-op-array-real :add a b))

(defmethod %t-add ((a tracer) (b tracer))
  (%trace-op-tracer-tracer :add a b))

(defmethod %t-add ((a real) (b tracer))
  (%trace-op-real-tracer :add a b))

(defmethod %t-add ((a tracer) (b real))
  (%trace-op-tracer-real :add a b))

(defmethod %t-add ((a array) (b tracer))
  (%trace-op-array-tracer :add a b))

(defmethod %t-add ((a tracer) (b array))
  (%trace-op-tracer-array :add a b))

;;; --- %t-sub: A - B（トレース対象の (- a b)）。 ---

(defgeneric %t-sub (a b)
  (:documentation "A - B（トレース対象の (- a b)）。"))

(defmethod %t-sub ((a real) (b real))
  (%trace-op-real-real #'- a b))

(defmethod %t-sub ((a array) (b array))
  (%trace-op-array-array :sub a b))

(defmethod %t-sub ((a real) (b array))
  (%trace-op-real-array :sub a b))

(defmethod %t-sub ((a array) (b real))
  (%trace-op-array-real :sub a b))

(defmethod %t-sub ((a tracer) (b tracer))
  (%trace-op-tracer-tracer :sub a b))

(defmethod %t-sub ((a real) (b tracer))
  (%trace-op-real-tracer :sub a b))

(defmethod %t-sub ((a tracer) (b real))
  (%trace-op-tracer-real :sub a b))

(defmethod %t-sub ((a array) (b tracer))
  (%trace-op-array-tracer :sub a b))

(defmethod %t-sub ((a tracer) (b array))
  (%trace-op-tracer-array :sub a b))

;;; --- %t-mul: A * B（トレース対象の (* a b)）。 ---

(defgeneric %t-mul (a b)
  (:documentation "A * B（トレース対象の (* a b)）。"))

(defmethod %t-mul ((a real) (b real))
  (%trace-op-real-real #'* a b))

(defmethod %t-mul ((a array) (b array))
  (%trace-op-array-array :mul a b))

(defmethod %t-mul ((a real) (b array))
  (%trace-op-real-array :mul a b))

(defmethod %t-mul ((a array) (b real))
  (%trace-op-array-real :mul a b))

(defmethod %t-mul ((a tracer) (b tracer))
  (%trace-op-tracer-tracer :mul a b))

(defmethod %t-mul ((a real) (b tracer))
  (%trace-op-real-tracer :mul a b))

(defmethod %t-mul ((a tracer) (b real))
  (%trace-op-tracer-real :mul a b))

(defmethod %t-mul ((a array) (b tracer))
  (%trace-op-array-tracer :mul a b))

(defmethod %t-mul ((a tracer) (b array))
  (%trace-op-tracer-array :mul a b))

;;; --- %t-div: A / B（トレース対象の (/ a b)）。 ---

(defgeneric %t-div (a b)
  (:documentation "A / B（トレース対象の (/ a b)）。"))

(defmethod %t-div ((a real) (b real))
  (%trace-op-real-real #'/ a b))

(defmethod %t-div ((a array) (b array))
  (%trace-op-array-array :div a b))

(defmethod %t-div ((a real) (b array))
  (%trace-op-real-array :div a b))

(defmethod %t-div ((a array) (b real))
  (%trace-op-array-real :div a b))

(defmethod %t-div ((a tracer) (b tracer))
  (%trace-op-tracer-tracer :div a b))

(defmethod %t-div ((a real) (b tracer))
  (%trace-op-real-tracer :div a b))

(defmethod %t-div ((a tracer) (b real))
  (%trace-op-tracer-real :div a b))

(defmethod %t-div ((a array) (b tracer))
  (%trace-op-array-tracer :div a b))

(defmethod %t-div ((a tracer) (b array))
  (%trace-op-tracer-array :div a b))

;;; --- %t-max: A と B の大きい方（トレース対象の (max a b)）。 ---

(defgeneric %t-max (a b)
  (:documentation "A と B の大きい方（トレース対象の (max a b)）。"))

(defmethod %t-max ((a real) (b real))
  (%trace-op-real-real #'max a b))

(defmethod %t-max ((a array) (b array))
  (%trace-op-array-array :max a b))

(defmethod %t-max ((a real) (b array))
  (%trace-op-real-array :max a b))

(defmethod %t-max ((a array) (b real))
  (%trace-op-array-real :max a b))

(defmethod %t-max ((a tracer) (b tracer))
  (%trace-op-tracer-tracer :max a b))

(defmethod %t-max ((a real) (b tracer))
  (%trace-op-real-tracer :max a b))

(defmethod %t-max ((a tracer) (b real))
  (%trace-op-tracer-real :max a b))

(defmethod %t-max ((a array) (b tracer))
  (%trace-op-array-tracer :max a b))

(defmethod %t-max ((a tracer) (b array))
  (%trace-op-tracer-array :max a b))

;;; --- %t-min: A と B の小さい方（トレース対象の (min a b)）。 ---

(defgeneric %t-min (a b)
  (:documentation "A と B の小さい方（トレース対象の (min a b)）。"))

(defmethod %t-min ((a real) (b real))
  (%trace-op-real-real #'min a b))

(defmethod %t-min ((a array) (b array))
  (%trace-op-array-array :min a b))

(defmethod %t-min ((a real) (b array))
  (%trace-op-real-array :min a b))

(defmethod %t-min ((a array) (b real))
  (%trace-op-array-real :min a b))

(defmethod %t-min ((a tracer) (b tracer))
  (%trace-op-tracer-tracer :min a b))

(defmethod %t-min ((a real) (b tracer))
  (%trace-op-real-tracer :min a b))

(defmethod %t-min ((a tracer) (b real))
  (%trace-op-tracer-real :min a b))

(defmethod %t-min ((a array) (b tracer))
  (%trace-op-array-tracer :min a b))

(defmethod %t-min ((a tracer) (b array))
  (%trace-op-tracer-array :min a b))

;;; --- %t-neg: -A（トレース対象の単項 (- a)）。 ---

(defgeneric %t-neg (a)
  (:documentation "-A（トレース対象の単項 (- a)）。"))

(defmethod %t-neg ((a real))
  (%trace-op-real #'- a))

(defmethod %t-neg ((a array))
  (%trace-op-array :neg a))

(defmethod %t-neg ((a tracer))
  (%trace-op-tracer :neg a))

;;; --- %t-exp: EXP(A)（トレース対象の (exp a)）。 ---

(defgeneric %t-exp (a)
  (:documentation "EXP(A)（トレース対象の (exp a)）。"))

(defmethod %t-exp ((a real))
  (%trace-op-real #'exp a))

(defmethod %t-exp ((a array))
  (%trace-op-array :exp a))

(defmethod %t-exp ((a tracer))
  (%trace-op-tracer :exp a))

;;; --- %t-log: LOG(A)（トレース対象の (log a)）。CL の (log -1.0) が複素数を返す性質を そのまま受け継ぐ（負の入力の扱いは呼び出し側／:EAGER の責任。ドキュメントの 既知の制約）。 ---

(defgeneric %t-log (a)
  (:documentation "LOG(A)（トレース対象の (log a)）。CL の (log -1.0) が複素数を返す性質を
そのまま受け継ぐ（負の入力の扱いは呼び出し側／:EAGER の責任。ドキュメントの
既知の制約）。"))

(defmethod %t-log ((a real))
  (%trace-op-real #'log a))

(defmethod %t-log ((a array))
  (%trace-op-array :log a))

(defmethod %t-log ((a tracer))
  (%trace-op-tracer :log a))

;;; --- %t-tanh: TANH(A)（トレース対象の (tanh a)）。 ---

(defgeneric %t-tanh (a)
  (:documentation "TANH(A)（トレース対象の (tanh a)）。"))

(defmethod %t-tanh ((a real))
  (%trace-op-real #'tanh a))

(defmethod %t-tanh ((a array))
  (%trace-op-array :tanh a))

(defmethod %t-tanh ((a tracer))
  (%trace-op-tracer :tanh a))

;;; --- %t-compare: A と B を DIRECTION（:LT :LE :GT :GE :EQ :NE）で比較する
;;; （トレース対象の (< a b) 等）。DIRECTION は A・B の型によらず常に
;;; キーワードで渡され、ディスパッチには関与しない（総称関数の第3引数）。

(defun %compare-direction-function (direction)
  "DIRECTION（:LT :LE :GT :GE :EQ :NE のいずれか）に対応する、2引数で
CL の (真/偽) を返す比較関数を返す。"
  (ecase direction
    (:lt #'<) (:le #'<=) (:gt #'>) (:ge #'>=) (:eq #'=) (:ne #'/=)))

(defun %trace-op-compare-real-real (a b direction)
  "A・B が両方 REAL のときの共通実装: DIRECTION に対応する CL の比較関数を
呼び、CL の（真/偽）を返す。"
  (funcall (%compare-direction-function direction) a b))

(defun %trace-op-compare-array-array (a b direction)
  "A・B が両方配列のときの共通実装: :COMPARE プリミティブの :EAGER を直接
呼び、:I1（BIT）の配列を返す。"
  (apply (primitive-eager (find-primitive :compare)) (list a b)
         (list (array-aval a) (array-aval b)) (list :direction direction)))

(defun %trace-op-compare-real-array (a b direction)
  "A が REAL、B が配列のときの共通実装。"
  (%trace-op-compare-array-array (%fill-array a b) b direction))

(defun %trace-op-compare-array-real (a b direction)
  "A が配列、B が REAL のときの共通実装。"
  (%trace-op-compare-array-array a (%fill-array b a) direction))

(defun %trace-op-compare-tracer-tracer (a b direction)
  "A・B が両方トレーサのときの共通実装: :COMPARE の EQN を1つ足す
（:I1 のトレーサを返す）。"
  (%trace-eqn :compare (list a b) :direction direction))

(defun %trace-op-compare-real-tracer (a b direction)
  "A が REAL、B がトレーサのときの共通実装。"
  (%trace-op-compare-tracer-tracer (%lift-number a b) b direction))

(defun %trace-op-compare-tracer-real (a b direction)
  "A がトレーサ、B が REAL のときの共通実装。"
  (%trace-op-compare-tracer-tracer a (%lift-number b a) direction))

(defun %trace-op-compare-array-tracer (a b direction)
  "A が配列、B がトレーサのときの共通実装。"
  (%trace-op-compare-tracer-tracer (%lift-array a b) b direction))

(defun %trace-op-compare-tracer-array (a b direction)
  "A がトレーサ、B が配列のときの共通実装。"
  (%trace-op-compare-tracer-tracer a (%lift-array b a) direction))

(defgeneric %t-compare (a b direction)
  (:documentation
   "A と B を DIRECTION（:LT :LE :GT :GE :EQ :NE のいずれか）で比較する。
A・B が両方トレーサなら :I1 のトレーサ、両方配列なら BIT の配列を返す。
片方が REAL／片方が配列またはトレーサのときは、REAL 側を相手に合わせて
リフト／埋めてから、揃った型どうしの実装に委ねる。"))

(defmethod %t-compare ((a real) (b real) direction)
  (%trace-op-compare-real-real a b direction))

(defmethod %t-compare ((a array) (b array) direction)
  (%trace-op-compare-array-array a b direction))

(defmethod %t-compare ((a real) (b array) direction)
  (%trace-op-compare-real-array a b direction))

(defmethod %t-compare ((a array) (b real) direction)
  (%trace-op-compare-array-real a b direction))

(defmethod %t-compare ((a tracer) (b tracer) direction)
  (%trace-op-compare-tracer-tracer a b direction))

(defmethod %t-compare ((a real) (b tracer) direction)
  (%trace-op-compare-real-tracer a b direction))

(defmethod %t-compare ((a tracer) (b real) direction)
  (%trace-op-compare-tracer-real a b direction))

(defmethod %t-compare ((a array) (b tracer) direction)
  (%trace-op-compare-array-tracer a b direction))

(defmethod %t-compare ((a tracer) (b array) direction)
  (%trace-op-compare-tracer-array a b direction))

;;; --- %t-if（T1 の暫定版）: TEST がトレーサ／配列なら、まだ SELECT に
;;; 変換できないため（T2 が対応する）TRACING-ERROR を signal する。それ
;;; 以外（ふつうの Lisp の値）は、ふつうの IF と同じく TEST の真偽で
;;; THEN-THUNK／ELSE-THUNK のどちらか一方だけを呼ぶ。

(defun %if-test-traced-p (test)
  "TEST がトレーサまたは配列（=トレース対象の値）なら真を返す。"
  (or (typep test 'tracer) (arrayp test)))

(defun %t-if (test then-thunk else-thunk)
  (if (%if-test-traced-p test)
      (error 'tracing-error
             :format-control "IF の条件にトレーサ／配列は使えない（select への変換は issue #32 の後続 PR で対応）: ~S"
             :format-arguments (list test))
      (if test (funcall then-thunk) (funcall else-thunk))))
