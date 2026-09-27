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
  "A・B が両方配列のときの共通実装: rank 0 のブロードキャストで shape を
合わせてから、PRIM-NAME の :EAGER を直接呼ぶ（それでも残る shape/dtype
の不一致は abstract-eval が PRIMITIVE-ERROR として報告する）。"
  (multiple-value-bind (a* b*) (%align-array-pair a b)
    (apply (primitive-eager (find-primitive prim-name)) (list a* b*) (list (array-aval a*) (array-aval b*)) nil)))

(defun %trace-op-real-array (prim-name a b)
  "A が REAL、B が配列のときの共通実装: A を B と同じ shape/dtype に埋めて
（%FILL-ARRAY）から、配列どうしの実装（%TRACE-OP-ARRAY-ARRAY）に委ねる。"
  (%trace-op-array-array prim-name (%fill-array a b) b))

(defun %trace-op-array-real (prim-name a b)
  "A が配列、B が REAL のときの共通実装（%TRACE-OP-REAL-ARRAY と対称）。"
  (%trace-op-array-array prim-name a (%fill-array b a)))

;;; --- rank 0 のブロードキャスト（issue #32、t2） ---
;;;
;;; 数値でも配列全体でもなく、すでに演算の結果である rank 0 のトレーサ／
;;; 配列（例: (REDUCE-SUM X) の結果）を、より rank の高いオペランドと
;;; 組み合わせるときは :BROADCAST-IN-DIM で shape を合わせる。それ以外の
;;; shape の不一致は、そのままプリミティブの abstract-eval に委ねる
;;; （PRIMITIVE-ERROR になる）。「数値と rank 0 の値だけがブロードキャスト
;;; され、それ以外はしない」というルール（README に明記する）。

(defun %broadcast-tracer-if-rank0 (tracer shape)
  "TRACER の shape が SHAPE とすでに等しければそのまま返す。TRACER が
rank 0 で SHAPE が rank 0 でなければ :BROADCAST-IN-DIM で SHAPE まで
広げる。それ以外（rank が2つとも0より大きく、かつ食い違う）は変えずに
返す（後続の演算の abstract-eval が PRIMITIVE-ERROR にする）。"
  (let ((tracer-shape (aval-shape (tracer-aval tracer))))
    (cond
      ((equal tracer-shape shape) tracer)
      ((null tracer-shape) (%trace-eqn :broadcast-in-dim (list tracer) :shape shape :dims '()))
      (t tracer))))

(defun %align-tracer-pair (a b)
  "A・B（両方トレーサ）のうち、片方だけが rank 0 ならもう片方の shape に
ブロードキャストした2つのトレーサを (VALUES A* B*) で返す。"
  (let ((shape-a (aval-shape (tracer-aval a))) (shape-b (aval-shape (tracer-aval b))))
    (cond
      ((equal shape-a shape-b) (values a b))
      ((null shape-a) (values (%broadcast-tracer-if-rank0 a shape-b) b))
      ((null shape-b) (values a (%broadcast-tracer-if-rank0 b shape-a)))
      (t (values a b)))))

(defun %broadcast-rank0-array (array target-shape)
  "ARRAY（rank 0 の配列）を TARGET-SHAPE まで広げた新しい配列を返す
（:BROADCAST-IN-DIM の :EAGER をそのまま呼ぶ）。"
  (apply (primitive-eager (find-primitive :broadcast-in-dim)) (list array) (list (array-aval array))
         (list :shape target-shape :dims '())))

(defun %array-broadcast-if-rank0 (array shape)
  "ARRAY の shape が SHAPE とすでに等しければそのまま返す。ARRAY が rank 0
で SHAPE が rank 0 でなければ %BROADCAST-RANK0-ARRAY で広げる。"
  (let ((array-shape (array-dimensions array)))
    (cond
      ((equal array-shape shape) array)
      ((null array-shape) (%broadcast-rank0-array array shape))
      (t array))))

(defun %align-array-pair (a b)
  "A・B（両方配列）のうち、片方だけが rank 0 ならもう片方の shape に
ブロードキャストした2つの配列を (VALUES A* B*) で返す。"
  (let ((shape-a (array-dimensions a)) (shape-b (array-dimensions b)))
    (cond
      ((equal shape-a shape-b) (values a b))
      ((null shape-a) (values (%array-broadcast-if-rank0 a shape-b) b))
      ((null shape-b) (values a (%array-broadcast-if-rank0 b shape-a)))
      (t (values a b)))))

(defun %trace-op-tracer-tracer (prim-name a b)
  "A・B が両方トレーサのときの共通実装: rank 0 のブロードキャストで shape
を合わせてから、PRIM-NAME の EQN を1つ足す。"
  (multiple-value-bind (a* b*) (%align-tracer-pair a b)
    (%trace-eqn prim-name (list a* b*))))

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
  "A・B が両方配列のときの共通実装: rank 0 のブロードキャストで shape を
合わせてから :COMPARE プリミティブの :EAGER を直接呼び、:I1（BIT）の配列を
返す。"
  (multiple-value-bind (a* b*) (%align-array-pair a b)
    (apply (primitive-eager (find-primitive :compare)) (list a* b*)
           (list (array-aval a*) (array-aval b*)) (list :direction direction))))

(defun %trace-op-compare-real-array (a b direction)
  "A が REAL、B が配列のときの共通実装。"
  (%trace-op-compare-array-array (%fill-array a b) b direction))

(defun %trace-op-compare-array-real (a b direction)
  "A が配列、B が REAL のときの共通実装。"
  (%trace-op-compare-array-array a (%fill-array b a) direction))

(defun %trace-op-compare-tracer-tracer (a b direction)
  "A・B が両方トレーサのときの共通実装: rank 0 のブロードキャストで shape
を合わせてから :COMPARE の EQN を1つ足す（:I1 のトレーサを返す）。"
  (multiple-value-bind (a* b*) (%align-tracer-pair a b)
    (%trace-eqn :compare (list a* b*) :direction direction)))

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

;;; --- %t-select: PRED（:I1 のトレーサ／ビット配列）の真偽で A・B のどちら
;;; かを選ぶ（issue #32、t2）。数値・rank 0 のトレーサ／配列は PRED の
;;; shape に合わせてブロードキャストし、dtype はもう一方の分岐から決める
;;; （両方数値なら dtype が決まらず TRACING-ERROR）。A・B のどちらかが :I1
;;; の値（トレーサ／ビット配列）なら TRACING-ERROR にする。これは AND/OR を
;;; トレーサの条件に使ったとき、SB-CLTL2:MACROEXPAND-ALL が
;;; (IF #:G #:G ELSE) に展開し、#:G（PRED と同じ :I1 の値）がそのまま分岐に
;;; 現れる（契約のピットフォール(1)）ケースを、分かりにくい PRIMITIVE-ERROR
;;; ではなく AND/OR を名指しするメッセージで報告するため。 ---

(defun %lift-number-to (number dtype shape)
  "NUMBER を DTYPE の定数トレーサにし、SHAPE が rank 0 でなければその形へ
:BROADCAST-IN-DIM で広げる（%LIFT-NUMBER の、既存のトレーサの dtype/shape
を経由しない汎用版。%T-SELECT が分岐を PRED の shape・もう一方の dtype に
合わせるのに使う）。"
  (let ((const-tracer (%lift-constant (%scalar-array number dtype) (make-aval '() dtype) *current-trace*)))
    (if (plusp (length shape))
        (%trace-eqn :broadcast-in-dim (list const-tracer) :shape shape :dims '())
        const-tracer)))

(defun %lift-array-to (array dtype)
  "ARRAY を DTYPE の定数トレーサにする（%LIFT-ARRAY の、既存のトレーサを
経由しない汎用版）。"
  (%lift-constant array (array-aval array dtype) *current-trace*))

(defun %select-branch-dtype (branch)
  "BRANCH（トレーサ・配列・実数のいずれか）の dtype を返す。実数なら NIL
（%SELECT-RESOLVE-DTYPE がもう一方の分岐から決める）。"
  (cond
    ((typep branch 'tracer) (aval-dtype (tracer-aval branch)))
    ((arrayp branch) (aval-dtype (array-aval branch)))
    (t nil)))

(defun %select-check-branch-type (branch)
  "BRANCH がトレーサ・配列・実数のいずれでもなければ TRACING-ERROR を
signal する。WHEN／UNLESS が省略した ELSE は NIL（Lisp のブール偽）に
展開されるが、NIL は数値としてリフトできないので、ここで分かりやすい
TRACING-ERROR にする（(WHEN tracer-test x) がトレーサの条件で失敗する、
という契約に明記されたドキュメント上の既知の制約）。"
  (unless (typep branch '(or tracer array real))
    (error 'tracing-error
           :format-control "SELECT/WHERE の分岐はトレーサ・配列・実数のいずれかでなければならない（WHEN/UNLESS が省略した ELSE は NIL になり使えない）: ~S"
           :format-arguments (list branch))))

(defun %select-check-branch-not-i1 (branch)
  "BRANCH が :I1 の値（トレーサ／配列）なら TRACING-ERROR を signal する。"
  (when (eq (%select-branch-dtype branch) :i1)
    (error 'tracing-error
           :format-control "SELECT/WHERE の分岐に :i1 の値は使えない（AND/OR がトレーサの条件に対して (IF X X ELSE) に展開されるため、X がそのままここに来ている可能性が高い。明示的な比較や WHERE を使うこと）: ~S"
           :format-arguments (list branch))))

(defun %select-resolve-dtype (a b)
  "A・B の一方が数値でなければその dtype を、両方数値なら
（決めようがないので）TRACING-ERROR を signal する。"
  (or (%select-branch-dtype a) (%select-branch-dtype b)
      (error 'tracing-error
             :format-control "SELECT/WHERE の両方の分岐が数値では dtype を決められない: ~S / ~S"
             :format-arguments (list a b))))

(defun %select-lift-branch-to-tracer (branch dtype shape)
  "BRANCH（トレーサ・配列・実数）を、DTYPE・SHAPE のトレーサにする。
トレーサ・配列は rank 0 なら SHAPE にブロードキャストする。"
  (typecase branch
    (tracer (%broadcast-tracer-if-rank0 branch shape))
    (array (%broadcast-tracer-if-rank0 (%lift-array-to branch dtype) shape))
    (t (%lift-number-to branch dtype shape))))

(defun %trace-select-tracer (pred a b)
  "PRED（:I1 のトレーサ）による %T-SELECT の本体。"
  (%select-check-branch-type a)
  (%select-check-branch-type b)
  (%select-check-branch-not-i1 a)
  (%select-check-branch-not-i1 b)
  (let* ((dtype (%select-resolve-dtype a b))
         (shape (aval-shape (tracer-aval pred)))
         (a* (%select-lift-branch-to-tracer a dtype shape))
         (b* (%select-lift-branch-to-tracer b dtype shape)))
    (%trace-eqn :select (list pred a* b*))))

(defun %select-lift-branch-to-array (branch dtype shape)
  "BRANCH（配列・実数）を、DTYPE・SHAPE の配列にする（%SELECT-LIFT-BRANCH-
TO-TRACER の eager 版）。"
  (typecase branch
    (array (%array-broadcast-if-rank0 branch shape))
    (t (%array-broadcast-if-rank0 (%filled-array '() dtype branch) shape))))

(defun %eager-select-array (pred a b)
  "PRED（:I1 のビット配列）による %T-SELECT の本体。"
  (%select-check-branch-type a)
  (%select-check-branch-type b)
  (%select-check-branch-not-i1 a)
  (%select-check-branch-not-i1 b)
  (let* ((dtype (%select-resolve-dtype a b))
         (shape (array-dimensions pred))
         (a* (%select-lift-branch-to-array a dtype shape))
         (b* (%select-lift-branch-to-array b dtype shape)))
    (apply (primitive-eager (find-primitive :select)) (list pred a* b*)
           (list (array-aval pred) (array-aval a*) (array-aval b*)) nil)))

(defgeneric %t-select (pred a b)
  (:documentation
   "PRED（:I1 のトレーサまたはビット配列）の真偽で A・B のどちらかを選ぶ。
A・B はトレーサ・配列・実数を任意に組み合わせられる。数値・rank 0 の
トレーサ／配列は PRED の shape に合わせてブロードキャストし、dtype は
もう一方の分岐から決める。両方とも数値、またはどちらかが :I1 の値だと
TRACING-ERROR になる。"))

(defmethod %t-select ((pred tracer) a b)
  (%trace-select-tracer pred a b))

(defmethod %t-select ((pred array) a b)
  (%eager-select-array pred a b))

;;; --- %t-if: TEST の種類で分岐する（issue #32、t1 の暫定版を t2 が
;;; 置き換える）。TEST が :I1 のトレーサ／ビット配列なら THEN-THUNK と
;;; ELSE-THUNK を両方呼び（配列は要素ごとの意味を持つので、両方の枝を
;;; 計算する必要がある。契約のピットフォール(3)）、%T-SELECT で選ぶ。TEST
;;; が :I1 でないトレーサ／配列なら TRACING-ERROR。それ以外（ふつうの
;;; Lisp の値）は、ふつうの IF と同じく TEST の真偽で THEN-THUNK／
;;; ELSE-THUNK のどちらか一方だけを呼ぶ。

(defun %if-test-kind (test)
  "TEST の種類を :TRACER-I1・:TRACER-OTHER・:ARRAY-I1・:ARRAY-OTHER・:PLAIN
のいずれかで返す。"
  (cond
    ((typep test 'tracer) (if (eq (aval-dtype (tracer-aval test)) :i1) :tracer-i1 :tracer-other))
    ((arrayp test) (if (eq (aval-dtype (array-aval test)) :i1) :array-i1 :array-other))
    (t :plain)))

(defun %t-if (test then-thunk else-thunk)
  (ecase (%if-test-kind test)
    ((:tracer-i1 :array-i1) (%t-select test (funcall then-thunk) (funcall else-thunk)))
    ((:tracer-other :array-other)
     (error 'tracing-error
            :format-control "IF の条件は :i1（比較の結果）でなければならない: ~S"
            :format-arguments (list test)))
    (:plain (if test (funcall then-thunk) (funcall else-thunk)))))
