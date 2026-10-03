;;;; trace: with-tracing のトレーサ本体（issue #32、t1）。
;;;;
;;;; TRACER は、トレース中に本物の配列の代わりに演算へ渡す値。演算のたびに
;;;; %TRACE-EQN が新しい EQN を作って TRACE に積み、その出力に対応する新しい
;;;; TRACER を返す。トレース終了後、TRACE-TO-GRAPH がそれを GRAPH に組み立てる
;;;; （StableHLO へは直接変換しない。CLAUDE.md: StableHLO は出力先であって
;;;; 内部表現ではない）。
;;;;
;;;; 契約は構造体の名前を TRACE にしているが、NABLA は (:USE #:CL) なので
;;;; その名前は CL:TRACE（デバッグ用マクロ）と同じシンボルになり、
;;;; DEFSTRUCT が COMMON-LISP パッケージのロックに触れてエラーになる
;;;; （確認済み）。ここでは %TRACE という内部名にして避ける（export しない
;;;; ので利用者には見えない）。

(in-package #:nabla)

(defstruct (%trace (:constructor %make-trace (invars &optional parent)) (:copier nil) (:conc-name trace-))
  "1回の WITH-TRACING 呼び出しに対応するトレース状態。EQNS と CONSTANTS は
逆順に PUSH し、TRACE-TO-GRAPH が最後に反転する（GRAPH の EQNS / CONSTANTS
は出現順のリストであるため）。

PARENT は、サブグラフのトレース（%TRACE-SUBGRAPH。契約 C2）のときだけ、
それを始めた時点の外側のトレース。それ以外は NIL（%CALL-WITH-FRESH-TRACE は
常に NIL。grad の既知の制限は変えない）。CAPTURED は、本体が閉包で捕まえた
祖先のトレースのトレーサを持ち上げた記録で、(外側のトレーサ . このトレースの
内側のトレーサ) の組を逆順に PUSH する（最初に使った順の反転で、明示的な
invars の後ろに足す追加の invars になる）。"
  (invars nil :type list :read-only t)
  (parent nil :type (or null %trace) :read-only t)
  (captured nil :type list)
  (eqns nil :type list)
  (constants nil :type list))

(defvar *current-trace* nil
  "TRACE-TO-GRAPH の実行中だけその %TRACE に束縛される特殊変数。それ以外では
NIL。TRACER が属するトレースと *CURRENT-TRACE* が違えば TRACING-ERROR に
なる（%TRACER-CHECK-CURRENT-TRACE 参照）。")

(define-condition tracing-error (error)
  ((format-control :initarg :format-control :reader tracing-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader tracing-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "トレースエラー: ~?"
             (tracing-error-format-control condition)
             (tracing-error-format-arguments condition))))
  (:documentation
   "トレース中の不変量が壊れたときに signal する。次の場合に起きる:
別の（または既に終わった）TRACE-TO-GRAPH 呼び出しに属する TRACER が混ざる、
トレース対象の関数がトレーサ・実数・配列のどれでもない値を返す、:I1 の
値を定数としてリフトしようとする、トレーサ／配列を IF の条件に使う
（T2 が SELECT に置き換えるまでは未対応）。"))

(defclass tracer ()
  ((var :initarg :var :reader tracer-var)
   (trace :initarg :trace :reader tracer-trace))
  (:documentation
   "トレース中に本物の配列の代わりに演算へ渡す値。VAR は対応する IR 上の
var、TRACE はこの TRACER が属する %TRACE。export しない（利用者が直接
TRACER を作ることはない。値はすべて WITH-TRACING が渡す）。"))

(defun tracer-aval (tracer)
  "TRACER に対応する var の aval を返す。"
  (var-aval (tracer-var tracer)))

(defmethod print-object ((tracer tracer) stream)
  (print-unreadable-object (tracer stream :type t)
    (let ((aval (tracer-aval tracer)))
      (format stream "~A[~{~D~^,~}]" (dtype-mlir-name (aval-dtype aval)) (aval-shape aval)))))

(defun %trace-ancestor-p (candidate trace)
  "CANDIDATE が TRACE の祖先（PARENT をたどって EQ になる、TRACE 自身は含まない）か。"
  (loop for ancestor = (trace-parent trace) then (trace-parent ancestor)
        while ancestor
        thereis (eq ancestor candidate)))

(defun %resolve-tracer (tracer)
  "TRACER を *CURRENT-TRACE* で使える TRACER にして返す（契約 C2）。
  - 今のトレースの TRACER ならそのまま返す。
  - 祖先のトレースの TRACER なら、今のトレースに新しい invar を作って持ち上げる
    （closure conversion）。同じ TRACER（EQ）は同じ invar にメモ化され、invar は
    明示的な invars の後ろに、最初に使った順で足される。
  - それ以外（無関係な、または終わったトレースのもの）は TRACING-ERROR。"
  (let ((current *current-trace*))
    (cond
      ((eq (tracer-trace tracer) current) tracer)
      ((and current (%trace-ancestor-p (tracer-trace tracer) current))
       (or (cdr (assoc tracer (trace-captured current) :test #'eq))
           (let ((inner (make-instance 'tracer :var (make-var (tracer-aval tracer)) :trace current)))
             (push (cons tracer inner) (trace-captured current))
             inner)))
      (t
       (error 'tracing-error
              :format-control "トレーサ ~S は現在のトレースに属していない（別の、または既に終わったトレースのもの）"
              :format-arguments (list tracer))))))

(defun %tracer-check-current-trace (tracers)
  "TRACERS の全員が *CURRENT-TRACE* で使える（今のトレースのものか、祖先の
トレースのものとして持ち上げられる）ことを確かめる。使えない TRACER が1つでも
あれば TRACING-ERROR を signal する。持ち上げ（%RESOLVE-TRACER）の副作用が
起きるので、使う側は %RESOLVE-TRACER の結果を使うこと。"
  (mapc #'%resolve-tracer tracers)
  nil)

(defun %trace-eqn* (prim-name tracers &rest params)
  "PRIM-NAME・TRACERS・PARAMS から EQN を作って *CURRENT-TRACE* に積み、その
全ての outvar に対応する新しい TRACER の「リスト」を返す（単一出力の
プリミティブでも長さ1のリスト。契約 C1）。TRACERS は今のトレースのもの、または
祖先のトレースのもの（%RESOLVE-TRACER で持ち上げられる）。"
  (let* ((trace *current-trace*)
         (resolved (mapcar #'%resolve-tracer tracers))
         (eqn (apply #'make-eqn prim-name (mapcar #'tracer-var resolved) params)))
    (push eqn (trace-eqns trace))
    (mapcar (lambda (var) (make-instance 'tracer :var var :trace trace)) (eqn-outvars eqn))))

(defun %trace-eqn (prim-name tracers &rest params)
  "PRIM-NAME・TRACERS・PARAMS から EQN を作って *CURRENT-TRACE* に積み、その
唯一の outvar に対応する新しい TRACER を返す。複数出力のプリミティブ
（PRIMITIVE-MULTIPLE-OUTPUTS-P）には使えず TRACING-ERROR になる（%TRACE-EQN* を
使う）。"
  (let ((prim (find-primitive prim-name)))
    (when (and prim (primitive-multiple-outputs-p prim))
      (error 'tracing-error
             :format-control "プリミティブ ~S は複数出力なので %TRACE-EQN では扱えない（%TRACE-EQN* を使う）"
             :format-arguments (list prim-name))))
  (first (apply #'%trace-eqn* prim-name tracers params)))

(defun %lift-constant (array aval trace)
  "ARRAY（AVAL の shape/dtype を持つ配列）を TRACE の定数として登録し、
新しい var に束縛した TRACER を返す。"
  (let ((var (make-var aval)))
    (push (cons var array) (trace-constants trace))
    (make-instance 'tracer :var var :trace trace)))

(defun %filled-array (shape dtype number)
  "SHAPE・DTYPE を持ち、要素がすべて NUMBER（DTYPE の計算型に COERCE した
もの）で埋まった配列を返す（%SCALAR-ARRAY と %FILL-ARRAY が共有する）。
DTYPE が :I1 なら（0/1 の意味が NUMBER の何を表すか自明でないため）
TRACING-ERROR を signal する。"
  (when (eq dtype :i1)
    (error 'tracing-error
           :format-control ":I1 の値 ~S はリフトできない（数値との対応が定義されていない）"
           :format-arguments (list number)))
  (let* ((compute-type (%compute-element-type dtype))
         (array (make-array shape :element-type compute-type :initial-element (coerce number compute-type))))
    (%encode-array array dtype)))

(defun %scalar-array (number dtype)
  "NUMBER を rank 0 の DTYPE 配列にエンコードして返す（%LIFT-NUMBER が使う）。
DTYPE が :I1 なら TRACING-ERROR を signal する。"
  (%filled-array '() dtype number))

(defun %fill-array (number like-array)
  "NUMBER を LIKE-ARRAY と同じ shape・dtype の配列にして返す（%LIFT-NUMBER の
eager 版）。LIKE-ARRAY が生の (UNSIGNED-BYTE 16) 配列（bf16/f16）だと dtype が
一意に決まらず、ARRAY-AVAL 自身が DTYPE-MISMATCH を signal する（bf16/f16 を
eager でそのまま扱えないという既存の制約どおり。CLAUDE.md）。"
  (let ((aval (array-aval like-array)))
    (%filled-array (aval-shape aval) (aval-dtype aval) number)))

(defun %lift-number (number like-tracer)
  "NUMBER を LIKE-TRACER と同じ dtype の定数 TRACER にする。LIKE-TRACER の
rank が 0 より大きければ :BROADCAST-IN-DIM で LIKE-TRACER の shape まで
広げる（新しい EQN が1つ増える）。rank 0 なら形が最初から一致しているので
EQN を足さず、定数 TRACER をそのまま返す（golden テストで pin する）。"
  (let* ((aval (tracer-aval like-tracer))
         (dtype (aval-dtype aval))
         (const-tracer (%lift-constant (%scalar-array number dtype) (make-aval '() dtype) *current-trace*)))
    (if (plusp (aval-rank aval))
        (%trace-eqn :broadcast-in-dim (list const-tracer) :shape (aval-shape aval) :dims '())
        const-tracer)))

(defun %lift-array (array like-tracer)
  "ARRAY を LIKE-TRACER と同じ dtype の定数として *CURRENT-TRACE* に足す。
ARRAY が生の (UNSIGNED-BYTE 16) 配列で LIKE-TRACER の dtype と食い違えば
ARRAY-AVAL 自身が DTYPE-MISMATCH を signal する。shape の不一致はここでは
検査せず、後続の演算の abstract-eval に委ねる（PRIMITIVE-ERROR になる）。"
  (let* ((dtype (aval-dtype (tracer-aval like-tracer)))
         (aval (array-aval array dtype)))
    (%lift-constant array aval *current-trace*)))

(defclass traceable-function (sb-mop:funcallable-standard-object)
  ((lambda-list :initarg :lambda-list :reader traceable-function-lambda-list)
   (function :initarg :function :reader %traceable-function-function))
  (:metaclass sb-mop:funcallable-standard-class)
  (:documentation
   "WITH-TRACING が返す関数オブジェクト。呼び出すと本体を eager に実行する
（トレースはしない）。LAMBDA-LIST は WITH-TRACING に渡した仮引数のリストで、
TRACE-TO-GRAPH がその長さを AVALS の長さと突き合わせるのに使う。export する
のはクラス名だけ（TYPEP で判定できれば十分。TRACEABLE-FUNCTION-LAMBDA-LIST は
内部）。"))

(defun %make-traceable-function (lambda-list function)
  "LAMBDA-LIST と FUNCTION（同じ引数を取る関数）から TRACEABLE-FUNCTION を
作る。WITH-TRACING の展開先。"
  (let ((instance (make-instance 'traceable-function :lambda-list lambda-list :function function)))
    (sb-mop:set-funcallable-instance-function instance function)
    instance))

(defmethod print-object ((f traceable-function) stream)
  (print-unreadable-object (f stream :type t)
    (format stream "~S" (traceable-function-lambda-list f))))

(defun %check-avals-length (fn avals)
  "AVALS の個数が FN（TRACEABLE-FUNCTION）の仮引数の個数と一致しなければ
TRACING-ERROR を signal する。"
  (let ((expected (length (traceable-function-lambda-list fn))))
    (unless (= (length avals) expected)
      (error 'tracing-error
             :format-control "avals の個数 ~D が関数の引数の個数 ~D と一致しない"
             :format-arguments (list (length avals) expected)))))

(defun %outvar-of (value trace)
  "TRACE-TO-GRAPH が、本体の戻り値の1つ VALUE を GRAPH の outvar（VAR）に
変換する。TRACER なら（同じ TRACE に属することを確かめてから）その var、
実数なら（DOUBLE-FLOAT は :F64、それ以外は :F32 の）rank 0 定数、配列なら
そのまま定数にする。それ以外は TRACING-ERROR。"
  (cond
    ((typep value 'tracer)
     (unless (eq (tracer-trace value) trace)
       ;; サブグラフでは、祖先のトレースのトレーサを返してよい（閉包で捕まえた
       ;; 値をそのまま返す本体。持ち上げて追加の入力にする）。
       (unless (and (eq trace *current-trace*) (%trace-ancestor-p (tracer-trace value) trace))
         (error 'tracing-error
                :format-control "戻り値のトレーサ ~S が別のトレースに属している"
                :format-arguments (list value))))
     (tracer-var (%resolve-tracer value)))
    ((typep value 'real)
     (let ((dtype (if (typep value 'double-float) :f64 :f32)))
       (tracer-var (%lift-constant (%scalar-array value dtype) (make-aval '() dtype) trace))))
    ((arrayp value)
     (tracer-var (%lift-constant value (array-aval value) trace)))
    (t
     (error 'tracing-error
            :format-control "トレース対象の関数はトレーサ・実数・配列以外を返せない: ~S"
            :format-arguments (list value)))))

(defun %call-with-trace (avals fn parent)
  "%CALL-WITH-FRESH-TRACE の本体。PARENT（NIL か、外側のトレース）を持つ新しい
トレースで FN を呼び、(VALUES GRAPH CAPTURED) を返す。CAPTURED は、FN が閉包で
捕まえて持ち上げた祖先のトレースのトレーサを、持ち上げた順に並べたリスト
（GRAPH の invars は、AVALS の分の後ろに、この順の追加の入力が並ぶ）。"
  (let* ((invars (mapcar #'make-var avals))
         (trace (%make-trace invars parent))
         (*current-trace* trace)
         (tracers (mapcar (lambda (var) (make-instance 'tracer :var var :trace trace)) invars)))
    (let* ((results (multiple-value-list (apply fn tracers)))
           ;; 戻り値の変換で持ち上げが起きうるので、captured は変換の後で読む。
           (outvars (mapcar (lambda (v) (%outvar-of v trace)) results))
           (captured (reverse (trace-captured trace))))
      (values
       (check-graph
        (make-graph (append invars (mapcar (lambda (entry) (tracer-var (cdr entry))) captured))
                    (reverse (trace-eqns trace))
                    outvars
                    (reverse (trace-constants trace))))
       (mapcar #'car captured)))))

(defun %call-with-fresh-trace (avals fn)
  "AVALS ごとに invar（VAR）を作り、新しい %TRACE を *CURRENT-TRACE* に
束縛したうえで、それぞれの invar に対応する TRACER を FN（普通の関数）に
適用する。戻り値（多値。0個なら outvars も0個）を %OUTVAR-OF で1つずつ
outvar に変換し、CHECK-GRAPH した GRAPH を返す。FN が呼び出したトレース対象の
演算は、すべて %TRACE-EQN 経由で *CURRENT-TRACE* に積まれる。TRACE-TO-GRAPH
と、変換（jvp など）が別のトレースを新しく始めるときに共有する。

親トレースを持たない（PARENT = NIL）ので、FN が外側のトレーサを閉包で捕まえると
TRACING-ERROR になる（grad の既知の制限）。制御構造の本体のように外側の値を
捕まえたいときは %TRACE-SUBGRAPH を使う。"
  (values (%call-with-trace avals fn nil)))

(defun %trace-subgraph (fn avals)
  "FN（WITH-TRACING が返す TRACEABLE-FUNCTION）を AVALS でトレースして、閉じた
GRAPH にする（契約 C2。cond / while-loop / scan の本体のトレースに使う）。
(VALUES GRAPH CAPTURED-OUTER-TRACERS) を返す。

FN が閉包で外側のトレース（今の *CURRENT-TRACE* とその祖先）のトレーサを
捕まえると、それは GRAPH の追加の入力に持ち上げられ（closure conversion）、
CAPTURED-OUTER-TRACERS にその外側のトレーサが、追加の入力と同じ順で入る。
GRAPH の invars は「AVALS の分、続けて captured の分」。呼び出し側は、
subgraph を持つ eqn の invars の末尾に CAPTURED を足す（足した後の %TRACE-EQN*
が、さらに外側のトレースへの持ち上げも自動で行う）。外側のトレースが無い
（*CURRENT-TRACE* が NIL の）ときも使える（CAPTURED は空）。"
  (unless (typep fn 'traceable-function)
    (error 'tracing-error
           :format-control "FN は TRACEABLE-FUNCTION でなければならない（WITH-TRACING で作る）: ~S"
           :format-arguments (list fn)))
  (%check-avals-length fn avals)
  (%call-with-trace avals (%traceable-function-function fn) *current-trace*))

(defun trace-to-graph (fn avals)
  "FN（WITH-TRACING が返す TRACEABLE-FUNCTION）を AVALS（FN の引数と同じ数の
AVAL のリスト）でトレースし、CHECK-GRAPH した GRAPH を返す。

手順は %CALL-WITH-FRESH-TRACE を参照（FN の型と AVALS の個数を確かめた
うえで、その薄い包みとして動く）。"
  (unless (typep fn 'traceable-function)
    (error 'tracing-error
           :format-control "FN は TRACEABLE-FUNCTION でなければならない（WITH-TRACING で作る）: ~S"
           :format-arguments (list fn)))
  (%check-avals-length fn avals)
  (%call-with-fresh-trace avals (%traceable-function-function fn)))
