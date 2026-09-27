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

(defstruct (%trace (:constructor %make-trace (invars)) (:copier nil) (:conc-name trace-))
  "1回の WITH-TRACING 呼び出しに対応するトレース状態。EQNS と CONSTANTS は
逆順に PUSH し、TRACE-TO-GRAPH が最後に反転する（GRAPH の EQNS / CONSTANTS
は出現順のリストであるため）。"
  (invars nil :type list :read-only t)
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

(defun %tracer-check-current-trace (tracers)
  "TRACERS の全員が *CURRENT-TRACE* に属することを確かめる。1つでも違えば
TRACING-ERROR を signal する（別の、または既に終わった WITH-TRACING 呼び出し
の TRACER が紛れ込んだ場合）。"
  (dolist (tracer tracers)
    (unless (eq (tracer-trace tracer) *current-trace*)
      (error 'tracing-error
             :format-control "トレーサ ~S は現在のトレースに属していない（別の、または既に終わったトレースのもの）"
             :format-arguments (list tracer)))))

(defun %trace-eqn (prim-name tracers &rest params)
  "PRIM-NAME・TRACERS（すべて *CURRENT-TRACE* に属する TRACER）・PARAMS から
EQN を作って *CURRENT-TRACE* に積み、その唯一の outvar に対応する新しい
TRACER を返す。"
  (%tracer-check-current-trace tracers)
  (let* ((trace *current-trace*)
         (eqn (apply #'make-eqn prim-name (mapcar #'tracer-var tracers) params)))
    (push eqn (trace-eqns trace))
    (make-instance 'tracer :var (first (eqn-outvars eqn)) :trace trace)))

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
       (error 'tracing-error
              :format-control "戻り値のトレーサ ~S が別のトレースに属している"
              :format-arguments (list value)))
     (tracer-var value))
    ((typep value 'real)
     (let ((dtype (if (typep value 'double-float) :f64 :f32)))
       (tracer-var (%lift-constant (%scalar-array value dtype) (make-aval '() dtype) trace))))
    ((arrayp value)
     (tracer-var (%lift-constant value (array-aval value) trace)))
    (t
     (error 'tracing-error
            :format-control "トレース対象の関数はトレーサ・実数・配列以外を返せない: ~S"
            :format-arguments (list value)))))

(defun trace-to-graph (fn avals)
  "FN（WITH-TRACING が返す TRACEABLE-FUNCTION）を AVALS（FN の引数と同じ数の
AVAL のリスト）でトレースし、CHECK-GRAPH した GRAPH を返す。

手順: AVALS ごとに invar（VAR）を作り、新しい %TRACE を *CURRENT-TRACE* に
束縛したうえで、それぞれの invar に対応する TRACER を FN の関数に適用する。
戻り値（多値。0個なら outvars も0個）を %OUTVAR-OF で1つずつ outvar に変換し、
GRAPH を組み立てる。FN 自身が呼び出したトレース対象の演算は、すべて
%TRACE-EQN 経由で *CURRENT-TRACE* に積まれる。"
  (%check-avals-length fn avals)
  (let* ((invars (mapcar #'make-var avals))
         (trace (%make-trace invars))
         (*current-trace* trace)
         (tracers (mapcar (lambda (var) (make-instance 'tracer :var var :trace trace)) invars)))
    (let ((results (multiple-value-list (apply (%traceable-function-function fn) tracers))))
      (check-graph
       (make-graph invars
                   (reverse (trace-eqns trace))
                   (mapcar (lambda (v) (%outvar-of v trace)) results)
                   (reverse (trace-constants trace)))))))
