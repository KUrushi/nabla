;;;; cond: 条件分岐 COND*（issue #130）。
;;;;
;;;; with-tracing の if は、トレーサの :i1 条件では両方の枝を評価して select で
;;;; 選ぶ（要素ごとの意味を保つ。docs/phase2-report.md §6）。COND* は、rank 0 の
;;;; :i1 の条件で「片方の枝だけを評価する」高階プリミティブ :cond を使う。
;;;; if を cond に落とさない理由は src/walk.lisp の %walk-if を参照。

(in-package #:nabla)

(define-condition cond-error (tracing-error)
  ()
  (:documentation
   "COND* の引数が不正なときに、トレース時に signal される TRACING-ERROR の子。
pred が rank 0 の :i1 でない、枝が TRACEABLE-FUNCTION でない、両枝の出力の aval
（個数・形状・dtype）が一致しない、枝の出力が無い、のいずれか。"))

(defun %cond-error (control &rest arguments)
  (error 'cond-error :format-control control :format-arguments arguments))

(defun %cond-bit-pred-p (pred)
  (and (arrayp pred) (null (array-dimensions pred)) (eq (aval-dtype (array-aval pred)) :i1)))

(defun %cond-lift-operand (operand)
  "OPERAND（トレーサ・:f32 / :f64 / :i1 の配列・実数）を現在のトレースのトレーサにする。
bf16 / f16 の生の配列は dtype を推論できないので受け付けない（トレーサで渡す）。"
  (typecase operand
    (tracer operand)
    ((or (array single-float) (array double-float) (array bit)) (%lift-array-to operand (aval-dtype (array-aval operand))))
    (real (%lift-number-to operand (if (typep operand 'double-float) :f64 :f32) '()))
    (t (%cond-error "COND* の operand はトレーサ・配列・実数でなければならない: ~S" operand))))

(defun cond* (pred then-fn else-fn &rest operands)
  "PRED が真なら (THEN-FN OPERANDS...) を、偽なら (ELSE-FN OPERANDS...) を評価する。
片方の枝しか評価されない（eager では選ばれた枝のサブグラフだけを評価する。
トレース中は両枝を1回ずつトレースしてサブグラフにする）。名前は CL の COND と
衝突しないよう COND* にしてある。

THEN-FN / ELSE-FN は WITH-TRACING で作った TRACEABLE-FUNCTION で、OPERANDS を
位置引数で受け取り、普通のトレース対象の関数と同じく1つの値または多値を返す。
COND* も同じ個数の値（多値）を返す。両枝の出力の aval（個数・形状・dtype）は
一致しなければならず、違えば COND-ERROR。枝が閉包で捕まえた外側のトレーサも使える
（closure conversion で eqn の入力に持ち上がる）。

PRED は次のいずれか。
  - トレーサ: rank 0 の :i1 でなければならず（違えば COND-ERROR）、:cond の eqn
    を作る（StableHLO では stablehlo.if）。
  - T / NIL / rank 0 の bit 配列: 選ばれた枝をそのまま呼ぶ（eqn は作らない）。
    それ以外の PRED は COND-ERROR。
with-tracing の if は、トレーサの条件のとき COND* ではなく select に落ちる
（要素ごとの意味を保つため）。スカラー条件で片枝だけを評価したいときに COND*
を明示的に呼ぶ。

jvp / transpose（grad / vmap）のルールはまだ無く、COND* を通した grad は
NO-JVP-RULE（名前は :COND）になる。"
  (flet ((check-branch (branch what)
           (unless (typep branch 'traceable-function)
             (%cond-error "COND* の ~A は TRACEABLE-FUNCTION（WITH-TRACING で作る）でなければならない: ~S"
                          what branch))))
    (check-branch then-fn "then-fn")
    (check-branch else-fn "else-fn"))
  (cond
    ((typep pred 'tracer)
     (let ((aval (tracer-aval pred)))
       (unless (and (eq (aval-dtype aval) :i1) (null (aval-shape aval)))
         (%cond-error "COND* の pred は rank 0 の :i1 でなければならない: ~S" aval)))
     (let* ((operands (mapcar #'%cond-lift-operand operands))
            (avals (mapcar #'tracer-aval operands)))
       (multiple-value-bind (then-graph then-captured) (%trace-subgraph then-fn avals)
         (multiple-value-bind (else-graph else-captured) (%trace-subgraph else-fn avals)
           (let ((then-out (mapcar #'var-aval (graph-outvars then-graph)))
                 (else-out (mapcar #'var-aval (graph-outvars else-graph))))
             (unless (equalp then-out else-out)
               (%cond-error "COND* の両枝の出力の aval が一致しない: then ~S / else ~S"
                            then-out else-out))
             (when (null then-out)
               (%cond-error "COND* の枝は1つ以上の値を返さなければならない"))
             (values-list
              (%trace-eqn* :cond (append (list pred) operands then-captured else-captured)
                           :then then-graph :else else-graph
                           :num-operands (length operands))))))))
    ((or (eq pred t) (eq pred nil) (%cond-bit-pred-p pred))
     (apply (if (or (eq pred t) (and (arrayp pred) (= 1 (row-major-aref pred 0)))) then-fn else-fn)
            operands))
    (t (%cond-error "COND* の pred はトレーサ・T・NIL・rank 0 の bit 配列のいずれかでなければならない: ~S"
                    pred))))
