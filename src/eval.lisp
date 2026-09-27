;;;; eval: graph を各プリミティブの :EAGER で評価するインタプリタ（issue #39）。
;;;;
;;;; wave 3 のトレーサー PBT（#32: EVAL-GRAPH した graph = 元の関数）、
;;;; wave 3 の emitter の medium テスト（#33: 実行系の実行結果 = EVAL-GRAPH）、
;;;; wave 4 の jit（#34: jit(f)(x) = f(x)）はどれも、この EVAL-GRAPH を
;;;; 前提にしている。

(in-package #:nabla)

(define-condition graph-input-mismatch (error)
  ((graph :initarg :graph :reader graph-input-mismatch-graph)
   (expected :initarg :expected :reader graph-input-mismatch-expected)
   (actual :initarg :actual :reader graph-input-mismatch-actual))
  (:report
   (lambda (condition stream)
     (format stream "graph への入力が invars と一致しない: 期待する aval ~S、実際 ~S"
             (graph-input-mismatch-expected condition)
             (graph-input-mismatch-actual condition))))
  (:documentation
   "EVAL-GRAPH に渡した配列が GRAPH-INVARS と合わないときに signal される。
個数が合わないときは EXPECTED が GRAPH-INVARS の aval のリスト、ACTUAL が
渡された配列の個数（整数）になる。個数は合うが aval が食い違うときは、
EXPECTED は同じく GRAPH-INVARS の aval のリスト、ACTUAL は食い違った1つの
配列の aval を要素に持つ1要素のリスト（(ARRAY-AVAL ARRAY DTYPE) が
DTYPE-MISMATCH を signal した場合は NIL を要素に持つ）になる。読み手
（GRAPH-INPUT-MISMATCH-GRAPH 等）は export しない（このコンディションが
持つ情報の詳しい形は、まだ公開 API として固めていないため）。"))

(define-condition primitive-not-evaluable (error)
  ((name :initarg :name :reader primitive-not-evaluable-name))
  (:report
   (lambda (condition stream)
     (format stream "プリミティブ ~S は :EAGER を持たないため EVAL-GRAPH できない"
             (primitive-not-evaluable-name condition))))
  (:documentation
   "EVAL-GRAPH が、:EAGER を持たないプリミティブを使う EQN に当たったときに
signal される。NAME はそのプリミティブ名（キーワード）。"))

(defun %eval-graph-invar-actual-aval (invar array)
  "ARRAY の aval を (VAR-AVAL INVAR) の dtype で決めて返す。ARRAY の実際の
要素型と食い違って ARRAY-DTYPE が DTYPE-MISMATCH を signal したときは NIL
を返す（EVAL-GRAPH はこれを GRAPH-INPUT-MISMATCH に変換する）。"
  (handler-case (array-aval array (aval-dtype (var-aval invar)))
    (dtype-mismatch () nil)))

(defun %eval-graph-bind-invars (graph arrays env)
  "GRAPH-INVARS の個数・aval を ARRAYS と照らし合わせ、問題が無ければ ENV
（var → array の EQ ハッシュ表）に束縛する。"
  (let ((invars (graph-invars graph)))
    (unless (= (length arrays) (length invars))
      (error 'graph-input-mismatch :graph graph
             :expected (mapcar #'var-aval invars) :actual (length arrays)))
    (loop for invar in invars
          for array in arrays
          do (let ((actual (%eval-graph-invar-actual-aval invar array)))
               (unless (equalp (var-aval invar) actual)
                 (error 'graph-input-mismatch :graph graph
                        :expected (mapcar #'var-aval invars) :actual (list actual)))
               (setf (gethash invar env) array)))))

(defun %eval-graph-lookup (graph env var)
  "ENV から VAR に束縛された配列を返す。未束縛なら MALFORMED-GRAPH を
signal する（CHECK-GRAPH を通らなかった graph が use-before-def を含む
場合に、ここで初めて検出される）。"
  (multiple-value-bind (array presentp) (gethash var env)
    (unless presentp
      (error 'malformed-graph :graph graph
             :format-control "var ~S が未定義のまま参照されている"
             :format-arguments (list var)))
    array))

(defun %eval-graph-step (graph env eqn)
  "1つの EQN を :EAGER で評価し、その唯一の outvar を ENV に束縛する。"
  (let* ((prim (eqn-prim eqn))
         (eager (primitive-eager prim))
         (outvars (eqn-outvars eqn)))
    (unless eager
      (error 'primitive-not-evaluable :name (primitive-name prim)))
    (unless (= (length outvars) 1)
      (error "EVAL-GRAPH は複数（または0個の）outvars を持つ eqn を扱わない（フェーズ1）: ~S" eqn))
    (let* ((invars (eqn-invars eqn))
           (in-arrays (mapcar (lambda (v) (%eval-graph-lookup graph env v)) invars))
           (in-avals (mapcar #'var-aval invars))
           (out-var (first outvars))
           (out-aval (var-aval out-var))
           (result (apply eager in-arrays in-avals (eqn-params eqn)))
           ;; #31 が保証する「abstract-eval の出力 aval と eager の出力 aval
           ;; が一致する」性質を、graph 評価の中でも検査する（issue #39 の
           ;; 推奨する不変量。壊れたプリミティブの eager をここで検出する）。
           ;; ARRAY-AVAL 自身が（RESULT の要素型が OUT-AVAL の dtype と
           ;; 食い違って）DTYPE-MISMATCH を signal することがあるので、
           ;; それも「aval が一致しない」場合として PRIMITIVE-ERROR に
           ;; まとめる（生の DTYPE-MISMATCH を漏らさない）。
           (result-aval (handler-case (array-aval result (aval-dtype out-aval))
                          (dtype-mismatch () nil))))
      (unless (equalp result-aval out-aval)
        (error 'primitive-error :name (primitive-name prim) :in-avals in-avals
               :format-control "eager の結果の aval が out-aval ~S と一致しない"
               :format-arguments (list out-aval)))
      (setf (gethash out-var env) result))))

(defun eval-graph (graph &rest arrays)
  "GRAPH（GRAPH 構造体）を ARRAYS で eager 評価し、GRAPH-OUTVARS に対応する
配列を多値で返す（出力が無ければ (VALUES)。同じ var が複数回出力に
現れれば、同じ（EQ な）配列を複数回返す）。

手順:

1. (LENGTH ARRAYS) が (LENGTH (GRAPH-INVARS GRAPH)) と一致しなければ
   GRAPH-INPUT-MISMATCH を signal する。
2. 各 invar について (ARRAY-AVAL ARRAY (AVAL-DTYPE (VAR-AVAL INVAR))) が
   (VAR-AVAL INVAR) と EQUALP で一致しなければ GRAPH-INPUT-MISMATCH を
   signal する。
3. INVARS と GRAPH-CONSTANTS の var を、それぞれの配列に束縛する
   （コピーはしない。EAGER は入力を書き換えない前提）。
4. GRAPH-EQNS を順に評価する。(PRIMITIVE-EAGER (EQN-PRIM EQN)) が NIL なら
   PRIMITIVE-NOT-EVALUABLE を signal する。eqn の outvars がちょうど1つで
   なければ（フェーズ1では常に1つのはずなので）ERROR を signal する。
   invars を環境から引いて EAGER に渡し（未束縛の var があれば
   MALFORMED-GRAPH）、結果の aval が出力 var の aval と一致することを
   確かめてから（違えば PRIMITIVE-ERROR）、outvar に束縛する。
5. GRAPH-OUTVARS に対応する配列を多値で返す。

CHECK-GRAPH は呼ばない。EVAL-GRAPH は「渡された graph が構造的に正しい
（CHECK-GRAPH を通る）」ことを呼び出し側の責任とする——トレーサーは
構成的に SSA な graph しか作らないため、eqn ごとに毎回検査するのは無駄
になる。渡された graph が壊れていた場合の挙動（MALFORMED-GRAPH を
signal する、など）は保証するが、壊れた graph を検出しきることは保証
しない。"
  (let ((env (make-hash-table :test 'eq)))
    (%eval-graph-bind-invars graph arrays env)
    (dolist (entry (graph-constants graph))
      (setf (gethash (car entry) env) (cdr entry)))
    (dolist (eqn (graph-eqns graph))
      (%eval-graph-step graph env eqn))
    (values-list (mapcar (lambda (v) (%eval-graph-lookup graph env v)) (graph-outvars graph)))))
