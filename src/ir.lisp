;;;; ir: nabla 自前の中間表現（var / eqn / graph）（issue #29 前半）。
;;;;
;;;; StableHLO は出力先であって内部表現ではない（CLAUDE.md）。トレース結果
;;;; はこの IR に変換し、grad / vmap はこの IR を書き換える変換として書く。

(in-package #:nabla)

(defstruct (var (:constructor make-var (aval)) (:copier nil) (:predicate var-p))
  "IR 上の値。同一性は EQ で判定する（名前は持たない。印字時の名前は
出現順に振る。u1b の PRINT-GRAPH を参照）。"
  (aval nil :type aval :read-only t))

(defstruct (eqn (:constructor %make-eqn (prim params invars outvars)) (:copier nil))
  "1つの演算の適用。PRIM は PRIMITIVE 構造体そのもの、PARAMS は
PRIMITIVE-PARAMS の宣言順に正規化した plist、INVARS / OUTVARS は VAR の
リスト。OUTVARS は、単一出力のプリミティブでは長さ1、
MULTIPLE-OUTPUTS-P が真のプリミティブ（契約 C1）では ABSTRACT-EVAL が返した
AVAL の個数。PARAMS の値には、サブグラフ（閉じた GRAPH。契約 C2）や、その
リストを持たせてよい。"
  (prim nil :type primitive :read-only t)
  (params nil :type list :read-only t)
  (invars nil :type list :read-only t)
  (outvars nil :type list :read-only t))

(defstruct (graph (:constructor make-graph (invars eqns outvars &optional constants)) (:copier nil))
  "トレース結果全体。INVARS / OUTVARS は VAR のリスト、EQNS は EQN の
リスト、CONSTANTS は ((var . simple-array) ...) のリスト。"
  (invars nil :type list :read-only t)
  (eqns nil :type list :read-only t)
  (outvars nil :type list :read-only t)
  (constants nil :type list :read-only t))

(defun %normalize-params (name declared-params params)
  "PARAMS（plist）のキー集合が DECLARED-PARAMS（宣言順のキーワードの
リスト）と一致すること（欠落も余分も無いこと）を確かめ、DECLARED-PARAMS の
順に並べ直した plist を返す。一致しなければ PRIMITIVE-ERROR を signal する。
PARAMS の要素数が奇数（plist として不正）のときも、GETF に渡す前に
PRIMITIVE-ERROR を signal する（奇数のまま渡すと SB-INT:SIMPLE-PROGRAM-ERROR
が漏れてしまうため）。"
  (unless (evenp (length params))
    (error 'primitive-error
           :name name
           :format-control "params の要素数が奇数で plist として不正（渡されたのは ~S）"
           :format-arguments (list params)))
  (let ((given-keys (loop for (k) on params by #'cddr collect k)))
    (if (and (= (length given-keys) (length declared-params))
             (null (set-difference given-keys declared-params))
             (null (set-difference declared-params given-keys)))
        (loop for key in declared-params
              collect key
              collect (getf params key))
        (error 'primitive-error
               :name name
               :format-control "params のキーが ~S と一致しない（渡されたのは ~S）"
               :format-arguments (list declared-params given-keys)))))

(defun make-eqn (prim-name invars &rest params)
  "PRIM-NAME（キーワード）・INVARS（VAR のリスト）・PARAMS から EQN を
作る。手順:

1. (FIND-PRIMITIVE PRIM-NAME) が NIL なら UNKNOWN-PRIMITIVE を signal する。
2. PARAMS の要素数が奇数（plist として不正）、またはキー集合が
   PRIMITIVE-PARAMS と一致しなければ PRIMITIVE-ERROR。

PARAMS は（&KEY ではなく）&REST で受け取る。&KEY &ALLOW-OTHER-KEYS にすると、
奇数個のキーワード引数は CL 自身の引数束縛の時点で
SB-INT:SIMPLE-PROGRAM-ERROR になってしまい、この関数の本体（step 2）に
たどり着く前に漏れてしまうため。
3. 宣言順に並べ直した plist を作る。
4. (APPLY ABSTRACT-EVAL (MAPCAR #'VAR-AVAL INVARS) PLIST) で出力 AVAL を
   計算し、AVAL 型であることを CHECK-TYPE で確かめる。
5. (MAKE-VAR OUT-AVAL) を1つ作り、EQN を返す。MULTIPLE-OUTPUTS-P が真の
   プリミティブでは、ABSTRACT-EVAL は AVAL のリストを返し、その要素ごとに
   VAR を作る（OUTVARS はその順）。

SSA（各 var はちょうど1回だけ定義される）は、この関数がその都度新しい
VAR を作ることによって構成的に保証される。"
  (let ((prim (or (find-primitive prim-name)
                  (error 'unknown-primitive :name prim-name))))
    (let* ((plist (%normalize-params prim-name (primitive-params prim) params))
           (in-avals (mapcar #'var-aval invars))
           (result (apply (primitive-abstract-eval prim) in-avals plist)))
      (if (primitive-multiple-outputs-p prim)
          (progn
            (unless (listp result)
              (error 'primitive-error :name prim-name :in-avals in-avals
                     :format-control "複数出力のプリミティブの abstract-eval は aval のリストを返さなければならない: ~S"
                     :format-arguments (list result)))
            (dolist (out-aval result) (check-type out-aval aval))
            (%make-eqn prim plist invars (mapcar #'make-var result)))
          (progn
            (check-type result aval)
            (%make-eqn prim plist invars (list (make-var result))))))))

(defun %param-subgraphs (params)
  "PARAMS（eqn の params の plist）の値のどこか（値そのもの、またはそのリストの
要素、入れ子のリストの要素）にある GRAPH（サブグラフ）を、出現順のリストにして返す。"
  (let ((found '()))
    (labels ((walk (form)
               (cond ((graph-p form) (push form found))
                     ((consp form) (walk (car form)) (walk (cdr form))))))
      (loop for (nil value) on params by #'cddr do (walk value)))
    (nreverse found)))

(defun %eqn-subgraphs (eqn)
  "EQN の params が持つサブグラフ（GRAPH）のリスト。無ければ NIL。"
  (%param-subgraphs (eqn-params eqn)))

(define-condition malformed-graph (error)
  ((graph :initarg :graph :reader malformed-graph-graph)
   (format-control :initarg :format-control :reader malformed-graph-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader malformed-graph-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "graph の構造が不正: ~?"
             (malformed-graph-format-control condition)
             (malformed-graph-format-arguments condition))))
  (:documentation
   "CHECK-GRAPH が graph の SSA 性・var の未定義参照・constants の aval
不一致を検出したときに signal する。constants の配列の要素型が var の dtype
と矛盾する場合（本来は DTYPE-MISMATCH になる場合）も、「graph の構造が
不正」という一貫した契約にするため、この条件に読み替えて signal する
（CHECK-GRAPH のドキュメント文字列 (c) を参照）。"))

(defun check-graph (graph)
  "GRAPH の構造上の不変量を検査し、問題が無ければ GRAPH をそのまま返す。
違反があれば MALFORMED-GRAPH を signal する。検査する内容:

(a) GRAPH-INVARS・GRAPH-CONSTANTS の var・各 eqn の outvars が、EQ で
    重複なく、ちょうど1回ずつ定義される。
(b) 各 eqn の invars と GRAPH-OUTVARS は、その時点までに定義済みの var
    だけを参照する。eqn の params が持つサブグラフも、それ自身が
    CHECK-GRAPH を通る（閉じた graph。外側の var は参照できない）。
(c) GRAPH-CONSTANTS の各 (var . array) について、ARRAY から作った AVAL
    が VAR-AVAL と EQUALP で一致する。ARRAY の要素型が VAR-AVAL の dtype と
    そもそも矛盾していて AVAL が作れない場合（ARRAY-AVAL が DTYPE-MISMATCH
    を signal する場合）も、この (c) の違反として MALFORMED-GRAPH に
    読み替える（呼び出し側は「graph の構造が壊れている」という1つの契約
    だけを気にすればよいようにするため）。"
  (let ((defined (make-hash-table :test 'eq))
        (invars (graph-invars graph))
        (constants (graph-constants graph))
        (eqns (graph-eqns graph)))
    (flet ((mark-defined (var)
             (when (gethash var defined)
               (error 'malformed-graph :graph graph
                      :format-control "var ~S が複数回定義されている" :format-arguments (list var)))
             (setf (gethash var defined) t))
           (check-defined (var)
             (unless (gethash var defined)
               (error 'malformed-graph :graph graph
                      :format-control "var ~S が未定義のまま参照されている" :format-arguments (list var)))))
      (dolist (v invars) (mark-defined v))
      (dolist (entry constants)
        (let ((var (car entry)) (array (cdr entry)))
          (mark-defined var)
          (let ((const-aval (handler-case (array-aval array (aval-dtype (var-aval var)))
                               (dtype-mismatch ()
                                 (error 'malformed-graph :graph graph
                                        :format-control "constant ~S の配列の要素型が var の dtype と矛盾する"
                                        :format-arguments (list var))))))
            (unless (equalp const-aval (var-aval var))
              (error 'malformed-graph :graph graph
                     :format-control "constant ~S の配列の aval が var の aval と一致しない"
                     :format-arguments (list var))))))
      (dolist (eqn eqns)
        (dolist (v (eqn-invars eqn)) (check-defined v))
        ;; サブグラフは閉じた graph なので、それ自身の不変量を再帰的に検査する
        ;; （外側の var への参照は、そもそも未定義参照として検出される）。
        (dolist (sub (%eqn-subgraphs eqn)) (check-graph sub))
        (dolist (v (eqn-outvars eqn)) (mark-defined v)))
      (dolist (v (graph-outvars graph)) (check-defined v))))
  graph)

(defmethod print-object ((var var) stream)
  (print-unreadable-object (var stream :type t)
    (format stream "~A[~{~D~^,~}]" (dtype-mlir-name (aval-dtype (var-aval var))) (aval-shape (var-aval var)))))

(defmethod print-object ((graph graph) stream)
  (print-unreadable-object (graph stream :type t)
    (format stream "~D in, ~D eqns, ~D out"
            (length (graph-invars graph)) (length (graph-eqns graph)) (length (graph-outvars graph)))))
