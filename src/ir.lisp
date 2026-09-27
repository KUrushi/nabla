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
リスト。フェーズ1では OUTVARS は常に長さ1。"
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
順に並べ直した plist を返す。一致しなければ PRIMITIVE-ERROR を signal する。"
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

(defun make-eqn (prim-name invars &rest params &key &allow-other-keys)
  "PRIM-NAME（キーワード）・INVARS（VAR のリスト）・PARAMS から EQN を
作る。手順:

1. (FIND-PRIMITIVE PRIM-NAME) が NIL なら UNKNOWN-PRIMITIVE を signal する。
2. PARAMS のキー集合が PRIMITIVE-PARAMS と一致しなければ PRIMITIVE-ERROR。
3. 宣言順に並べ直した plist を作る。
4. (APPLY ABSTRACT-EVAL (MAPCAR #'VAR-AVAL INVARS) PLIST) で出力 AVAL を
   計算し、AVAL 型であることを CHECK-TYPE で確かめる。
5. (MAKE-VAR OUT-AVAL) を1つ作り、EQN を返す。

SSA（各 var はちょうど1回だけ定義される）は、この関数がその都度新しい
VAR を作ることによって構成的に保証される。"
  (let ((prim (or (find-primitive prim-name)
                  (error 'unknown-primitive :name prim-name))))
    (let* ((plist (%normalize-params prim-name (primitive-params prim) params))
           (in-avals (mapcar #'var-aval invars))
           (out-aval (apply (primitive-abstract-eval prim) in-avals plist)))
      (check-type out-aval aval)
      (%make-eqn prim plist invars (list (make-var out-aval))))))

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
不一致を検出したときに signal する。"))

(defun check-graph (graph)
  "GRAPH の構造上の不変量を検査し、問題が無ければ GRAPH をそのまま返す。
違反があれば MALFORMED-GRAPH を signal する。検査する内容:

(a) GRAPH-INVARS・GRAPH-CONSTANTS の var・各 eqn の outvars が、EQ で
    重複なく、ちょうど1回ずつ定義される。
(b) 各 eqn の invars と GRAPH-OUTVARS は、その時点までに定義済みの var
    だけを参照する。
(c) GRAPH-CONSTANTS の各 (var . array) について、ARRAY から作った AVAL
    が VAR-AVAL と EQUALP で一致する。"
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
          (unless (equalp (array-aval array (aval-dtype (var-aval var))) (var-aval var))
            (error 'malformed-graph :graph graph
                   :format-control "constant ~S の配列の aval が var の aval と一致しない"
                   :format-arguments (list var)))))
      (dolist (eqn eqns)
        (dolist (v (eqn-invars eqn)) (check-defined v))
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
