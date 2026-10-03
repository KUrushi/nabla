;;;; ad/partial-eval: 複数出力の高階プリミティブ（scan など）の eqn を、既知の部分
;;;; （主値だけで決まる）と未知の部分（接線に依存する）の2つの eqn に分ける partial eval の
;;;; フック（issue #139）。
;;;;
;;;; LINEARIZE-GRAPH は jvp 変換した graph を「接線の入力に推移的に依存するか」で主値側と
;;;; 線形側に分ける。scan の jvp は主値と接線を1つの eqn で計算するので、そのままだと
;;;; eqn が丸ごと線形側に入ってしまう。そこで分ける前に、プリミティブごとの partial eval
;;;; ルールで、その eqn を「主値だけを入力にとる eqn」と「接線に依存する eqn」に置き換える。
;;;; JAX の pe.custom_partial_eval_rules（_scan_partial_eval など）に相当する。
;;;;
;;;; ルールは (eqn unknown-flags) を受け、(VALUES 置き換えの eqn のリスト 追加の定数) を返す。
;;;; UNKNOWN-FLAGS は EQN-INVARS ごとの「接線に依存するか」。置き換えの eqn は元の eqn の
;;;; outvars（VAR の同一性）をそのまま使い、定義の順に並べる。NIL を返すと分けない。
;;;; 追加の定数は ((var . 配列) ...) で、graph-constants に足される。

(in-package #:nabla)

(defvar *partial-eval-rules* (make-hash-table :test 'eq)
  "プリミティブ名（キーワード）から partial eval ルールへの表。")

(defun set-partial-eval-rule (name function)
  "プリミティブ NAME の partial eval ルールを FUNCTION にする。"
  (setf (gethash name *partial-eval-rules*) function))

(defun %partial-eval-split (eqns seeds)
  "EQNS（定義の順の EQN のリスト）を SEEDS（接線の入力の VAR のリスト）への依存で見て、
SEEDS に依存する入力を持ち partial eval ルールのある eqn をルールで置き換えたリストを作る。
(VALUES 置き換え後の eqn のリスト（順序を保つ） 追加の定数 依存する VAR の表) を返す。
表は SEEDS と、依存する eqn の出力を T にした EQ のハッシュ表。EQNS は書き換えない。"
  (let ((dependent (make-hash-table :test 'eq))
        (ordered '())
        (constants '()))
    (dolist (v seeds) (setf (gethash v dependent) t))
    (labels ((unknown-flags (eqn) (mapcar (lambda (v) (and (gethash v dependent) t)) (eqn-invars eqn)))
             (add (eqn)
               (when (some (lambda (v) (gethash v dependent)) (eqn-invars eqn))
                 (dolist (v (eqn-outvars eqn)) (setf (gethash v dependent) t)))
               (push eqn ordered)))
      (dolist (eqn eqns)
        (let* ((flags (unknown-flags eqn))
               (rule (and (some #'identity flags)
                          (gethash (primitive-name (eqn-prim eqn)) *partial-eval-rules*))))
          (multiple-value-bind (pieces extra) (and rule (funcall rule eqn flags))
            (cond (pieces (setf constants (append constants extra))
                          (mapc #'add pieces))
                  (t (add eqn)))))))
    (values (nreverse ordered) constants dependent)))
