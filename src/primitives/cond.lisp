;;;; primitives/cond: cond プリミティブ（issue #130）。
;;;;
;;;; 2つのサブグラフ（:THEN / :ELSE）を持つ高階プリミティブ（契約 C1 / C2）。
;;;; eqn の invars は「pred、operands（N 個）、then の捕捉値、else の捕捉値」の
;;;; 順に並ぶ（捕捉値は closure conversion で外側から持ち上げたトレーサ）。
;;;; then のサブグラフの invars は「operands、then の捕捉値」、else のサブグラフ
;;;; の invars は「operands、else の捕捉値」。パラメタ :NUM-OPERANDS が N。
;;;;
;;;; StableHLO は stablehlo.if（リージョンが2つ）。stablehlo.case は添え字が
;;;; i32 なので pred の変換が要る。実行系が stablehlo.if を受け付けることは
;;;; 実行系の medium テスト（cond-test） で確かめる。枝のリージョンは外側の SSA 値を直接
;;;; 参照できるので、枝の invars は外側の入力の名前に結びつける
;;;; （%STABLEHLO-REGION-LINES の :ARG-NAMES）。
;;;;
;;;; eager は pred を見て、選ばれた枝のサブグラフだけを EVAL-GRAPH する。
;;;; jvp / transpose ルールは issue #134。それまでは no-jvp-rule になる。
;;;; 公開 API（COND* と枝の検査）は src/cond.lisp。

(in-package #:nabla)

(defun %cond-split-args (items num-operands then-graph else-graph)
  "ITEMS（pred を除いた「operands、then の捕捉値、else の捕捉値」の並び）を、
(VALUES THEN-ARGS ELSE-ARGS) に分ける。THEN-ARGS は then のサブグラフの invars に、
ELSE-ARGS は else のサブグラフの invars に対応する。"
  (let* ((then-captured (- (length (graph-invars then-graph)) num-operands))
         (operands (subseq items 0 num-operands))
         (then-captures (subseq items num-operands (+ num-operands then-captured)))
         (else-captures (subseq items (+ num-operands then-captured))))
    (unless (= (length else-captures) (- (length (graph-invars else-graph)) num-operands))
      (error 'primitive-error :name :cond
             :format-control "入力の個数 ~D が operands ~D と枝の捕捉値の個数に合わない"
             :format-arguments (list (length items) num-operands)))
    (values (append operands then-captures) (append operands else-captures))))

(defun %cond-abstract-eval (in-avals &key then else num-operands)
  (unless (and (graph-p then) (graph-p else) (typep num-operands '(integer 0)))
    (error 'primitive-error :name :cond :in-avals in-avals
           :format-control ":THEN / :ELSE は graph、:NUM-OPERANDS は非負整数でなければならない"
           :format-arguments nil))
  (let ((pred (first in-avals)))
    (unless (and pred (eq (aval-dtype pred) :i1) (null (aval-shape pred)))
      (error 'primitive-error :name :cond :in-avals in-avals
             :format-control "pred は rank 0 の :i1 でなければならない: ~S"
             :format-arguments (list pred))))
  (multiple-value-bind (then-args else-args) (%cond-split-args (rest in-avals) num-operands then else)
    (flet ((check (graph args)
             (unless (equalp args (mapcar #'var-aval (graph-invars graph)))
               (error 'primitive-error :name :cond :in-avals in-avals
                      :format-control "入力の aval が枝の入力と一致しない: ~S / ~S"
                      :format-arguments (list args (mapcar #'var-aval (graph-invars graph)))))))
      (check then then-args)
      (check else else-args)))
  (let ((then-out (mapcar #'var-aval (graph-outvars then)))
        (else-out (mapcar #'var-aval (graph-outvars else))))
    (unless (and then-out (equalp then-out else-out))
      (error 'primitive-error :name :cond :in-avals in-avals
             :format-control "両枝の出力の aval が一致しない（または出力が無い）: ~S / ~S"
             :format-arguments (list then-out else-out)))
    then-out))

(defun %cond-emit (in-names in-avals out-names out-avals &key then else num-operands)
  (declare (ignore in-avals))
  (multiple-value-bind (then-names else-names)
      (%cond-split-args (rest in-names) num-operands then else)
    (format nil "~{~A~^, ~} = \"stablehlo.if\"(~A) ({~%~{  ~A~^~%~}~%}, {~%~{  ~A~^~%~}~%}) : (tensor<i1>) -> (~{~A~^, ~})"
            out-names
            (first in-names)
            (%stablehlo-region-lines then :arg-names then-names)
            (%stablehlo-region-lines else :arg-names else-names)
            (mapcar #'tensor-type-string out-avals))))

(defun %cond-eager (arrays in-avals &key then else num-operands)
  (declare (ignore in-avals))
  (multiple-value-bind (then-args else-args) (%cond-split-args (rest arrays) num-operands then else)
    (if (= 1 (row-major-aref (first arrays) 0))
        (multiple-value-list (apply #'eval-graph then then-args))
        (multiple-value-list (apply #'eval-graph else else-args)))))

(defprimitive cond (:then :else :num-operands)
  :multiple-outputs t
  :abstract-eval #'%cond-abstract-eval
  :emit #'%cond-emit
  :eager #'%cond-eager)
