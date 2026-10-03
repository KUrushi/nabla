;;;; primitives/cond: cond プリミティブ（issue #130）。
;;;;
;;;; 2つのサブグラフ（:THEN / :ELSE）を持つ高階プリミティブ（契約 C1 / C2）。
;;;; JAX と同じく、両方の枝は「同一の入力シグネチャ」を持つ: eqn の invars は
;;;; 「pred、operands、captures（両枝の捕捉値の和集合）」の順で、:THEN と :ELSE の
;;;; サブグラフの invars はどちらも「operands、captures」と同じ aval の並び。枝が
;;;; 使わない捕捉値の位置には使われない invar が置かれる（%COND-UNIFY-BRANCHES、
;;;; src/cond.lisp）。jvp（#134）や vmap（#140）が枝を同じ形で書き換えられる。
;;;;
;;;; StableHLO は stablehlo.if（リージョンが2つ）。stablehlo.case は添え字が
;;;; i32 なので pred の変換が要る。実行系が stablehlo.if を受け付けることは
;;;; 実行系の medium テスト（cond-test）で確かめる。枝のリージョンは外側の SSA 値を直接
;;;; 参照できるので、枝の invars は外側の入力の名前に結びつける
;;;; （%STABLEHLO-REGION-LINES の :ARG-NAMES）。
;;;;
;;;; eager は pred を見て、選ばれた枝のサブグラフだけを EVAL-GRAPH する。
;;;; jvp / transpose ルールは src/ad/rules-control.lisp（issue #134）。
;;;; 公開 API（COND* と枝の検査）は src/cond.lisp。

(in-package #:nabla)

(defun %cond-abstract-eval (in-avals &key then else)
  (unless (and (graph-p then) (graph-p else))
    (error 'primitive-error :name :cond :in-avals in-avals
           :format-control ":THEN / :ELSE は graph でなければならない"
           :format-arguments nil))
  (let ((pred (first in-avals)))
    (unless (and pred (eq (aval-dtype pred) :i1) (null (aval-shape pred)))
      (error 'primitive-error :name :cond :in-avals in-avals
             :format-control "pred は rank 0 の :i1 でなければならない: ~S"
             :format-arguments (list pred))))
  (dolist (graph (list then else))
    (unless (equalp (rest in-avals) (mapcar #'var-aval (graph-invars graph)))
      (error 'primitive-error :name :cond :in-avals in-avals
             :format-control "入力の aval が枝の入力と一致しない（両枝は同じ入力シグネチャを持つ）: ~S / ~S"
             :format-arguments (list (rest in-avals) (mapcar #'var-aval (graph-invars graph))))))
  (let ((then-out (mapcar #'var-aval (graph-outvars then)))
        (else-out (mapcar #'var-aval (graph-outvars else))))
    (unless (and then-out (equalp then-out else-out))
      (error 'primitive-error :name :cond :in-avals in-avals
             :format-control "両枝の出力の aval が一致しない（または出力が無い）: ~S / ~S"
             :format-arguments (list then-out else-out)))
    then-out))

(defun %cond-indent (lines)
  "LINES（複数行の文字列を含みうる）の全ての行を2文字インデントした、改行区切りの文字列。"
  (format nil "~{  ~A~^~%~}"
          (loop for line in lines
                append (loop for start = 0 then (1+ pos)
                             for pos = (position #\Newline line :start start)
                             collect (subseq line start pos)
                             while pos))))

(defun %cond-emit (in-names in-avals out-names out-avals &key then else)
  (declare (ignore in-avals))
  (format nil "~{~A~^, ~} = \"stablehlo.if\"(~A) ({~%~A~%}, {~%~A~%}) : (tensor<i1>) -> (~{~A~^, ~})"
          out-names
          (first in-names)
          (%cond-indent (%stablehlo-region-lines then :arg-names (rest in-names)))
          (%cond-indent (%stablehlo-region-lines else :arg-names (rest in-names)))
          (mapcar #'tensor-type-string out-avals)))

(defun %cond-eager (arrays in-avals &key then else)
  (declare (ignore in-avals))
  (multiple-value-list
   (apply #'eval-graph (if (= 1 (row-major-aref (first arrays) 0)) then else) (rest arrays))))

(defprimitive cond (:then :else)
  :multiple-outputs t
  ;; #'関数 を直接渡すと、関数オブジェクトがロード時に固定されて再定義
  ;; （mutation testing）が効かなくなるので、名前で呼ぶ lambda で包む。
  :abstract-eval (lambda (in-avals &rest params) (apply #'%cond-abstract-eval in-avals params))
  :emit (lambda (in-names in-avals out-names out-avals &rest params)
          (apply #'%cond-emit in-names in-avals out-names out-avals params))
  :eager (lambda (arrays in-avals &rest params) (apply #'%cond-eager arrays in-avals params)))
