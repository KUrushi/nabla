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

(defun %cond-passthrough-name (graph var operand-names)
  "GRAPH（枝のサブグラフ）の出力 VAR が枝の入力そのもの（素通し）なら、その入力に結びつけた
外側の SSA 名（OPERAND-NAMES の対応する要素）。そうでなければ NIL。"
  (let ((position (position var (graph-invars graph))))
    (and position (nth position operand-names))))

(defun %cond-derived-name (kind out-name)
  "出力 OUT-NAME（\"%7\" など）から作る補助の SSA 名 \"%cond_<KIND>_7\"。MLIR の SSA 名は数字だけ
（%7）か英字・_ で始まる名前で、\"%7_if\" のように数字で始めて文字を続けることはできない。"
  (format nil "%cond_~A_~A" kind (subseq out-name 1)))

(defun %cond-branch-region (graph operand-names keep dummies)
  "枝 GRAPH のリージョンの行のリスト。stablehlo.return は、GRAPH の出力のうち KEEP
（出力の位置のリスト）のものだけを返す。DUMMIES は (位置 . 名前) のリストで、その位置の値の
代わりに、同じ名前の 0 の定数を返す（リージョンの中で定義する）。KEEP の各位置は
DUMMIES に現れるか、GRAPH の出力そのものを返すか（キーが無ければ）のどちらか。"
  (let* ((lines (%stablehlo-region-lines graph :arg-names operand-names))
         (body (butlast lines))
         (names (%scan-return-names (car (last lines))))
         (avals (mapcar #'var-aval (graph-outvars graph))))
    (append body
            (loop for (position . name) in dummies
                  collect (%scan-zero-constant-line name (nth position avals)))
            (list (format nil "stablehlo.return ~{~A~^, ~} : ~{~A~^, ~}"
                          (loop for position in keep
                                collect (let ((dummy (assoc position dummies)))
                                          (if dummy (cdr dummy) (nth position names))))
                          (loop for position in keep
                                collect (tensor-type-string (nth position avals))))))))

(defun %cond-emit (in-names in-avals out-names out-avals &key then else)
  "stablehlo.if を出す。枝が入力をそのまま返す出力（素通し）は、stablehlo.if の結果にせず、
if の外の stablehlo.select(pred, then 側, else 側) にする（issue #226）。

特定の版のバックエンド（記録は docs/ の上流バグの表の 8 と、docs/ の repros/ にある
if-passthrough-operands*.mlir）は、片方の枝が入力を stablehlo.return する出力を持つ if を、もう片方の
枝を選んだ実行時に \"ref is null\" で落とすことがある（出力の並びによる）。素通しを
optimization_barrier に通しても直らない。入力のコピーを枝の中で作る（0 を足すなど）と通るが、
dtype ごとに違う演算が要り、浮動小数点の -0.0 を変えないよう気を付ける必要がある。select は dtype に
依らず、pred が実際の値なので畳み込まれない。出力の種類（素通しは枝の入力に結びつけた外側の名前）:
  - どちらの枝も計算: if の結果のまま
  - 片方だけ素通し: if は計算する側の値を返し（素通しの側は 0 の定数 %cond_z_NAME を返す）、
    その結果 %cond_if_NAME と素通しの入力を select する
  - 両方素通し: if の結果から外し、2つの入力を select する
結果が1つも残らなければ stablehlo.if 自体を出さない。"
  (declare (ignore in-avals))
  (let* ((pred (first in-names))
         (operands (rest in-names))
         (then-passes (mapcar (lambda (v) (%cond-passthrough-name then v operands)) (graph-outvars then)))
         (else-passes (mapcar (lambda (v) (%cond-passthrough-name else v operands)) (graph-outvars else)))
         (if-names '()) (if-avals '())
         (then-keep '()) (else-keep '())
         (then-dummies '()) (else-dummies '())
         (selects '()))
    (loop for position from 0
          for out in out-names for aval in out-avals
          for then-pass in then-passes for else-pass in else-passes
          for type = (tensor-type-string aval)
          do (flet ((select (on-true on-false)
                      (push (format nil "~A = stablehlo.select ~A, ~A, ~A : (tensor<i1>, ~A, ~A) -> ~A"
                                    out pred on-true on-false type type type)
                            selects))
                    (keep (temp)
                      (push temp if-names)
                      (push aval if-avals)
                      (push position then-keep)
                      (push position else-keep)))
               (cond
                 ((and (null then-pass) (null else-pass))
                  (keep out))
                 ((and then-pass else-pass)
                  (select then-pass else-pass))
                 (then-pass
                  (let ((temp (%cond-derived-name "if" out)))
                    (keep temp)
                    (push (cons position (%cond-derived-name "z" out)) then-dummies)
                    (select then-pass temp)))
                 (t
                  (let ((temp (%cond-derived-name "if" out)))
                    (keep temp)
                    (push (cons position (%cond-derived-name "z" out)) else-dummies)
                    (select temp else-pass))))))
    (let ((if-text
            (when if-names
              (format nil "~{~A~^, ~} = \"stablehlo.if\"(~A) ({~%~A~%}, {~%~A~%}) : (tensor<i1>) -> (~{~A~^, ~})"
                      (reverse if-names)
                      pred
                      (%cond-indent (%cond-branch-region then operands (reverse then-keep) then-dummies))
                      (%cond-indent (%cond-branch-region else operands (reverse else-keep) else-dummies))
                      (mapcar #'tensor-type-string (reverse if-avals))))))
      (format nil "~{~A~^~%~}" (append (and if-text (list if-text)) (reverse selects))))))

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
