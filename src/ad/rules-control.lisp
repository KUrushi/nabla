;;;; ad/rules-control: 制御構造（while-loop と cond）の jvp ルールと、cond の transpose ルール（issue #134）。
;;;;
;;;; JAX の jax._src.lax.control_flow.loops._while_loop_jvp の写し。複数出力の
;;;; プリミティブの規約（契約 C1。src/ad/jvp.lisp）に従い、ルールは
;;;;   (lambda (primals tangents &key cond body n-carries) ...)
;;;;     → (VALUES 主値の出力のリスト 接線のリスト)
;;;; で、主値の eqn も自分で足す。

(in-package #:nabla)

;;; ---- while-loop ----

(defun %while-jvp-body (body nonzero)
  "BODY を NONZERO（全オペランドの入力の接線が非ゼロか）で jvp 変換する。
(VALUES JVP-GRAPH OUT-NONZERO) を返す。JVP-GRAPH の入力は主値の全オペランド +
NONZERO が真のものの接線、出力は主値の全出力 + 全出力の接線（ゼロは実体化）。"
  (%jvp-graph-with-out-nonzero body nonzero))

(defun %while-jvp-fixpoint (body carry-nonzero const-nonzero)
  "carry の接線が非ゼロかどうかの不動点を求める。最初は接線がゼロの carry も、本体を
通ると非ゼロになりうる（JAX と同じ）ので、本体の jvp の出力の接線が非ゼロの carry を
足して集合が変わらなくなるまで繰り返す。集合は単調に増えるので有限回で止まる。
(VALUES 不動点の carry の非ゼロのリスト JVP-GRAPH OUT-NONZERO) を返す。"
  (loop
    (multiple-value-bind (jvp out-nonzero)
        (%while-jvp-body body (append carry-nonzero const-nonzero))
      (let ((next (loop for flag in carry-nonzero
                        for out in out-nonzero
                        collect (or flag out))))
        (when (equal next carry-nonzero)
          (return (values carry-nonzero jvp out-nonzero)))
        (setf carry-nonzero next)))))

(defun %while-jvp-pick (list flags)
  "LIST の要素のうち、FLAGS が真の位置のものだけを返す。"
  (loop for x in list for flag in flags when flag collect x))

(defun %while-loop-jvp (primals tangents &key cond body n-carries)
  "while-loop の jvp ルール。非ゼロの接線を持つ carry に、接線の carry を足した
while-loop にする。loop 不変のオペランド（n-carries 番目以降）の接線は、非ゼロのものだけを
loop 不変のオペランドとして足す（本体は素通しのまま。EQ の素通しの約束を守る）。
接線はすべて、本体の jvp 変換の中で、その被演算子について線形なプリミティブにしか
流れない。"
  (let* ((n n-carries)
         (carries (subseq primals 0 n))
         (consts (nthcdr n primals))
         (carry-tangents (subseq tangents 0 n))
         (const-tangents (nthcdr n tangents))
         (const-nonzero (mapcar (lambda (tg) (not (symbolic-zero-p tg))) const-tangents)))
    (multiple-value-bind (carry-nonzero jvp)
        (%while-jvp-fixpoint body
                             (mapcar (lambda (tg) (not (symbolic-zero-p tg))) carry-tangents)
                             const-nonzero)
      (let* ((m (length consts))
             (nt (count t carry-nonzero))
             (jvp-invars (graph-invars jvp))
             (jvp-outvars (graph-outvars jvp))
             ;; JVP の入力: [carry 主値 n][const 主値 m][carry 接線 nt][const 接線]
             ;; 新しい並び: [carry 主値 n][carry 接線 nt][const 主値 m][const 接線]
             (p-carry (subseq jvp-invars 0 n))
             (p-const (subseq jvp-invars n (+ n m)))
             (t-carry (subseq jvp-invars (+ n m) (+ n m nt)))
             (t-const (nthcdr (+ n m nt) jvp-invars))
             ;; JVP の出力: [carry 主値 n][const 主値 m][全 carry 接線 n][全 const 接線 m]
             (o-carry (subseq jvp-outvars 0 n))
             (o-const (subseq jvp-outvars n (+ n m)))
             (o-t-carry (%while-jvp-pick (subseq jvp-outvars (+ n m) (+ n m n)) carry-nonzero))
             (o-t-const (%while-jvp-pick (nthcdr (+ n m n) jvp-outvars) const-nonzero))
             (new-body (make-graph (append p-carry t-carry p-const t-const)
                                   (graph-eqns jvp)
                                   (append o-carry o-t-carry o-const o-t-const)
                                   (graph-constants jvp)))
             ;; cond は接線を使わないので、接線の入力は未使用の入力として足す。
             (cond-invars (graph-invars cond))
             (new-cond (make-graph (append (subseq cond-invars 0 n)
                                           (mapcar (lambda (v) (make-var (var-aval v))) t-carry)
                                           (nthcdr n cond-invars)
                                           (mapcar (lambda (v) (make-var (var-aval v))) t-const))
                                   (graph-eqns cond)
                                   (graph-outvars cond)
                                   (graph-constants cond)))
             (operands (append carries
                               (mapcar #'instantiate-zero (%while-jvp-pick carry-tangents carry-nonzero))
                               consts
                               (mapcar #'instantiate-zero (%while-jvp-pick const-tangents const-nonzero))))
             (outs (%trace-eqn* :while-loop operands
                                :cond new-cond :body new-body :n-carries (+ n nt)))
             (out-carries (subseq outs 0 n))
             (out-t-carries (subseq outs n (+ n nt)))
             (out-consts (subseq outs (+ n nt) (+ n nt m)))
             (tangent-iter out-t-carries))
        (values (append out-carries out-consts)
                (append (loop for flag in carry-nonzero
                              for primal in carries
                              collect (if flag
                                          (pop tangent-iter)
                                          (make-symbolic-zero (tracer-aval primal))))
                        ;; 素通しの出力の接線は入力の接線そのもの。
                        const-tangents))))))

;; 名前で呼ぶ lambda で包む（関数オブジェクトを直接渡すと、再定義した %WHILE-LOOP-JVP が
;; 反映されず、mutation testing の変異が効かない）。
(set-jvp-rule :while-loop (lambda (primals tangents &rest params)
                            (apply '%while-loop-jvp primals tangents params)))
;;; ---- cond ----
;;;
;;; JAX の _cond_jvp / _cond_transpose / _cond_partial_eval の写し。eqn の invars は
;;; pred ++ operands ++ captures、:THEN / :ELSE の invars はどちらも operands ++ captures
;;; （src/primitives/cond.lisp）。
;;;
;;; jvp ルールは、主値と接線を1つの cond に混ぜずに、2つの eqn を出す:
;;;   主値の cond  : pred, 主値 → 主値の出力 ++ 残差（各枝の残差を並べ、他方の枝の分はゼロ）
;;;   線形な cond  : pred, 残差, 接線 → 接線の出力（枝は接線について線形）
;;; 枝ごとに jvp 変換して LINEARIZE の分割（%LINEARIZE-JVP-GRAPH）で主値の部分と線形な部分に
;;; 分けるので、linearize が接線の入力に依存する eqn（線形な cond）だけを線形側に置けて、
;;; その cond を transpose できる（逆モード）。JAX が partial eval で行う分割を、静的な
;;; graph なので jvp の時点で済ませている。

(defun %cond-pick (list flags)
  (loop for x in list for flag in flags when flag collect x))

(defun %cond-rebuild (in-avals fn)
  "IN-AVALS を入力に、FN（トレーサを受けて出力のトレーサを多値で返す）をトレースした閉じた graph。"
  (%call-with-fresh-trace in-avals fn))

(defun %cond-primal-branch (linearization m offset residual-avals)
  "枝の LINEARIZATION から、入力が operands ++ captures、出力が主値 M 個 ++ RESIDUAL-AVALS
の全残差（この枝の残差は OFFSET から並ぶ位置、他の位置はゼロ）の graph を作る。"
  (let ((primal (linearization-primal-graph linearization)))
    (%cond-rebuild (mapcar #'var-aval (graph-invars primal))
                   (lambda (&rest tracers)
                     (let* ((results (inline-graph primal tracers))
                            (own (nthcdr m results))
                            (slots (loop for aval in residual-avals
                                         for i from 0
                                         collect (if (and (<= offset i) (< i (+ offset (length own))))
                                                     (nth (- i offset) own)
                                                     (instantiate-zero (make-symbolic-zero aval))))))
                       (values-list (append (subseq results 0 m) slots)))))))

(defun %cond-linear-branch (linearization offset residual-avals)
  "枝の LINEARIZATION から、入力が全残差 RESIDUAL-AVALS ++ 接線、出力が接線の graph を作る
（この枝の残差は OFFSET から並ぶ位置で、他の位置の残差は使わない）。"
  (let* ((linear (linearization-linear-graph linearization))
         (n-own (linearization-n-residuals linearization))
         (tangent-avals (mapcar #'var-aval (nthcdr n-own (graph-invars linear)))))
    (%cond-rebuild (append residual-avals tangent-avals)
                   (lambda (&rest tracers)
                     (let ((own (subseq tracers offset (+ offset n-own)))
                           (tangents (nthcdr (length residual-avals) tracers)))
                       (values-list (inline-graph linear (append own tangents))))))))

(defun %cond-jvp (primals tangents &key then else)
  "cond の jvp ルール（ファイル冒頭の説明）。"
  (let* ((pred (first primals))
         (operands (rest primals))
         (operand-tangents (rest tangents))
         (nonzero (mapcar (lambda (tg) (not (symbolic-zero-p tg))) operand-tangents))
         (m (length (graph-outvars then))))
    (multiple-value-bind (jvp-then then-nz) (%jvp-graph-with-out-nonzero then nonzero)
      (multiple-value-bind (jvp-else else-nz) (%jvp-graph-with-out-nonzero else nonzero)
        (let ((out-nonzero (mapcar (lambda (a b) (or a b)) then-nz else-nz)))
          (if (notany #'identity out-nonzero)
              ;; どの出力にも接線が流れない: 主値の cond だけ。
              (let ((outs (%trace-eqn* :cond primals :then then :else else)))
                (values outs (mapcar (lambda (o) (make-symbolic-zero (tracer-aval o))) outs)))
              (flet ((split (jvp)
                       ;; 接線の出力は out-nonzero の位置だけ（ゼロの接線は jvp-graph が
                       ;; 実体化済みなので、枝の間で aval が揃う）。
                       (%linearize-jvp-graph
                        (make-graph (graph-invars jvp) (graph-eqns jvp)
                                    (append (subseq (graph-outvars jvp) 0 m)
                                            (%cond-pick (nthcdr m (graph-outvars jvp)) out-nonzero))
                                    (graph-constants jvp))
                        (length operands) m)))
                (let* ((lin-then (split jvp-then))
                       (lin-else (split jvp-else))
                       (n-then (linearization-n-residuals lin-then))
                       (residual-avals
                         (flet ((avals (lin)
                                  (let ((linear (linearization-linear-graph lin)))
                                    (mapcar #'var-aval
                                            (subseq (graph-invars linear) 0 (linearization-n-residuals lin))))))
                           (append (avals lin-then) (avals lin-else))))
                       (primal-outs (%trace-eqn* :cond primals
                                                 :then (%cond-primal-branch lin-then m 0 residual-avals)
                                                 :else (%cond-primal-branch lin-else m n-then residual-avals)))
                       (residuals (nthcdr m primal-outs))
                       (tangent-outs (%trace-eqn* :cond
                                                  (append (list pred) residuals
                                                          (mapcar #'instantiate-zero
                                                                  (%cond-pick operand-tangents nonzero)))
                                                  :then (%cond-linear-branch lin-then 0 residual-avals)
                                                  :else (%cond-linear-branch lin-else n-then residual-avals)))
                       (iter tangent-outs))
                  (values (subseq primal-outs 0 m)
                          (loop for flag in out-nonzero
                                for out in primal-outs
                                collect (if flag
                                            (pop iter)
                                            (make-symbolic-zero (tracer-aval out)))))))))))))

(set-jvp-rule :cond (lambda (primals tangents &rest params)
                      (apply '%cond-jvp primals tangents params)))

(defun %cond-transpose (ct invars &key then else)
  "cond の transpose ルール。線形な入力（UNDEFINED-PRIMAL）を持つ cond の各枝を
TRANSPOSE-GRAPH で転置した枝を持つ cond にする。既知の入力（残差）はそのまま使い、
出力の余接線を枝の入力に足す。pred は既知でなければならない。"
  (let* ((pred (first invars))
         (operands (rest invars))
         (linear (mapcar #'undefined-primal-p operands)))
    (when (undefined-primal-p pred)
      (error 'autodiff-error
             :format-control "cond の pred が線形な入力になっている（pred は既知でなければならない）"))
    (let* ((known (%cond-pick operands (mapcar #'not linear)))
           (n-known (length known))
           (cts (mapcar #'instantiate-zero ct)))
      (flet ((transpose-branch (graph)
               (let* ((vars (graph-invars graph))
                      (reordered (append (%cond-pick vars (mapcar #'not linear))
                                         (%cond-pick vars linear))))
                 (transpose-graph (make-graph reordered (graph-eqns graph)
                                              (graph-outvars graph) (graph-constants graph))
                                  n-known))))
        (let* ((results (%trace-eqn* :cond (append (list pred) known cts)
                                     :then (transpose-branch then) :else (transpose-branch else)))
               (iter results))
          (cons nil (loop for flag in linear
                          collect (and flag (pop iter)))))))))

(set-transpose-rule :cond (lambda (ct invars &rest params)
                            (apply '%cond-transpose ct invars params)))
