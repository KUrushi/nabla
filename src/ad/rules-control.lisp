;;;; ad/rules-control: 制御構造（while-loop。cond は後続）の jvp ルール（issue #134）。
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
