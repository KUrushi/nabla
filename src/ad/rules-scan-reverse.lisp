;;;; ad/rules-scan-reverse: scan の逆モード微分（issue #139）。partial eval と transpose。
;;;;
;;;; JAX の jax._src.lax.control_flow.loops._scan_partial_eval / _scan_transpose に対応する。
;;;;
;;;; partial eval（LINEARIZE-GRAPH が jvp した scan の eqn を分けるときに呼ばれる。
;;;; partial-eval.lisp）: 本体を「接線に依存する（未知）」入力で分け、
;;;;   - 既知の scan: 主値だけを計算する。各ステップの残差を ys として積んで出す
;;;;   - 未知の scan: 接線について線形。残差を xs（ステップごと）と consts（ループ不変）で受ける
;;;; に置き換える。未知の carry の集合は、本体を通ると増えうるので、増えなくなる
;;;; （不動点）まで広げる。ループ不変な残差（consts と本体の定数だけで決まる値）は
;;;; 積まず、scan の外で（必要なら計算して）consts として渡す。
;;;; 閉包で捕まえた配列は scan の consts ではなく本体の graph-constants なので、
;;;; ループ不変として扱う（残差にせず、未知の本体にも定数として持たせる）。
;;;; 未知の scan の並びは JAX と同じ: consts = ループ不変な残差 ++ 未知の consts、
;;;; carry = 未知の carry、xs = 未知の xs ++ 積んだ残差。
;;;;
;;;; transpose: 線形な scan を reverse を反転した scan にする。carry の余接線は carry、
;;;; xs の余接線は ys、consts の余接線は carry に足し込むアキュムレータ（ステップの総和）。
;;;; 本体は TRANSPOSE-GRAPH で転置する。ys の余接線が symbolic zero なら xs に積まず、
;;;; 本体の中でゼロを作る。主値と接線が混ざった scan（jvp ルールの出力そのもの）は
;;;; 線形ではないので、carry が線形入力に依存しなければ AUTODIFF-ERROR にする。

(in-package #:nabla)

;;; ---- partial eval ----

(defun %scan-pe-hoist (needed-vars known-eqns invariant-p env constants)
  "ループ不変な既知の var NEEDED-VARS を、scan の外で計算し直す。ENV は本体の var から
外側の var への表（本体の consts の入力は登録済み。NEEDED-VARS の外側の var を足す）。
本体の定数は新しい外側の定数にする。(VALUES 外側の eqn のリスト 追加の定数) を返す。"
  (let ((live (make-hash-table :test 'eq))
        (chosen '())
        (hoisted '())
        (extra '()))
    (dolist (v needed-vars) (setf (gethash v live) t))
    (dolist (eqn (reverse known-eqns))
      (when (and (some (lambda (v) (gethash v live)) (eqn-outvars eqn))
                 (every invariant-p (eqn-invars eqn)))
        (push eqn chosen)
        (dolist (v (eqn-invars eqn)) (setf (gethash v live) t))))
    (flet ((outer (v)
             (or (gethash v env)
                 (let ((entry (assoc v constants :test #'eq)))
                   (unless entry
                     (error 'autodiff-error
                            :format-control "scan の partial eval: ループ不変な var ~S の出所が分からない"
                            :format-arguments (list v)))
                   (let ((new (make-var (var-aval v))))
                     (push (cons new (cdr entry)) extra)
                     (setf (gethash v env) new))))))
      (dolist (eqn chosen)
        (let ((ins (mapcar #'outer (eqn-invars eqn)))
              (outs (mapcar (lambda (v) (make-var (var-aval v))) (eqn-outvars eqn))))
          (push (%make-eqn (eqn-prim eqn) (eqn-params eqn) ins outs) hoisted)
          (loop for v in (eqn-outvars eqn) for o in outs do (setf (gethash v env) o)))))
    (values (nreverse hoisted) (nreverse extra))))

(defun %scan-pe-fixpoint (body flag-consts flag-init flag-xs num-consts num-carry)
  "未知の carry の不動点を求める。(VALUES 未知の carry のフラグ 本体の eqn（入れ子の
partial eval 済み） 追加の定数 依存する var の表) を返す。"
  (multiple-value-bind (b-consts b-carry b-xs) (%scan-split (graph-invars body) num-consts num-carry)
    (let ((unk-carry (copy-list flag-init))
          (outs (subseq (graph-outvars body) 0 num-carry)))
      (loop
        (multiple-value-bind (ordered extra dependent)
            (%partial-eval-split (graph-eqns body)
                                 (append (loop for v in b-consts for f in flag-consts when f collect v)
                                         (loop for v in b-carry for f in unk-carry when f collect v)
                                         (loop for v in b-xs for f in flag-xs when f collect v)))
          (let ((new (loop for f in unk-carry for out in outs
                           collect (or f (and (gethash out dependent) t)))))
            (when (equal new unk-carry)
              (return (values unk-carry ordered extra dependent)))
            (setf unk-carry new)))))))

(defun %scan-partial-eval (eqn flags)
  "scan の EQN（FLAGS は eqn の invars ごとの「接線に依存するか」）を、主値だけの scan と
接線に依存する scan に分ける。(VALUES eqn のリスト 追加の定数)。"
  (destructuring-bind (&key num-consts num-carry length reverse body) (eqn-params eqn)
    (multiple-value-bind (b-consts b-carry b-xs) (%scan-split (graph-invars body) num-consts num-carry)
      (multiple-value-bind (flag-consts flag-init flag-xs) (%scan-split flags num-consts num-carry)
        (multiple-value-bind (o-consts o-init o-xs) (%scan-split (eqn-invars eqn) num-consts num-carry)
          (multiple-value-bind (unk-carry ordered extra dependent)
              (%scan-pe-fixpoint body flag-consts flag-init flag-xs num-consts num-carry)
            (let* ((constants (append (graph-constants body) extra))
                   (unknown-p (lambda (v) (gethash v dependent)))
                   (known-eqns (remove-if (lambda (e) (some unknown-p (eqn-invars e))) ordered))
                   (unknown-eqns (remove-if-not (lambda (e) (some unknown-p (eqn-invars e))) ordered))
                   (carry-outs (subseq (graph-outvars body) 0 num-carry))
                   (ys-outs (nthcdr num-carry (graph-outvars body)))
                   (ys-unk (mapcar (lambda (v) (and (gethash v dependent) t)) ys-outs))
                   (unknown-outs (append (loop for v in carry-outs for f in unk-carry when f collect v)
                                         (loop for v in ys-outs for f in ys-unk when f collect v)))
                   ;; 残差: 未知側が使う、既知側で定義された var（定数は未知の本体にも持たせる）
                   (residuals (let ((seen (make-hash-table :test 'eq)) (list '()))
                                (flet ((note (v)
                                         (unless (or (gethash v dependent)
                                                     (assoc v constants :test #'eq)
                                                     (gethash v seen))
                                           (setf (gethash v seen) t)
                                           (push v list))))
                                  (dolist (e unknown-eqns) (mapc #'note (eqn-invars e)))
                                  (mapc #'note unknown-outs))
                                (nreverse list)))
                   ;; ループ不変: 既知の consts の入力と本体の定数だけで決まる var
                   (invariant (let ((table (make-hash-table :test 'eq)))
                                (loop for v in b-consts for f in flag-consts
                                      unless f do (setf (gethash v table) t))
                                (loop for entry in constants do (setf (gethash (car entry) table) t))
                                (dolist (e known-eqns)
                                  (when (every (lambda (v) (gethash v table)) (eqn-invars e))
                                    (dolist (v (eqn-outvars e)) (setf (gethash v table) t))))
                                table))
                   (invariant-p (lambda (v) (gethash v invariant)))
                   (inv-res (remove-if-not invariant-p residuals))
                   (stacked-res (remove-if invariant-p residuals))
                   (env (make-hash-table :test 'eq)))
              (loop for v in b-consts for o in o-consts do (setf (gethash v env) o))
              (multiple-value-bind (hoisted hoist-constants)
                  (%scan-pe-hoist inv-res known-eqns invariant-p env constants)
                (let* (;; 既知の xs の要素そのものが残差なら、積み直さず外側の xs をそのまま渡す
                       ;; （JAX の _scan_partial_eval も既知の xs を転送する）。
                       (stacked-outer (mapcar (lambda (v)
                                                (let ((i (position v b-xs :test #'eq)))
                                                  (if i
                                                      (nth i o-xs)
                                                      (make-var (%scan-stacked-aval length (var-aval v))))))
                                              stacked-res))
                       (new-stacked (remove-if (lambda (v) (member v b-xs :test #'eq)) stacked-res))
                       (stacked-vars (loop for o in stacked-outer for v in stacked-res
                                           unless (member v b-xs :test #'eq) collect o))
                       (flag-idx (lambda (flags value)
                                   (loop for f in flags for i from 0 when (eq (and f t) value) collect i)))
                       (known-const-idx (funcall flag-idx flag-consts nil))
                       (unk-const-idx (funcall flag-idx flag-consts t))
                       (known-xs-idx (funcall flag-idx flag-xs nil))
                       (unk-xs-idx (funcall flag-idx flag-xs t))
                       (known-carry-idx (funcall flag-idx unk-carry nil))
                       (unk-carry-idx (funcall flag-idx unk-carry t))
                       (known-ys-idx (funcall flag-idx ys-unk nil))
                       (unk-ys-idx (funcall flag-idx ys-unk t))
                       (outer-outs (eqn-outvars eqn))
                       (outer-ys (nthcdr num-carry outer-outs))
                       (known-body-outs (append (%scan-pick carry-outs known-carry-idx)
                                                (%scan-pick ys-outs known-ys-idx)
                                                new-stacked))
                       (pieces (reverse hoisted)))
                  ;; 既知の scan（主値と残差）
                  (when known-body-outs
                    (push (%make-eqn
                           (eqn-prim eqn)
                           (list :num-consts (length known-const-idx)
                                 :num-carry (length known-carry-idx)
                                 :length length :reverse reverse
                                 :body (dce-graph
                                        (make-graph (append (%scan-pick b-consts known-const-idx)
                                                            (%scan-pick b-carry known-carry-idx)
                                                            (%scan-pick b-xs known-xs-idx))
                                                    known-eqns known-body-outs constants)))
                           (append (%scan-pick o-consts known-const-idx)
                                   (%scan-pick o-init known-carry-idx)
                                   (%scan-pick o-xs known-xs-idx))
                           (append (%scan-pick outer-outs known-carry-idx)
                                   (%scan-pick outer-ys known-ys-idx)
                                   stacked-vars))
                          pieces))
                  ;; 未知（線形）の scan
                  (when unknown-outs
                    (push (%make-eqn
                           (eqn-prim eqn)
                           (list :num-consts (+ (length inv-res) (length unk-const-idx))
                                 :num-carry (length unk-carry-idx)
                                 :length length :reverse reverse
                                 :body (dce-graph
                                        (make-graph (append inv-res
                                                            (%scan-pick b-consts unk-const-idx)
                                                            (%scan-pick b-carry unk-carry-idx)
                                                            (%scan-pick b-xs unk-xs-idx)
                                                            stacked-res)
                                                    unknown-eqns unknown-outs constants)))
                           (append (mapcar (lambda (v) (gethash v env)) inv-res)
                                   (%scan-pick o-consts unk-const-idx)
                                   (%scan-pick o-init unk-carry-idx)
                                   (%scan-pick o-xs unk-xs-idx)
                                   stacked-outer)
                           (append (%scan-pick outer-outs unk-carry-idx)
                                   (%scan-pick outer-ys unk-ys-idx)))
                          pieces))
                  (values (nreverse pieces) hoist-constants))))))))))

(defun %scan-pick (list indices)
  "LIST の INDICES 番目の要素のリスト。"
  (mapcar (lambda (i) (nth i list)) indices))

(set-partial-eval-rule :scan '%scan-partial-eval)

;;; ---- transpose ----

(defun %scan-linear-carry-check (body lin-consts lin-xs carry-vars known-init)
  "線形な scan の carry がどれも線形入力に依存することを確かめる（依存しない carry は、
カウンタのような主値の carry で、転置できない）。違えば AUTODIFF-ERROR。KNOWN-INIT は
carry ごとの「初期値が既知（線形でない）か」。依存する carry を、増えなくなるまで足していく。"
  (let ((seeds (append lin-consts lin-xs
                       (loop for v in carry-vars for k in known-init unless k collect v)))
        (outs (subseq (graph-outvars body) 0 (length carry-vars))))
    (loop
      (let ((dep (make-hash-table :test 'eq)))
        (dolist (v seeds) (setf (gethash v dep) t))
        (dolist (e (nth-value 1 (partition-eqns-by-dependence (graph-eqns body) seeds)))
          (dolist (v (eqn-outvars e)) (setf (gethash v dep) t)))
        (let ((added (loop for c in carry-vars for out in outs
                           when (and (not (member c seeds :test #'eq)) (gethash out dep)) collect c)))
          (if added
              (setf seeds (append seeds added))
              (progn
                (loop for c in carry-vars
                      unless (member c seeds :test #'eq)
                        do (error 'autodiff-error
                                  :format-control "scan の carry ~S が線形入力に依存しない。主値と接線が混ざった scan は転置できない（LINEARIZE-GRAPH が分けた線形な scan だけを転置する）"
                                  :format-arguments (list c)))
                (return))))))))

(defun %scan-transpose (ct invars &key num-consts num-carry length reverse body)
  "線形な scan の transpose ルール。CT は出力（最終 carry ++ ys）の余接線のリスト。"
  (multiple-value-bind (consts init xs) (%scan-split invars num-consts num-carry)
    (multiple-value-bind (b-consts b-carry b-xs) (%scan-split (graph-invars body) num-consts num-carry)
      (let* ((lin-c (mapcar #'undefined-primal-p consts))
             (lin-x (mapcar #'undefined-primal-p xs))
             (known-init (mapcar (lambda (v) (not (undefined-primal-p v))) init))
             (res-c-vars (loop for v in b-consts for l in lin-c unless l collect v))
             (res-x-vars (loop for v in b-xs for l in lin-x unless l collect v))
             (lin-c-vars (loop for v in b-consts for l in lin-c when l collect v))
             (lin-x-vars (loop for v in b-xs for l in lin-x when l collect v))
             (ct-carry (subseq ct 0 num-carry))
             (ct-ys (nthcdr num-carry ct))
             (ys-vars (nthcdr num-carry (graph-outvars body)))
             (ys-nonzero (mapcar (lambda (c) (not (symbolic-zero-p c))) ct-ys))
             (n-res-c (length res-c-vars))
             (n-res-x (length res-x-vars))
             (n-lin-c (length lin-c-vars)))
        (%scan-linear-carry-check body lin-c-vars lin-x-vars b-carry known-init)
        (let* ((permuted (make-graph (append res-c-vars res-x-vars lin-c-vars b-carry lin-x-vars)
                                     (graph-eqns body) (graph-outvars body) (graph-constants body)))
               (transposed (transpose-graph permuted (+ n-res-c n-res-x)))
               (res-c (loop for v in consts for l in lin-c unless l collect v))
               (res-x (loop for v in xs for l in lin-x unless l collect v))
               ;; 新しい本体: consts = 既知の consts、carry = consts の余接線の和 ++ carry の余接線、
               ;; xs = 既知の xs ++ ys の余接線（ゼロでないもの）、出力 = carry ++ xs の余接線
               (new-body
                 (%call-with-fresh-trace
                  (append (mapcar #'var-aval res-c-vars)
                          (mapcar #'var-aval lin-c-vars)
                          (mapcar #'var-aval b-carry)
                          (mapcar #'var-aval res-x-vars)
                          (loop for v in ys-vars for f in ys-nonzero when f collect (var-aval v)))
                  (lambda (&rest tracers)
                    (let* ((rc (subseq tracers 0 n-res-c))
                           (acc (subseq tracers n-res-c (+ n-res-c n-lin-c)))
                           (cc (subseq tracers (+ n-res-c n-lin-c) (+ n-res-c n-lin-c num-carry)))
                           (rest (nthcdr (+ n-res-c n-lin-c num-carry) tracers))
                           (rx (subseq rest 0 n-res-x))
                           (cy (nthcdr n-res-x rest))
                           (cy-full (loop for v in ys-vars for f in ys-nonzero
                                          collect (if f
                                                      (pop cy)
                                                      (instantiate-zero (make-symbolic-zero (var-aval v))))))
                           (results (inline-graph transposed (append rc rx cc cy-full)))
                           (ct-lc (subseq results 0 n-lin-c))
                           (ct-cin (subseq results n-lin-c (+ n-lin-c num-carry)))
                           (ct-lx (nthcdr (+ n-lin-c num-carry) results)))
                      (values-list (append (mapcar #'add-tangents acc ct-lc) ct-cin ct-lx))))))
               (results
                 (%trace-eqn* :scan
                              (append res-c
                                      (loop for v in lin-c-vars
                                            collect (instantiate-zero (make-symbolic-zero (var-aval v))))
                                      (mapcar #'instantiate-zero ct-carry)
                                      res-x
                                      (loop for c in ct-ys for f in ys-nonzero when f collect c))
                              :num-consts n-res-c
                              :num-carry (+ n-lin-c num-carry)
                              :length length :reverse (not reverse) :body new-body))
               (acc-final (subseq results 0 n-lin-c))
               (ct-init (subseq results n-lin-c (+ n-lin-c num-carry)))
               (ct-xs (nthcdr (+ n-lin-c num-carry) results)))
          (append (loop for l in lin-c collect (and l (pop acc-final)))
                  (loop for k in known-init for c in ct-init collect (and (not k) c))
                  (loop for l in lin-x collect (and l (pop ct-xs)))))))))

(set-transpose-rule :scan (lambda (&rest arguments) (apply '%scan-transpose arguments)))
