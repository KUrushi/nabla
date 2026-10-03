;;;; ad/rules-batch-control: 制御構造（cond / while-loop / scan）のバッチ化ルール（issue #140）。
;;;;
;;;; JAX の jax._src.lax.control_flow（_cond_batching_rule / _while_loop_batching_rule）に倣い、
;;;; 本体のサブグラフを再帰的に vmap（%VMAP-WALK-VALUES）して、バッチ化した新しい
;;;; サブグラフを持つ同じプリミティブの eqn を発行する。scan は下の SCAN のルール。
;;;;
;;;; 本体のサブグラフは「入力ごとのバッチ軸」を受けて「出力ごとのバッチ軸」を返す。
;;;; cond の枝・while の carry のように入出力の形が揃っていなければならない場所では、
;;;; 「バッチされる出力の集合」を決めて、その出力のバッチ軸を先頭（0）に揃える
;;;; （バッチされていない出力はバッチ軸を足して揃える）。

(in-package #:nabla)

(defun %vmap-batched-aval (aval dim size)
  "AVAL に、位置 DIM（NIL ならそのまま）へ長さ SIZE のバッチ軸を足した aval。"
  (if dim
      (let ((shape (aval-shape aval)))
        (make-aval (append (subseq shape 0 dim) (list size) (nthcdr dim shape)) (aval-dtype aval)))
      aval))

(defun %vmap-batch-size (args batch-dims)
  "バッチされた ARGS（トレーサ）のバッチ軸の長さ。"
  (loop for arg in args for dim in batch-dims
        when dim return (nth dim (aval-shape (tracer-aval arg)))))

(defun %vmap-subgraph (graph in-dims size want)
  "閉じた GRAPH を、入力のバッチ軸 IN-DIMS（invar ごとに整数か NIL。バッチ軸を持つ形が
invar の形）で再帰的にバッチ化した新しい閉じた graph を作り、(VALUES NEW-GRAPH OUT-DIMS)
を返す。SIZE はバッチ軸の長さ。WANT は出力ごとに T か NIL: T の出力はバッチ軸を
先頭（0）に揃える（バッチされていなければバッチ軸を足す）。NIL の出力は、バッチ化の結果
そのまま（GRAPH の invar をそのまま返す出力はそのまま invar になる。while-loop の loop
不変の素通しの検査が依存する）。
OUT-DIMS は、WANT に関わらず、揃える前の自然なバッチ軸（無ければ NIL）。T で揃えた出力の
実際の軸は常に 0 なので、呼び出し側は WANT から分かる。"
  (let ((out-dims '()))
    (let ((new (%call-with-fresh-trace
                (loop for invar in (graph-invars graph) for dim in in-dims
                      collect (%vmap-batched-aval (var-aval invar) dim size))
                (lambda (&rest tracers)
                  (multiple-value-bind (outs dims) (%vmap-walk-values graph tracers in-dims size)
                    (let ((forced
                            (loop for out in outs for dim in dims for w in want
                                  collect (cond ((not w) out)
                                                (dim (%vmap-move-axis out dim 0))
                                                (t (%vmap-broadcast-batch out 0 size))))))
                      (setf out-dims dims)
                      (values-list forced)))))))
      (values new out-dims))))

(defun %vmap-subgraph-out-dims (graph in-dims size)
  "GRAPH を IN-DIMS でバッチ化したときの、出力の自然なバッチ軸のリスト（graph は捨てる）。"
  (nth-value 1 (%vmap-subgraph graph in-dims size (make-list (length (graph-outvars graph))))))

(defun %vmap-force-batch-0 (arg dim size)
  "ARG（バッチ軸 DIM。NIL ならバッチされていない）のバッチ軸を先頭へ揃える。"
  (if dim (%vmap-move-axis arg dim 0) (%vmap-broadcast-batch arg 0 size)))

(defun %vmap-select-by-batched-pred (pred triples)
  "PRED（形 [B] の :i1）で、TRIPLES（各要素は (ON-TRUE ON-FALSE)。どちらも形 [B, ...]）の
各組を要素ごとに select して、結果のリストを返す。形が [B] の組には PRED をそのまま使い
（恒等の broadcast-in-dim を作らない）、それ以外は形ごとに1回だけ PRED を broadcast して
使い回す。"
  (let ((broadcasts '()))
    (loop for (on-true on-false) in triples
          collect (let* ((shape (aval-shape (tracer-aval on-true)))
                         (p (if (equal shape (aval-shape (tracer-aval pred)))
                                pred
                                (or (cdr (assoc shape broadcasts :test #'equal))
                                    (let ((b (%trace-eqn :broadcast-in-dim (list pred)
                                                         :shape shape :dims '(0))))
                                      (push (cons shape b) broadcasts)
                                      b)))))
                    (%trace-eqn :select (list p on-true on-false))))))

;;; --- cond ---

(def-batch-rule cond (args batch-dims &key then else)
  (let* ((size (%vmap-batch-size args batch-dims))
         (operand-dims (rest batch-dims))
         (n-out (length (graph-outvars then))))
    (if (null (first batch-dims))
        ;; 条件がバッチされていない: 両枝を同じ入力のバッチ軸でバッチ化し、出力のバッチ軸を
        ;; 揃える（どちらかの枝でバッチされる出力は、両枝でバッチして先頭に置く）。
        (let ((batched (mapcar (lambda (a b) (and (or a b) t))
                               (%vmap-subgraph-out-dims then operand-dims size)
                               (%vmap-subgraph-out-dims else operand-dims size))))
          (values (%trace-eqn* :cond args
                               :then (%vmap-subgraph then operand-dims size batched)
                               :else (%vmap-subgraph else operand-dims size batched))
                  (mapcar (lambda (b) (and b 0)) batched)))
        ;; 条件がバッチされている: 片方の枝だけを評価できないので、両枝を（バッチ化して）
        ;; インライン化して評価し、要素ごとに select で選ぶ。
        (let* ((want (make-list n-out :initial-element t))
               (operands (rest args))
               (then-outs (inline-graph (%vmap-subgraph then operand-dims size want) operands))
               (else-outs (inline-graph (%vmap-subgraph else operand-dims size want) operands)))
          (values (%vmap-select-by-batched-pred (first args) (mapcar #'list then-outs else-outs))
                  (make-list n-out :initial-element 0))))))

;;; --- while-loop ---

(defun %vmap-while-fixpoint (cond body n-carries batch-dims size)
  "while-loop のバッチされる carry の集合の不動点を求める。(VALUES CARRY-BATCHED PRED-BATCHED)。
carry の入力のバッチ軸は先頭に揃える前提（バッチされる carry は軸 0）。バッチされる carry の
集合を入力のバッチされ方から始めて、本体の出力でバッチされるもの（最初はバッチされない
carry が本体を通るとバッチされる）を足し、条件がバッチされれば全 carry を足す、を
変化しなくなるまで繰り返す（JAX の _while_loop_batching_rule と同じ）。"
  (let* ((total (length batch-dims))
         (invariant-dims (nthcdr n-carries batch-dims))
         (batched (mapcar (lambda (d) (and d t)) (subseq batch-dims 0 n-carries))))
    (loop
      (let* ((in-dims (append (mapcar (lambda (b) (and b 0)) batched) invariant-dims))
             (pred-batched (and (first (%vmap-subgraph-out-dims cond in-dims size)) t))
             (out-dims (nth-value 1 (%vmap-subgraph
                                     body in-dims size
                                     (append batched (make-list (- total n-carries))))))
             (next (loop for b in batched for d in out-dims
                         collect (or b pred-batched (and d t)))))
        (when (equal next batched)
          (return (values batched pred-batched)))
        (setf batched next)))))

(defun %vmap-any-true (pred)
  "PRED（形 [B] の :i1）のどれかが真か（rank 0 の :i1）。:i1 の縮約を避け、f32 にして
reduce-max が正かで調べる。"
  (let* ((as-float (%trace-eqn :convert (list pred) :dtype :f32))
         (largest (%trace-eqn :reduce-max (list as-float) :axes '(0)))
         (zero (%lift-constant (%scalar-array 0.0 :f32) (make-aval '() :f32) *current-trace*)))
    (%trace-eqn :compare (list largest zero) :direction :gt)))

(def-batch-rule while-loop (args batch-dims &key cond body n-carries)
  (let ((size (%vmap-batch-size args batch-dims))
        (invariants (nthcdr n-carries args)))
    (multiple-value-bind (batched pred-batched)
        (%vmap-while-fixpoint cond body n-carries batch-dims size)
      (let* ((carries (loop for arg in (subseq args 0 n-carries)
                            for dim in batch-dims
                            for b in batched
                            collect (if b (%vmap-force-batch-0 arg dim size) arg)))
             (operands (append carries invariants))
             (in-dims (append (mapcar (lambda (b) (and b 0)) batched) (nthcdr n-carries batch-dims)))
             (body-want (append batched (make-list (length invariants))))
             (out-dims in-dims))
        (multiple-value-bind (cond-b) (%vmap-subgraph cond in-dims size '(nil))
          (let ((body-b (%vmap-subgraph body in-dims size body-want)))
            (values
             (if (not pred-batched)
                 (%trace-eqn* :while-loop operands :cond cond-b :body body-b :n-carries n-carries)
                 ;; 条件がバッチされている: どれかの要素の条件が真の間回し、条件が偽になった要素の
                 ;; carry は select で据え置く（全 carry がバッチされている）。
                 (let ((avals (loop for arg in operands collect (tracer-aval arg))))
                   (%trace-eqn*
                    :while-loop operands
                    :cond (%call-with-fresh-trace
                           avals
                           (lambda (&rest tracers)
                             (%vmap-any-true (first (inline-graph cond-b tracers)))))
                    :body (%call-with-fresh-trace
                           avals
                           (lambda (&rest tracers)
                             (let ((new (inline-graph body-b tracers))
                                   (pred (first (inline-graph cond-b tracers))))
                               (values-list
                                (append (%vmap-select-by-batched-pred
                                         pred (loop for k below n-carries
                                                    collect (list (nth k new) (nth k tracers))))
                                        (nthcdr n-carries tracers))))))
                    :n-carries n-carries)))
             out-dims)))))))

;;; --- scan ---

(defun %vmap-scan-body-in-dims (batch-dims num-consts num-carry carry-batched)
  "scan の本体の入力のバッチ軸（consts ++ carry ++ x_t）。consts は元のまま、carry は
バッチされるなら 0、x_t は xs のバッチ軸を 1 に動かすので（走査の軸 0 を除くと）0。"
  (append (subseq batch-dims 0 num-consts)
          (mapcar (lambda (b) (and b 0)) carry-batched)
          (mapcar (lambda (d) (and d 0)) (nthcdr (+ num-consts num-carry) batch-dims))))

(defun %vmap-scan-fixpoint (body num-consts num-carry batch-dims size)
  "scan のバッチされる carry の集合の不動点。(VALUES CARRY-BATCHED YS-BATCHED)。
本体の入力のバッチ軸は %VMAP-SCAN-BODY-IN-DIMS。入力のバッチされ方から始め、
本体の出力でバッチされる carry を足して、変化しなくなるまで繰り返す。"
  (let* ((batched (mapcar (lambda (d) (and d t))
                          (subseq batch-dims num-consts (+ num-consts num-carry)))))
    (loop
      (let* ((in-dims (%vmap-scan-body-in-dims batch-dims num-consts num-carry batched))
             (out-dims (nth-value 1 (%vmap-subgraph body in-dims size
                                                    (make-list (length (graph-outvars body))))))
             (next (loop for b in batched for d in out-dims collect (or b (and d t)))))
        (when (equal next batched)
          (return (values batched (mapcar (lambda (d) (and d t)) (nthcdr num-carry out-dims)))))
        (setf batched next)))))

(def-batch-rule scan (args batch-dims &key num-consts num-carry length reverse body)
  ;; xs / ys は走査の軸が先頭（0）なので、バッチ軸は 1 に置く（xs は 1 に動かし、ys は 1 に出る）。
  (let ((size (%vmap-batch-size args batch-dims)))
    (multiple-value-bind (carry-batched ys-batched)
        (%vmap-scan-fixpoint body num-consts num-carry batch-dims size)
      (let* ((consts (subseq args 0 num-consts))
             (inits (loop for k from num-consts below (+ num-consts num-carry)
                          for b in carry-batched
                          collect (if b
                                      (%vmap-force-batch-0 (nth k args) (nth k batch-dims) size)
                                      (nth k args))))
             (xs (loop for arg in (nthcdr (+ num-consts num-carry) args)
                       for dim in (nthcdr (+ num-consts num-carry) batch-dims)
                       collect (if dim (%vmap-move-axis arg dim 1) arg)))
             (in-dims (%vmap-scan-body-in-dims batch-dims num-consts num-carry carry-batched))
             (body-b (%vmap-subgraph body in-dims size (append carry-batched ys-batched))))
        (values (%trace-eqn* :scan (append consts inits xs)
                             :num-consts num-consts :num-carry num-carry
                             :length length :reverse reverse :body body-b)
                (append (mapcar (lambda (b) (and b 0)) carry-batched)
                        (mapcar (lambda (b) (and b 1)) ys-batched)))))))
