;;;; ad/rules-batch-control: 制御構造（cond / while-loop）のバッチ化ルール（issue #140）。
;;;;
;;;; JAX の jax._src.lax.control_flow（_cond_batching_rule / _while_loop_batching_rule）に倣い、
;;;; 本体のサブグラフを再帰的に vmap（%VMAP-WALK-VALUES）して、バッチ化した新しい
;;;; サブグラフを持つ同じプリミティブの eqn を発行する。scan は scan（#132）が入ってから足す。
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
そのまま（バッチ軸の位置は OUT-DIMS で返す。GRAPH の invar をそのまま返す出力は
そのまま invar になる。while-loop の loop 不変の素通しの検査が依存する）。
OUT-DIMS の要素は、WANT が T なら 0、そうでなければ自然なバッチ軸（無ければ NIL）。"
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
                      (setf out-dims (loop for dim in dims for w in want collect (if w 0 dim)))
                      (values-list forced)))))))
      (values new out-dims))))

(defun %vmap-subgraph-out-dims (graph in-dims size)
  "GRAPH を IN-DIMS でバッチ化したときの、出力の自然なバッチ軸のリスト（graph は捨てる）。"
  (nth-value 1 (%vmap-subgraph graph in-dims size (make-list (length (graph-outvars graph))))))

(defun %vmap-force-batch-0 (arg dim size)
  "ARG（バッチ軸 DIM。NIL ならバッチされていない）のバッチ軸を先頭へ揃える。"
  (if dim (%vmap-move-axis arg dim 0) (%vmap-broadcast-batch arg 0 size)))

(defun %vmap-select-by-batched-pred (pred on-true on-false)
  "PRED（形 [B]）を ON-TRUE / ON-FALSE（形 [B, ...]）の形へ broadcast して select する。"
  (let ((shape (aval-shape (tracer-aval on-true))))
    (%trace-eqn :select
                (list (%trace-eqn :broadcast-in-dim (list pred) :shape shape :dims '(0))
                      on-true on-false))))

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
          (values (mapcar (lambda (a b) (%vmap-select-by-batched-pred (first args) a b))
                          then-outs else-outs)
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
                                (append (loop for k below n-carries
                                              collect (%vmap-select-by-batched-pred
                                                       pred (nth k new) (nth k tracers)))
                                        (nthcdr n-carries tracers))))))
                    :n-carries n-carries)))
             out-dims)))))))
