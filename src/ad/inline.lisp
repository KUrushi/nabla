;;;; ad/inline: graph を現在のトレースへ再発行する INLINE-GRAPH と、
;;;; 出力に効かない eqn を落とす DCE-GRAPH（issue #77、77b）。
;;;;
;;;; どちらも内部関数（export しない）。grad / jvp は f を新しいトレース
;;;; （%CALL-WITH-FRESH-TRACE）で GRAPH にし、変換したあとの結果を外側の
;;;; トレースへ INLINE-GRAPH で戻す。

(in-package #:nabla)

(defun inline-graph (graph tracers)
  "GRAPH を、現在のトレース（*CURRENT-TRACE*）に再発行し、GRAPH-OUTVARS に
対応する TRACER のリストを返す。TRACERS は GRAPH-INVARS に1対1で対応する
（個数と aval が一致しなければ TRACING-ERROR。*CURRENT-TRACE* が NIL の
ときと、TRACERS が現在のトレースに属さないときも TRACING-ERROR）。

GRAPH-CONSTANTS の配列は（コピーせず）%LIFT-CONSTANT で現在のトレースの
定数として登録し直し、GRAPH-EQNS は順に %TRACE-EQN で発行し直す。var は
そのたびに新しく作られるので、同じ GRAPH を何度インライン化してもよい
（GRAPH 自体は書き換えない）。eqn の params が持つサブグラフは中身を書き換えず
そのまま（同じ GRAPH を）再発行する eqn に渡す。同じ var が出力に複数回現れれば、同じ
（EQ な）TRACER が複数回返る。"
  (unless *current-trace*
    (error 'tracing-error
           :format-control "INLINE-GRAPH はトレース中（*CURRENT-TRACE* が束縛されている間）にしか呼べない"))
  (let ((invars (graph-invars graph)))
    (unless (= (length tracers) (length invars))
      (error 'tracing-error
             :format-control "INLINE-GRAPH: トレーサの個数 ~D が graph の入力の個数 ~D と一致しない"
             :format-arguments (list (length tracers) (length invars))))
    (loop for tracer in tracers
          for invar in invars
          unless (equalp (tracer-aval tracer) (var-aval invar))
            do (error 'tracing-error
                      :format-control "INLINE-GRAPH: トレーサ ~S の aval が graph の入力 ~S の aval と一致しない"
                      :format-arguments (list tracer invar)))
    (%tracer-check-current-trace tracers)
    (let ((env (make-hash-table :test 'eq)))
      (loop for tracer in tracers
            for invar in invars
            do (setf (gethash invar env) tracer))
      (loop for (var . array) in (graph-constants graph)
            do (setf (gethash var env) (%lift-constant array (var-aval var) *current-trace*)))
      (dolist (eqn (graph-eqns graph))
        ;; 複数出力の eqn（契約 C1）も %TRACE-EQN* で扱う。params が持つ
        ;; サブグラフ（閉じた graph。契約 C2）は変換せず、そのまま共有する。
        (let ((results (apply #'%trace-eqn* (primitive-name (eqn-prim eqn))
                              (mapcar (lambda (v) (gethash v env)) (eqn-invars eqn))
                              (eqn-params eqn))))
          (loop for out in (eqn-outvars eqn)
                for result in results
                do (setf (gethash out env) result))))
      (mapcar (lambda (v) (gethash v env)) (graph-outvars graph)))))

(defun dce-graph (graph)
  "GRAPH-OUTVARS から逆向きにたどって到達できる eqn と定数だけを残した新しい
GRAPH を CHECK-GRAPH して返す（死んだコードの削除）。GRAPH-INVARS は、
使われていなくても消さない（関数の引数の数を変えないため）。残した eqn と
定数の順序は元のまま。GRAPH 自体は書き換えない。"
  (let ((live (make-hash-table :test 'eq))
        (kept '()))
    (dolist (v (graph-outvars graph)) (setf (gethash v live) t))
    (dolist (eqn (reverse (graph-eqns graph)))
      (when (some (lambda (v) (gethash v live)) (eqn-outvars eqn))
        (push eqn kept)
        (dolist (v (eqn-invars eqn)) (setf (gethash v live) t))))
    (check-graph
     (make-graph (graph-invars graph)
                 kept
                 (graph-outvars graph)
                 (remove-if-not (lambda (entry) (gethash (car entry) live))
                                (graph-constants graph))))))
