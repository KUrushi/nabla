;;;; ad/transpose: 線形な graph を逆向きにたどって余接線を求める TRANSPOSE-GRAPH
;;;; と、linearize + transpose で vjp を作る VJP-GRAPH（issue #82）。
;;;;
;;;; 内部関数（export しない）。公開 API は grad / value-and-grad（#86）。
;;;; 各 eqn の transpose ルール（PRIMITIVE-TRANSPOSE。規約は DEF-TRANSPOSE-RULE）が
;;;; 余接線を現在のトレースに eqn として足す。

(in-package #:nabla)

(defun %transpose-ct-aval-check (eqn ct var)
  "余接線 CT の aval が、線形入力 VAR の aval と一致することを確かめる。"
  (unless (equalp (tangent-aval ct) (var-aval var))
    (error 'autodiff-error
           :format-control "プリミティブ ~S の transpose ルールが返した余接線の aval ~S が、入力の aval ~S と一致しない"
           :format-arguments (list (primitive-name (eqn-prim eqn)) (tangent-aval ct) (var-aval var)))))

(defun transpose-graph (graph n-known)
  "GRAPH（線形な graph）を転置した新しい GRAPH を CHECK-GRAPH して返す。GRAPH 自体は
書き換えない。

GRAPH の入力の先頭 N-KNOWN 個は既知の値（残差。係数として使う）、残りが線形な入力
（未定の主値、UNDEFINED-PRIMAL）。新しい graph の入力は、既知の値（同じ aval）に続けて、
GRAPH の出力ごとの余接線（出力と同じ aval）。出力は、線形な入力ごとの余接線（入力と同じ aval）。
どの出力にも効かない線形入力の余接線は、ゼロの配列（INSTANTIATE-ZERO）になる。

線形な入力に依存しない eqn（既知の値だけの eqn）は順方向に再発行し、依存する eqn は
逆順にたどる。各 eqn の出力の余接線がゼロ（まだ何も流れていない）なら飛ばし、そうでなければ
REQUIRE-TRANSPOSE-RULE のルールを
  (apply rule ct invars (eqn-params eqn))
で呼ぶ（無ければ NO-TRANSPOSE-RULE）。INVARS は既知の入力なら新しい graph の中の
トレーサ、線形な入力なら UNDEFINED-PRIMAL。ルールが返した余接線は ADD-TANGENTS で
入力ごとに足し合わせる（同じ var が複数回使われても、複数の出力に使われても和になる）。
線形な入力に依存しない出力（定数など）の余接線は捨てる。余接線が SYMBOLIC-ZERO の
var の eqn も飛ばす。使われない定数は持ち上げたあと DCE-GRAPH で落とす。"
  (let* ((invars (graph-invars graph))
         (known-invars (subseq invars 0 n-known))
         (linear-invars (nthcdr n-known invars))
         (known-avals (mapcar #'var-aval known-invars))
         (out-avals (mapcar #'var-aval (graph-outvars graph))))
    (dce-graph
     (%call-with-fresh-trace
     (append known-avals out-avals)
     (lambda (&rest tracers)
       (let ((known (make-hash-table :test 'eq))
             (linear (make-hash-table :test 'eq))
             (cts (make-hash-table :test 'eq))
             (cotangent-tracers (nthcdr n-known tracers)))
         (loop for var in known-invars for tracer in tracers
               do (setf (gethash var known) tracer))
         (dolist (var linear-invars) (setf (gethash var linear) t))
         (loop for (var . array) in (graph-constants graph)
               do (setf (gethash var known) (%lift-constant array (var-aval var) *current-trace*)))
         ;; 順方向: 線形な入力に依存しない eqn は既知の値として再発行する。
         (let ((linear-eqns '()))
           (dolist (eqn (graph-eqns graph))
             (assert (= 1 (length (eqn-outvars eqn))))
             (if (some (lambda (v) (gethash v linear)) (eqn-invars eqn))
                 (progn (setf (gethash (first (eqn-outvars eqn)) linear) t)
                        (push eqn linear-eqns))
                 (setf (gethash (first (eqn-outvars eqn)) known)
                       (apply #'%trace-eqn (primitive-name (eqn-prim eqn))
                              (mapcar (lambda (v) (gethash v known)) (eqn-invars eqn))
                              (eqn-params eqn)))))
           ;; 出力の余接線を、出力の var に足す（同じ var が複数回出力なら和）。
           (loop for var in (graph-outvars graph) for ct in cotangent-tracers
                 when (gethash var linear)
                   do (setf (gethash var cts)
                            (add-tangents (gethash var cts (make-symbolic-zero (var-aval var))) ct)))
           ;; 逆向き: LINEAR-EQNS は逆順に積まれているので、そのままたどる。
           (dolist (eqn linear-eqns)
             (let* ((out (first (eqn-outvars eqn)))
                    (ct (gethash out cts)))
               (when (and ct (not (symbolic-zero-p ct)))
                 (let* ((rule (require-transpose-rule (eqn-prim eqn)))
                        (rule-invars (mapcar (lambda (v)
                                               (if (gethash v linear)
                                                   (make-undefined-primal (var-aval v))
                                                   (gethash v known)))
                                             (eqn-invars eqn)))
                        (results (apply rule ct rule-invars (eqn-params eqn))))
                   (unless (and (listp results) (= (length results) (length rule-invars)))
                     (error 'autodiff-error
                            :format-control "プリミティブ ~S の transpose ルールは、入力と同じ長さ ~D のリストを返さなければならない: ~S"
                            :format-arguments (list (primitive-name (eqn-prim eqn)) (length rule-invars) results)))
                   (loop for var in (eqn-invars eqn)
                         for result in results
                         do (cond
                              ((gethash var linear)
                               (unless result
                                 (error 'autodiff-error
                                        :format-control "プリミティブ ~S の transpose ルールが、線形な入力 ~S の余接線を返さなかった"
                                        :format-arguments (list (primitive-name (eqn-prim eqn)) var)))
                               (%transpose-ct-aval-check eqn result var)
                               (setf (gethash var cts)
                                     (add-tangents (gethash var cts (make-symbolic-zero (var-aval var)))
                                                   result)))
                              (result
                               (error 'autodiff-error
                                      :format-control "プリミティブ ~S の transpose ルールが、既知の入力 ~S の位置に NIL でない値を返した"
                                      :format-arguments (list (primitive-name (eqn-prim eqn)) var))))))))))
         (values-list (mapcar (lambda (var)
                                (instantiate-zero (gethash var cts (make-symbolic-zero (var-aval var)))))
                              linear-invars))))))))

(defun vjp-graph (graph &key (nonzero nil nonzero-p))
  "GRAPH の vjp（reverse モードの微分）を計算する新しい GRAPH を CHECK-GRAPH して返す。
GRAPH 自体は書き換えない。LINEARIZE-GRAPH で jvp を主値と線形部分に分け、線形部分を
TRANSPOSE-GRAPH で転置して、1つの graph にまとめる（出力に効かない eqn は DCE する）。

新しい graph の入力は、GRAPH の入力の主値に続けて、GRAPH の出力ごとの余接線（出力と同じ aval）。
出力は、GRAPH の出力の主値に続けて、入力ごとの余接線（入力と同じ aval）。余接線が返るのは
NONZERO（JVP-GRAPH と同じ意味。既定は浮動小数点の入力だけ）が真の入力だけで、入力の順に並ぶ。
:I1 など接線がゼロの出力の余接線は無視される。"
  (let* ((lin (if nonzero-p (linearize-graph graph :nonzero nonzero) (linearize-graph graph)))
         (transposed (transpose-graph (linearization-linear-graph lin) (linearization-n-residuals lin)))
         (primal (linearization-primal-graph lin))
         (m (linearization-n-outputs lin)))
    (dce-graph
     (%call-with-fresh-trace
      (append (mapcar #'var-aval (graph-invars graph))
              (mapcar #'var-aval (graph-outvars graph)))
      (lambda (&rest tracers)
        (let* ((n (length (graph-invars graph)))
               (primal-results (inline-graph primal (subseq tracers 0 n)))
               (residuals (nthcdr m primal-results))
               (cotangents (inline-graph transposed (append residuals (nthcdr n tracers)))))
          (values-list (append (subseq primal-results 0 m) cotangents))))))))
