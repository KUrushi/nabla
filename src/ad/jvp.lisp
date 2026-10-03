;;;; ad/jvp: graph を jvp（前向きモード微分）で書き換える JVP-GRAPH
;;;; （issue #77、77c）。
;;;;
;;;; 内部関数（export しない）。公開 API は grad / value-and-grad（#86）。
;;;; 元の graph を新しいトレース（%CALL-WITH-FRESH-TRACE）の中で1 eqn ずつ
;;;; 再発行し、各 eqn の jvp ルール（PRIMITIVE-JVP）が接線の eqn を同じ
;;;; トレースに足す。主値と接線は1つの graph の中に並ぶ。

(in-package #:nabla)

(defun %jvp-rule-tangent (eqn rule primals out tangents)
  "EQN の jvp ルール RULE を呼び、出力の接線を返す。接線の aval が主値出力 OUT
の aval と equalp でなければ AUTODIFF-ERROR（どのプリミティブか分かる）。"
  (let ((tangent (apply rule primals out tangents (eqn-params eqn))))
    (unless (equalp (tangent-aval tangent) (tracer-aval out))
      (error 'autodiff-error
             :format-control "プリミティブ ~S の jvp ルールが返した接線の aval ~S が、主値の出力の aval ~S と一致しない"
             :format-arguments (list (primitive-name (eqn-prim eqn))
                                     (tangent-aval tangent) (tracer-aval out))))
    tangent))

(defun %jvp-multiple-output-rule (eqn rule primals tangents)
  "複数出力の EQN（契約 C1）の jvp ルール RULE を
  (apply rule primals tangents params)
で呼び、(VALUES 主値の出力のトレーサのリスト 接線のリスト) を返す。ルールが主値の
eqn も自分で足す（JVP-GRAPH は事前に足さない。足すと while / scan / cond のような
高階プリミティブが2回走る eqn になってしまうため）。個数と、主値・接線の aval が
EQN の outvars と合わなければ AUTODIFF-ERROR。"
  (multiple-value-bind (outs tangents-out) (apply rule primals tangents (eqn-params eqn))
    (let ((name (primitive-name (eqn-prim eqn)))
          (n (length (eqn-outvars eqn))))
      (unless (and (listp outs) (listp tangents-out)
                   (= (length outs) n) (= (length tangents-out) n))
        (error 'autodiff-error
               :format-control "プリミティブ ~S の jvp ルールは、出力と同じ長さ ~D の主値のリストと接線のリストの2値を返さなければならない: ~S / ~S"
               :format-arguments (list name n outs tangents-out)))
      (loop for var in (eqn-outvars eqn)
            for out in outs
            for tangent in tangents-out
            do (unless (equalp (tracer-aval out) (var-aval var))
                 (error 'autodiff-error
                        :format-control "プリミティブ ~S の jvp ルールが返した主値の aval ~S が、出力の aval ~S と一致しない"
                        :format-arguments (list name (tracer-aval out) (var-aval var))))
               (unless (equalp (tangent-aval tangent) (var-aval var))
                 (error 'autodiff-error
                        :format-control "プリミティブ ~S の jvp ルールが返した接線の aval ~S が、主値の出力の aval ~S と一致しない"
                        :format-arguments (list name (tangent-aval tangent) (var-aval var)))))
      (values outs tangents-out))))

(defun %jvp-graph-with-out-nonzero (graph nonzero)
  "JVP-GRAPH の本体。(VALUES JVP-GRAPH OUT-NONZERO) を返す。OUT-NONZERO は GRAPH の
出力ごとに、接線が SYMBOLIC-ZERO でなければ T のリスト（JVP-GRAPH は出力のゼロの
接線を実体化して隠すが、while-loop / scan の不動点の計算はどの出力が非ゼロかを要る）。"
  (let ((out-nonzero '()))
    (values (%jvp-graph-1 graph nonzero (lambda (flags) (setf out-nonzero flags)))
            out-nonzero)))

(defun jvp-graph (graph &key (nonzero (mapcar (lambda (v) (and (%float-dtype-p (aval-dtype (var-aval v))) t))
                                              (graph-invars graph))))
  "GRAPH を jvp 変換した新しい GRAPH を CHECK-GRAPH して返す。GRAPH 自体は
書き換えない。

新しい graph の入力は、GRAPH の入力の主値に続けて、NONZERO が真の入力の
接線（その入力と同じ aval）。NONZERO は GRAPH-INVARS と同じ長さの真偽値の
リストで、偽の入力の接線は SYMBOLIC-ZERO として扱われ、graph の入力には
ならない。既定は、浮動小数点の入力なら T、:I1 など浮動小数点でない入力は
NIL（その接線空間は自明で、接線は常に SYMBOLIC-ZERO）。非浮動小数点の入力に
T を渡すと AUTODIFF-ERROR。出力は GRAPH の出力の主値に続けて、その接線
（ゼロと分かっていれば INSTANTIATE-ZERO でゼロの配列を作る）。出力は常に「主値 ++ 接線、同じ個数」で、:I1 の出力の
接線も全 false の :I1 配列になる。

この関数の中では DCE しない（結果は「元と同じ主値の eqn + 接線の eqn」）。
接線だけを取り出す linearize（#82）が、接線の部分に DCE-GRAPH をかける。

各 eqn は主値を %TRACE-EQN で再発行する。全入力の接線がゼロなら出力の接線も
ゼロ（ルールを呼ばない）。そうでなければ REQUIRE-JVP-RULE のルールを
  (apply rule primals out tangents (eqn-params eqn))
で呼ぶ（無ければ NO-JVP-RULE）。複数出力のプリミティブ（契約 C1）は主値の eqn を
事前に足さず、ルールを (apply rule primals tangents params) で呼んで
(VALUES 主値の出力のリスト 接線のリスト) を受け取る（ルールが主値の eqn を足す。
全入力の接線がゼロなら、複数出力でもルールを呼ばず主値だけ再発行する）。定数の接線はゼロ。"
  (values (%jvp-graph-1 graph nonzero nil)))

(defun %jvp-graph-1 (graph nonzero report-out-nonzero)
  "JVP-GRAPH の実体。REPORT-OUT-NONZERO が関数なら、出力ごとの「接線が非ゼロか」の
リストを渡して呼ぶ。"
  (let ((invars (graph-invars graph)))
    (unless (= (length nonzero) (length invars))
      (error 'autodiff-error
             :format-control "NONZERO の長さ ~D が graph の入力の個数 ~D と一致しない"
             :format-arguments (list (length nonzero) (length invars))))
    (loop for invar in invars
          for flag in nonzero
          when (and flag (not (%float-dtype-p (aval-dtype (var-aval invar)))))
            do (error 'autodiff-error
                      :format-control "浮動小数点でない入力 ~S には接線を渡せない（nonzero は NIL にする）"
                      :format-arguments (list invar)))
    (%call-with-fresh-trace
     (append (mapcar #'var-aval invars)
             (loop for invar in invars for flag in nonzero
                   when flag collect (var-aval invar)))
     (lambda (&rest tracers)
       (let ((env (make-hash-table :test 'eq))
             (tangent-tracers (nthcdr (length invars) tracers)))
         (loop for invar in invars
               for primal in tracers
               for flag in nonzero
               do (setf (gethash invar env)
                        (cons primal (if flag
                                         (pop tangent-tracers)
                                         (make-symbolic-zero (var-aval invar))))))
         (loop for (var . array) in (graph-constants graph)
               do (setf (gethash var env)
                        (cons (%lift-constant array (var-aval var) *current-trace*)
                              (make-symbolic-zero (var-aval var)))))
         (dolist (eqn (graph-eqns graph))
           (let* ((entries (mapcar (lambda (v) (gethash v env)) (eqn-invars eqn)))
                  (primals (mapcar #'car entries))
                  (tangents (mapcar #'cdr entries))
                  (prim (eqn-prim eqn))
                  (zero-p (every #'symbolic-zero-p tangents)))
             (multiple-value-bind (outs out-tangents)
                 (cond
                   ;; 複数出力で接線がゼロでない: ルールが主値も足す。
                   ((and (primitive-multiple-outputs-p prim) (not zero-p))
                    (%jvp-multiple-output-rule eqn (require-jvp-rule prim) primals tangents))
                   (t
                    (let ((outs (apply #'%trace-eqn* (primitive-name prim) primals (eqn-params eqn))))
                      (values outs
                              (if zero-p
                                  (mapcar (lambda (o) (make-symbolic-zero (tracer-aval o))) outs)
                                  (list (%jvp-rule-tangent eqn (require-jvp-rule prim)
                                                           primals (first outs) tangents)))))))
               (loop for var in (eqn-outvars eqn)
                     for out in outs
                     for tangent in out-tangents
                     do (setf (gethash var env) (cons out tangent))))))
         (let ((entries (mapcar (lambda (v) (gethash v env)) (graph-outvars graph))))
           (when report-out-nonzero
             (funcall report-out-nonzero
                      (mapcar (lambda (entry) (not (symbolic-zero-p (cdr entry)))) entries)))
           (values-list (append (mapcar #'car entries)
                                (mapcar (lambda (entry) (instantiate-zero (cdr entry))) entries)))))))))
