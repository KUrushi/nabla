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

(defun %jvp-rule-tangents (eqn rule primals outs tangents)
  "複数出力の EQN（契約 C1）の jvp ルール RULE を
  (apply rule primals outs tangents params)
で呼び、出力ごとの接線のリストを返す。個数または接線の aval が主値の出力と
合わなければ AUTODIFF-ERROR。"
  (let ((result (apply rule primals outs tangents (eqn-params eqn))))
    (unless (and (listp result) (= (length result) (length outs)))
      (error 'autodiff-error
             :format-control "プリミティブ ~S の jvp ルールは、出力と同じ長さ ~D の接線のリストを返さなければならない: ~S"
             :format-arguments (list (primitive-name (eqn-prim eqn)) (length outs) result)))
    (loop for tangent in result
          for out in outs
          unless (equalp (tangent-aval tangent) (tracer-aval out))
            do (error 'autodiff-error
                      :format-control "プリミティブ ~S の jvp ルールが返した接線の aval ~S が、主値の出力の aval ~S と一致しない"
                      :format-arguments (list (primitive-name (eqn-prim eqn))
                                              (tangent-aval tangent) (tracer-aval out))))
    result))

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
で呼ぶ（無ければ NO-JVP-RULE）。定数の接線はゼロ。"
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
                  (outs (apply #'%trace-eqn* (primitive-name prim) primals (eqn-params eqn)))
                  (out-tangents
                    (cond ((every #'symbolic-zero-p tangents)
                           (mapcar (lambda (o) (make-symbolic-zero (tracer-aval o))) outs))
                          ((primitive-multiple-outputs-p prim)
                           (%jvp-rule-tangents eqn (require-jvp-rule prim) primals outs tangents))
                          (t (list (%jvp-rule-tangent eqn (require-jvp-rule prim)
                                                      primals (first outs) tangents))))))
             (loop for var in (eqn-outvars eqn)
                   for out in outs
                   for tangent in out-tangents
                   do (setf (gethash var env) (cons out tangent)))))
         (let ((entries (mapcar (lambda (v) (gethash v env)) (graph-outvars graph))))
           (values-list (append (mapcar #'car entries)
                                (mapcar (lambda (entry) (instantiate-zero (cdr entry))) entries)))))))))
