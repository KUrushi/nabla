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

(defun jvp-graph (graph &key (nonzero (make-list (length (graph-invars graph)) :initial-element t)))
  "GRAPH を jvp 変換した新しい GRAPH を CHECK-GRAPH して返す。GRAPH 自体は
書き換えない。

新しい graph の入力は、GRAPH の入力の主値に続けて、NONZERO が真の入力の
接線（その入力と同じ aval）。NONZERO は GRAPH-INVARS と同じ長さの真偽値の
リストで、偽の入力の接線は SYMBOLIC-ZERO として扱われ、graph の入力には
ならない。既定はすべて T。出力は GRAPH の出力の主値に続けて、その接線
（ゼロと分かっていれば INSTANTIATE-ZERO でゼロの配列を作る）。

各 eqn は主値を %TRACE-EQN で再発行する。全入力の接線がゼロなら出力の接線も
ゼロ（ルールを呼ばない）。そうでなければ REQUIRE-JVP-RULE のルールを
  (apply rule primals out tangents (eqn-params eqn))
で呼ぶ（無ければ NO-JVP-RULE）。定数の接線はゼロ。

制限: :I1 など浮動小数点でない dtype の値にも接線の出力を作るので、:I1 の
出力があるとゼロの実体化（INSTANTIATE-ZERO）が TRACING-ERROR になる。
compare / select を扱う #80 以降で、整数・真偽値の接線の扱いを決める。"
  (let ((invars (graph-invars graph)))
    (unless (= (length nonzero) (length invars))
      (error 'autodiff-error
             :format-control "NONZERO の長さ ~D が graph の入力の個数 ~D と一致しない"
             :format-arguments (list (length nonzero) (length invars))))
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
                  (out (apply #'%trace-eqn (primitive-name (eqn-prim eqn)) primals (eqn-params eqn)))
                  (tangent (if (every #'symbolic-zero-p tangents)
                               (make-symbolic-zero (tracer-aval out))
                               (%jvp-rule-tangent eqn (require-jvp-rule (eqn-prim eqn))
                                                  primals out tangents))))
             ;; フェーズ1では eqn の outvars は常に1つ（src/ir.lisp の EQN を参照）。
             (assert (= 1 (length (eqn-outvars eqn))))
             (setf (gethash (first (eqn-outvars eqn)) env) (cons out tangent))))
         (let ((entries (mapcar (lambda (v) (gethash v env)) (graph-outvars graph))))
           (values-list (append (mapcar #'car entries)
                                (mapcar (lambda (entry) (instantiate-zero (cdr entry))) entries)))))))))
