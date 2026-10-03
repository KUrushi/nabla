;;;; ad/rules-scan: scan の jvp ルール（issue #135）。
;;;;
;;;; JAX の jax._src.lax.control_flow.loops._scan_jvp に対応する。本体のサブグラフを
;;;; jvp 変換し、接線を持つ入力を足した1つの scan にする。主値と接線は同じループ
;;;; （1つの scan）で一緒に計算する（ループを2回回さない）。
;;;;
;;;; 並びは JAX と同じ（#139 の linearize / transpose と、#140 のバッチ化がこの並びを
;;;; 前提にする）:
;;;;   consts      = consts ++ 接線を持つ consts の接線
;;;;   carry       = carry  ++ 接線を持つ carry の接線
;;;;   xs          = xs     ++ 接線を持つ xs の接線
;;;;   本体の出力  = carry ++ carry の接線 ++ ys ++ 接線を持つ ys の接線
;;;;   eqn の出力  = 最終 carry ++ 最終 carry の接線 ++ ys ++ ys の接線
;;;; symbolic zero の接線（と、浮動小数点でない入力の接線）は入力にも出力にもならない。
;;;; carry の接線を持つかどうかは、本体を通ると変わりうる（初期の接線がゼロでも、
;;;; 本体で非ゼロの接線を受ける carry は、次のステップで非ゼロになる）。そこで、
;;;; 非ゼロの接線を持つ carry の集合を、増えなくなる（不動点）まで広げて決める。
;;;; 集合は単調に増えるだけで carry の個数が上限なので、必ず止まる。
;;;;
;;;; 本体の jvp は %JVP-GRAPH-CORE の FORCE で、「接線を持つ carry の出力は必ず出力にし、
;;;; ys の接線はゼロでなければ出力にする」とする。

(in-package #:nabla)

(defun %scan-jvp-fixpoint (body nonzero-consts nonzero-carry nonzero-xs num-carry)
  "BODY の jvp を、非ゼロの接線を持つ carry の集合の不動点で求める。
(VALUES NONZERO-CARRY（不動点） JVP-GRAPH TANGENT-FLAGS) を返す。TANGENT-FLAGS は
本体の出力（carry ++ ys）ごとの、接線を出力にしたか。"
  (let ((nonzero-carry (copy-list nonzero-carry))
        (n-outs (length (graph-outvars body))))
    (loop
      (multiple-value-bind (jvp flags)
          (%jvp-graph-core body (append nonzero-consts nonzero-carry nonzero-xs)
                           (append nonzero-carry (make-list (- n-outs num-carry) :initial-element nil)))
        (let ((new (loop for flag in nonzero-carry
                         for included in (subseq flags 0 num-carry)
                         collect (or flag included))))
          (when (equal new nonzero-carry)
            (return (values nonzero-carry jvp flags)))
          (setf nonzero-carry new))))))

(defun %scan-jvp-select (list flags)
  "LIST のうち、FLAGS が真の位置の要素のリスト。"
  (loop for item in list for flag in flags when flag collect item))

(set-jvp-rule
 :scan
 (lambda (primals tangents &key num-consts num-carry length reverse body)
   (multiple-value-bind (consts init xs) (%scan-split primals num-consts num-carry)
     (multiple-value-bind (consts-dot init-dot xs-dot) (%scan-split tangents num-consts num-carry)
       (let ((nonzero-consts (mapcar (lambda (tangent) (not (symbolic-zero-p tangent))) consts-dot))
             (nonzero-xs (mapcar (lambda (tangent) (not (symbolic-zero-p tangent))) xs-dot))
             (n-xs (length xs)))
         (multiple-value-bind (nonzero-carry jvp-body tangent-flags)
             (%scan-jvp-fixpoint body nonzero-consts
                                 (mapcar (lambda (tangent) (not (symbolic-zero-p tangent))) init-dot)
                                 nonzero-xs num-carry)
           (let* ((n-ys (- (length (graph-outvars body)) num-carry))
                  (n-consts-dot (count t nonzero-consts))
                  (n-carry-dot (count t nonzero-carry))
                  ;; jvp 本体の入力は 主値 (consts carry xs) ++ 接線 (consts carry xs)。
                  ;; JAX の並び consts consts' carry carry' xs xs' に直す。
                  (invars (graph-invars jvp-body))
                  (primal-in (subseq invars 0 (+ num-consts num-carry n-xs)))
                  (tangent-in (nthcdr (+ num-consts num-carry n-xs) invars))
                  (outvars (graph-outvars jvp-body))
                  (n-outs (+ num-carry n-ys))
                  (primal-out (subseq outvars 0 n-outs))
                  (tangent-out (nthcdr n-outs outvars))
                  (new-body
                    (check-graph
                     (make-graph
                      (append (subseq primal-in 0 num-consts)
                              (subseq tangent-in 0 n-consts-dot)
                              (subseq primal-in num-consts (+ num-consts num-carry))
                              (subseq tangent-in n-consts-dot (+ n-consts-dot n-carry-dot))
                              (nthcdr (+ num-consts num-carry) primal-in)
                              (nthcdr (+ n-consts-dot n-carry-dot) tangent-in))
                      (graph-eqns jvp-body)
                      (append (subseq primal-out 0 num-carry)
                              (subseq tangent-out 0 n-carry-dot)
                              (nthcdr num-carry primal-out)
                              (nthcdr n-carry-dot tangent-out))
                      (graph-constants jvp-body))))
                  (results
                    (%trace-eqn* :scan
                                 ;; init の接線がゼロでも、不動点で非ゼロになった carry の接線の
                                 ;; 初期値は、symbolic zero を実体化したゼロの配列。
                                 (append consts (%scan-jvp-select consts-dot nonzero-consts)
                                         init
                                         (mapcar #'instantiate-zero
                                                 (%scan-jvp-select init-dot nonzero-carry))
                                         xs (%scan-jvp-select xs-dot nonzero-xs))
                                 :num-consts (+ num-consts n-consts-dot)
                                 :num-carry (+ num-carry n-carry-dot)
                                 :length length :reverse reverse :body new-body))
                  (final-carry (subseq results 0 num-carry))
                  (final-carry-dot (subseq results num-carry (+ num-carry n-carry-dot)))
                  (ys (subseq results (+ num-carry n-carry-dot) (+ num-carry n-carry-dot n-ys)))
                  (ys-dot (subseq results (+ num-carry n-carry-dot n-ys))))
             (flet ((spread (outs flags dots)
                      ;; 接線を出力にしたものはそのトレーサ、そうでないものは symbolic zero。
                      (loop for out in outs for flag in flags
                            collect (if flag
                                        (pop dots)
                                        (make-symbolic-zero (tracer-aval out))))))
               (values (append final-carry ys)
                       (append (spread final-carry nonzero-carry final-carry-dot)
                               (spread ys (nthcdr num-carry tangent-flags) ys-dot)))))))))))
