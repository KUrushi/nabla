;;;; ad/linearize: jvp した graph を「主値の計算」と「接線についての線形な
;;;; 計算」に分ける LINEARIZE-GRAPH（issue #82）。
;;;;
;;;; 内部関数（export しない）。静的な graph なので、JAX の partial eval の
;;;; 代わりに「接線の入力に推移的に依存するか」だけで分ける（依存する eqn は
;;;; 構成上、接線について線形。DEF-JVP-RULE の制約を参照）。分けた線形部分は
;;;; TRANSPOSE-GRAPH（transpose.lisp）が逆向きにたどって vjp にする。

(in-package #:nabla)

(defun partition-eqns-by-dependence (eqns seeds)
  "EQNS（定義の順に並んだ EQN のリスト）を、SEEDS（VAR のリスト）に推移的に依存する
eqn（線形側）と、依存しない eqn（主値側）に分け、(values 主値側 線形側) を返す。
どちらも EQNS の中での順序を保つ。eqn が SEEDS に依存するとは、その入力の
どれかが SEEDS の var か、依存する eqn の出力であること。EQNS は書き換えない。"
  (let ((dependent (make-hash-table :test 'eq))
        (primal '())
        (linear '()))
    (dolist (v seeds) (setf (gethash v dependent) t))
    (dolist (eqn eqns)
      (if (some (lambda (v) (gethash v dependent)) (eqn-invars eqn))
          (progn (dolist (v (eqn-outvars eqn)) (setf (gethash v dependent) t))
                 (push eqn linear))
          (push eqn primal)))
    (values (nreverse primal) (nreverse linear))))

(defun %linearize-mixed-eqn-p (eqn)
  "EQN が、主値と接線を1つの eqn で計算する jvp の結果（while-loop / scan の jvp など）か:
複数出力で transpose ルールを持たない。cond の jvp は主値の cond と線形な cond に分けて出す
ので当てはまらない（src/ad/rules-control.lisp）。scan の jvp は、ここへ来る前に
%PARTIAL-EVAL-SPLIT（partial-eval.lisp）が主値の scan と線形な scan に分け、scan は transpose
ルールを持つので当てはまらない（issue #139）。"
  (and (primitive-multiple-outputs-p (eqn-prim eqn))
       (null (primitive-transpose (eqn-prim eqn)))))

(defun %linearize-reject-mixed-eqn (eqn)
  "EQN（%LINEARIZE-MIXED-EQN-P）を線形部分に分けられないので NO-TRANSPOSE-RULE
（AUTODIFF-ERROR の子。REQUIRE-TRANSPOSE-RULE が出す）にする。

注意: 主値と接線を1つの eqn で計算する高階プリミティブ（scan / while-loop）は、transpose
ルールを足しただけでは足りない。linearize が eqn を主値の高階プリミティブと線形の高階
プリミティブに分ける（JAX の partial eval。scan は #139）必要がある。分ける実装を入れる
までは、この検査が壊れた graph（MALFORMED-GRAPH）の代わりに止める。"
  (require-transpose-rule (eqn-prim eqn)))

(defstruct (linearization (:constructor make-linearization (primal-graph linear-graph n-outputs n-residuals))
                          (:copier nil))
  "LINEARIZE-GRAPH の結果。f の jvp を、主値だけを計算する graph と、接線について
線形な graph に分けたもの。

- PRIMAL-GRAPH: 入力は f の入力の主値。出力は f の出力の主値（N-OUTPUTS 個）に
  続けて、線形 graph が必要とする残差（residuals、N-RESIDUALS 個）。残差は主値 graph の
  var（f の入力そのものや中間値）で、線形 graph の係数になる。
- LINEAR-GRAPH: 入力は残差（N-RESIDUALS 個）に続けて、接線の入力（jvp-graph の NONZERO が
  真の入力ごとに1つ、その入力と同じ aval）。出力は f の出力ごとの接線（N-OUTPUTS 個、
  出力と同じ aval）。残差と接線の入力の両方に依存しない定数は、graph の定数として持つ。
  eqn はどれも接線の入力に推移的に依存する（DCE 済み）。接線に依存しない出力（ゼロの接線）は、
  その値を作る残差か定数がそのまま出力になる。
  2つの graph は var を共有する（残差は主値 graph の出力であり、線形 graph の入力）。"
  (primal-graph nil :type graph :read-only t)
  (linear-graph nil :type graph :read-only t)
  (n-outputs 0 :type (integer 0) :read-only t)
  (n-residuals 0 :type (integer 0) :read-only t))

(defun linearize-graph (graph &key (nonzero nil nonzero-p))
  "GRAPH を JVP-GRAPH（NONZERO は同じ意味）で jvp 変換し、PARTITION-EQNS-BY-DEPENDENCE
で接線の入力に依存する eqn とそれ以外に分けて、LINEARIZATION を返す。GRAPH 自体は
書き換えない。

主値 graph と線形 graph を順に評価すると、JVP-GRAPH を評価した結果（主値 ++ 接線）と
一致する。線形 graph には DCE-GRAPH をかけ、残差は線形 graph が実際に使うものだけに
絞る（主値 graph にも DCE をかける）。"
  (%linearize-jvp-graph (if nonzero-p (jvp-graph graph :nonzero nonzero) (jvp-graph graph))
                        (length (graph-invars graph))
                        (length (graph-outvars graph))))

(defun %linearize-jvp-graph (jvp n-primals n-outputs)
  "LINEARIZE-GRAPH の後半。JVP（JVP-GRAPH の結果と同じ形の graph。入力は主値 N-PRIMALS 個に
続けて接線、出力は主値 N-OUTPUTS 個に続けて接線。接線の個数は入力・出力とも任意）を
LINEARIZATION に分ける。cond の jvp ルールが、枝ごとに主値と接線を分けるのにも使う。"
  (let* ((primal-invars (subseq (graph-invars jvp) 0 n-primals))
         (tangent-invars (nthcdr n-primals (graph-invars jvp)))
         (primal-outvars (subseq (graph-outvars jvp) 0 n-outputs))
         (tangent-outvars (nthcdr n-outputs (graph-outvars jvp))))
   ;; scan など、主値と接線を1つの eqn で計算する eqn を、主値だけの eqn と接線に依存する
   ;; eqn に分ける（partial-eval.lisp。issue #139）。
   (multiple-value-bind (jvp-eqns extra-constants) (%partial-eval-split (graph-eqns jvp) tangent-invars)
    (let* ((constants (append (graph-constants jvp) extra-constants))
           (constant-table (make-hash-table :test 'eq))
           (seen (make-hash-table :test 'eq)))
    (loop for (var . nil) in constants do (setf (gethash var constant-table) t))
    (multiple-value-bind (primal-eqns linear-eqns)
        (partition-eqns-by-dependence jvp-eqns tangent-invars)
      ;; 主値の出力そのものが、主値と接線を混ぜた eqn の下流（線形側）にあるときは、DCE の
      ;; 前に拒否する（主値 graph がその出力を作れない。結果に効かない mixed eqn は
      ;; 下の DCE の後の検査で見逃す: JAX と同じく消えるだけ）。
      (let ((mixed (find-if #'%linearize-mixed-eqn-p linear-eqns)))
        (when (and mixed
                   (some (lambda (v) (member v primal-outvars :test #'eq))
                         (loop for eqn in linear-eqns append (eqn-outvars eqn))))
          (%linearize-reject-mixed-eqn mixed)))
      (let ((linear-defined (make-hash-table :test 'eq))
            (candidates '()))
        (dolist (v tangent-invars) (setf (gethash v linear-defined) t))
        (dolist (eqn linear-eqns)
          (dolist (v (eqn-outvars eqn)) (setf (gethash v linear-defined) t)))
        ;; 線形側が使う、線形側で定義されていない var のうち、定数でないもの = 残差の候補。
        (flet ((note (v)
                 (unless (or (gethash v linear-defined) (gethash v constant-table) (gethash v seen))
                   (setf (gethash v seen) t)
                   (push v candidates))))
          (dolist (eqn linear-eqns) (mapc #'note (eqn-invars eqn)))
          ;; ゼロの接線の出力は、jvp-graph が主値側の eqn（定数 + broadcast）で
          ;; instantiate したもの。線形側には eqn が無いので、その var は残差に
          ;; なり、線形 graph はそれをそのまま出力する（主値 graph が残差として出す）。
          (mapc #'note tangent-outvars))
        (setf candidates (nreverse candidates))
        (let* ((linear (dce-graph (make-graph (append candidates tangent-invars)
                                              linear-eqns tangent-outvars constants)))
               ;; DCE 後の線形側に、主値と接線を1つの eqn で計算するものが残っているなら拒否する。
               (checked (let ((mixed (find-if #'%linearize-mixed-eqn-p (graph-eqns linear))))
                          (when mixed (%linearize-reject-mixed-eqn mixed))))
               (used (let ((table (make-hash-table :test 'eq)))
                       (dolist (eqn (graph-eqns linear)) (dolist (v (eqn-invars eqn)) (setf (gethash v table) t)))
                       (dolist (v (graph-outvars linear)) (setf (gethash v table) t))
                       table))
               (residuals (remove-if-not (lambda (v) (gethash v used)) candidates))
               (linear (check-graph (make-graph (append residuals tangent-invars)
                                                (graph-eqns linear) (graph-outvars linear)
                                                (graph-constants linear))))
               (primal (dce-graph (make-graph primal-invars primal-eqns
                                              (append primal-outvars residuals) constants))))
          (declare (ignore checked))
          (make-linearization primal linear n-outputs (length residuals)))))))))
