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
  (let* ((jvp (if nonzero-p (jvp-graph graph :nonzero nonzero) (jvp-graph graph)))
         (n-primals (length (graph-invars graph)))
         (n-outputs (length (graph-outvars graph)))
         (primal-invars (subseq (graph-invars jvp) 0 n-primals))
         (tangent-invars (nthcdr n-primals (graph-invars jvp)))
         (primal-outvars (subseq (graph-outvars jvp) 0 n-outputs))
         (tangent-outvars (nthcdr n-outputs (graph-outvars jvp)))
         (constants (graph-constants jvp))
         (constant-table (make-hash-table :test 'eq))
         (seen (make-hash-table :test 'eq)))
    (loop for (var . nil) in constants do (setf (gethash var constant-table) t))
    (multiple-value-bind (primal-eqns linear-eqns)
        (partition-eqns-by-dependence (graph-eqns jvp) tangent-invars)
      ;; 複数出力の eqn が線形側に入り、transpose ルールを持たないのは、主値と接線を
      ;; 1つの eqn で計算している（while-loop の jvp など）とき。主値の部分を
      ;; 主値 graph へ分けられず壊れた graph になるので、どのプリミティブか分かる
      ;; エラーにする（cond は後で partial eval で分けてからここに来る）。
      (dolist (eqn linear-eqns)
        (when (and (primitive-multiple-outputs-p (eqn-prim eqn))
                   (null (primitive-transpose (eqn-prim eqn))))
          (error 'autodiff-error
                 :format-control "プリミティブ ~S の jvp は主値と接線を1つの eqn で計算するので、線形部分に分けられない（逆モードは対応していない。前進モードの jvp だけ使える）"
                 :format-arguments (list (primitive-name (eqn-prim eqn))))))
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
          (make-linearization primal linear n-outputs (length residuals)))))))
