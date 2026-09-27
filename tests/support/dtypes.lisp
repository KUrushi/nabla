;;;; dtype の一覧と、dtype ごとの既定の許容誤差。

(in-package #:nabla.tests.support)

(defparameter *dtypes* '(:f32 :f64 :bf16 :f16)
  "nabla がサポートする dtype の一覧。")

(defun dtype-tolerance (dtype)
  "DTYPE の既定の許容誤差を (values rtol atol) で返す。

数値はいずれも SKILL.md の表のとおり:
f64 は 1e-12/1e-12、f32 は 1e-5/1e-6、bf16/f16 は 1e-2/1e-3。i1（issue #37）
は 0 / 1 の厳密な値しか取らないので許容誤差は 0。"
  (ecase dtype
    (:f64 (values 1d-12 1d-12))
    (:f32 (values 1d-5 1d-6))
    (:bf16 (values 1d-2 1d-3))
    (:f16 (values 1d-2 1d-3))
    (:i1 (values 0d0 0d0))))

(defparameter *float-dtype-precision-rank* '((:bf16 . 0) (:f16 . 0) (:f32 . 1) (:f64 . 2))
  "浮動小数点 dtype を精度の低い順に並べた順位（小さいほど精度が低い）。
GRAPH-WORST-FLOAT-DTYPE が「最も粗い精度」を選ぶのに使う。i1 など、
浮動小数点でない dtype はここに現れない。")

(defun graph-worst-float-dtype (graph)
  "GRAPH の OUTVARS から eqn の INVARS を逆にたどって届く var（invars・
constants・中間 eqn の出力）だけを対象に、浮動小数点 dtype で最も精度が
低いもの（bf16/f16 < f32 < f64）を返す。浮動小数点 dtype が1つも無ければ
NIL を返す。

数値比較の許容誤差は、graph の最終的な出力 dtype ではなく、その出力の
計算に実際に使われた最も粗い精度で決めないといけない。たとえば bf16 の
入力を reduce-sum → tanh してから f32 に convert する graph は、出力こそ
f32 だが値そのものは bf16 の丸め誤差を引き継いでいる。さらに IREE は
エレメントワイズ演算を融合して1回だけ丸めるのに対し eager は演算ごとに
丸めるため、bf16/f16 サイズの丸め誤差が出力の dtype（ここでは f32）の
厳しい許容誤差では吸収できず、スプリアスな失敗になる（issue #33 の
リグレッション。STABLEHLO/IREE-MATCHES-EVAL-GRAPH 参照）。

OUTVARS から辿れない var（PRIMITIVE-GRAPH-RECIPE は各ステップで過去の
任意の IDX を参照できるので、出力に一切寄与しない枝が生まれうる。
MAKE-GRAPH はそれを刈らない）は無視する。GRAPH 全体を無条件に走査すると、
出力が実は f32 だけで計算されているのに、無関係な bf16 の枝に釣られて
粗い許容誤差を選んでしまい、emit-stablehlo の本当の f32 精度のバグを
見逃しかねない。"
  (let ((worst nil)
        (worst-rank nil)
        (visited (make-hash-table :test #'eq))
        (producer (make-hash-table :test #'eq)))
    (dolist (eqn (nb:graph-eqns graph))
      (dolist (v (nb:eqn-outvars eqn))
        (setf (gethash v producer) eqn)))
    (labels ((consider (dtype)
               (let ((rank (cdr (assoc dtype *float-dtype-precision-rank*))))
                 (when (and rank (or (null worst-rank) (< rank worst-rank)))
                   (setf worst dtype worst-rank rank))))
             (visit (var)
               (unless (gethash var visited)
                 (setf (gethash var visited) t)
                 (consider (nb:aval-dtype (nb:var-aval var)))
                 (let ((producing-eqn (gethash var producer)))
                   (when producing-eqn
                     (dolist (in (nb:eqn-invars producing-eqn)) (visit in)))))))
      (dolist (v (nb:graph-outvars graph)) (visit v)))
    worst))
