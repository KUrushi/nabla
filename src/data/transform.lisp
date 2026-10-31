;;;; map / filter の変換（issue #281）。
;;;;
;;;; レコード（PyTree）1 つにかける純粋な変換（Grain の MapTransform / FilterTransform）。
;;;; 変換のリストはローダ（#285）の :transforms に渡し、ローダとワーカー（#286）は同じ
;;;; APPLY-TRANSFORMS でレコードを組み立てる。バッチにまとめるのは変換ではなくローダの
;;;; :batch-size の仕事（batch.lisp の BATCH-RECORDS）で、変換のリストには入れられない
;;;; （「batch はパイプラインの最後に 1 つだけ」を構造で守る）。

(in-package #:nabla.data)

(defstruct (map-transform (:constructor map-transform (function)) (:copier nil))
  "レコード → レコードの変換。MAP-TRANSFORM で作る。"
  (function nil :type (or function symbol) :read-only t))

(setf (documentation 'map-transform 'function)
      "レコードを受けて新しいレコードを返す関数 FUNCTION の変換を作る（Grain の MapTransform）。
FUNCTION はレコードの木をまるごと受ける（葉ごとではない）。変換はワーカースレッドから同時に
呼ばれるので、FUNCTION は純粋（同じレコードには同じ結果を返し、レコードも大域の状態も
書き換えない）でなければならない。")

(defstruct (filter-transform (:constructor filter-transform (predicate)) (:copier nil))
  "レコードを残すかどうかの述語。FILTER-TRANSFORM で作る。"
  (predicate nil :type (or function symbol) :read-only t))

(setf (documentation 'filter-transform 'function)
      "述語 PREDICATE が真を返すレコードだけを残す変換を作る（Grain の FilterTransform）。
MAP-TRANSFORM の FUNCTION と同じく、PREDICATE は純粋でなければならない。")

(defun apply-transforms (transforms record)
  "変換のリスト TRANSFORMS を先頭から順に RECORD にかける。どの filter も RECORD を落とさな
ければ (values 変換後のレコード t)、落としたら (values nil nil) を返す（落とした後の変換は
呼ばない）。TRANSFORMS に MAP-TRANSFORM / FILTER-TRANSFORM でないものがあれば TYPE-ERROR。"
  (dolist (transform transforms (values record t))
    (etypecase transform
      (map-transform
       (setf record (funcall (map-transform-function transform) record)))
      (filter-transform
       (unless (funcall (filter-transform-predicate transform) record)
         (return (values nil nil)))))))
