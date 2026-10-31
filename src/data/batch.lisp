;;;; レコードをバッチに積む（issue #281）。
;;;;
;;;; バッチはレコードの PyTree の葉を新しい先頭の軸で積んだ同じ形の木（Grain の Batch）。
;;;; 積むのはローダの :batch-size の仕事で、変換（transform.lisp）のリストには入れられない。
;;;; 端数のバッチを捨てる（drop-remainder）のもローダの仕事で、ここでは 1 個以上の任意の
;;;; 個数を積む。端数を詰める pad は v1 では持たない。

(in-package #:nabla.data)

(define-condition batch-error (data-error)
  ((reason :initarg :reason :reader batch-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader batch-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "batch: ~A (~S)" (batch-error-detail c) (batch-error-reason c))))
  (:documentation
   "BATCH-RECORDS がレコードを積めないときに signal する。REASON は次のどれか。
  :no-records             レコードが 1 つも無い
  :no-leaves              レコードに葉が無い（バッチの軸を作れない）
  :bad-leaf               配列でない葉
  :treedef-mismatch       レコードの木の形（treedef）が最初のレコードと違う
  :shape-mismatch         葉の形が最初のレコードの同じ位置の葉と違う
  :element-type-mismatch  葉の element-type が最初のレコードの同じ位置の葉と違う"))

(defun %batch-error (reason control &rest arguments)
  (error 'batch-error :reason reason :detail (apply #'format nil control arguments)))

(defun %same-element-type-p (a b)
  (equal (array-element-type a) (array-element-type b)))

(defun %check-record-leaves (k leaves first-leaves)
  "K 番目のレコードの葉 LEAVES が、最初のレコードの葉 FIRST-LEAVES と形・element-type で揃う
ことを確かめる。"
  (loop for leaf in leaves
        for first in first-leaves
        do (unless (arrayp leaf)
             (%batch-error :bad-leaf "~D 番目のレコードの葉 ~S が配列でない" k leaf))
           (unless (%same-element-type-p leaf first)
             (%batch-error :element-type-mismatch
                           "~D 番目のレコードの葉の element-type ~S が最初のレコードの ~S と違う"
                           k (array-element-type leaf) (array-element-type first)))
           (unless (equal (array-dimensions leaf) (array-dimensions first))
             (%batch-error :shape-mismatch "~D 番目のレコードの葉の形 ~S が最初のレコードの ~S と違う"
                           k (array-dimensions leaf) (array-dimensions first)))))

(defun %stack-leaves (rows)
  "形と element-type の揃った配列のリスト ROWS を、新しい先頭の軸で積んだ配列。"
  (let* ((first (first rows))
         (element-type (array-element-type first))
         (row-size (array-total-size first))
         (stacked (make-array (cons (length rows) (array-dimensions first))
                              :element-type element-type))
         (flat (make-array (array-total-size stacked) :element-type element-type
                                                      :displaced-to stacked)))
    ;; 行ごとに REPLACE でまとめて写す（要素ごとの ROW-MAJOR-AREF より桁違いに速い）。
    (loop for row in rows
          for k from 0
          do (replace flat (make-array row-size :element-type element-type :displaced-to row)
                      :start1 (* k row-size)))
    stacked))

(defun batch-records (records)
  "レコード（PyTree）のリスト RECORDS を、各葉を新しい先頭の軸で順に積んだ同じ形の木にする。
バッチの葉の j 番目の行は、j 番目のレコードの同じ位置の葉（形 (length RECORDS) . 葉の形、
同じ element-type の新しい配列。rank 0 の葉は 1 次元になる）。

レコードは 1 つ以上で、葉を 1 つ以上持ち、どれも最初のレコードと同じ treedef（NABLA:TREEDEF=）、
同じ位置の葉が同じ形・同じ element-type の配列でなければならない（そうでなければ BATCH-ERROR）。
(unsigned-byte 16) の葉（bf16 / f16）は element-type の一致だけを見る（どちらの dtype かは
data 層では決めない）。ホストの Lisp 配列だけを扱い、device-array は渡さない。"
  (when (null records)
    (%batch-error :no-records "積むレコードが無い"))
  (multiple-value-bind (first-leaves treedef) (nabla:tree-flatten (first records))
    (when (null first-leaves)
      (%batch-error :no-leaves "レコード ~S に葉が無いので、バッチの軸を作れない" (first records)))
    (dolist (leaf first-leaves)
      (unless (arrayp leaf)
        (%batch-error :bad-leaf "0 番目のレコードの葉 ~S が配列でない" leaf)))
    (let ((rows (cons (coerce first-leaves 'simple-vector)
                      (loop for record in (rest records)
                            for k from 1
                            collect (multiple-value-bind (leaves record-treedef)
                                        (nabla:tree-flatten record)
                                      (unless (nabla:treedef= treedef record-treedef)
                                        (%batch-error :treedef-mismatch
                                                      "~D 番目のレコードの木の形が最初のレコードと違う" k))
                                      (%check-record-leaves k leaves first-leaves)
                                      (coerce leaves 'simple-vector))))))
      (nabla:tree-unflatten treedef
                            (loop for i from 0 below (length first-leaves)
                                  collect (%stack-leaves
                                           (mapcar (lambda (leaves) (svref leaves i)) rows)))))))
