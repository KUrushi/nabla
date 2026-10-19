;;;; attention のマスクの部品（issue #208）。ホスト側の bit 配列を作る。
;;;;
;;;; Flax の make_causal_mask / make_attention_mask / combine_masks に相当するが、bit 配列を返し、
;;;; make-causal-mask は入力の配列でなく長さを取って (1 長さ 長さ) を返す（先頭のバッチ軸には右から揃える
;;;; ブロードキャストが効く）。multi-head-attention の mask 引数に渡す（1 が「見える」）。

(in-package #:nabla.nn)

;;; --- マスクの部品（ホスト側の bit 配列）---

(defun %check-bit-array (name mask)
  (unless (and (arrayp mask) (equal (array-element-type mask) 'bit))
    (error "~A: マスクは bit の配列でなければならない: ~S" name mask)))

(defun make-causal-mask (length)
  "長さ LENGTH の因果マスク: 形 (1 LENGTH LENGTH) の bit 配列で、クエリ位置 q とキー位置 k について
k <= q のとき 1（見える）。先頭の 1 は、multi-head-attention の [... ヘッド q k] に右から揃えて
ブロードキャストされる（ヘッドとバッチ軸に共通）。LENGTH は 1 以上の整数。"
  (unless (and (integerp length) (plusp length))
    (error "make-causal-mask: 長さは正の整数でなければならない: ~S" length))
  (let ((mask (make-array (list 1 length length) :element-type 'bit :initial-element 0)))
    (dotimes (q length mask)
      (dotimes (k (1+ q))
        (setf (aref mask 0 q k) 1)))))

(defun make-attention-mask (query-valid key-valid)
  "QUERY-VALID [... q] と KEY-VALID [... k]（bit の配列。1 が有効な位置。先頭の軸は同じ）から、
形 [... 1 q k] のマスクを作る（q 番目のクエリと k 番目のキーがどちらも有効なとき 1）。padding の
マスクに使う。1 はヘッドの軸。"
  (%check-bit-array "make-attention-mask" query-valid)
  (%check-bit-array "make-attention-mask" key-valid)
  (let ((query-shape (array-dimensions query-valid))
        (key-shape (array-dimensions key-valid)))
    (unless (and query-shape key-shape (equal (butlast query-shape) (butlast key-shape)))
      (error "make-attention-mask: 先頭の軸が同じで rank 1 以上の配列でなければならない: ~S と ~S"
             query-shape key-shape))
    (let* ((lead (butlast query-shape))
           (q (car (last query-shape)))
           (k (car (last key-shape)))
           (rows (reduce #'* lead))
           (mask (make-array (append lead (list 1 q k)) :element-type 'bit)))
      (dotimes (row rows mask)
        (dotimes (qi q)
          (dotimes (ki k)
            (setf (row-major-aref mask (+ (* row q k) (* qi k) ki))
                  (logand (row-major-aref query-valid (+ (* row q) qi))
                          (row-major-aref key-valid (+ (* row k) ki))))))))))

(defun %broadcast-shape (name shapes)
  "SHAPES（右に揃える）をブロードキャストした形。合わなければ error。"
  (let* ((rank (reduce #'max (mapcar #'length shapes)))
         (padded (mapcar (lambda (shape) (append (make-list (- rank (length shape)) :initial-element 1) shape))
                         shapes)))
    (loop for axis below rank
          collect (let ((sizes (remove 1 (mapcar (lambda (shape) (nth axis shape)) padded))))
                    (unless (= 1 (length (remove-duplicates sizes)))
                      (when sizes
                        (error "~A: マスクの形がブロードキャストできない: ~S" name shapes)))
                    (if sizes (first sizes) 1)))))

(defun combine-masks (&rest masks)
  "MASKS（bit の配列。NIL は無視する）の論理積。ランクが違えば右に揃えてブロードキャストする
\(大きさ 1 の軸は広がる）。全部 NIL なら NIL。因果マスクと padding マスクを合わせるのに使う。"
  (let ((masks (remove nil masks)))
    (when masks
      (dolist (mask masks) (%check-bit-array "combine-masks" mask))
      (let* ((shape (%broadcast-shape "combine-masks" (mapcar #'array-dimensions masks)))
             (result (make-array shape :element-type 'bit :initial-element 1))
             (views (mapcar (lambda (mask)
                              (let ((dims (array-dimensions mask)))
                                (make-array (append (make-list (- (length shape) (length dims)) :initial-element 1) dims)
                                            :element-type 'bit :displaced-to mask)))
                            masks)))
        (dotimes (i (array-total-size result) result)
          (let ((index (let ((rest i))
                         (reverse (loop for size in (reverse shape)
                                        collect (multiple-value-bind (quotient remainder) (floor rest size)
                                                  (setf rest quotient)
                                                  remainder))))))
            (setf (row-major-aref result i)
                  (reduce #'logand
                          (mapcar (lambda (view)
                                    (apply #'aref view (mapcar (lambda (size j) (if (= size 1) 0 j))
                                                               (array-dimensions view) index)))
                                  views)))))))))
