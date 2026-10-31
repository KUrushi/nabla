;;;; data-source と array-source（issue #279）。
;;;;
;;;; data-source は Grain の RandomAccessDataSource に当たる「長さがあり、添字でレコードを
;;;; 取れる」抽象。サンプラーが決めた添字をローダがここで引く（設計タブ「Grain相当層の設計」）。
;;;; レコードは PyTree（core の tree-flatten が扱う木）で、ローダがバッチにして jit に渡す。

(in-package #:nabla.data)

;;; --- コンディション ---

(define-condition data-error (error)
  ()
  (:documentation
   "nabla/data が signal するコンディションの親。"))

(define-condition data-source-error (data-error)
  ((reason :initarg :reason :reader data-source-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader data-source-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "data-source: ~A (~S)"
                     (data-source-error-detail c) (data-source-error-reason c))))
  (:documentation
   "データソースを作れない、またはレコードを取れないときに signal する。REASON は次のどれか。
  :no-leaves          MAKE-ARRAY-SOURCE の木に葉が無い（長さが決まらない）
  :bad-leaf           配列でない葉
  :rank-zero-leaf     rank 0 の葉（レコードの軸が無い）
  :length-mismatch    葉の先頭の軸の長さが揃わない
  :index-out-of-range SOURCE-ITEM の添字が 0 以上 SOURCE-LENGTH 未満の整数でない"))

(defun %data-source-error (reason control &rest arguments)
  (error 'data-source-error :reason reason :detail (apply #'format nil control arguments)))

;;; --- プロトコル ---

(defclass data-source ()
  ()
  (:documentation
   "長さがあり、添字でレコードを取れるデータの抽象（Grain の RandomAccessDataSource）。
サブクラスは SOURCE-LENGTH と SOURCE-ITEM のメソッドを定義する。レコードは PyTree で、
同じ添字には常に同じレコードを返す（決定的な再開の前提）。"))

(defgeneric source-length (source)
  (:documentation
   "SOURCE のレコードの個数（0 以上の整数）を返す。"))

(defgeneric source-item (source index)
  (:documentation
   "SOURCE の INDEX 番目（0 始まり）のレコードを返す。INDEX が 0 以上 SOURCE-LENGTH 未満の
整数でなければ、:index-out-of-range の DATA-SOURCE-ERROR を signal する。"))

(defun %check-index (source index)
  (unless (and (integerp index) (< -1 index (source-length source)))
    (%data-source-error :index-out-of-range
                        "添字 ~S は 0 以上 ~D 未満の整数でない" index (source-length source))))

;;; --- array-source ---

(defclass array-source (data-source)
  ((leaves :initarg :leaves :reader %array-source-leaves
           :documentation "木の葉（先頭の軸がレコードの軸の配列）のリスト。")
   (treedef :initarg :treedef :reader %array-source-treedef)
   (length :initarg :length :reader source-length))
  (:documentation
   "メモリ上の配列の木から作るデータソース。葉の先頭の軸がレコードの軸で、INDEX 番目の
レコードは、各葉をその軸の INDEX 番目の行に置き換えた同じ形の木。MAKE-ARRAY-SOURCE で作る。"))

(defun make-array-source (tree)
  "PyTree TREE から ARRAY-SOURCE を作る。TREE の葉はどれも rank 1 以上の配列で、先頭の軸の
長さ（＝レコードの個数。0 でもよい）が揃っている必要がある。element-type は問わない
（(unsigned-byte 8) の画素のように nabla の dtype に無いものも、ローダの変換で直せばよい）。
葉が無い・配列でない葉・rank 0 の葉・長さの食い違いは DATA-SOURCE-ERROR。
TREE の配列はコピーせずに持つので、作った後で書き換えないこと。"
  (multiple-value-bind (leaves treedef) (nabla:tree-flatten tree)
    (when (null leaves)
      (%data-source-error :no-leaves "木 ~S に葉が無いので、レコードの個数が決まらない" tree))
    (dolist (leaf leaves)
      (unless (arrayp leaf)
        (%data-source-error :bad-leaf "葉 ~S が配列でない" leaf))
      (when (zerop (array-rank leaf))
        (%data-source-error :rank-zero-leaf "葉 ~S が rank 0 で、レコードの軸が無い" leaf)))
    (let ((length (array-dimension (first leaves) 0)))
      (dolist (leaf (rest leaves))
        (unless (= length (array-dimension leaf 0))
          (%data-source-error :length-mismatch "葉の先頭の軸の長さが揃わない（~D と ~D）"
                              length (array-dimension leaf 0))))
      (make-instance 'array-source :leaves leaves :treedef treedef :length length))))

(defun %leading-row (array index)
  "ARRAY の先頭の軸の INDEX 番目の行を、同じ element-type の新しい配列にして返す。"
  (let* ((row (make-array (rest (array-dimensions array))
                          :element-type (array-element-type array)))
         (row-size (array-total-size row))
         (start (* index row-size)))
    (dotimes (j row-size row)
      (setf (row-major-aref row j) (row-major-aref array (+ start j))))))

(defmethod source-item ((source array-source) index)
  (%check-index source index)
  (nabla:tree-unflatten (%array-source-treedef source)
                        (mapcar (lambda (leaf) (%leading-row leaf index))
                                (%array-source-leaves source))))
