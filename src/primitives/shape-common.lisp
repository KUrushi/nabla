;;;; shape-common: reshape / broadcast-in-dim / transpose（issue #31 p4）が
;;;; 共有する小さなヘルパー。
;;;;
;;;; チェーンB（p4〜p6）は src/primitives/common.lisp（チェーンA所有）を
;;;; 編集しない約束なので、このファイルにチェーンB専用の複製を持つ
;;;; （wave 3 で重複を解消する予定。契約 §1 の DAMP の裏返し）。すべて
;;;; %SHAPE- 接頭辞にして、common.lisp の %-prefixed ヘルパーと名前が
;;;; 衝突しないようにしてある。
;;;;
;;;; reshape / broadcast-in-dim / transpose はどれも値をデコードせず、raw
;;;; storage を row-major-aref で操作する（CLAUDE.md の bf16/f16 表現の約束
;;;; と無関係に、あらゆる dtype—:i1 を含む—で同じコードが動く）。出力側の
;;;; index を driver にして入力側の index を逆算する（gather）ことで、
;;;; rank 0・size 0 の次元も特別扱いなしに扱える。

(in-package #:nabla)

(defun %shape-check-arity (name in-avals n)
  "IN-AVALS の個数が N でなければ PRIMITIVE-ERROR を signal する。"
  (unless (= (length in-avals) n)
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "入力の個数が違う: ~S 個渡されたが ~S 個必要"
           :format-arguments (list (length in-avals) n))))

(defun %shape-strides (shape)
  "SHAPE（非負整数のリスト）に対応する row-major のストライドのリストを
返す（末尾の次元のストライドが1）。"
  (let* ((rank (length shape))
         (strides (make-list rank :initial-element 1))
         (acc 1))
    (loop for i from (1- rank) downto 0
          do (setf (nth i strides) acc)
             (setf acc (* acc (nth i shape))))
    strides))

(defun %shape-row-major-index (subscripts shape &optional (strides (%shape-strides shape)))
  "SUBSCRIPTS（各次元の添字のリスト）と SHAPE から row-major のインデックス
を返す。同じ SHAPE に対して繰り返し呼ぶ場合（%SHAPE-EAGER-FILL のように
出力の各要素ごとに呼ぶ場合など）は、STRIDES を一度だけ計算して渡すと
そのたびの再計算を避けられる。"
  (reduce #'+ (mapcar #'* subscripts strides) :initial-value 0))

(defun %shape-subscripts (index shape &optional (strides (%shape-strides shape)))
  "row-major の INDEX を、SHAPE の各次元ごとの添字のリストに変換する。
STRIDES は %SHAPE-ROW-MAJOR-INDEX と同じ、再計算を避けるための任意引数。"
  (mapcar (lambda (stride dim) (mod (floor index stride) dim))
          strides
          shape))

(defun %shape-eager-fill (out-shape dtype index->in-index array)
  "OUT-SHAPE・DTYPE の新しい配列を作り、各出力インデックス I について
(row-major-aref result I) に (row-major-aref array (funcall index->in-index I))
を代入して返す。ARRAY・出力とも raw storage をそのまま row-major-aref で
読み書きする（デコードしない）ので、あらゆる dtype（:i1 含む）で使える。"
  (let ((result (make-array out-shape :element-type (dtype-element-type dtype))))
    (dotimes (i (array-total-size result) result)
      (setf (row-major-aref result i)
            (row-major-aref array (funcall index->in-index i))))))
