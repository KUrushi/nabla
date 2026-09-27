;;;; reference-*: テストの中でループを書かずに済むための、素朴な参照実装
;;;; （issue #8 の補足）。
;;;;
;;;; 入力は CL の実数（single-float / double-float）を要素に持つ Lisp 配列。
;;;; 計算は DOUBLE-FLOAT で行い、返り値も (simple-array double-float shape)
;;;; にする（f32 の実測値と比べるときは、allclose の :dtype :f32 が両辺を
;;;; double-float にデコードしてから比べるので、そのまま使える）。
;;;;
;;;; ここは JAX との数値一致フィクスチャや PBT の対象そのものではなく、
;;;; 「期待値」を計算するオラクルなので、賢いことはせず素朴なループで書く
;;;; （CLAUDE.md / このスキルの「テストの中にロジックを書かない」の裏返し
;;;; ——ロジックはここに集め、各テスト本体には書かない）。

(in-package #:nabla.tests.support)

(defun reference-add (a b)
  "A + B（要素ごとの和）を返す。A と B は同じ shape を持つこと。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-add: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (+ (coerce (row-major-aref a i) 'double-float)
               (coerce (row-major-aref b i) 'double-float))))))

;;; issue #31 p5: dot-general の参照実装。REFERENCE-MATMUL は
;;; REFERENCE-DOT-GENERAL を lhs-contracting (1) / rhs-contracting (0) /
;;; batch なしで呼ぶ薄いラッパーにする。

(defun reference-dot-general (a b &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  "A・B（DOUBLE-FLOAT にデコード済みの配列）の stablehlo.dot_general を
素朴に計算して返す。LHS-CONTRACTING / RHS-CONTRACTING / LHS-BATCH /
RHS-BATCH は A・B それぞれの次元番号のリスト（対応するペアは同じ長さ・
同じサイズであること）。出力の shape は batch dims（LHS-BATCH の順）→
A の free dims（昇順）→ B の free dims（昇順）の順（用語集の
「contracting dims / batch dims」参照）。"
  (let* ((a-shape (array-dimensions a))
         (b-shape (array-dimensions b))
         (a-excluded (append lhs-batch lhs-contracting))
         (b-excluded (append rhs-batch rhs-contracting))
         (a-free (sort (set-difference (loop for i below (length a-shape) collect i) a-excluded) #'<))
         (b-free (sort (set-difference (loop for i below (length b-shape) collect i) b-excluded) #'<))
         (n-batch (length lhs-batch))
         (n-a-free (length a-free))
         (batch-sizes (mapcar (lambda (d) (nth d a-shape)) lhs-batch))
         (a-free-sizes (mapcar (lambda (d) (nth d a-shape)) a-free))
         (b-free-sizes (mapcar (lambda (d) (nth d b-shape)) b-free))
         (contract-sizes (mapcar (lambda (d) (nth d a-shape)) lhs-contracting))
         (out-shape (append batch-sizes a-free-sizes b-free-sizes))
         (result (make-array out-shape :element-type 'double-float :initial-element 0.0d0)))
    (dotimes (i (array-total-size result) result)
      (let* ((out-subs (%row-major-index->subscripts i out-shape))
             (batch-subs (subseq out-subs 0 n-batch))
             (a-free-subs (subseq out-subs n-batch (+ n-batch n-a-free)))
             (b-free-subs (subseq out-subs (+ n-batch n-a-free)))
             (a-subs (make-list (length a-shape) :initial-element 0))
             (b-subs (make-list (length b-shape) :initial-element 0)))
        (loop for d in lhs-batch for s in batch-subs do (setf (nth d a-subs) s))
        (loop for d in rhs-batch for s in batch-subs do (setf (nth d b-subs) s))
        (loop for d in a-free for s in a-free-subs do (setf (nth d a-subs) s))
        (loop for d in b-free for s in b-free-subs do (setf (nth d b-subs) s))
        (let ((sum 0.0d0))
          (dotimes (c (reduce #'* contract-sizes :initial-value 1))
            (let ((c-subs (%row-major-index->subscripts c contract-sizes)))
              (loop for d in lhs-contracting for s in c-subs do (setf (nth d a-subs) s))
              (loop for d in rhs-contracting for s in c-subs do (setf (nth d b-subs) s))
              (incf sum (* (coerce (apply #'aref a a-subs) 'double-float)
                           (coerce (apply #'aref b b-subs) 'double-float)))))
          (setf (row-major-aref result i) sum))))))

(defun reference-matmul (a b)
  "A @ B（行列積）を返す。A・B は rank 2 で、A の列数と B の行数が一致すること。
REFERENCE-DOT-GENERAL を lhs-contracting (1) / rhs-contracting (0) /
batch なしで呼ぶ薄いラッパー。"
  (unless (and (= (array-rank a) 2) (= (array-rank b) 2))
    (error "reference-matmul: rank 2 の配列だけを受け付ける: ~A と ~A"
           (array-dimensions a) (array-dimensions b)))
  (destructuring-bind (m k) (array-dimensions a)
    (declare (ignore m))
    (destructuring-bind (k2 n) (array-dimensions b)
      (declare (ignore n))
      (unless (= k k2)
        (error "reference-matmul: A の列数 ~A と B の行数 ~A が一致しない" k k2))))
  (reference-dot-general a b :lhs-contracting '(1) :rhs-contracting '(0) :lhs-batch '() :rhs-batch '()))

(defun reference-reduce-sum (a axis)
  "A の dimension AXIS に沿った総和を返す（結果の rank は A の rank - 1）。"
  (let* ((shape (array-dimensions a))
         (rank (length shape)))
    (unless (< -1 axis rank)
      (error "reference-reduce-sum: axis ~A が rank ~A の範囲外" axis rank))
    (let* ((out-shape (append (subseq shape 0 axis) (subseq shape (1+ axis))))
           (result (make-array out-shape :element-type 'double-float :initial-element 0.0d0)))
      (dotimes (i (array-total-size a) result)
        (let* ((in-index (%row-major-index->subscripts i shape))
               (out-index (append (subseq in-index 0 axis) (subseq in-index (1+ axis)))))
          (incf (apply #'aref result out-index)
                (coerce (row-major-aref a i) 'double-float)))))))

(defun %row-major-index->subscripts (row-major-index shape)
  "SHAPE を持つ配列の ROW-MAJOR-INDEX を、各次元ごとの添字のリストにする
（row-major なので末尾の次元から割っていく）。"
  (let ((reversed-subscripts nil))
    (dolist (dim (reverse shape) reversed-subscripts)
      (push (mod row-major-index dim) reversed-subscripts)
      (setf row-major-index (floor row-major-index dim)))))

;;; issue #31 p4: reshape / broadcast-in-dim / transpose の参照実装。
;;;
;;; 3つとも値を解釈しない構造だけの演算なので、DOUBLE-FLOAT にデコード
;;; 済みの配列を受け取り DOUBLE-FLOAT の配列を返す（decode は要素ごとに
;;; 同じ規則で行うので、reshape / broadcast-in-dim / transpose のような
;;; 要素の並べ替えとは可換。decode してから並べ替えても、並べ替えてから
;;; decode しても同じ結果になる）。

(defun reference-reshape (a shape)
  "A の要素を row-major 順のまま SHAPE に並べ替えて返す。A の要素数と SHAPE
の積が一致すること。"
  (unless (= (array-total-size a) (reduce #'* shape :initial-value 1))
    (error "reference-reshape: 要素数が一致しない: ~A → ~A" (array-dimensions a) shape))
  (let ((result (make-array shape :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (coerce (row-major-aref a i) 'double-float)))))

(defun reference-broadcast-in-dim (a shape dims)
  "A（rank (length DIMS)）を StableHLO の broadcast_in_dim の意味で SHAPE に
広げる。DIMS[i] は A の次元 i が対応する SHAPE 側の次元、A の次元 i は
1 か SHAPE[DIMS[i]] のどちらかであること。"
  (let ((in-shape (array-dimensions a))
        (result (make-array shape :element-type 'double-float)))
    (dotimes (i (array-total-size result) result)
      (let* ((out-subscripts (%row-major-index->subscripts i shape))
             (in-subscripts (loop for operand-dim in in-shape
                                   for target-dim in dims
                                   collect (if (= operand-dim 1) 0 (nth target-dim out-subscripts)))))
        (setf (row-major-aref result i)
              (coerce (apply #'aref a in-subscripts) 'double-float))))))

(defun reference-transpose (a perm)
  "A の次元を PERM（0 始まりの permutation）の順に並べ替えて返す。"
  (let* ((in-shape (array-dimensions a))
         (out-shape (mapcar (lambda (p) (nth p in-shape)) perm))
         (result (make-array out-shape :element-type 'double-float)))
    (dotimes (i (array-total-size result) result)
      (let ((out-subscripts (%row-major-index->subscripts i out-shape))
            (in-subscripts (make-list (length in-shape))))
        (loop for p in perm
              for s in out-subscripts
              do (setf (nth p in-subscripts) s))
        (setf (row-major-aref result i) (coerce (apply #'aref a in-subscripts) 'double-float))))))
