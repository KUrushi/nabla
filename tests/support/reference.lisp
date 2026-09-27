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

;; issue #31 p1
(defun reference-sub (a b)
  "A - B（要素ごとの差）を返す。A と B は同じ shape を持つこと。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-sub: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (- (coerce (row-major-aref a i) 'double-float)
               (coerce (row-major-aref b i) 'double-float))))))

;; issue #31 p1
(defun reference-mul (a b)
  "A * B（要素ごとの積）を返す。A と B は同じ shape を持つこと。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-mul: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (* (coerce (row-major-aref a i) 'double-float)
               (coerce (row-major-aref b i) 'double-float))))))

;; issue #31 p1
(defun reference-div (a b)
  "A / B（要素ごとの商）を返す。A と B は同じ shape を持つこと。B に0を
渡すとそのまま DOUBLE-FLOAT の除算に委ねる（reference-* には浮動小数点
トラップのマスクを意図的に足していない。0除算を確かめるテストは eager
側を直接呼ぶ）。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-div: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (/ (coerce (row-major-aref a i) 'double-float)
               (coerce (row-major-aref b i) 'double-float))))))

(defun reference-matmul (a b)
  "A @ B（行列積）を返す。A・B は rank 2 で、A の列数と B の行数が一致すること。"
  (unless (and (= (array-rank a) 2) (= (array-rank b) 2))
    (error "reference-matmul: rank 2 の配列だけを受け付ける: ~A と ~A"
           (array-dimensions a) (array-dimensions b)))
  (destructuring-bind (m k) (array-dimensions a)
    (destructuring-bind (k2 n) (array-dimensions b)
      (unless (= k k2)
        (error "reference-matmul: A の列数 ~A と B の行数 ~A が一致しない" k k2))
      (let ((result (make-array (list m n) :element-type 'double-float :initial-element 0.0d0)))
        (dotimes (i m result)
          (dotimes (j n)
            (let ((sum 0.0d0))
              (dotimes (p k)
                (incf sum (* (coerce (aref a i p) 'double-float)
                             (coerce (aref b p j) 'double-float))))
              (setf (aref result i j) sum))))))))

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
