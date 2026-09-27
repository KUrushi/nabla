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

(defun reference-matmul (a b)
  "A @ B（行列積）を返す。A・B は rank 2 で、A の列数と B の行数が一致すること。

issue #31 p5 以降は REFERENCE-DOT-GENERAL の薄いラッパー（contracting
(1)/(0)、batch 無し）。"
  (unless (and (= (array-rank a) 2) (= (array-rank b) 2))
    (error "reference-matmul: rank 2 の配列だけを受け付ける: ~A と ~A"
           (array-dimensions a) (array-dimensions b)))
  (unless (= (second (array-dimensions a)) (first (array-dimensions b)))
    (error "reference-matmul: A の列数 ~A と B の行数 ~A が一致しない"
           (second (array-dimensions a)) (first (array-dimensions b))))
  (reference-dot-general a b '(1) '(0) '() '()))

(defun reference-reduce-sum (a axes)
  "A の AXES に沿った総和を返す。AXES は整数1つでも、整数のリストでもよい
（後方互換: 既存の呼び出し元は整数1つを渡す。issue #31 p6 で拡張）。"
  (let* ((axes (if (listp axes) axes (list axes)))
         (shape (array-dimensions a))
         (rank (length shape)))
    (dolist (axis axes)
      (unless (< -1 axis rank)
        (error "reference-reduce-sum: axis ~A が rank ~A の範囲外" axis rank)))
    (let* ((out-shape (loop for d in shape for i from 0 unless (member i axes) collect d))
           (result (make-array out-shape :element-type 'double-float :initial-element 0.0d0)))
      (dotimes (i (array-total-size a) result)
        (let* ((in-index (%row-major-index->subscripts i shape))
               (out-index (loop for s in in-index for d from 0 unless (member d axes) collect s)))
          (incf (apply #'aref result out-index)
                (coerce (row-major-aref a i) 'double-float)))))))

;;; issue #31 p6: reduce-max の参照実装。

(defun reference-reduce-max (a axes)
  "A の AXES（整数1つ、または整数のリスト）に沿った最大値を返す。NaN が
含まれるスライスの出力要素は NaN になる（reduce-max の :EAGER と同じ
意味論のオラクル）。"
  (let* ((axes (if (listp axes) axes (list axes)))
         (shape (array-dimensions a))
         (rank (length shape)))
    (dolist (axis axes)
      (unless (< -1 axis rank)
        (error "reference-reduce-max: axis ~A が rank ~A の範囲外" axis rank)))
    (let* ((out-shape (loop for d in shape for i from 0 unless (member i axes) collect d))
           (result (make-array out-shape :element-type 'double-float
                                :initial-element sb-ext:double-float-negative-infinity)))
      (dotimes (i (array-total-size a) result)
        (let* ((in-index (%row-major-index->subscripts i shape))
               (out-index (loop for s in in-index for d from 0 unless (member d axes) collect s))
               (v (coerce (row-major-aref a i) 'double-float))
               (cur (apply #'aref result out-index)))
          (setf (apply #'aref result out-index)
                (cond
                  ((sb-ext:float-nan-p cur) cur)
                  ((sb-ext:float-nan-p v) v)
                  (t (max cur v)))))))))

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

;;; issue #31 p5: dot-general の参照実装。REFERENCE-MATMUL（上）はこれの
;;; 薄いラッパー（contracting (1)/(0)、batch 無し）。
;;;
;;; eager 実装（src/primitives/dot.lisp）とは別の書き方（再帰による
;;; 縮約次元のネストしたループ）にして、同じバグを2箇所で踏まないように
;;; している。DOUBLE-FLOAT にデコード済みの配列を受け取り DOUBLE-FLOAT の
;;; 配列を返す。

(defun %dot-general-free-dims (rank batch contracting)
  "0..RANK-1 のうち BATCH にも CONTRACTING にも含まれない次元を昇順で返す。"
  (loop for d below rank
        unless (member d batch) unless (member d contracting)
        collect d))

(defun %dot-general-build-subscripts (rank batch-dims batch-subs free-dims free-subs
                                       contract-dims contract-subs)
  "RANK 個の添字のリストを組み立てる。BATCH-DIMS[i] 番目の次元に
BATCH-SUBS[i] を、FREE-DIMS[i] 番目に FREE-SUBS[i] を、CONTRACT-DIMS[i]
番目に CONTRACT-SUBS[i] を入れる。"
  (let ((subscripts (make-list rank)))
    (loop for d in batch-dims for s in batch-subs do (setf (nth d subscripts) s))
    (loop for d in free-dims for s in free-subs do (setf (nth d subscripts) s))
    (loop for d in contract-dims for s in contract-subs do (setf (nth d subscripts) s))
    subscripts))

(defun reference-dot-general (a b lhs-contracting rhs-contracting lhs-batch rhs-batch)
  "StableHLO の dot_general の意味で A・B を縮約する。出力の次元順序は
（LHS-BATCH の順のバッチ次元、続いて A の自由次元を昇順、続いて B の
自由次元を昇順）。A・B は DOUBLE-FLOAT の配列であること。"
  (let* ((a-shape (array-dimensions a))
         (b-shape (array-dimensions b))
         (a-rank (length a-shape))
         (b-rank (length b-shape))
         (lhs-free (%dot-general-free-dims a-rank lhs-batch lhs-contracting))
         (rhs-free (%dot-general-free-dims b-rank rhs-batch rhs-contracting))
         (batch-shape (mapcar (lambda (d) (nth d a-shape)) lhs-batch))
         (lhs-free-shape (mapcar (lambda (d) (nth d a-shape)) lhs-free))
         (rhs-free-shape (mapcar (lambda (d) (nth d b-shape)) rhs-free))
         (contract-shape (mapcar (lambda (d) (nth d a-shape)) lhs-contracting))
         (out-shape (append batch-shape lhs-free-shape rhs-free-shape))
         (result (make-array out-shape :element-type 'double-float :initial-element 0.0d0)))
    (dotimes (i (array-total-size result) result)
      (let* ((out-subscripts (%row-major-index->subscripts i out-shape))
             (batch-subs (subseq out-subscripts 0 (length batch-shape)))
             (lhs-free-subs (subseq out-subscripts (length batch-shape)
                                    (+ (length batch-shape) (length lhs-free-shape))))
             (rhs-free-subs (subseq out-subscripts (+ (length batch-shape) (length lhs-free-shape))))
             (sum 0.0d0))
        (labels ((sum-contract (remaining-shape contract-subs)
                   (if (null remaining-shape)
                       (let ((a-subscripts (%dot-general-build-subscripts
                                            a-rank lhs-batch batch-subs lhs-free lhs-free-subs
                                            lhs-contracting (reverse contract-subs)))
                             (b-subscripts (%dot-general-build-subscripts
                                            b-rank rhs-batch batch-subs rhs-free rhs-free-subs
                                            rhs-contracting (reverse contract-subs))))
                         (incf sum (* (coerce (apply #'aref a a-subscripts) 'double-float)
                                      (coerce (apply #'aref b b-subscripts) 'double-float))))
                       (dotimes (k (first remaining-shape))
                         (sum-contract (rest remaining-shape) (cons k contract-subs))))))
          (sum-contract contract-shape nil))
        (setf (row-major-aref result i) sum)))))
