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

;; issue #31 p2
(defun reference-neg (a)
  "-A（要素ごとの符号反転）を返す。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (- (coerce (row-major-aref a i) 'double-float))))))

;; issue #31 p2
(defun reference-exp (a)
  "EXP(A)（要素ごとの指数関数）を返す。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (exp (coerce (row-major-aref a i) 'double-float))))))

;; issue #31 p2
(defun reference-log (a)
  "LOG(A)（要素ごとの自然対数）を返す。A の要素はすべて正であること
（負・0の扱いは NaN/-inf の性質として eager 側を直接呼んで確かめる。
このスキルの「テストの中にロジックを書かない」/このファイル冒頭の注記
どおり、参照実装は素朴なままにする）。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (log (coerce (row-major-aref a i) 'double-float))))))

;; issue #31 p2
(defun reference-tanh (a)
  "TANH(A)（要素ごとの双曲線正接）を返す。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (tanh (coerce (row-major-aref a i) 'double-float))))))

;; issue #196
(defun reference-sqrt (a)
  "SQRT(A)（要素ごとの平方根）を返す。A は非負。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (sqrt (coerce (row-major-aref a i) 'double-float))))))

;; issue #196
(defun reference-rsqrt (a)
  "RSQRT(A) = 1/SQRT(A)（要素ごと）を返す。A は正。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (/ 1d0 (sqrt (coerce (row-major-aref a i) 'double-float)))))))

;; issue #223。実装（libm の log1p / expm1）とは別の式で、double の log(1 + x) / exp(x) - 1。
;; 入力が f32 / bf16 / f16 なら double で十分に正確（f64 の小さな値は別の例ベースのテストで見る）。
(defun reference-log1p (a)
  "LOG(1 + A)（要素ごと、double の素朴な式）を返す。A は -1 より大きい。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (log (+ 1d0 (coerce (row-major-aref a i) 'double-float)))))))

(defun reference-expm1 (a)
  "EXP(A) - 1（要素ごと、double の素朴な式）を返す。"
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i) (- (exp (coerce (row-major-aref a i) 'double-float)) 1d0)))))

;; issue #31 p2
(defun reference-max (a b)
  "MAX(A, B)（要素ごとの大きい方）を返す。A と B は同じ shape を持つこと。
NaN の伝播は eager 側を直接呼んで確かめる（CL の MAX は NaN を伝播しない
ため、この参照実装は NaN を含まない入力にだけ使う）。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-max: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (max (coerce (row-major-aref a i) 'double-float)
                 (coerce (row-major-aref b i) 'double-float))))))

;; issue #31 p2
(defun reference-min (a b)
  "MIN(A, B)（要素ごとの小さい方）を返す。REFERENCE-MAX と同じ注記が
当てはまる。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-min: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (min (coerce (row-major-aref a i) 'double-float)
                 (coerce (row-major-aref b i) 'double-float))))))

;; issue #31 p3
(defun reference-compare (a b direction)
  "A DIRECTION B（要素ごとの比較）を BIT の配列で返す。A と B は同じ shape を
持つこと。DIRECTION は :LT :LE :GT :GE :EQ :NE のいずれか。NaN を含まない
入力にだけ使う（NaN の扱いは eager 側を直接呼んで確かめる。このファイル
冒頭の注記どおり参照実装は素朴なままにする）。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "reference-compare: 形状が違う: ~A と ~A" (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'bit))
        (op (ecase direction
              (:lt #'<) (:le #'<=) (:gt #'>) (:ge #'>=) (:eq #'=) (:ne #'/=))))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (if (funcall op
                         (coerce (row-major-aref a i) 'double-float)
                         (coerce (row-major-aref b i) 'double-float))
                1 0)))))

;; issue #31 p3
(defun reference-select (pred a b)
  "PRED（BIT の配列）のビットに応じて A または B の要素をそのまま返す
（PRED・A・B は同じ shape を持つこと。A・B は DOUBLE-FLOAT の配列）。"
  (unless (and (equal (array-dimensions pred) (array-dimensions a))
               (equal (array-dimensions pred) (array-dimensions b)))
    (error "reference-select: 形状が違う: ~A / ~A / ~A"
           (array-dimensions pred) (array-dimensions a) (array-dimensions b)))
  (let ((result (make-array (array-dimensions a) :element-type 'double-float)))
    (dotimes (i (array-total-size a) result)
      (setf (row-major-aref result i)
            (if (= 1 (row-major-aref pred i)) (row-major-aref a i) (row-major-aref b i))))))

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

;;; --- JAX の sqrt / rsqrt フィクスチャ（issue #196、tests/fixtures/sqrt/） ---

(defun load-sqrt-jax-fixture (dtype)
  "tests/fixtures/sqrt/jax-sqrt.lisp のうち DTYPE の行を (入力 sqrt rsqrt)
（すべてビットパターンの整数）のリストで返す。READ は *read-eval* nil で行う。"
  (let ((rows (with-standard-io-syntax
                (let ((*read-eval* nil))
                  (with-open-file (stream (asdf:system-relative-pathname
                                           "nabla" "tests/fixtures/sqrt/jax-sqrt.lisp"))
                    (read stream))))))
    (loop for row in rows
          when (eq (first row) dtype) collect (rest row))))

(defun sqrt-fixture-bits->number (dtype bits)
  "BITS（DTYPE のビットパターン）を CL の浮動小数点数にデコードする
（bf16 / f16 は double-float、f32 は single-float、f64 は double-float）。"
  (flet ((signed (n width) (if (logbitp (1- width) n) (- n (ash 1 width)) n)))
    (ecase dtype
      (:f32 (sb-kernel:make-single-float (signed bits 32)))
      (:f64 (sb-kernel:make-double-float (signed (ash bits -32) 32) (logand bits #xffffffff)))
      ((:bf16 :f16) (coerce (nabla::decode-float16 bits dtype) 'double-float)))))

(defun sqrt-fixture-input-array (dtype rows)
  "ROWS の入力（ビット）から DTYPE の1次元の配列を作る（bf16 / f16 は
(unsigned-byte 16) のビットのまま）。"
  (let ((array (make-array (length rows)
                           :element-type (ecase dtype
                                           (:f32 'single-float)
                                           (:f64 'double-float)
                                           ((:bf16 :f16) '(unsigned-byte 16))))))
    (loop for row in rows for i from 0
          do (setf (aref array i)
                   (if (member dtype '(:bf16 :f16))
                       (first row)
                       (sqrt-fixture-bits->number dtype (first row)))))
    array))

(defun sqrt-fixture-match-p (dtype actual expected)
  "ACTUAL（配列の1要素。bf16 / f16 はビット）が EXPECTED（フィクスチャのビット）
と一致するか。NaN は NaN と、±無限大・±0 は符号まで厳密に、それ以外は dtype の
許容誤差（dtype-tolerance）で比べる。"
  (let ((a (if (member dtype '(:bf16 :f16))
               (coerce (nabla::decode-float16 actual dtype) 'double-float)
               actual))
        (e (sqrt-fixture-bits->number dtype expected)))
    (cond ((sb-ext:float-nan-p e) (sb-ext:float-nan-p a))
          ((sb-ext:float-nan-p a) nil)
          ((sb-ext:float-infinity-p e) (and (sb-ext:float-infinity-p a) (= a e)))
          ((sb-ext:float-infinity-p a) nil)
          ((zerop e) (and (zerop a) (= (float-sign a) (float-sign e))))
          (t (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
               (<= (abs (- a e)) (+ atol (* rtol (abs e)))))))))

;;; issue #197: convolution の参照実装。プリミティブの eager 実装とは別の作り方をする:
;;; lhs を膨らませて padding した配列（負の padding は切り落とす）と、rhs を膨らませた
;;; 配列を実際に作ってから、窓の開始位置を数え上げて素直に相関を取る。

(defun %conv-reference-place (array spatial dilations pads)
  "ARRAY（[N, 空間..., C]）の各空間軸 i の添字 x の要素を、pads[i].low + x * dilations[i]
の位置に置いた 0 埋めの配列を返す（範囲外になる要素は捨てる）。"
  (let* ((shape (array-dimensions array))
         (new-spatial (loop for d in (subseq shape 1 (1+ spatial))
                            for dil in dilations for p in pads
                            collect (+ (if (zerop d) 0 (1+ (* (1- d) dil))) (first p) (second p))))
         (result (make-array (append (list (first shape)) new-spatial (last shape))
                             :element-type 'double-float :initial-element 0.0d0)))
    (dotimes (i (array-total-size array) result)
      (let* ((subs (%row-major-index->subscripts i shape))
             (placed (loop for x in (subseq subs 1 (1+ spatial))
                           for dil in dilations for p in pads
                           collect (+ (first p) (* x dil)))))
        (when (every (lambda (pos size) (< -1 pos size)) placed new-spatial)
          (setf (apply #'aref result (first subs) (append placed (last subs)))
                (coerce (apply #'aref array subs) 'double-float)))))))

(defun reference-conv (lhs rhs strides padding lhs-dilation rhs-dilation groups &optional (batch-groups 1))
  "channels-last の畳み込み（lhs [N, 空間..., C]、rhs [空間..., C/G, O]。DOUBLE-FLOAT の
配列）。STRIDES・LHS-DILATION・RHS-DILATION は空間次元ごとの整数、PADDING は (low high)。
BATCH-GROUPS（B。issue #203）が 1 より大きいとき、lhs のバッチ軸を B 個の連続した塊に分け
（グループ g は N/B 個）、出力チャネルを B 個の連続した塊に分けて、g 番目の出力チャネルの塊が
g 番目のバッチの塊だけを読む。出力のバッチ軸は N/B。GROUPS と同時に 1 より大きくしない。"
  (let* ((spatial (- (array-rank lhs) 2))
         (padded (%conv-reference-place lhs spatial lhs-dilation padding))
         (padded-spatial (subseq (array-dimensions padded) 1 (1+ spatial)))
         (kernel-spatial (subseq (array-dimensions rhs) 0 spatial))
         (kernel-dilated (loop for k in kernel-spatial for d in rhs-dilation
                               collect (1+ (* (1- k) d))))
         (in-per-group (array-dimension rhs spatial))
         (out-channels (array-dimension rhs (1+ spatial)))
         (out-per-group (floor out-channels groups))
         ;; 窓の開始位置を数え上げる（出力サイズの式を使わない）。
         (out-spatial (loop for p in padded-spatial for k in kernel-dilated for s in strides
                            collect (loop for start from 0 by s while (<= (+ start k) p) count t)))
         (out-batch (floor (array-dimension lhs 0) batch-groups))
         (out-per-bgroup (floor out-channels batch-groups))
         (out-shape (append (list out-batch) out-spatial (list out-channels)))
         (result (make-array out-shape :element-type 'double-float :initial-element 0.0d0)))
    (dotimes (i (array-total-size result) result)
      (let* ((subs (%row-major-index->subscripts i out-shape))
             (n (first subs))
             (starts (loop for o in (subseq subs 1 (1+ spatial)) for s in strides collect (* o s)))
             (oc (car (last subs)))
             (group (floor oc out-per-group))
             (lhs-n (+ (* (floor oc out-per-bgroup) out-batch) n))
             (sum 0.0d0))
        (nb::%dot-for-each-index
         kernel-spatial
         (lambda (k-pos)
           (let ((pos (loop for start in starts for k in k-pos for d in rhs-dilation
                            collect (+ start (* k d)))))
             (dotimes (c in-per-group)
               (incf sum (* (apply #'aref padded lhs-n (append pos (list (+ (* group in-per-group) c))))
                            (coerce (apply #'aref rhs (append k-pos (list c oc))) 'double-float)))))))
        (setf (row-major-aref result i) sum)))))

(defun read-conv-fixture (&optional (file "conv"))
  "tests/fixtures/conv/FILE.lisp を読み、ケースごとの plist
(:name :strides :padding :lhs-dilation :rhs-dilation :groups :lhs :rhs :out) のリストを返す。
既定の \"conv\" は JAX の conv_general_dilated の f32 の入出力（生成は tests/fixtures/conv/generate.py）。
\"conv-grad\"（issue #203。生成は generate-grad.py）は :batch-groups と、余接線 :ct に対する
勾配 :dlhs :drhs も持つ。:lhs :rhs :out :ct :dlhs :drhs は single-float の配列。"
  (flet ((to-array (entry)
           (destructuring-bind (shape bits) entry
             (let ((array (make-array shape :element-type 'single-float)))
               (loop for i from 0 for bit in bits
                     do (setf (row-major-aref array i) (nb::%make-single-float bit)))
               array))))
    (with-open-file (stream (asdf:system-relative-pathname "nabla" (format nil "tests/fixtures/conv/~A.lisp" file)))
      (loop for (name . plist) in (let ((*package* (find-package '#:nabla.tests.support))) (read stream))
            collect (list* :name name
                           (loop for (key value) on plist by #'cddr
                                 append (list key (if (member key '(:lhs :rhs :out :ct :dlhs :drhs)) (to-array value) value))))))))
