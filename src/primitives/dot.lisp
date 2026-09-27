;;;; dot: dot-general プリミティブ（issue #31 p5）。
;;;;
;;;; StableHLO の stablehlo.dot_general に対応する。2つの入力の次元を
;;;; contracting dims（総和を取って消える次元）・batch dims（総和を取らず
;;;; 両方の入力とも同じ大きさのまま出力にも残る次元）・free dims（どちら
;;;; でもない次元）の3種類に分け、出力の shape は
;;;; 「batch dims（lhs の順） → lhs の free dims（昇順） → rhs の free
;;;; dims（昇順）」の順に並ぶ（用語集の「contracting dims / batch dims」参照）。
;;;;
;;;; チェーンB（p4〜p6）は src/primitives/common.lisp（チェーンA所有）を
;;;; 編集しない約束なので、shape-common.lisp（p4）の %SHAPE- 接頭辞の
;;;; ヘルパー（%shape-check-arity / %shape-strides / %shape-subscripts）を
;;;; そのまま再利用する。dtype の decode/encode は bf16 / f16 専用の
;;;; DECODE-FLOAT16-ARRAY / ENCODE-FLOAT16-ARRAY（src/float16.lisp）を
;;;; 直接呼ぶ（ガイダンスのとおり、common.lisp の %DECODE-ARRAY は使わない）。

(in-package #:nabla)

(defun %dot-float-dtype-p (dtype)
  "DTYPE が dot-general の入力として許される4つの浮動小数点 dtype
（:f32 :f64 :bf16 :f16）のいずれかかどうかを返す。"
  (and (member dtype '(:f32 :f64 :bf16 :f16)) t))

(defun %dot-in-range-p (dims rank)
  "DIMS（整数のリスト）がすべて [0, RANK) の範囲内かどうかを返す。"
  (every (lambda (d) (typep d `(integer 0 (,rank)))) dims))

(defun %dot-check-params (name in-avals lhs-contracting rhs-contracting lhs-batch rhs-batch)
  "dot-general の入力・パラメタを検証し、共通の dtype を返す。不正なら
PRIMITIVE-ERROR を signal する。"
  (%shape-check-arity name in-avals 2)
  (let* ((lhs (first in-avals))
         (rhs (second in-avals))
         (lhs-shape (aval-shape lhs))
         (rhs-shape (aval-shape rhs))
         (lhs-rank (length lhs-shape))
         (rhs-rank (length rhs-shape))
         (lhs-dtype (aval-dtype lhs))
         (rhs-dtype (aval-dtype rhs)))
    (unless (and (%dot-float-dtype-p lhs-dtype) (%dot-float-dtype-p rhs-dtype))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "dot-general は float dtype の入力しか受け付けない: ~S"
             :format-arguments (list (list lhs-dtype rhs-dtype))))
    (unless (eq lhs-dtype rhs-dtype)
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "2つの入力の dtype が違う: ~S と ~S"
             :format-arguments (list lhs-dtype rhs-dtype)))
    (unless (and (listp lhs-contracting) (listp rhs-contracting) (listp lhs-batch) (listp rhs-batch))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "lhs/rhs-contracting・lhs/rhs-batch はリストでなければならない: ~S"
             :format-arguments (list (list lhs-contracting rhs-contracting lhs-batch rhs-batch))))
    (unless (= (length lhs-contracting) (length rhs-contracting))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "lhs-contracting ~S と rhs-contracting ~S の長さが違う"
             :format-arguments (list lhs-contracting rhs-contracting)))
    (unless (= (length lhs-batch) (length rhs-batch))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "lhs-batch ~S と rhs-batch ~S の長さが違う"
             :format-arguments (list lhs-batch rhs-batch)))
    (unless (and (%dot-in-range-p lhs-contracting lhs-rank) (%dot-in-range-p lhs-batch lhs-rank))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "lhs の次元指定が rank ~S の範囲外: contracting ~S, batch ~S"
             :format-arguments (list lhs-rank lhs-contracting lhs-batch)))
    (unless (and (%dot-in-range-p rhs-contracting rhs-rank) (%dot-in-range-p rhs-batch rhs-rank))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "rhs の次元指定が rank ~S の範囲外: contracting ~S, batch ~S"
             :format-arguments (list rhs-rank rhs-contracting rhs-batch)))
    (let ((lhs-all (append lhs-contracting lhs-batch))
          (rhs-all (append rhs-contracting rhs-batch)))
      (unless (= (length lhs-all) (length (remove-duplicates lhs-all)))
        (error 'primitive-error :name name :in-avals in-avals
               :format-control "lhs の contracting/batch の次元が重複するか両方に属している: ~S"
               :format-arguments (list lhs-all)))
      (unless (= (length rhs-all) (length (remove-duplicates rhs-all)))
        (error 'primitive-error :name name :in-avals in-avals
               :format-control "rhs の contracting/batch の次元が重複するか両方に属している: ~S"
               :format-arguments (list rhs-all))))
    (loop for lc in lhs-contracting for rc in rhs-contracting
          unless (= (nth lc lhs-shape) (nth rc rhs-shape))
          do (error 'primitive-error :name name :in-avals in-avals
                    :format-control "contracting dim のサイズが一致しない: lhs[~S]=~S, rhs[~S]=~S"
                    :format-arguments (list lc (nth lc lhs-shape) rc (nth rc rhs-shape))))
    (loop for lb in lhs-batch for rb in rhs-batch
          unless (= (nth lb lhs-shape) (nth rb rhs-shape))
          do (error 'primitive-error :name name :in-avals in-avals
                    :format-control "batch dim のサイズが一致しない: lhs[~S]=~S, rhs[~S]=~S"
                    :format-arguments (list lb (nth lb lhs-shape) rb (nth rb rhs-shape))))
    lhs-dtype))

(defun %dot-shape-groups (lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch)
  "batch / lhs の free dims / rhs の free dims / contracting のサイズと、
free dims 自体の次元番号（昇順）を multiple values で返す:
(VALUES BATCH-SIZES LHS-FREE LHS-FREE-SIZES RHS-FREE RHS-FREE-SIZES CONTRACT-SIZES)。"
  (let* ((lhs-excluded (append lhs-batch lhs-contracting))
         (rhs-excluded (append rhs-batch rhs-contracting))
         (lhs-free (sort (set-difference (loop for i below (length lhs-shape) collect i) lhs-excluded) #'<))
         (rhs-free (sort (set-difference (loop for i below (length rhs-shape) collect i) rhs-excluded) #'<))
         (batch-sizes (mapcar (lambda (b) (nth b lhs-shape)) lhs-batch))
         (lhs-free-sizes (mapcar (lambda (d) (nth d lhs-shape)) lhs-free))
         (rhs-free-sizes (mapcar (lambda (d) (nth d rhs-shape)) rhs-free))
         (contract-sizes (mapcar (lambda (d) (nth d lhs-shape)) lhs-contracting)))
    (values batch-sizes lhs-free lhs-free-sizes rhs-free rhs-free-sizes contract-sizes)))

(defun %dot-decode (array dtype)
  "ARRAY が :bf16 / :f16 のビット列表現なら SINGLE-FLOAT にデコードし、
:f32 / :f64 ならそのまま（すでに計算に使える型）ARRAY を返す。"
  (if (member dtype '(:bf16 :f16)) (decode-float16-array array dtype) array))

(defprimitive dot-general (:lhs-contracting :rhs-contracting :lhs-batch :rhs-batch)
  :abstract-eval
  (lambda (in-avals &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
    (let ((dtype (%dot-check-params :dot-general in-avals lhs-contracting rhs-contracting lhs-batch rhs-batch)))
      (multiple-value-bind (batch-sizes lhs-free lhs-free-sizes rhs-free rhs-free-sizes)
          (%dot-shape-groups (aval-shape (first in-avals)) (aval-shape (second in-avals))
                             lhs-contracting rhs-contracting lhs-batch rhs-batch)
        (declare (ignore lhs-free rhs-free))
        (make-aval (append batch-sizes lhs-free-sizes rhs-free-sizes) dtype))))
  :emit
  (lambda (in-names in-avals out-name out-aval &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
    (let ((batching-clause
            (if (or lhs-batch rhs-batch)
                (format nil "batching_dims = [~{~D~^, ~}] x [~{~D~^, ~}], " lhs-batch rhs-batch)
                "")))
      (format nil "~A = stablehlo.dot_general ~A, ~A, ~Acontracting_dims = [~{~D~^, ~}] x [~{~D~^, ~}] : (~A, ~A) -> ~A"
              out-name (first in-names) (second in-names) batching-clause
              lhs-contracting rhs-contracting
              (tensor-type-string (first in-avals)) (tensor-type-string (second in-avals))
              (tensor-type-string out-aval))))
  :eager
  (lambda (arrays in-avals &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
    (let* ((lhs-aval (first in-avals))
           (dtype (aval-dtype lhs-aval))
           (compute-type (if (eq dtype :f64) 'double-float 'single-float))
           (lhs-shape (aval-shape lhs-aval))
           (rhs-shape (aval-shape (second in-avals)))
           (lhs-rank (length lhs-shape))
           (rhs-rank (length rhs-shape))
           (lhs-array (%dot-decode (first arrays) dtype))
           (rhs-array (%dot-decode (second arrays) dtype)))
      (multiple-value-bind (batch-sizes lhs-free lhs-free-sizes rhs-free rhs-free-sizes contract-sizes)
          (%dot-shape-groups lhs-shape rhs-shape lhs-contracting rhs-contracting lhs-batch rhs-batch)
        (let* ((out-shape (append batch-sizes lhs-free-sizes rhs-free-sizes))
               (n-batch (length lhs-batch))
               (n-lhs-free (length lhs-free))
               (out-strides (%shape-strides out-shape))
               (contract-strides (%shape-strides contract-sizes))
               (n-contract (reduce #'* contract-sizes :initial-value 1))
               (result (make-array out-shape :element-type compute-type
                                    :initial-element (coerce 0 compute-type))))
          (sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero)
            (dotimes (out-i (array-total-size result))
              (let* ((out-subs (%shape-subscripts out-i out-shape out-strides))
                     (batch-subs (subseq out-subs 0 n-batch))
                     (lhs-free-subs (subseq out-subs n-batch (+ n-batch n-lhs-free)))
                     (rhs-free-subs (subseq out-subs (+ n-batch n-lhs-free)))
                     (lhs-subs (make-list lhs-rank :initial-element 0))
                     (rhs-subs (make-list rhs-rank :initial-element 0)))
                (loop for b in lhs-batch for s in batch-subs do (setf (nth b lhs-subs) s))
                (loop for rb in rhs-batch for s in batch-subs do (setf (nth rb rhs-subs) s))
                (loop for d in lhs-free for s in lhs-free-subs do (setf (nth d lhs-subs) s))
                (loop for d in rhs-free for s in rhs-free-subs do (setf (nth d rhs-subs) s))
                (let ((sum (coerce 0 compute-type)))
                  (dotimes (c-i n-contract)
                    (let ((c-subs (%shape-subscripts c-i contract-sizes contract-strides)))
                      (loop for lc in lhs-contracting for s in c-subs do (setf (nth lc lhs-subs) s))
                      (loop for rc in rhs-contracting for s in c-subs do (setf (nth rc rhs-subs) s))
                      (incf sum (* (apply #'aref lhs-array lhs-subs) (apply #'aref rhs-array rhs-subs)))))
                  (setf (row-major-aref result out-i) sum)))))
          (if (member dtype '(:bf16 :f16))
              (encode-float16-array result dtype)
              result))))))
