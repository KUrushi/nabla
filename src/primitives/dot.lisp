;;;; dot: dot-general プリミティブ（issue #31 p5）。
;;;;
;;;; StableHLO の dot_general と同じ意味を持つ、唯一の縮約演算。
;;;; パラメタは :lhs-contracting / :rhs-contracting（対応する縮約次元の
;;;; リスト）と :lhs-batch / :rhs-batch（対応するバッチ次元のリスト）。
;;;; 出力の次元順序は「バッチ次元（lhs-batch の順）→ lhs の自由次元
;;;; （昇順）→ rhs の自由次元（昇順）」（StableHLO / JAX と同じ順序。
;;;; 契約 §2・§4 のピットフォール(1)）。
;;;;
;;;; チェーンB（p4〜p6）は src/primitives/common.lisp（チェーンA所有）を
;;;; 編集しない約束なので、decode-float16-array / encode-float16 は
;;;; nb::（同じパッケージなので接頭辞なし）を直接呼ぶ（契約のガイダンス
;;;; (3)）。%DOT- 接頭辞のヘルパーはこのファイルだけで使う。

(in-package #:nabla)

(defun %dot-float-dtype-p (dtype)
  "DTYPE が浮動小数点の dtype（:f32 :f64 :bf16 :f16 のいずれか）なら真。"
  (member dtype '(:f32 :f64 :bf16 :f16)))

(defun %dot-check-integer-list (name in-avals param-name value)
  "VALUE が整数の正リスト（proper list）でなければ PRIMITIVE-ERROR を
signal する（トレーサから渡された任意の Lisp オブジェクトが、この後の
LENGTH / NTH / MEMBER に生の型エラーとして流れ込むのを防ぐ。契約 §0）。
LISTP だけでは `(1 . 2)` のようなドットリストを弾けない
（EVERY はドットリストでもエラーにならずに NIL を返すことがあり、その
まま通すと後段の LENGTH が生の TYPE-ERROR を signal してしまう）ので、
SB-INT:PROPER-LIST-P で正リストであることも確かめる。"
  (unless (and (sb-int:proper-list-p value) (every #'integerp value))
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "~S は整数のリストでなければならない: ~S"
           :format-arguments (list param-name value))))

(defun %dot-check-dims-in-range (name in-avals param-name dims rank)
  (unless (every (lambda (d) (typep d `(integer 0 (,rank)))) dims)
    (error 'primitive-error :name name :in-avals in-avals
           :format-control "~S ~S は rank ~S の範囲外の次元を含む"
           :format-arguments (list param-name dims rank))))

(defun %dot-check-no-overlap (name in-avals side batch contracting)
  (let ((used (append batch contracting)))
    (unless (= (length used) (length (remove-duplicates used)))
      (error 'primitive-error :name name :in-avals in-avals
             :format-control "~A の batch/contracting の次元に重複がある: ~S"
             :format-arguments (list side used)))))

(defun %dot-free-dims (rank batch contracting)
  "0..RANK-1 のうち BATCH にも CONTRACTING にも含まれない次元を昇順で返す
（StableHLO の自由次元の並び）。"
  (loop for d below rank
        unless (member d batch) unless (member d contracting)
        collect d))

(defun %dot-dims-string (dims)
  (format nil "[~{~D~^, ~}]" dims))

(defun %dot-accumulate-in-f32-p (dtype)
  "DTYPE が :BF16 / :F16 なら真。IREE（llvm-cpu）の dot_general は入力
dtype のまま累積し、eager 実装（single-float 累積）と縮約が長いときに
許容誤差を超えてずれる（issue #54）ので、この場合だけ f32 で累積させて
から元の dtype に戻す2行の :emit にする。"
  (member dtype '(:bf16 :f16)))

(defun %dot-aux-name (tag out-name)
  "OUT-NAME（\"%7\" のような SSA 名）の数字部分を使った補助 SSA 名
（\"%acc_7\"）を返す。src/primitives/reduce.lisp の %reduce-aux-name と
同じ規約。SUBSEQ の開始位置（\"%\" 1文字だけ落とす）を FORMAT の外に
出しているのは、format 呼び出しの内側は変異させない runner の arid
node 判定に埋もれさせず、この 1 を変異させられるようにするため。"
  (let ((digits (subseq out-name 1)))
    (format nil "%~A_~A" tag digits)))

(defun %dot-batching-clause (lhs-batch rhs-batch)
  "LHS-BATCH・RHS-BATCH のどちらかが空でなければ batching_dims 節を、
両方空なら空文字列を返す。"
  (if (or lhs-batch rhs-batch)
      (format nil "batching_dims = ~A x ~A, "
              (%dot-dims-string lhs-batch) (%dot-dims-string rhs-batch))
      ""))

(defun %dot-general-line (out-name lhs-name rhs-name lhs-batch rhs-batch
                          lhs-contracting rhs-contracting lhs-type rhs-type result-type)
  "stablehlo.dot_general 1行分のテキストを組み立てる（結果型 RESULT-TYPE
は出力の dtype と異なっていてもよい。f32 累積の中間結果を作るときに使う）。"
  (format nil "~A = stablehlo.dot_general ~A, ~A, ~Acontracting_dims = ~A x ~A : (~A, ~A) -> ~A"
          out-name lhs-name rhs-name (%dot-batching-clause lhs-batch rhs-batch)
          (%dot-dims-string lhs-contracting) (%dot-dims-string rhs-contracting)
          lhs-type rhs-type result-type))

(defun %dot-convert-line (out-name in-name in-type out-type)
  "stablehlo.convert 1行分のテキストを組み立てる。"
  (format nil "~A = stablehlo.convert ~A : (~A) -> ~A" out-name in-name in-type out-type))

(defun %dot-emit-lines (in-names in-avals out-name out-aval
                        lhs-batch rhs-batch lhs-contracting rhs-contracting)
  "dot-general の :emit 本体。OUT-AVAL の dtype が bf16/f16 なら、f32 の
結果型を持つ dot_general と convert の2行を、そうでなければ dot_general
1行だけを返す（issue #54）。"
  (let ((lhs-type (tensor-type-string (first in-avals)))
        (rhs-type (tensor-type-string (second in-avals)))
        (out-type (tensor-type-string out-aval)))
    (if (%dot-accumulate-in-f32-p (aval-dtype out-aval))
        (let* ((acc-name (%dot-aux-name "acc" out-name))
               (acc-type (tensor-type-string (make-aval (aval-shape out-aval) :f32))))
          (format nil "~A~%~A"
                  (%dot-general-line acc-name (first in-names) (second in-names)
                                     lhs-batch rhs-batch lhs-contracting rhs-contracting
                                     lhs-type rhs-type acc-type)
                  (%dot-convert-line out-name acc-name acc-type out-type)))
        (%dot-general-line out-name (first in-names) (second in-names)
                           lhs-batch rhs-batch lhs-contracting rhs-contracting
                           lhs-type rhs-type out-type))))

(defun %dot-build-subscripts (rank batch-dims batch-subs free-dims free-subs
                              contract-dims contract-subs)
  "RANK 個の添字のリストを組み立てる。BATCH-DIMS[i] 番目の次元に
BATCH-SUBS[i] を、FREE-DIMS[i] 番目に FREE-SUBS[i] を、CONTRACT-DIMS[i]
番目に CONTRACT-SUBS[i] を入れる。"
  (let ((subscripts (make-list rank)))
    (loop for d in batch-dims for s in batch-subs do (setf (nth d subscripts) s))
    (loop for d in free-dims for s in free-subs do (setf (nth d subscripts) s))
    (loop for d in contract-dims for s in contract-subs do (setf (nth d subscripts) s))
    subscripts))

(defun %dot-for-each-index (sizes fn)
  "SIZES（非負整数のリスト）が表す全通りの添字（row-major 順、末尾の次元が
最速で回る）ごとに、添字のリストを引数として FN を呼ぶ。SIZES が空なら
（rank 0 として）空リストで1回だけ呼ぶ。SIZES のどれかが0なら、一度も
呼ばない。"
  (labels ((rec (remaining acc)
             (if (null remaining)
                 (funcall fn (reverse acc))
                 (dotimes (i (first remaining))
                   (rec (rest remaining) (cons i acc))))))
    (rec sizes nil)))

(defprimitive dot-general (:lhs-contracting :rhs-contracting :lhs-batch :rhs-batch)
  :abstract-eval
  (lambda (in-avals &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
    (unless (= (length in-avals) 2)
      (error 'primitive-error :name :dot-general :in-avals in-avals
             :format-control "入力の個数が違う: ~S 個渡されたが 2 個必要"
             :format-arguments (list (length in-avals))))
    (%dot-check-integer-list :dot-general in-avals :lhs-contracting lhs-contracting)
    (%dot-check-integer-list :dot-general in-avals :rhs-contracting rhs-contracting)
    (%dot-check-integer-list :dot-general in-avals :lhs-batch lhs-batch)
    (%dot-check-integer-list :dot-general in-avals :rhs-batch rhs-batch)
    (let* ((lhs (first in-avals))
           (rhs (second in-avals))
           (lhs-shape (aval-shape lhs))
           (rhs-shape (aval-shape rhs))
           (lhs-rank (length lhs-shape))
           (rhs-rank (length rhs-shape)))
      (unless (and (eq (aval-dtype lhs) (aval-dtype rhs)) (%dot-float-dtype-p (aval-dtype lhs)))
        (error 'primitive-error :name :dot-general :in-avals in-avals
               :format-control "dot-general は同じ浮動小数点 dtype の2入力が必要: ~S と ~S"
               :format-arguments (list (aval-dtype lhs) (aval-dtype rhs))))
      (unless (= (length lhs-contracting) (length rhs-contracting))
        (error 'primitive-error :name :dot-general :in-avals in-avals
               :format-control "lhs-contracting と rhs-contracting の長さが違う: ~S と ~S"
               :format-arguments (list lhs-contracting rhs-contracting)))
      (unless (= (length lhs-batch) (length rhs-batch))
        (error 'primitive-error :name :dot-general :in-avals in-avals
               :format-control "lhs-batch と rhs-batch の長さが違う: ~S と ~S"
               :format-arguments (list lhs-batch rhs-batch)))
      (%dot-check-dims-in-range :dot-general in-avals :lhs-contracting lhs-contracting lhs-rank)
      (%dot-check-dims-in-range :dot-general in-avals :lhs-batch lhs-batch lhs-rank)
      (%dot-check-dims-in-range :dot-general in-avals :rhs-contracting rhs-contracting rhs-rank)
      (%dot-check-dims-in-range :dot-general in-avals :rhs-batch rhs-batch rhs-rank)
      (%dot-check-no-overlap :dot-general in-avals "lhs" lhs-batch lhs-contracting)
      (%dot-check-no-overlap :dot-general in-avals "rhs" rhs-batch rhs-contracting)
      (loop for lb in lhs-batch for rb in rhs-batch
            unless (= (nth lb lhs-shape) (nth rb rhs-shape))
            do (error 'primitive-error :name :dot-general :in-avals in-avals
                      :format-control "batch dim のサイズが一致しない: lhs[~S]=~S rhs[~S]=~S"
                      :format-arguments (list lb (nth lb lhs-shape) rb (nth rb rhs-shape))))
      (loop for lc in lhs-contracting for rc in rhs-contracting
            unless (= (nth lc lhs-shape) (nth rc rhs-shape))
            do (error 'primitive-error :name :dot-general :in-avals in-avals
                      :format-control "contracting dim のサイズが一致しない: lhs[~S]=~S rhs[~S]=~S"
                      :format-arguments (list lc (nth lc lhs-shape) rc (nth rc rhs-shape))))
      (let* ((lhs-free (%dot-free-dims lhs-rank lhs-batch lhs-contracting))
             (rhs-free (%dot-free-dims rhs-rank rhs-batch rhs-contracting))
             (out-shape (append (mapcar (lambda (b) (nth b lhs-shape)) lhs-batch)
                                 (mapcar (lambda (d) (nth d lhs-shape)) lhs-free)
                                 (mapcar (lambda (d) (nth d rhs-shape)) rhs-free))))
        (make-aval out-shape (aval-dtype lhs)))))
  :emit
  (lambda (in-names in-avals out-name out-aval &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
    (%dot-emit-lines in-names in-avals out-name out-aval
                     lhs-batch rhs-batch lhs-contracting rhs-contracting))
  :eager
  (lambda (arrays in-avals &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
    (let* ((dtype (aval-dtype (first in-avals)))
           (lhs-array (first arrays))
           (rhs-array (second arrays))
           (lhs-shape (array-dimensions lhs-array))
           (rhs-shape (array-dimensions rhs-array))
           (lhs-rank (length lhs-shape))
           (rhs-rank (length rhs-shape))
           (lhs-free (%dot-free-dims lhs-rank lhs-batch lhs-contracting))
           (rhs-free (%dot-free-dims rhs-rank rhs-batch rhs-contracting))
           (batch-sizes (mapcar (lambda (b) (nth b lhs-shape)) lhs-batch))
           (lhs-free-sizes (mapcar (lambda (d) (nth d lhs-shape)) lhs-free))
           (rhs-free-sizes (mapcar (lambda (d) (nth d rhs-shape)) rhs-free))
           (contract-sizes (mapcar (lambda (c) (nth c lhs-shape)) lhs-contracting))
           (out-shape (append batch-sizes lhs-free-sizes rhs-free-sizes))
           ;; 契約 §0: f64 は DOUBLE-FLOAT、それ以外（f32・bf16・f16）は
           ;; SINGLE-FLOAT で累算する。bf16 / f16 は一度だけデコードし、
           ;; 出力は RNE で丸めて戻す。
           (compute-type (if (eq dtype :f64) 'double-float 'single-float))
           (lhs-values (if (member dtype '(:bf16 :f16))
                           (decode-float16-array lhs-array dtype)
                           lhs-array))
           (rhs-values (if (member dtype '(:bf16 :f16))
                           (decode-float16-array rhs-array dtype)
                           rhs-array))
           (result (make-array out-shape :element-type (dtype-element-type dtype)))
           (out-index 0))
      (sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero)
        (%dot-for-each-index
         batch-sizes
         (lambda (batch-subs)
           (%dot-for-each-index
            lhs-free-sizes
            (lambda (lhs-free-subs)
              (%dot-for-each-index
               rhs-free-sizes
               (lambda (rhs-free-subs)
                 (let ((sum (coerce 0 compute-type)))
                   (%dot-for-each-index
                    contract-sizes
                    (lambda (contract-subs)
                      (let* ((lhs-subs (%dot-build-subscripts lhs-rank lhs-batch batch-subs
                                                              lhs-free lhs-free-subs
                                                              lhs-contracting contract-subs))
                             (rhs-subs (%dot-build-subscripts rhs-rank rhs-batch batch-subs
                                                              rhs-free rhs-free-subs
                                                              rhs-contracting contract-subs)))
                         (incf sum (* (coerce (apply #'aref lhs-values lhs-subs) compute-type)
                                      (coerce (apply #'aref rhs-values rhs-subs) compute-type))))))
                   (setf (row-major-aref result out-index)
                         (if (member dtype '(:bf16 :f16)) (encode-float16 sum dtype) sum))
                   (incf out-index))))))))
        result))))
