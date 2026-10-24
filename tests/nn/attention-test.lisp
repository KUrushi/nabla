;;;; multi-head attention と dense-general（issue #208、small）。
;;;;
;;;; 守らせる性質:
;;;; - Flax との数値一致: 同じパラメータ・入力・マスク・余接線で、順伝播の出力と、パラメータ・入力
;;;;   （自己 attention は x、交差 attention は x と kv）についての勾配が Flax の
;;;;   nn.MultiHeadDotProductAttention のフィクスチャ（tests/fixtures/attention/flax-attention.lisp、
;;;;   生成は同じディレクトリの generate.py）と f32 の許容誤差で一致する（eager。jit は medium、
;;;;   tests/nn/attention-iree-test.lisp）。マスクなし・因果・padding（全部隠れた行を含む）・
;;;;   バッチ軸なしと2つ・交差の各ケース。
;;;; - 参照実装（ループで書いた素朴な attention。f64）との一致: ヘッド数・長さ・マスクを変えて。
;;;; - 因果マスクのとき、出力の位置 i の勾配は i より後の入力に流れない（ちょうど 0。先頭からの位置は
;;;;   0 でない）。padding マスクで隠したキーの値を変えても出力は変わらない。マスクなしの自己 attention は
;;;;   系列の並べ替えに同変。全部 1 のマスクはマスクなしと同じ。
;;;; - init: 形・スロット名（Flax と同じ query / key / value / out の kernel / bias）・分布。
;;;; - dropout: deterministic のとき rate によらず決定的、そうでないとき同じキーで同じ結果・別のキーで別の結果、
;;;;   キーが無ければ error。
;;;; - マスクの部品（make-causal-mask / make-attention-mask / combine-masks）とエラー（層の名前入り）。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)

;;; --- Flax フィクスチャ ---

(defvar *attention-fixture* nil)

(defun %attention-fixture ()
  (or *attention-fixture*
      (setf *attention-fixture*
            (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/attention/flax-attention.lisp"))
              (read stream)))))

(defun %att-case (name)
  (or (find name (getf (%attention-fixture) :cases) :key (lambda (c) (getf c :name)) :test #'string=)
      (error "フィクスチャにケース ~S が無い" name)))

(defparameter *attention-case-names*
  '("self" "self-causal" "self-padding" "cross" "cross-padding" "self-rank2-causal"
    "self-two-batch-dims-causal-padding")
  "フィクスチャのケース名。medium からも使う。")

(defun %att-projection (case prefix)
  (let ((bias (%fx-entry case (concatenate 'string prefix ".bias"))))
    (make-dense-general :use-bias (and bias t)
                        :kernel (%fx-array case (concatenate 'string prefix ".kernel"))
                        :bias (and bias (%fx-array case (concatenate 'string prefix ".bias")))
                        :contract-axes (if (string= prefix "out") 2 1))))

(defun %att-model (case)
  (make-multi-head-attention :broadcast-dropout t :query (%att-projection case "query") :key (%att-projection case "key")
                             :value (%att-projection case "value") :out (%att-projection case "out")
                             :dropout-rate 0))

(defun %att-mask (case)
  "ケースのマスク（0/1 の整数）を bit 配列にしたもの。無ければ NIL。"
  (let ((entry (%fx-entry case "mask")))
    (when entry
      (destructuring-bind (name shape values) entry
        (declare (ignore name))
        (let ((mask (make-array shape :element-type 'bit)))
          (loop for i from 0 for v in values do (setf (row-major-aref mask i) v))
          mask)))))

(defun %att-param-names (case)
  (loop for (name) in (getf case :arrays)
        when (and (> (length name) 5) (string= "grad." name :end2 5)
                  (string/= name "grad.x") (string/= name "grad.kv"))
          collect (subseq name 5)))

(defun %att-loss-and-forward (case)
  "(values 損失 順伝播 引数の個数)。引数は モデル x [kv] mask ct の順。"
  (if (%fx-entry case "kv")
      (values (nabla:with-tracing (m x kv mask ct) (nabla:reduce-sum (* (call m x kv mask nil t) ct)))
              (nabla:with-tracing (m x kv mask) (call m x kv mask nil t))
              '(0 1 2))
      (values (nabla:with-tracing (m x mask ct) (nabla:reduce-sum (* (call m x x mask nil t) ct)))
              (nabla:with-tracing (m x mask) (call m x x mask nil t))
              '(0 1))))

(defun %att-inputs (case)
  "モデル以外の引数: x [kv] mask（forward の引数）。"
  (append (list (%fx-array case "x"))
          (when (%fx-entry case "kv") (list (%fx-array case "kv")))
          (list (%att-mask case))))

(defun %att-expected-grads (case)
  (append (mapcar (lambda (p) (%fx-array case (concatenate 'string "grad." p))) (%att-param-names case))
          (list (%fx-array case "grad.x"))
          (when (%fx-entry case "grad.kv") (list (%fx-array case "grad.kv")))))

;; key.bias の勾配は数学的に 0 である: softmax はスコアの定数ずらしに不変で、key の bias は 1 つのクエリの
;; 全スコアに同じずれ（q . bias_k / sqrt(d)）を足すだけなので、損失は key.bias によらない。Flax の勾配も
;; 実測は丸め誤差で 0 ではなく（1e-7 台）、nabla も同様（最大 1.09e-6）。この葉だけは、値の一致ではなく
;; 「どちらも絶対値が 1e-5 未満（0 と見なせる）」で比べる。ほかの葉と出力は f32 の既定の許容誤差。
(defparameter *attention-zero-gradient-bound* 1d-5)

(defun %max-abs (array)
  (loop for i below (array-total-size array) maximize (abs (row-major-aref array i))))

(defun %att-grad-names (case)
  "勾配のリストの各要素の名前: パラメータ（フィクスチャの並び）、x、kv（あれば）。"
  (append (%att-param-names case) (list "x") (when (%fx-entry case "grad.kv") (list "kv"))))

(defun %att-check-leaf (case-name leaf-name actual expected)
  "葉 LEAF-NAME の勾配が Flax と一致するか確かめる（key.bias は 0 と見なせるかどうか）。"
  (if (string= leaf-name "key.bias")
      (progn
        (is (< (%max-abs actual) *attention-zero-gradient-bound*)
            "~A: key.bias の勾配は 0 のはずなのに最大 ~E" case-name (%max-abs actual))
        (is (< (%max-abs expected) *attention-zero-gradient-bound*)
            "~A: Flax の key.bias の勾配が 0 でない（最大 ~E）" case-name (%max-abs expected)))
      (is (%close-f32 actual expected) "~A: ~A の勾配が Flax と一致しない" case-name leaf-name)))

(defun %check-attention-case (name)
  (let ((case (%att-case name)))
    (multiple-value-bind (loss forward argnums) (%att-loss-and-forward case)
      (let* ((model (%att-model case))
             (inputs (%att-inputs case))
             (ct (%fx-array case "ct"))
             (grads (apply (nabla:grad loss :argnums argnums) model (append inputs (list ct))))
             (expected (%att-expected-grads case))
             (actual (append (nabla:tree-leaves (first grads)) (rest grads))))
        (is (%close-f32 (apply forward model inputs) (%fx-array case "out"))
            "~A: 順伝播の出力が Flax と一致しない" name)
        (is (= (length expected) (length actual)) "~A: 勾配の個数が違う" name)
        (loop for a in actual for e in expected for leaf in (%att-grad-names case)
              do (%att-check-leaf name leaf a e))))))

(defparameter *attention-no-bias-case-names* '("self-no-bias" "cross-padding-no-bias")
  "use_bias=False のフィクスチャのケース名（フィクスチャの再生成が要る）。medium からも使う。")

(test attention/without-bias-matches-flax
  "Flax の MultiHeadDotProductAttention(use_bias=False) のフィクスチャ（自己 attention と、padding マスク付きの交差
attention）で、出力と kernel・入力の勾配が一致する。4 つの射影の bias はどれも NIL。"
  (dolist (name *attention-no-bias-case-names*)
    (let ((model (%att-model (%att-case name))))
      (is (every #'null (mapcar (lambda (f) (dense-general-bias (funcall f model)))
                                '(multi-head-attention-query multi-head-attention-key
                                  multi-head-attention-value multi-head-attention-out)))
          "~A: bias が NIL でない" name))
    (%check-attention-case name)))

(test attention/forward-and-grads-match-flax
  "マスクなし・因果・padding（全部のキーが隠れた行を含む）・交差（入力の幅が違う）・バッチ軸なし・
バッチ軸2つ（因果 + padding）の各ケースで、順伝播の出力と、パラメータと入力の勾配が Flax と一致する。"
  (dolist (name *attention-case-names*)
    (%check-attention-case name)))

;;; --- 参照実装（ループ）---

(defun %ref-project (x kernel bias)
  "X (b t f)、KERNEL (f h d)、BIAS (h d) → (b t h d)（2重ループの素朴な実装。f64）。"
  (destructuring-bind (b tt f) (array-dimensions x)
    (destructuring-bind (h d) (rest (array-dimensions kernel))
      (let ((y (make-array (list b tt h d) :element-type 'double-float)))
        (dotimes (bi b y)
          (dotimes (ti tt)
            (dotimes (hi h)
              (dotimes (di d)
                (setf (aref y bi ti hi di)
                      (+ (aref bias hi di)
                         (loop for fi below f sum (* (aref x bi ti fi) (aref kernel fi hi di)))))))))))))

(defun %ref-attention (model x kv mask)
  "MODEL (f64) を X (b t f)・KV (b s g)・MASK (b 1 t s の bit 配列か NIL) に適用した (b t out)。
マスクで隠した位置は f64 の最小値（Flax の jnp.finfo(dtype).min と同じ）を入れてから softmax。"
  (let* ((q (%ref-project x (dense-general-kernel (multi-head-attention-query model))
                          (dense-general-bias (multi-head-attention-query model))))
         (k (%ref-project kv (dense-general-kernel (multi-head-attention-key model))
                          (dense-general-bias (multi-head-attention-key model))))
         (v (%ref-project kv (dense-general-kernel (multi-head-attention-value model))
                          (dense-general-bias (multi-head-attention-value model))))
         (out-kernel (dense-general-kernel (multi-head-attention-out model)))
         (out-bias (dense-general-bias (multi-head-attention-out model))))
    (destructuring-bind (b tt h d) (array-dimensions q)
      (let ((s (second (array-dimensions k)))
            (y (make-array (list b tt (first (array-dimensions out-bias))) :element-type 'double-float)))
        (dotimes (bi b y)
          (dotimes (ti tt)
            (let ((context (make-array (list h d) :element-type 'double-float :initial-element 0d0)))
              (dotimes (hi h)
                (let* ((scores (loop for si below s
                                     collect (if (and mask (zerop (aref mask bi 0 ti si)))
                                                 most-negative-double-float
                                                 (/ (loop for di below d sum (* (aref q bi ti hi di) (aref k bi si hi di)))
                                                    (sqrt (coerce d 'double-float))))))
                       (top (reduce #'max scores))
                       (weights (mapcar (lambda (sc) (exp (- sc top))) scores))
                       (total (reduce #'+ weights)))
                  (loop for si from 0 for w in weights
                        do (dotimes (di d)
                             (incf (aref context hi di) (* (/ w total) (aref v bi si hi di)))))))
              (dotimes (oi (first (array-dimensions out-bias)))
                (setf (aref y bi ti oi)
                      (+ (aref out-bias oi)
                         (loop for hi below h
                               sum (loop for di below d
                                         sum (* (aref context hi di) (aref out-kernel hi di oi))))))))))))))

(defun %random-attention (seed in kv-in heads dim out)
  "f64 の乱数のパラメータの multi-head-attention（bias も 0 でない）。"
  (flet ((proj (s fan features) (make-dense-general :use-bias t :kernel (%rand (append (list fan) features) s :f64)
                                                    :bias (%rand features (+ s 1) :f64)
                                                    :contract-axes 1)))
    (make-multi-head-attention
     :broadcast-dropout t :query (proj seed in (list heads dim)) :key (proj (+ seed 10) kv-in (list heads dim))
     :value (proj (+ seed 20) kv-in (list heads dim))
     :out (make-dense-general :use-bias t :kernel (%rand (list heads dim out) (+ seed 30) :f64)
                              :bias (%rand (list out) (+ seed 31) :f64)
                              :contract-axes 2)
     :dropout-rate 0)))

(defun %random-bits (shape seed)
  (let ((a (make-array shape :element-type 'bit))
        (r (%rand shape seed :f64)))
    (dotimes (i (array-total-size a) a)
      (setf (row-major-aref a i) (if (plusp (row-major-aref r i)) 1 0)))))

(def-layer-property attention/matches-the-loop-reference
  "ヘッド数・ヘッドの幅・長さ（q と kv で別）・入力の幅（q と kv で別）・マスクの有無を変えても、
f64 の multi-head-attention はループで書いた素朴な参照実装（マスクで隠した位置に f64 の最小値を入れて
softmax）と一致する。全部のキーが隠れた行も含む。"
  (seed batch tq tk heads dim masked) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 2)
                                             (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4)
                                             (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 3)
                                             (uniform-integer :lo 0 :hi 1))
  (let* ((in 3) (kv-in 4) (out 2)
         (model (%random-attention seed in kv-in heads dim out))
         (x (%rand (list batch tq in) (+ seed 100) :f64))
         (kv (%rand (list batch tk kv-in) (+ seed 200) :f64))
         (mask (when (= masked 1) (%random-bits (list batch 1 tq tk) (+ seed 300)))))
    (allclose (call model x kv mask nil t) (%ref-attention model x kv mask) :dtype :f64 :rtol 1d-10 :atol 1d-10)))

;;; --- 因果マスクと padding マスクの性質 ---

(defun %row-ct (shape row seed)
  "SHAPE (... t f) の f32 の配列で、位置 ROW の行（最後から2番目の軸）だけ乱数、ほかは 0。"
  (let ((r (%rand shape seed)) (a (make-array shape :element-type 'single-float :initial-element 0.0)))
    (dotimes (i (array-total-size a) a)
      (when (= row (mod (floor i (car (last shape))) (car (last shape 2))))
        (setf (row-major-aref a i) (row-major-aref r i))))))

(defun %position-grad-norms (grad)
  "GRAD (... t f) の位置ごとの絶対値の和（t 個のリスト）。"
  (let* ((shape (array-dimensions grad)) (f (car (last shape))) (tt (car (last shape 2)))
         (sums (make-list tt :initial-element 0d0)))
    (dotimes (i (array-total-size grad) sums)
      (incf (nth (mod (floor i f) tt) sums) (abs (row-major-aref grad i))))))

(def-layer-property attention/causal-output-position-ignores-later-inputs
  "因果マスクのとき、出力の位置 i の損失の勾配は、入力の位置 i より後ではちょうど 0（softmax の
重みが 0 になるため）で、位置 0 から i には流れる（性質が空にならないことの対照）。バッチ軸が0個と1個、
ヘッドが1と2で確かめる。"
  (seed len row heads batched) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 5)
                                      (uniform-integer :lo 0 :hi 4) (uniform-integer :lo 1 :hi 2)
                                      (uniform-integer :lo 0 :hi 1))
  (let* ((row (mod row len))
         (model (init-multi-head-attention (nabla:prng-key seed) heads 4))
         (shape (if (= batched 1) (list 2 len 4) (list len 4)))
         (x (%rand shape (+ seed 1)))
         (ct (%row-ct shape row (+ seed 2)))
         (mask (make-causal-mask len))
         (loss (nabla:with-tracing (m x ct mask) (nabla:reduce-sum (* (call m x x mask nil t) ct))))
         (norms (%position-grad-norms (second (funcall (nabla:grad loss :argnums '(0 1)) model x ct mask)))))
    (and (every #'zerop (nthcdr (1+ row) norms))
         (plusp (nth row norms)))))

(def-layer-property attention/padding-hides-key-values
  "padding マスクで隠したキー・値の位置の kv を別の値に変えても、出力は（ちょうど）変わらない。"
  (seed tk) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 2 :hi 5))
  (let* ((model (init-multi-head-attention (nabla:prng-key seed) 2 4))
         (x (%rand (list 2 3 4) (+ seed 1)))
         (kv (%rand (list 2 tk 4) (+ seed 2)))
         (other (%rand (list 2 tk 4) (+ seed 3)))
         (valid (let ((v (make-array (list 2 tk) :element-type 'bit :initial-element 1)))
                  (setf (aref v 0 (1- tk)) 0 (aref v 1 0) 0)
                  v))
         (mask (make-attention-mask (make-array '(2 3) :element-type 'bit :initial-element 1) valid))
         (changed (make-array (list 2 tk 4) :element-type 'single-float)))
    (dotimes (b 2)
      (dotimes (s tk)
        (dotimes (f 4)
          (setf (aref changed b s f) (if (zerop (aref valid b s)) (aref other b s f) (aref kv b s f))))))
    (equalp (call model x kv mask nil t) (call model x changed mask nil t))))

(def-layer-property attention/self-attention-without-mask-is-permutation-equivariant
  "マスクなしの自己 attention は、系列の位置を並べ替えると出力も同じ並べ替えになる。"
  (seed shift) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 3))
  (let* ((len 4)
         (model (init-multi-head-attention (nabla:prng-key seed) 2 4))
         (x (%rand (list len 4) (+ seed 1)))
         (rotated (make-array (list len 4) :element-type 'single-float))
         (y (call model x x nil nil t))
         (y-rotated (make-array (list len 4) :element-type 'single-float)))
    (dotimes (i len)
      (dotimes (f 4)
        (setf (aref rotated i f) (aref x (mod (+ i shift) len) f)
              (aref y-rotated i f) (aref y (mod (+ i shift) len) f))))
    (allclose (call model rotated rotated nil nil t) y-rotated :dtype :f32 :rtol 1d-5 :atol 1d-5)))

(def-layer-property attention/all-ones-mask-is-no-mask
  "全部 1 のマスクはマスクなしと同じ出力。"
  (seed) (tuple (uniform-integer :lo 0 :hi 100000))
  (let* ((model (init-multi-head-attention (nabla:prng-key seed) 2 4))
         (x (%rand (list 2 3 4) (+ seed 1)))
         (ones (make-array '(2 1 3 3) :element-type 'bit :initial-element 1)))
    (allclose (call model x x ones nil t) (call model x x nil nil t) :dtype :f32 :rtol 1d-6 :atol 1d-6)))

(test attention/large-scores-do-not-overflow
  "スコアが大きくても（最大値を引くので）出力は有限で、重みは 1 に正規化される: kv の値が全部同じ
ベクトル v なら、出力は（スコアによらず）v の out への射影そのもの。"
  (let* ((model (make-multi-head-attention
                 :broadcast-dropout t
                 :query (make-dense-general :use-bias t :kernel (make-array '(2 1 2) :element-type 'single-float :initial-element 1000.0)
                                            :bias (make-array '(1 2) :element-type 'single-float :initial-element 0.0)
                                            :contract-axes 1)
                 :key (make-dense-general :use-bias t :kernel (make-array '(2 1 2) :element-type 'single-float :initial-element 1000.0)
                                          :bias (make-array '(1 2) :element-type 'single-float :initial-element 0.0)
                                          :contract-axes 1)
                 :value (make-dense-general :use-bias t :kernel (make-array '(2 1 2) :element-type 'single-float :initial-element 0.0)
                                            :bias (make-array '(1 2) :element-type 'single-float :initial-contents '((1.0 2.0)))
                                            :contract-axes 1)
                 :out (make-dense-general :use-bias t :kernel (make-array '(1 2 2) :element-type 'single-float
                                                                       :initial-contents '(((1.0 0.0) (0.0 1.0))))
                                          :bias (make-array '(2) :element-type 'single-float :initial-element 0.0)
                                          :contract-axes 2)
                 :dropout-rate 0))
         (x (%rand '(3 2) 4))
         (y (call model x x nil nil t)))
    (is (every (lambda (v) (and (< (abs v) 10) (= v v))) (%elements y)))
    (loop for i below 3 do (is (allclose (make-array 2 :element-type 'single-float :initial-contents (list (aref y i 0) (aref y i 1)))
                                         (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0))
                                         :dtype :f32 :rtol 1d-5 :atol 1d-5)))))

;;; --- init ---

(defun %zero-bias-attention (model)
  "bias が NIL の MODEL と同じ kernel で bias が 0 の multi-head-attention。"
  (flet ((twin (f) (%zero-bias-twin (funcall f model))))
    (make-multi-head-attention :broadcast-dropout t :query (twin 'multi-head-attention-query) :key (twin 'multi-head-attention-key)
                               :value (twin 'multi-head-attention-value) :out (twin 'multi-head-attention-out)
                               :dropout-rate 0)))

(def-layer-property attention/without-bias-is-the-zero-bias-model
  "use-bias が偽の multi-head-attention は、4 つの射影（query / key / value / out）のどれにも bias が無い（NIL）。
出力は、同じ kernel で bias が 0 のモデルと、ループの参照実装（f64、マスクの有無）の両方と一致し、葉は 4 つの kernel
だけ（パス名は query.kernel key.kernel value.kernel out.kernel）、勾配の葉も 4 つで 0 bias のモデルの kernel の勾配と
等しい。"
  (seed batch tq tk heads dim masked) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 2)
                                             (uniform-integer :lo 1 :hi 4) (uniform-integer :lo 1 :hi 4)
                                             (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 3)
                                             (uniform-integer :lo 0 :hi 1))
  (let* ((features (* heads dim))
         (plain (init-multi-head-attention (nabla:prng-key seed) heads features :use-bias nil :dtype :f64))
         (twin (%zero-bias-attention plain))
         (x (%rand (list batch tq features) (+ seed 1) :f64))
         (kv (%rand (list batch tk features) (+ seed 2) :f64))
         (mask (when (plusp masked) (%random-bits (list batch 1 tq tk) (+ seed 3))))
         (out (call plain x kv mask nil t))
         (ct (%rand (array-dimensions out) (+ seed 4) :f64))
         (loss (nabla:with-tracing (m x kv mask ct) (nabla:reduce-sum (* (call m x kv mask nil t) ct))))
         (grad (funcall (nabla:grad loss) plain x kv mask ct))
         (twin-grad (funcall (nabla:grad loss) twin x kv mask ct)))
    (and (every #'null (mapcar (lambda (f) (dense-general-bias (funcall f plain)))
                               '(multi-head-attention-query multi-head-attention-key
                                 multi-head-attention-value multi-head-attention-out)))
         (every #'null (mapcar (lambda (f) (dense-general-use-bias (funcall f plain)))
                               '(multi-head-attention-query multi-head-attention-key
                                 multi-head-attention-value multi-head-attention-out)))
         (equal '("query.kernel" "key.kernel" "value.kernel" "out.kernel") (nabla.nn:tree-paths plain))
         (equalp out (call twin x kv mask nil t))
         (allclose out (%ref-attention twin x kv mask) :dtype :f64)
         (= 4 (length (nabla:tree-leaves grad)))
         (every #'equalp
                (mapcar #'dense-general-kernel
                        (list (multi-head-attention-query grad) (multi-head-attention-key grad)
                              (multi-head-attention-value grad) (multi-head-attention-out grad)))
                (mapcar #'dense-general-kernel
                        (list (multi-head-attention-query twin-grad) (multi-head-attention-key twin-grad)
                              (multi-head-attention-value twin-grad) (multi-head-attention-out twin-grad)))))))

(test attention/use-bias-init-and-misuse
  "init-multi-head-attention の use-bias の既定は真（4 つの射影に bias）。真偽値でなければ層の名前入りの error。
use-bias が偽でも qkv / kv / out の幅の指定は効く。"
  (let ((with (init-multi-head-attention (nabla:prng-key 1) 2 4))
        (without (init-multi-head-attention (nabla:prng-key 1) 2 4 :qkv-features 6 :kv-features 5 :out-features 3
                                                                    :use-bias nil)))
    (is (every #'dense-general-use-bias (list (multi-head-attention-query with) (multi-head-attention-out with))))
    (is (equal '(4 2 3) (array-dimensions (dense-general-kernel (multi-head-attention-query without)))))
    (is (equal '(5 2 3) (array-dimensions (dense-general-kernel (multi-head-attention-key without)))))
    (is (equal '(2 3 3) (array-dimensions (dense-general-kernel (multi-head-attention-out without)))))
    (is (equal '("query.kernel" "query.bias" "key.kernel" "key.bias" "value.kernel" "value.bias"
                 "out.kernel" "out.bias")
               (nabla.nn:tree-paths with))))
  (is (search "multi-head-attention"
              (or (%error-message (lambda () (init-multi-head-attention (nabla:prng-key 1) 2 4 :use-bias 1))) ""))))

(test attention/init-shapes-names-and-distribution-match-flax
  "init-multi-head-attention (4 ヘッド、qkv = 64、入力 64) の形が Flax と同じ: query/key/value の kernel (64 4 16)、
bias (4 16)、out の kernel (4 16 64)、bias (64)。kernel の標準偏差は Flax の lecun_normal（fan_in は
縮約する軸の積: query は 64、out は 4×16 = 64）、bias は 0。スロット名は Flax のパラメータ名と同じ。
qkv-features と out-features の既定は入力の幅。"
  (let* ((model (init-multi-head-attention (nabla:prng-key 208) 4 64 :qkv-features 64))
         (q (multi-head-attention-query model))
         (o (multi-head-attention-out model)))
    (is (equal '(64 4 16) (array-dimensions (dense-general-kernel q))))
    (is (equal '(4 16) (array-dimensions (dense-general-bias q))))
    (is (equal '(64 4 16) (array-dimensions (dense-general-kernel (multi-head-attention-key model)))))
    (is (equal '(64 4 16) (array-dimensions (dense-general-kernel (multi-head-attention-value model)))))
    (is (equal '(4 16 64) (array-dimensions (dense-general-kernel o))))
    (is (equal '(64) (array-dimensions (dense-general-bias o))))
    (is (every #'zerop (%elements (dense-general-bias q))))
    (is (every #'zerop (%elements (dense-general-bias o))))
    (is (equal '(("query" . :param) ("key" . :param) ("value" . :param) ("out" . :param)
                 ("dropout-rate" . :static) ("broadcast-dropout" . :static))
               (%slot-names 'multi-head-attention)))
    (destructuring-bind (&key query-kernel out-kernel &allow-other-keys)
        (loop for (name . rest) in (getf (%attention-fixture) :init-stats)
              append (list name rest))
      (destructuring-bind (shape flax-std flax-max) query-kernel
        (is (equal '(64 4 16) shape))
        (multiple-value-bind (std max) (%stats (dense-general-kernel q))
          (is (<= (abs (- std flax-std)) (* 0.03 flax-std)) "query の標準偏差 ~F と Flax の ~F" std flax-std)
          (is (<= max (* 1.0001 (* 2 (/ (sqrt (/ 1d0 64)) 0.87962566103423978d0)))))
          (is (<= (abs (- max flax-max)) (* 0.1 flax-max)))))
      (destructuring-bind (shape flax-std flax-max) out-kernel
        (declare (ignore flax-max))
        (is (equal '(4 16 64) shape))
        (is (<= (abs (- (%stats (dense-general-kernel o)) flax-std)) (* 0.03 flax-std)))))
    (let ((defaults (init-multi-head-attention (nabla:prng-key 1) 2 6)))
      (is (equal '(6 2 3) (array-dimensions (dense-general-kernel (multi-head-attention-query defaults)))))
      (is (equal '(2 3 6) (array-dimensions (dense-general-kernel (multi-head-attention-out defaults))))))
    (let ((cross (init-multi-head-attention (nabla:prng-key 1) 4 5 :qkv-features 8 :kv-features 7 :out-features 3
                                                                  :dropout-rate 0.1)))
      (is (equal '(5 4 2) (array-dimensions (dense-general-kernel (multi-head-attention-query cross)))))
      (is (equal '(7 4 2) (array-dimensions (dense-general-kernel (multi-head-attention-key cross)))))
      (is (equal '(4 2 3) (array-dimensions (dense-general-kernel (multi-head-attention-out cross)))))
      (is (= 0.1 (multi-head-attention-dropout-rate cross))))
    (is (%same-leaves-p (init-multi-head-attention (nabla:prng-key 5) 2 4) (init-multi-head-attention (nabla:prng-key 5) 2 4)))
    (is (not (%same-leaves-p (init-multi-head-attention (nabla:prng-key 5) 2 4)
                             (init-multi-head-attention (nabla:prng-key 6) 2 4))))
    (is (not (equalp (dense-general-kernel q) (dense-general-kernel (multi-head-attention-key model)))))))

;;; --- dropout ---

(test attention/dropout-is-deterministic-or-keyed
  "deterministic が真なら rate によらず dropout なしと同じ出力（キー不要）。偽で rate > 0 なら、同じキーで
同じ出力・別のキーで別の出力・deterministic の出力とも違う。偽でキーが無いと error。rate = 0 なら偽でもキー不要で
deterministic と同じ。"
  (let* ((base (init-multi-head-attention (nabla:prng-key 1) 2 4))
         (model (make-multi-head-attention :broadcast-dropout t :query (multi-head-attention-query base) :key (multi-head-attention-key base)
                                           :value (multi-head-attention-value base) :out (multi-head-attention-out base)
                                           :dropout-rate 0.5))
         (x (%rand '(2 5 4) 2))
         (k1 (nabla:prng-key 10)) (k2 (nabla:prng-key 11))
         (plain (call base x x nil nil t)))
    (is (equalp plain (call model x x nil nil t)))
    (is (equalp plain (call base x x nil nil nil)))
    (is (equalp (call model x x nil k1 nil) (call model x x nil k1 nil)))
    (is (not (equalp (call model x x nil k1 nil) (call model x x nil k2 nil))))
    (is (not (equalp plain (call model x x nil k1 nil))))
    (signals error (call model x x nil nil nil))))

;;; --- エラー ---

(test attention/errors-name-the-layer
  "入力の幅・マスクの形・マスクの dtype・ヘッド数が不正なときは、層の名前を含むエラーになる。"
  (let* ((model (init-multi-head-attention (nabla:prng-key 1) 2 4))
         (x (%rand '(3 4) 1)))
    (flet ((names-layer (thunk) (search "multi-head-attention" (or (%error-message thunk) ""))))
      (is (search "inputs-q" (or (%error-message (lambda () (call model (%rand '(3 5) 1) x nil nil t))) ""))
          "q の幅（エラーが inputs-q を名指しする）")
      (is (search "inputs-kv" (or (%error-message (lambda () (call model x (%rand '(3 5) 1) nil nil t))) ""))
          "kv の幅（エラーが inputs-kv を名指しする）")
      (is (names-layer (lambda () (call model (%rand '(3 5) 1) x nil nil t))) "q の幅")
      (is (names-layer (lambda () (call model x (%rand '(3 5) 1) nil nil t))) "kv の幅")
      (is (names-layer (lambda () (call model x x (make-array '(1 2 3) :element-type 'bit :initial-element 1) nil t)))
          "マスクの形（キー長が合わない）")
      (is (names-layer (lambda () (call model x x (%rand '(1 3 3) 1) nil t))) "マスクが bit でない")
      (is (names-layer (lambda () (init-multi-head-attention (nabla:prng-key 1) 3 4))) "qkv がヘッド数で割れない")
      (is (names-layer (lambda () (init-multi-head-attention (nabla:prng-key 1) 2 4 :dropout-rate 1.5)))
          "rate が範囲外")
      (is (names-layer (lambda () (call model (%rand '(4) 1) (%rand '(4) 1) nil nil t))) "rank 1 の入力")
      ;; 配列でない真のマスク（t やリスト）は tracing-error でなく層名入りの error
      (is (names-layer (lambda () (call model x x t nil t))) "マスクが t")
      (is (names-layer (lambda () (call model x x '((1 0 0) (1 1 0) (1 1 1)) nil t))) "マスクがリスト"))))

(test attention/leading-axes-of-q-and-kv-must-match
  "inputs-q と inputs-kv の先頭の軸（バッチ軸）が違うと、層の名前を含む error。"
  (let ((model (init-multi-head-attention (nabla:prng-key 1) 2 4)))
    (is (search "multi-head-attention"
                (or (%error-message (lambda () (call model (%rand '(2 3 4) 1) (%rand '(3 3 4) 1) nil nil t))) "")))))

(test attention/checkpoint-paths-match-flax-names
  "multi-head-attention の葉のパス名は Flax の flatten_dict と同じ（query / key / value / out の kernel / bias）。
dropout-rate と contract-axes は :static なので現れない。"
  (is (equal '("query.kernel" "query.bias" "key.kernel" "key.bias" "value.kernel" "value.bias"
               "out.kernel" "out.bias")
             (nabla.nn:tree-paths (init-multi-head-attention (nabla:prng-key 1) 2 4)))))

;;; --- broadcast_dropout（issue #241）---
;;;
;;; attention の重みは直接見えないので、重みが一様（query / key の kernel が 0 ならスコアは全部 0 で softmax は
;;; 1/幅）で、value が単位行列（v[b k h d] = [k = d]）、out が (h d) を平らにする恒等になるように作ったモデルで
;;; 調べる。すると出力 [... q (h 幅 + k)] = 重み[... h q k] = マスク[... h q k] / (幅 (1 - rate))（落ちた位置は 0）で、
;;; マスクがそのまま読める。Flax の dropout_rng でビット単位の比較はできない（nabla の PRNG は JAX と違う）ので、
;;; 保持率・平均の保存・マスクが共有される軸と独立な軸を確かめる。

(defun %all-indices (dims)
  "DIMS（大きさのリスト）の添字のリストをすべて（辞書順で）並べたもの。DIMS が空なら (())。"
  (if (null dims)
      (list '())
      (loop for i below (first dims)
            append (mapcar (lambda (rest) (cons i rest)) (%all-indices (rest dims))))))

(defun %dropout-probe-model (heads width broadcast rate)
  "ファイル冒頭のコメントの、重みが出力にそのまま現れる multi-head-attention（入力の幅 = ヘッドの幅 = WIDTH）。"
  (flet ((zeros (&rest dims) (make-array dims :element-type 'single-float :initial-element 0.0))
         (projection (kernel contract-axes)
           (make-dense-general :kernel kernel :bias nil :contract-axes contract-axes :use-bias nil)))
    (let ((value-kernel (make-array (list width heads width) :element-type 'single-float :initial-element 0.0))
          (out-kernel (make-array (list heads width (* heads width)) :element-type 'single-float
                                                                      :initial-element 0.0)))
      (dotimes (f width) (dotimes (h heads) (setf (aref value-kernel f h f) 1.0)))
      (dotimes (h heads) (dotimes (d width) (setf (aref out-kernel h d (+ (* h width) d)) 1.0)))
      (make-multi-head-attention :query (projection (zeros width heads width) 1)
                                 :key (projection (zeros width heads width) 1)
                                 :value (projection value-kernel 1)
                                 :out (projection out-kernel 2)
                                 :dropout-rate rate :broadcast-dropout broadcast))))

(defun %dropout-probe (batch heads queries width broadcast rate seed)
  "プローブのモデルを BATCH（先頭の軸の大きさのリスト）つきの入力に適用して、出力の配列を返す。"
  (let ((kv (make-array (append batch (list width width)) :element-type 'single-float :initial-element 0.0))
        (q (make-array (append batch (list queries width)) :element-type 'single-float :initial-element 0.0)))
    (dolist (index (%all-indices batch))
      (dotimes (k width) (setf (apply #'aref kv (append index (list k k))) 1.0)))
    (call (%dropout-probe-model heads width broadcast rate) q kv nil (nabla:prng-key seed) nil)))

(defun %probe-mask (out batch heads queries width)
  "プローブの出力から、(バッチ添字 ヘッド) ごとの q × k のマスク（保持なら 1、落ちたなら 0 の bit のリスト）の
連想リストを作る。"
  (loop for index in (%all-indices batch)
        append (loop for h below heads
                     collect (cons (append index (list h))
                                   (loop for q below queries
                                         append (loop for k below width
                                                      collect (if (plusp (apply #'aref out (append index (list q (+ (* h width) k)))))
                                                                  1 0)))))))

(def-layer-property attention/broadcast-dropout-shares-one-mask-over-batch-axes-and-heads
  "broadcast-dropout が真（Flax の既定）のとき、dropout のマスクはバッチ軸（0 個・1 個・2 個）とヘッドで共有される:
どの (バッチ添字, ヘッド) でも q × k のマスクが同じ。マスクは重みの要素ごとに 0 / 1 で、保持した重みは 1/(幅 (1 - rate))、
落ちた重みは 0。"
  (seed rate-percent batch-rank) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 10 :hi 90)
                                        (uniform-integer :lo 0 :hi 2))
  (let* ((rate (/ rate-percent 100f0))
         (batch (subseq '(2 3) 0 batch-rank))
         (out (%dropout-probe batch 2 4 5 t rate seed))
         (masks (%probe-mask out batch 2 4 5))
         (kept-value (/ 1 (* 5 (- 1 rate)))))
    (and (every (lambda (entry) (equal (cdr entry) (cdr (first masks)))) masks)
         (loop for i below (array-total-size out)
               always (let ((v (row-major-aref out i)))
                        (or (zerop v) (< (abs (- v kept-value)) (* 1d-5 kept-value))))))))

(def-layer-property attention/dropout-without-broadcast-draws-independent-masks
  "broadcast-dropout が偽のとき、マスクは重みの要素ごとに独立: 2 つのバッチ × 2 つのヘッドの 4 つの q × k
のマスク（8 × 8）はどの 2 つも違う（rate 30% から 70% では、同じになる確率は 1e-15 未満）。"
  (seed rate-percent) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 30 :hi 70))
  (let* ((out (%dropout-probe '(2) 2 8 8 nil (/ rate-percent 100f0) seed))
         (masks (mapcar #'cdr (%probe-mask out '(2) 2 8 8))))
    (= 4 (length (remove-duplicates masks :test #'equal)))))

(defun %probe-statistics (broadcast rate seed)
  "プローブ（バッチ 2 × ヘッド 2 × q 32 × k 32）の (values 保持率 出力の平均 × 幅 独立な標本の数)。
出力の平均 × 幅 は重みの総和の平均 = mean(mask) / (1 - rate) で、保存されていれば 1 に近い。"
  (let* ((out (%dropout-probe '(2) 2 32 32 broadcast rate seed))
         (n (array-total-size out))
         (kept (loop for i below n count (plusp (row-major-aref out i)))))
    (values (/ kept n)
            (* 32 (/ (loop for i below n sum (row-major-aref out i)) n))
            (if broadcast (* 32 32) n))))

(def-layer-property attention/dropout-keeps-the-expected-fraction-and-preserves-the-mean
  "broadcast-dropout が真でも偽でも、attention の dropout の保持率は 1 - rate に近い（二項分布の 4.5 標準偏差以内。
独立な標本の数は、共有するときは q × k の 1024、しないときは重みの全要素）。重みの総和の平均 × 幅 は
保持率 / (1 - rate) に等しい（保持した重みが 1/(1 - rate) 倍になる）ので、平均が 1 に保たれることは、保持率の検査を
別の形で見たもの（独立な性質ではない）。誤って落ちる確率は 1e-5 未満。"
  (seed rate-percent broadcast) (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 10 :hi 90)
                                       (uniform-integer :lo 0 :hi 1))
  (let ((rate (/ rate-percent 100d0)))
    (multiple-value-bind (fraction mean n) (%probe-statistics (plusp broadcast) (coerce rate 'single-float) seed)
      (let* ((keep (- 1 rate))
             (sd (sqrt (/ (* keep rate) n))))
        (and (< (abs (- fraction keep)) (* 4.5 sd))
             (< (abs (- mean 1)) (* 4.5 (/ sd keep))))))))

(test attention/broadcast-dropout-default-and-same-key-same-mask
  "init-multi-head-attention の broadcast-dropout の既定は真（Flax と同じ）、偽も選べる。broadcast-dropout が真でも偽でも、
同じキーなら同じ出力・別のキーなら別の出力で、deterministic が真なら broadcast-dropout によらず dropout なしと同じ出力。
真偽値でなければ層名入りの error。"
  (let ((on (init-multi-head-attention (nabla:prng-key 1) 2 4 :dropout-rate 0.5))
        (off (init-multi-head-attention (nabla:prng-key 1) 2 4 :dropout-rate 0.5 :broadcast-dropout nil))
        (x (%rand '(2 6 4) 2)))
    (is (eq t (multi-head-attention-broadcast-dropout on)))
    (is (null (multi-head-attention-broadcast-dropout off)))
    (dolist (model (list on off))
      (is (equalp (call model x x nil (nabla:prng-key 3) nil) (call model x x nil (nabla:prng-key 3) nil)))
      (is (not (equalp (call model x x nil (nabla:prng-key 3) nil) (call model x x nil (nabla:prng-key 4) nil))))
      (is (equalp (call model x x nil nil t) (call (init-multi-head-attention (nabla:prng-key 1) 2 4) x x nil nil t))))
    (is (not (equalp (call on x x nil (nabla:prng-key 3) nil) (call off x x nil (nabla:prng-key 3) nil)))
        "共有するマスクと独立なマスクでは同じキーでも出力が違う")
    (is (search "multi-head-attention"
                (or (%error-message (lambda () (init-multi-head-attention (nabla:prng-key 1) 2 4 :broadcast-dropout 1)))
                    "")))))

(test attention/broadcast-dropout-with-rate-zero-and-no-batch-axes
  "rate = 0 なら broadcast-dropout によらず dropout なし。バッチ軸が無い入力（[長さ 幅]）でも、共有される軸は
ヘッドだけで動く（マスクがヘッドで共有される）。"
  (let ((x (%rand '(5 4) 2)))
    (dolist (broadcast '(t nil))
      (let ((model (init-multi-head-attention (nabla:prng-key 1) 2 4 :broadcast-dropout broadcast)))
        (is (equalp (call model x x nil (nabla:prng-key 3) nil) (call model x x nil nil t))
            "broadcast-dropout ~S: rate 0 は dropout なし" broadcast))))
  (let ((masks (mapcar #'cdr (%probe-mask (%dropout-probe '() 3 6 4 t 0.5f0 11) '() 3 6 4))))
    (is (= 3 (length masks)) "ヘッドの数")
    (is (= 24 (length (first masks))) "q × k")
    (is (= 1 (length (remove-duplicates masks :test #'equal))) "3 つのヘッドが同じマスク")
    (is (< 0 (reduce #'+ (first masks)) 24) "全部保持でも全部落ちでもない")))
