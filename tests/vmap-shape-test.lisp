;;;; 形状演算・縮約・dot-general のバッチ化ルールの性質（issue #129）。
;;;;
;;;; 守らせる性質: 「vmap f の結果は、バッチ軸で切り出した各要素に f を eager で適用して
;;;; out-axes の位置に積み直したもの（REFERENCE-VMAP）と一致する」。プリミティブごとに、
;;;; params・バッチ軸の位置・out-axes・（dot-general では）どの引数をバッチするかを
;;;; ランダムに生成する。f は1つのプリミティブを直接呼ぶ関数（%PRIMITIVE-FUNCTION）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %primitive-function (name params arity)
  "プリミティブ NAME を PARAMS（plist）で1回だけ呼ぶ ARITY 引数の TRACEABLE-FUNCTION。
トレーサが渡れば eqn を足し、配列だけなら eager 実装を呼ぶ。"
  (nb::%make-traceable-function
   (loop for i below arity collect (intern (format nil "X~D" i)))
   (lambda (&rest args)
     (if (some (lambda (a) (typep a 'nb::tracer)) args)
         (apply #'nb::%trace-eqn name args params)
         (apply (nb::primitive-eager (nb::find-primitive name))
                args (mapcar #'nb:array-aval args) params)))))

(defun %shuffle-from-seed (items seed)
  "ITEMS を SEED から決まる順に並べ替える（SEED から作る LCG の値でソートする）。"
  (let ((state seed))
    (mapcar #'cdr
            (sort (mapcar (lambda (item)
                            (setf state (mod (+ (* state 1103515245) 12345) 2147483648))
                            (cons state item))
                          items)
                  #'< :key #'car))))

(defun %case-generator ()
  (generator (tuple (uniform-integer :lo 0 :hi 3)      ; 0 内側の rank
                    (uniform-integer :lo 1 :hi 4)      ; 1 バッチの長さ
                    (uniform-integer :lo 0 :hi 100000) ; 2 seed
                    (uniform-integer :lo 0 :hi 99)     ; 3 バッチ軸の選択
                    (uniform-integer :lo 0 :hi 99)     ; 4 out-axes の選択
                    (uniform-integer :lo 0 :hi 99))))  ; 5 その他

(defun %nonempty-axes-from-seed (rank seed)
  "0..RANK-1 の空でない部分集合（昇順）。RANK は1以上。"
  (let ((mask (1+ (mod seed (1- (expt 2 rank))))))
    (loop for i below rank when (logbitp i mask) collect i)))

(defun %unary-vmap-property (build &key (min-rank 0))
  "BUILD: (inner-shape seed code) → (values primitive-name params out-inner-rank)。
1引数のプリミティブの vmap が参照実装と一致する性質。内側の rank は MIN-RANK 以上にする。"
  (lambda (case)
    (destructuring-bind (rank size seed code-axis code-out code) case
      (let ((inner (%vmap-inner-shape (max rank min-rank) seed)))
        (multiple-value-bind (name params out-rank) (funcall build inner seed code)
          (let* ((f (%primitive-function name params 1))
                 (axis (mod code-axis (1+ (length inner))))
                 (out (mod code-out (+ out-rank 1)))
                 (x (%vmap-batched-array inner size axis seed))
                 (expected (reference-vmap f (list x) :in-axes axis :out-axes out))
                 (actual (funcall (nb:vmap f :in-axes axis :out-axes out) x)))
            (allclose actual (first expected) :dtype :f64)))))))

(defmacro def-unary-vmap-test (test-name doc (&key (min-rank 0)) build-lambda)
  `(test ,test-name
     ,doc
     (is (check-it (%case-generator)
                   (%unary-vmap-property ,build-lambda :min-rank ,min-rank)
                   :regression-id ,test-name
                   :regression-file (regression-path "vmap-shape")))))

(def-unary-vmap-test vmap-shape/transpose-matches-per-slice-reference
  "transpose: perm と batch 軸の位置をランダムにしても参照実装と一致する。" ()
  (lambda (inner seed code)
    (declare (ignore code))
    (values :transpose
            (list :perm (%shuffle-from-seed (loop for i below (length inner) collect i) seed))
            (length inner))))

(def-unary-vmap-test vmap-shape/reshape-matches-per-slice-reference
  "reshape: 平坦化・軸の逆順・先頭に長さ1・先頭の軸の分割をランダムに選んでも一致する。" ()
  (lambda (inner seed code)
    (declare (ignore seed))
    (let* ((total (reduce #'* inner))
           (new (ecase (mod code 4)
                  (0 (list total))
                  (1 (reverse inner))
                  (2 (cons 1 inner))
                  (3 (if inner
                         (list (first inner) (/ total (first inner)))
                         '(1 1))))))
      (values :reshape (list :shape new) (length new)))))

(def-unary-vmap-test vmap-shape/reduce-sum-matches-per-slice-reference
  "reduce-sum: 縮約する軸の部分集合とバッチ軸の位置をランダムにしても一致する（内側 rank 1 以上）。"
  (:min-rank 1)
  (lambda (inner seed code)
    (declare (ignore code))
    (let ((axes (%nonempty-axes-from-seed (length inner) seed)))
      (values :reduce-sum (list :axes axes) (- (length inner) (length axes))))))

(def-unary-vmap-test vmap-shape/reduce-max-matches-per-slice-reference
  "reduce-max: 縮約する軸の部分集合とバッチ軸の位置をランダムにしても一致する（内側 rank 1 以上）。"
  (:min-rank 1)
  (lambda (inner seed code)
    (declare (ignore code))
    (let ((axes (%nonempty-axes-from-seed (length inner) seed)))
      (values :reduce-max (list :axes axes) (- (length inner) (length axes))))))

(def-unary-vmap-test vmap-shape/broadcast-in-dim-matches-per-slice-reference
  "broadcast-in-dim: 入力の軸を出力のどの位置に置くか・足す軸の長さをランダムにしても一致する。" ()
  (lambda (inner seed code)
    (let* ((out-rank (+ (length inner) (mod code 3)))
           (dims (sort (subseq (%shuffle-from-seed (loop for i below out-rank collect i) seed)
                               0 (length inner))
                       #'<))
           (shape (loop for i below out-rank
                        collect (let ((p (position i dims)))
                                  (if p (nth p inner) (1+ (mod (+ seed i) 3)))))))
      (values :broadcast-in-dim (list :shape shape :dims dims) out-rank))))

;;; --- dot-general ---

(defun %dot-case (seed code)
  "SEED / CODE から dot-general の params と lhs / rhs の内側の形を作る。
lhs・rhs の各軸は「バッチ・縮約・自由」のどれかで、軸の並びもランダム。
(values params lhs-shape rhs-shape out-rank)"
  (let* ((n-batch (mod code 3))
         (n-contract (mod (floor code 3) 3))
         (n-lfree (mod (floor code 9) 3))
         (n-rfree (mod (floor code 27) 3))
         (batch-sizes (loop for i below n-batch collect (1+ (mod (+ seed i) 3))))
         (contract-sizes (loop for i below n-contract collect (1+ (mod (+ seed 5 i) 3))))
         (lfree-sizes (loop for i below n-lfree collect (1+ (mod (+ seed 11 i) 3))))
         (rfree-sizes (loop for i below n-rfree collect (1+ (mod (+ seed 17 i) 3))))
         (lhs-roles (%shuffle-from-seed
                     (append (loop for i below n-batch collect (list :b i))
                             (loop for i below n-contract collect (list :c i))
                             (loop for i below n-lfree collect (list :f i)))
                     seed))
         (rhs-roles (%shuffle-from-seed
                     (append (loop for i below n-batch collect (list :b i))
                             (loop for i below n-contract collect (list :c i))
                             (loop for i below n-rfree collect (list :f i)))
                     (+ seed 3)))
         (lhs-shape (loop for (role i) in lhs-roles
                          collect (ecase role (:b (nth i batch-sizes)) (:c (nth i contract-sizes))
                                    (:f (nth i lfree-sizes)))))
         (rhs-shape (loop for (role i) in rhs-roles
                          collect (ecase role (:b (nth i batch-sizes)) (:c (nth i contract-sizes))
                                    (:f (nth i rfree-sizes)))))
         (params (list :lhs-contracting (loop for i below n-contract
                                              collect (position (list :c i) lhs-roles :test #'equal))
                       :rhs-contracting (loop for i below n-contract
                                              collect (position (list :c i) rhs-roles :test #'equal))
                       :lhs-batch (loop for i below n-batch
                                        collect (position (list :b i) lhs-roles :test #'equal))
                       :rhs-batch (loop for i below n-batch
                                        collect (position (list :b i) rhs-roles :test #'equal)))))
    (values params lhs-shape rhs-shape (+ n-batch n-lfree n-rfree))))

(defun %dot-general-vmap-property (case)
  "CASE (size seed code mode code-l code-r code-out) の dot-general の vmap が参照実装と一致する。
MODE は 0 lhs だけ / 1 rhs だけ / 2 両方 をバッチする。"
  (destructuring-bind (size seed code mode code-l code-r code-out) case
    (multiple-value-bind (params lhs-shape rhs-shape out-rank) (%dot-case seed code)
      (let* ((f (%primitive-function :dot-general params 2))
             (al (and (member mode '(0 2)) (mod code-l (1+ (length lhs-shape)))))
             (ar (and (member mode '(1 2)) (mod code-r (1+ (length rhs-shape)))))
             (out (mod code-out (+ out-rank 1)))
             (x (%vmap-batched-array lhs-shape size al seed))
             (y (%vmap-batched-array rhs-shape size ar (1+ seed)))
             (expected (reference-vmap f (list x y) :in-axes (list al ar) :out-axes out))
             (actual (funcall (nb:vmap f :in-axes (list al ar) :out-axes out) x y)))
        (allclose actual (first expected) :dtype :f64)))))

(defun %dot-general-case-generator ()
  (generator (tuple (uniform-integer :lo 1 :hi 3)       ; バッチの長さ
                    (uniform-integer :lo 0 :hi 100000)  ; seed
                    (uniform-integer :lo 0 :hi 80)      ; 次元の構成
                    (uniform-integer :lo 0 :hi 2)       ; 0 lhs だけ 1 rhs だけ 2 両方
                    (uniform-integer :lo 0 :hi 99)      ; lhs のバッチ軸
                    (uniform-integer :lo 0 :hi 99)      ; rhs のバッチ軸
                    (uniform-integer :lo 0 :hi 99))))   ; out-axes

(test vmap-shape/dot-general-matches-per-slice-reference
  "dot-general: 既存の batch 次元・縮約次元・自由次元の数と並び、どの引数をバッチするか
（lhs だけ / rhs だけ / 両方）、バッチ軸の位置、out-axes をランダムにしても一致する。"
  (is (check-it (%dot-general-case-generator) #'%dot-general-vmap-property
                :regression-id vmap-shape/dot-general-matches-per-slice-reference
                :regression-file (regression-path "vmap-shape"))))

(test vmap-shape/dot-general-covers-every-batching-mode
  "固定の例: 行列積を only-lhs / only-rhs / 両側 / 既存 batch 次元あり で vmap したものが
参照実装と一致する（PBT が各モードを必ず通ることの下限）。"
  (dolist (case '((2 7 3 0 0 0 0)    ; code 3: n-contract 1 のみ → 行列積風、lhs だけ
                  (2 7 3 1 0 1 1)    ; rhs だけ
                  (2 7 3 2 1 0 2)    ; 両側
                  (2 9 31 2 0 1 0)   ; batch 次元あり、両側
                  (2 9 31 0 2 0 1)   ; batch 次元あり、lhs だけ
                  (2 9 31 1 0 2 1))) ; batch 次元あり、rhs だけ
    (is (%dot-general-vmap-property case) "case ~S" case)))

(test vmap-shape/reduce-sum-keeps-natural-batch-position-without-extra-transpose
  "reduce-sum は出力のバッチ軸を自然な位置に置くので、in-axes 1・out-axes 0 でも
eqn は reduce-sum 1つだけ（余分な transpose を足さない）。"
  (let* ((f (%primitive-function :reduce-sum '(:axes (0)) 1))
         (graph (nb:trace-to-graph (nb:vmap f :in-axes 1 :out-axes 0)
                                   (list (nb:make-aval '(3 4) :f64)))))
    (is (equal '(:reduce-sum)
               (mapcar (lambda (e) (nb::primitive-name (nb::eqn-prim e))) (nb:graph-eqns graph))))))
