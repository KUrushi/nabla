;;;; primitive-recipes: emit-stablehlo の medium PBT が使う「レシピ」形式
;;;; （issue #33、wave 3 s1）。tests/graph-recipes.lisp（nabla.tests パッケージ、
;;;; テスト専用の %TEST- プリミティブ向け）と同じ考え方だが、こちらは実際に
;;;; 登録されているプリミティブ（add / neg / compare / dot-general など）
;;;; だけを使い、生成した graph をそのまま EMIT-STABLEHLO → IREE に渡せる
;;;; ようにする。IREE の TO-DEVICE / 出力の型制約（:f32 :bf16 :f16 のみ）に
;;;; 合わせ、入出力の dtype はこの3つに限る。:div と :log は inf/NaN を
;;;; 作りうるので対象外（例ベースのテストで別に確かめる。契約 §3）。
;;;;
;;;; レシピはステップのリスト:
;;;;   (:in dtype shape)                    入力 var を1つ作る
;;;;   (:const dtype shape seed)            定数 var を1つ作る
;;;;   (:binary prim-keyword idx idx)       add/sub/mul/max/min
;;;;   (:unary prim-keyword idx)            neg/tanh/exp
;;;;   (:compare-select idx idx direction idx)  compare の pred を select に
;;;;                                        すぐ使う（pred 自身は出力候補に
;;;;                                        残さない）
;;;;   (:convert idx dtype)
;;;;   (:reshape idx shape)                 要素数を変えないフラット化
;;;;   (:transpose idx perm)
;;;;   (:broadcast idx)                     先頭にサイズ1の次元を1つ足す
;;;;   (:reduce prim-keyword idx axis)      reduce-sum/reduce-max
;;;;   (:dot idx seed n)                    lhs (... K) と、新しく作る定数
;;;;                                        rhs (K N) の2次元 dot-general
;;;;   (:out idx)                           出力 var を1つ選ぶ
;;;;
;;;; IDX は「それまでに定義され、後続の演算に使ってよい var の番号」
;;;; （0始まり）。compare の pred はここに現れない（select にしか使わない）。

(in-package #:nabla.tests.support)

(defparameter *primitive-recipe-dtypes* '(:f32 :bf16 :f16)
  "IREE の TO-DEVICE / 出力が受け付ける3つの dtype。")

(defun %pr-random-shape (max-rank max-dim)
  (loop repeat (random (1+ max-rank)) collect (1+ (random max-dim))))

(defun %pr-random-dtype ()
  (nth (random (length *primitive-recipe-dtypes*)) *primitive-recipe-dtypes*))

(defun %pr-matching-index (avals idx)
  "AVALS（(shape . dtype) のリスト）の中から IDX と EQUAL な aval を持つ、
IDX 以外のインデックスを探す。無ければ IDX 自身を返す（自分自身との
二項演算・select になる。契約が明示的に許す「compare of an expression
with itself」と同じ考え方）。"
  (let ((target (nth idx avals)))
    (or (loop for i from 0
              for a in avals
              when (and (/= i idx) (equal a target))
                return i)
        idx)))

(defun %generate-primitive-recipe (max-ops)
  "レシピ（ステップのリスト）を1つランダムに生成して返す。少なくとも1つの
:IN ステップと、末尾に1つの :OUT ステップを持つ。"
  (let ((steps '())
        ;; AVALS は (shape . dtype) のリスト。インデックスは「後続の演算に
        ;; 使ってよい var」と対応する（compare の pred はここに現れない）。
        (avals '())
        (n-in (1+ (random 2))))
    (dotimes (_ n-in)
      (let ((shape (%pr-random-shape 3 4))
            (dtype (%pr-random-dtype)))
        (push (list :in dtype shape) steps)
        (setf avals (append avals (list (cons shape dtype))))))
    (when (zerop (random 2))
      (let ((shape (%pr-random-shape 3 4))
            (dtype (%pr-random-dtype)))
        (push (list :const dtype shape (random (expt 2 31))) steps)
        (setf avals (append avals (list (cons shape dtype))))))
    (dotimes (_ (random (1+ max-ops)))
      (let* ((n (length avals))
             (idx (random n))
             (aval (nth idx avals))
             (shape (car aval))
             (dtype (cdr aval))
             (rank (length shape))
             (choices (append '(:binary :unary :compare-select :convert :broadcast)
                               (when (plusp rank) '(:reshape :transpose :reduce))
                               (when (= rank 2) '(:dot))))
             (kind (nth (random (length choices)) choices)))
        (ecase kind
          (:binary
           (let* ((prim (nth (random 5) '(:add :sub :mul :max :min)))
                  (idx2 (%pr-matching-index avals idx)))
             (push (list :binary prim idx idx2) steps)
             (setf avals (append avals (list aval)))))
          (:unary
           (let ((prim (nth (random 3) '(:neg :tanh :exp))))
             (push (list :unary prim idx) steps)
             (setf avals (append avals (list aval)))))
          (:compare-select
           (let ((idx2 (%pr-matching-index avals idx))
                 (idx3 (%pr-matching-index avals idx))
                 (direction (nth (random 6) '(:lt :le :gt :ge :eq :ne))))
             (push (list :compare-select idx idx2 direction idx3) steps)
             (setf avals (append avals (list aval)))))
          (:convert
           (let ((new-dtype (%pr-random-dtype)))
             (push (list :convert idx new-dtype) steps)
             (setf avals (append avals (list (cons shape new-dtype))))))
          (:broadcast
           (push (list :broadcast idx) steps)
           (setf avals (append avals (list (cons (cons 1 shape) dtype)))))
          (:reshape
           (let ((flat (list (reduce #'* shape :initial-value 1))))
             (push (list :reshape idx flat) steps)
             (setf avals (append avals (list (cons flat dtype))))))
          (:transpose
           (let ((perm (%shuffled-list rank)))
             (push (list :transpose idx perm) steps)
             (setf avals (append avals (list (cons (mapcar (lambda (p) (nth p shape)) perm) dtype))))))
          (:reduce
           (let* ((prim (nth (random 2) '(:reduce-sum :reduce-max)))
                  (axis (random rank))
                  (new-shape (append (subseq shape 0 axis) (subseq shape (1+ axis)))))
             (push (list :reduce prim idx axis) steps)
             (setf avals (append avals (list (cons new-shape dtype))))))
          (:dot
           (let* ((k (second shape))
                  (n (1+ (random 4)))
                  (seed (random (expt 2 31))))
             (push (list :dot idx seed n) steps)
             (setf avals (append avals (list (cons (list (first shape) n) dtype)))))))))
    (push (list :out (1- (length avals))) steps)
    (nreverse steps)))

(defun %shuffled-list (n)
  "0..N-1 をランダムに並べ替えたリストを返す（TRANSPOSE の perm に使う）。"
  (let ((list (loop for i below n collect i)))
    (loop for i from (1- n) downto 1
          do (rotatef (nth i list) (nth (random (1+ i)) list)))
    list))

(defun %recipe-step-aval (step avals)
  "STEP が AVALS（(shape . dtype) のリスト、0始まりの IDX に対応）に
新しく積むはずの (shape . dtype) を返す。:OUT は AVALS を伸ばさないので
NIL を返す。%GENERATE-PRIMITIVE-RECIPE の各 ECASE 節が計算しているのと
同じ規則（レシピの各ステップの意味そのもの）を、ここに独立にもう一度
書き下している。REPLAY-RECIPE-AVALS がこれを使い、既存のレシピ（乱数を
経由せず、既に確定した IDX・SHAPE・DTYPE を持つ）から同じ AVALS を
計算し直す。BUILD-PRIMITIVE-GRAPH が IDX の指す var を取り違えていないか
を、生成器の実装から独立に検査する（issue #33 のリグレッション: :DOT が
rhs 定数を VARS に混ぜて後続の IDX をずらしたバグ）。"
  (ecase (first step)
    (:in (destructuring-bind (dtype shape) (rest step) (cons shape dtype)))
    (:const (destructuring-bind (dtype shape seed) (rest step)
              (declare (ignore seed))
              (cons shape dtype)))
    (:binary (destructuring-bind (prim idx1 idx2) (rest step)
               (declare (ignore prim idx2))
               (nth idx1 avals)))
    (:unary (destructuring-bind (prim idx) (rest step)
              (declare (ignore prim))
              (nth idx avals)))
    (:compare-select (destructuring-bind (idx1 idx2 direction idx3) (rest step)
                        (declare (ignore idx2 direction idx3))
                        (nth idx1 avals)))
    (:convert (destructuring-bind (idx dtype) (rest step)
                (cons (car (nth idx avals)) dtype)))
    (:broadcast (destructuring-bind (idx) (rest step)
                  (let ((aval (nth idx avals)))
                    (cons (cons 1 (car aval)) (cdr aval)))))
    (:reshape (destructuring-bind (idx shape) (rest step)
                (cons shape (cdr (nth idx avals)))))
    (:transpose (destructuring-bind (idx perm) (rest step)
                  (let* ((aval (nth idx avals))
                         (shape (car aval)))
                    (cons (mapcar (lambda (p) (nth p shape)) perm) (cdr aval)))))
    (:reduce (destructuring-bind (prim idx axis) (rest step)
               (declare (ignore prim))
               (let* ((aval (nth idx avals))
                      (shape (car aval)))
                 (cons (append (subseq shape 0 axis) (subseq shape (1+ axis))) (cdr aval)))))
    (:dot (destructuring-bind (idx seed n) (rest step)
            (declare (ignore seed))
            (let* ((aval (nth idx avals))
                   (shape (car aval)))
              (cons (list (first shape) n) (cdr aval)))))
    (:out nil)))

(defun replay-recipe-avals (recipe)
  "RECIPE（BUILD-PRIMITIVE-GRAPH と同じ形式）を辿り、各 IDX が指すはずの
(shape . dtype) を、%GENERATE-PRIMITIVE-RECIPE とは独立にレシピ自身の
記述だけから計算して (shape . dtype) のリストとして返す。:OUT は含まない
（IDX の参照先にならないため）。"
  (let ((avals '()))
    (dolist (step recipe)
      (let ((next (%recipe-step-aval step avals)))
        (when next (setf avals (append avals (list next))))))
    avals))

(defclass %primitive-graph-recipe-generator (check-it:generator)
  ((max-ops :initarg :max-ops :reader %primitive-graph-recipe-max-ops))
  (:documentation "PRIMITIVE-GRAPH-RECIPE の named generator の実体。"))

(defmethod check-it:generate ((generator %primitive-graph-recipe-generator))
  (%generate-primitive-recipe (%primitive-graph-recipe-max-ops generator)))

(defmethod check-it:shrink ((generator %primitive-graph-recipe-generator) test)
  ;; レシピの縮小は行わない（tests/graph-recipes.lisp の GRAPH-RECIPE と同じ
  ;; 考え方）。
  (declare (ignore test))
  (check-it:cached-value generator))

(check-it:def-generator primitive-graph-recipe (&key (max-ops 4))
  (make-instance '%primitive-graph-recipe-generator :max-ops max-ops))

(defun build-primitive-graph (recipe)
  "RECIPE（このファイル冒頭のレシピ形式）から NB::GRAPH を組み立てて返す。
実プリミティブの MAKE-EQN を呼ぶので、abstract-eval が形状・dtype の
不変量を検査する。CHECK-GRAPH は呼ばない。

第2の値として、RECIPE の各 IDX に対応する var を、生成した順（AVALS の
インデックスと1対1）に並べたベクタを返す。REPLAY-RECIPE-AVALS が
独立に計算する期待値と突き合わせるテスト専用の値で、通常の呼び出し側は
無視してよい。"
  (let ((vars (make-array 0 :adjustable t :fill-pointer 0))
        (invars '())
        (constants '())
        (eqns '())
        (outvars '()))
    (flet ((push-result (eqn)
             (push eqn eqns)
             (vector-push-extend (first (nb:eqn-outvars eqn)) vars)))
      (dolist (step recipe)
        (ecase (first step)
          (:in (destructuring-bind (dtype shape) (rest step)
                 (let ((v (nb::make-var (nb:make-aval shape dtype))))
                   (vector-push-extend v vars)
                   (push v invars))))
          (:const (destructuring-bind (dtype shape seed) (rest step)
                    (let* ((array (make-random-array (make-array-spec shape dtype) :seed seed))
                           (v (nb::make-var (nb:array-aval array dtype))))
                      (vector-push-extend v vars)
                      (push (cons v array) constants))))
          (:binary (destructuring-bind (prim idx1 idx2) (rest step)
                     (push-result (nb::make-eqn prim (list (aref vars idx1) (aref vars idx2))))))
          (:unary (destructuring-bind (prim idx) (rest step)
                    (push-result (nb::make-eqn prim (list (aref vars idx))))))
          (:compare-select
           (destructuring-bind (idx1 idx2 direction idx3) (rest step)
             (let* ((pred-eqn (nb::make-eqn :compare (list (aref vars idx1) (aref vars idx2)) :direction direction))
                    (pred (first (nb:eqn-outvars pred-eqn))))
               (push pred-eqn eqns)
               (push-result (nb::make-eqn :select (list pred (aref vars idx1) (aref vars idx3)))))))
          (:convert (destructuring-bind (idx dtype) (rest step)
                      (push-result (nb::make-eqn :convert (list (aref vars idx)) :dtype dtype))))
          (:broadcast (destructuring-bind (idx) (rest step)
                        (let* ((in-shape (nb:aval-shape (nb:var-aval (aref vars idx))))
                               (out-shape (cons 1 in-shape))
                               (dims (loop for i below (length in-shape) collect (1+ i))))
                          (push-result (nb::make-eqn :broadcast-in-dim (list (aref vars idx))
                                                      :shape out-shape :dims dims)))))
          (:reshape (destructuring-bind (idx shape) (rest step)
                      (push-result (nb::make-eqn :reshape (list (aref vars idx)) :shape shape))))
          (:transpose (destructuring-bind (idx perm) (rest step)
                        (push-result (nb::make-eqn :transpose (list (aref vars idx)) :perm perm))))
          (:reduce (destructuring-bind (prim idx axis) (rest step)
                     (push-result (nb::make-eqn prim (list (aref vars idx)) :axes (list axis)))))
          (:dot (destructuring-bind (idx seed n) (rest step)
                  ;; RHS-VAR は %GENERATE-PRIMITIVE-RECIPE の AVALS には現れない
                  ;; （AVALS が増えるのは dot の出力1個分だけ）ので、VARS にも
                  ;; 積まない。積むと後続ステップの IDX が VARS 上で1つずれ、
                  ;; 無関係な var を参照してしまう（issue #33 のリグレッション）。
                  (let* ((lhs-var (aref vars idx))
                         (lhs-aval (nb:var-aval lhs-var))
                         (dtype (nb:aval-dtype lhs-aval))
                         (k (second (nb:aval-shape lhs-aval)))
                         (rhs-array (make-random-array (make-array-spec (list k n) dtype) :seed seed))
                         (rhs-var (nb::make-var (nb:array-aval rhs-array dtype))))
                    (push (cons rhs-var rhs-array) constants)
                    (push-result (nb::make-eqn :dot-general (list lhs-var rhs-var)
                                                :lhs-contracting '(1) :rhs-contracting '(0)
                                                :lhs-batch '() :rhs-batch '())))))
          (:out (setf outvars (list (aref vars (second step)))))))
      (values (nb::make-graph (nreverse invars) (nreverse eqns) outvars (nreverse constants))
              vars))))

(defun primitive-recipe-eqn-count (recipe)
  "RECIPE から BUILD-PRIMITIVE-GRAPH した graph の eqn 数を返す。emitter の
「loc(\"eqn-N\") の個数 = eqn 数」という性質のオラクルに使う。"
  (length (nb:graph-eqns (build-primitive-graph recipe))))
