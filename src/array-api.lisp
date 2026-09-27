;;;; array-api: 配列レベルの公開 API（issue #32、t2）。
;;;;
;;;; DOT / RESHAPE / TRANSPOSE / BROADCAST-IN-DIM / REDUCE-SUM / REDUCE-MAX /
;;;; CONVERT / WHERE の8個の総称関数。ARRAY メソッドは対応するプリミティブの
;;;; :EAGER を直接呼び、TRACER メソッドは %TRACE-EQN で EQN を1つ足す
;;;; （trace-ops.lisp の %T-* と同じ形）。パラメタ（デフォルトの perm・axes・
;;;; dot の contracting 次元）を計算する判断は、すべて一行のメソッドから
;;;; 小さい DEFUN に切り出す（mutation testing の的にするため）。
;;;;
;;;; スカラー・rank 0 のブロードキャストはここには無い（このファイルの
;;;; API はすべて配列どうしの演算で、暗黙の型変換をしない。数値との
;;;; ブロードキャストが要るのは算術演算子（+ など）だけで、それは
;;;; trace-ops.lisp の役目）。

(in-package #:nabla)

;;; --- dot: 最後の軸（A）と最初の軸（B）を縮約する、バッチ無しの
;;; dot-general。 ---

(defun %dot-check-rank (rank)
  "RANK が0（スカラー）なら TRACING-ERROR を signal する。DOT は rank 1
以上の配列にしか意味を持たない（縮約する軸が無い）。"
  (when (zerop rank)
    (error 'tracing-error
           :format-control "DOT は rank 0 の値を受け付けない（縮約する軸が無い）"
           :format-arguments nil)))

(defun %dot-params (rank-a)
  "A の RANK-A から dot-general のパラメタ（最後の軸 vs 最初の軸、バッチ
無し）を計算する。"
  (list :lhs-contracting (list (1- rank-a)) :rhs-contracting '(0)
        :lhs-batch '() :rhs-batch '()))

(defgeneric dot (a b)
  (:documentation
   "A の最後の軸と B の最初の軸を縮約する（バッチ次元は無い）
dot-general。A・B は rank 1 以上でなければならない（rank 0 は
TRACING-ERROR）。A が rank 1（ベクタ）なら結果は B の残りの軸だけになる
（1次元どうしなら rank 0 のスカラー）。"))

(defmethod dot ((a array) (b array))
  (%dot-check-rank (array-rank a))
  (apply (primitive-eager (find-primitive :dot-general)) (list a b)
         (list (array-aval a) (array-aval b)) (%dot-params (array-rank a))))

(defmethod dot ((a tracer) (b tracer))
  (%dot-check-rank (aval-rank (tracer-aval a)))
  (apply #'%trace-eqn :dot-general (list a b) (%dot-params (aval-rank (tracer-aval a)))))

;;; --- reshape ---

(defgeneric reshape (x shape)
  (:documentation "X を SHAPE（要素数が同じ非負整数のリスト）に reshape する。"))

(defmethod reshape ((x array) shape)
  (apply (primitive-eager (find-primitive :reshape)) (list x) (list (array-aval x)) (list :shape shape)))

(defmethod reshape ((x tracer) shape)
  (%trace-eqn :reshape (list x) :shape shape))

;;; --- transpose ---

(defun %transpose-default-perm (rank)
  "RANK 個の軸を逆順にした perm（デフォルトの転置）を返す。"
  (reverse (loop for i below rank collect i)))

(defgeneric transpose (x &optional perm)
  (:documentation
   "X の軸を PERM（省略時は軸を逆順にする）で並べ替える。PERM は X の rank
と同じ長さの permutation でなければならない。"))

(defmethod transpose ((x array) &optional perm)
  (let ((perm (or perm (%transpose-default-perm (array-rank x)))))
    (apply (primitive-eager (find-primitive :transpose)) (list x) (list (array-aval x)) (list :perm perm))))

(defmethod transpose ((x tracer) &optional perm)
  (let ((perm (or perm (%transpose-default-perm (aval-rank (tracer-aval x))))))
    (%trace-eqn :transpose (list x) :perm perm)))

;;; --- broadcast-in-dim ---

(defgeneric broadcast-in-dim (x shape dims)
  (:documentation
   "X（各次元が1かSHAPEの対応する次元と等しい）を SHAPE まで広げる。DIMS
は X の各軸が SHAPE のどの軸に対応するかを表す（X の rank と同じ長さ）。"))

(defmethod broadcast-in-dim ((x array) shape dims)
  (apply (primitive-eager (find-primitive :broadcast-in-dim)) (list x) (list (array-aval x))
         (list :shape shape :dims dims)))

(defmethod broadcast-in-dim ((x tracer) shape dims)
  (%trace-eqn :broadcast-in-dim (list x) :shape shape :dims dims))

;;; --- reduce-sum / reduce-max ---

(defun %reduce-default-axes (rank)
  "RANK 個の軸すべて（0..RANK-1）を返す（AXES省略時のデフォルト、全軸を
潰す）。"
  (loop for i below rank collect i))

(defgeneric reduce-sum (x &key axes)
  (:documentation
   "X を AXES（省略時は全軸）に沿って総和で潰す。全軸を潰すと rank 0 に
なる。"))

(defmethod reduce-sum ((x array) &key axes)
  (let ((axes (or axes (%reduce-default-axes (array-rank x)))))
    (apply (primitive-eager (find-primitive :reduce-sum)) (list x) (list (array-aval x)) (list :axes axes))))

(defmethod reduce-sum ((x tracer) &key axes)
  (let ((axes (or axes (%reduce-default-axes (aval-rank (tracer-aval x))))))
    (%trace-eqn :reduce-sum (list x) :axes axes)))

(defgeneric reduce-max (x &key axes)
  (:documentation
   "X を AXES（省略時は全軸）に沿って最大値で潰す。全軸を潰すと rank 0 に
なる。"))

(defmethod reduce-max ((x array) &key axes)
  (let ((axes (or axes (%reduce-default-axes (array-rank x)))))
    (apply (primitive-eager (find-primitive :reduce-max)) (list x) (list (array-aval x)) (list :axes axes))))

(defmethod reduce-max ((x tracer) &key axes)
  (let ((axes (or axes (%reduce-default-axes (aval-rank (tracer-aval x))))))
    (%trace-eqn :reduce-max (list x) :axes axes)))

;;; --- convert ---

(defgeneric convert (x dtype)
  (:documentation "X の要素を DTYPE（浮動小数点の dtype）に変換する。"))

(defmethod convert ((x array) dtype)
  (apply (primitive-eager (find-primitive :convert)) (list x) (list (array-aval x)) (list :dtype dtype)))

(defmethod convert ((x tracer) dtype)
  (%trace-eqn :convert (list x) :dtype dtype))

;;; --- where ---

(defgeneric where (pred a b)
  (:documentation
   "PRED（:I1 の値）が真の要素は A、偽の要素は B を選ぶ（%T-SELECT と同じ。
trace-ops.lisp の SELECT/WHERE の分岐に関するドキュメントを参照）。"))

(defmethod where ((pred array) a b)
  (%t-select pred a b))

(defmethod where ((pred tracer) a b)
  (%t-select pred a b))
