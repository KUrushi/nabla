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

(defun %dot-check-ranks (rank-a rank-b)
  "A・B 両方の rank を %DOT-CHECK-RANK で確かめる（A だけでなく B が
rank 0 の場合も TRACING-ERROR にする）。"
  (%dot-check-rank rank-a)
  (%dot-check-rank rank-b))

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
（1次元どうしなら rank 0 のスカラー）。A・B の片方が配列、もう片方が
トレーサなら、配列をトレーサと同じ dtype の定数としてリフトしてから
トレースする（+ 等の算術演算子と同じ規約。dtype が食い違えば
DTYPE-MISMATCH）。"))

(defmethod dot ((a array) (b array))
  (%dot-check-ranks (array-rank a) (array-rank b))
  (apply (primitive-eager (find-primitive :dot-general)) (list a b)
         (list (array-aval a) (array-aval b)) (%dot-params (array-rank a))))

(defmethod dot ((a tracer) (b tracer))
  (%dot-check-ranks (aval-rank (tracer-aval a)) (aval-rank (tracer-aval b)))
  (apply #'%trace-eqn :dot-general (list a b) (%dot-params (aval-rank (tracer-aval a)))))

(defmethod dot ((a array) (b tracer))
  (dot (%lift-array a b) b))

(defmethod dot ((a tracer) (b array))
  (dot a (%lift-array b a)))

;;; --- dot-general: バッチ次元と縮約次元を明示する（issue #208）---

(defgeneric dot-general (a b &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  (:documentation
   "StableHLO の dot_general。LHS-CONTRACTING / RHS-CONTRACTING は縮約する軸（A と B で
同じ個数、同じ長さ）、LHS-BATCH / RHS-BATCH はバッチ軸（既定は空。先頭でなくてもよい）。結果の軸は
バッチ軸（LHS-BATCH の順）、A の残りの軸、B の残りの軸の順。attention の
`(dot-general q k :lhs-batch '(0 2) :rhs-batch '(0 2) :lhs-contracting '(3) :rhs-contracting '(3))`
のように使う。軸の検査は dot-general プリミティブが行い、食い違えば PRIMITIVE-ERROR。
A・B の片方が配列、もう片方がトレーサなら、DOT と同じく配列をトレーサと同じ dtype の定数に
リフトする。"))

(defmethod dot-general ((a array) (b array) &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  (let ((avals (list (array-aval a) (array-aval b)))
        (params (list :lhs-contracting lhs-contracting :rhs-contracting rhs-contracting
                      :lhs-batch lhs-batch :rhs-batch rhs-batch)))
    ;; eager は検査しないので、先に形状推論で不正なパラメタを PRIMITIVE-ERROR にする。
    (apply (primitive-abstract-eval (find-primitive :dot-general)) avals params)
    (apply (primitive-eager (find-primitive :dot-general)) (list a b) avals params)))

(defmethod dot-general ((a tracer) (b tracer) &key lhs-contracting rhs-contracting lhs-batch rhs-batch)
  (%trace-eqn :dot-general (list a b) :lhs-contracting lhs-contracting :rhs-contracting rhs-contracting
                                      :lhs-batch lhs-batch :rhs-batch rhs-batch))

(defmethod dot-general ((a array) (b tracer) &rest params &key &allow-other-keys)
  (apply #'dot-general (%lift-array a b) b params))

(defmethod dot-general ((a tracer) (b array) &rest params &key &allow-other-keys)
  (apply #'dot-general a (%lift-array b a) params))

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

(defun %reduce-resolve-axes (axes-supplied-p axes rank)
  "AXES-SUPPLIED-P・AXES・RANK から実際に潰す軸のリストを決める。AXES が
明示的に省略された（AXES-SUPPLIED-P が偽）ときだけ全軸をデフォルトにする。
AXES が明示的に空リストで渡されたとき（AXES-SUPPLIED-P が真）は NIL を
返し、呼び出し側が「reduce しない（X をそのまま返す）」と解釈する。
src/primitives/reduce.lisp が空の :AXES を PRIMITIVE-ERROR にする契約上の
逸脱を、このレイヤーで吸収する（プリミティブより上のレイヤーが axes を
空に正規化して、そもそも eqn を作らない）。"
  (cond
    ((not axes-supplied-p) (%reduce-default-axes rank))
    ((null axes) nil)
    (t axes)))

(defgeneric reduce-sum (x &key axes)
  (:documentation
   "X を AXES（省略時は全軸）に沿って総和で潰す。全軸を潰すと rank 0 に
なる。AXES を明示的に空リストで渡すと reduce しない（X をそのまま返す。
eqn も足さない）。"))

(defmethod reduce-sum ((x array) &key (axes nil axes-p))
  (let ((axes (%reduce-resolve-axes axes-p axes (array-rank x))))
    (if axes
        (apply (primitive-eager (find-primitive :reduce-sum)) (list x) (list (array-aval x)) (list :axes axes))
        x)))

(defmethod reduce-sum ((x tracer) &key (axes nil axes-p))
  (let ((axes (%reduce-resolve-axes axes-p axes (aval-rank (tracer-aval x)))))
    (if axes
        (%trace-eqn :reduce-sum (list x) :axes axes)
        x)))

(defgeneric reduce-max (x &key axes)
  (:documentation
   "X を AXES（省略時は全軸）に沿って最大値で潰す。全軸を潰すと rank 0 に
なる。AXES を明示的に空リストで渡すと reduce しない（X をそのまま返す。
eqn も足さない）。"))

(defmethod reduce-max ((x array) &key (axes nil axes-p))
  (let ((axes (%reduce-resolve-axes axes-p axes (array-rank x))))
    (if axes
        (apply (primitive-eager (find-primitive :reduce-max)) (list x) (list (array-aval x)) (list :axes axes))
        x)))

(defmethod reduce-max ((x tracer) &key (axes nil axes-p))
  (let ((axes (%reduce-resolve-axes axes-p axes (aval-rank (tracer-aval x)))))
    (if axes
        (%trace-eqn :reduce-max (list x) :axes axes)
        x)))

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
trace-ops.lisp の SELECT/WHERE の分岐に関するドキュメントを参照）。
PRED が rank 0 で A・B が rank 1 以上なら、PRED を A・B の shape に
ブロードキャストする。PRED が Lisp のブール値（T／NIL）なら、ふつうの IF
と同じく片方の分岐を静的に選び、SELECT をトレースしない（結果の shape・
dtype は同じ真偽の rank 0 の PRED を渡したときと同じ）。PRED がそれ以外
（数値など）なら TRACING-ERROR。"))

(defmethod where (pred a b)
  (%t-select pred a b))

;;; --- rsqrt（issue #196） ---

(defun rsqrt (x)
  "X の平方根の逆数 1/sqrt(X)（要素ごと）。X は浮動小数点の配列・トレーサ・
数値。配列には eager 実装、トレーサには rsqrt の eqn、数値には CL の
(/ 1 (sqrt x)) を使う。配列とトレーサでは、負の要素は NaN、0 は +無限大
になる。数値は CL の演算なので負の値や 0 では複素数・除算エラーになる。
WITH-TRACING は CL の sqrt と同じく、この関数の呼び出しも書き換える。"
  (%t-rsqrt x))

;;; --- expm1 / log1p（issue #223） ---

(defun expm1 (x)
  "exp(X) - 1（要素ごと）。X が小さくても桁落ちしない。X は浮動小数点の配列・トレーサ・
数値。配列には eager 実装、トレーサには expm1 の eqn、数値には（CL に expm1 が無いので）
eager 実装と同じ計算を使う。数値の結果は X が double-float なら double-float、それ以外は
single-float。WITH-TRACING は CL の exp と同じく、この関数の呼び出しも書き換える。"
  (%t-expm1 x))

(defun log1p (x)
  "log(1 + X)（要素ごと）。X が小さくても桁落ちしない。X は浮動小数点の配列・トレーサ・
数値。配列とトレーサでは、X < -1 は NaN、X = -1 は -無限大になる（数値も同じ）。
配列には eager 実装、トレーサには log1p の eqn を使う。WITH-TRACING は CL の log と同じく、
この関数の呼び出しも書き換える。"
  (%t-log1p x))

;;; --- stop-gradient（issue #80） ---

(defgeneric stop-gradient (x)
  (:documentation "X と同じ値を返すが、自動微分（jvp / grad）では定数として
扱われる（接線がゼロになる）。JAX の lax.stop_gradient と同じ。X は任意の
dtype の配列かトレーサ。配列には中身が等しい新しい配列を返す。"))

(defmethod stop-gradient ((x array))
  (%trace-op-array :stop-gradient x))

(defmethod stop-gradient ((x tracer))
  (%trace-eqn :stop-gradient (list x)))

;;; --- conv-general-dilated（issue #197。プリミティブは src/primitives/conv.lisp）---

(defgeneric conv-general-dilated (lhs rhs &key strides padding lhs-dilation rhs-dilation feature-group-count)
  (:documentation
   "LHS（[N, 空間..., C]）と RHS（[空間..., C/G, O]）の畳み込み（Flax と同じ channels-last の
NHWC / HWIO。JAX の conv_general_dilated 相当）。空間次元の数 S は rank - 2（1 以上）。
STRIDES・LHS-DILATION・RHS-DILATION は長さ S の正の整数のリスト（既定はすべて 1）、
PADDING は長さ S の (low high) のリスト（明示的な値。既定はすべて (0 0)）、
FEATURE-GROUP-COUNT は G（既定 1）。不正な値は PRIMITIVE-ERROR。float dtype のみ。
LHS・RHS の片方が配列でもう片方がトレーサなら、配列をトレーサと同じ dtype の定数としてリフトする。"))

(defun %conv-params (lhs-rank strides padding lhs-dilation rhs-dilation groups)
  "省略されたパラメタを既定値（stride・dilation は 1、padding は (0 0)）で埋めた plist。"
  ;; rank が 3 未満だと SPATIAL は負になり、既定値のリストは空になる。abstract-eval が拒否する。
  (let ((spatial (- lhs-rank 2)))
    (flet ((ones (value) (or value (loop repeat spatial collect 1))))
      (list :strides (ones strides)
            :padding (or padding (loop repeat spatial collect (list 0 0)))
            :lhs-dilation (ones lhs-dilation) :rhs-dilation (ones rhs-dilation)
            :feature-group-count groups
            ;; 公開 API には出さない（issue #203）。rhs の転置が G > 1 のときに現れる内部のパラメタで、
            ;; make-eqn がキーの完全一致を求めるので、ここで既定値を明示する。
            :batch-group-count 1))))

(defmethod conv-general-dilated ((lhs array) (rhs array) &key strides padding lhs-dilation rhs-dilation
                                                            (feature-group-count 1))
  (let ((avals (list (array-aval lhs) (array-aval rhs)))
        (params (%conv-params (array-rank lhs) strides padding lhs-dilation rhs-dilation
                              feature-group-count)))
    ;; eager は検査しないので、先に形状推論で不正なパラメタを PRIMITIVE-ERROR にする。
    (apply (primitive-abstract-eval (find-primitive :convolution)) avals params)
    (apply (primitive-eager (find-primitive :convolution)) (list lhs rhs) avals params)))

(defmethod conv-general-dilated ((lhs tracer) (rhs tracer) &key strides padding lhs-dilation rhs-dilation
                                                              (feature-group-count 1))
  (apply #'%trace-eqn :convolution (list lhs rhs)
         (%conv-params (aval-rank (tracer-aval lhs)) strides padding lhs-dilation rhs-dilation
                       feature-group-count)))

(defmethod conv-general-dilated ((lhs array) (rhs tracer) &rest keys)
  (apply #'conv-general-dilated (%lift-array lhs rhs) rhs keys))

(defmethod conv-general-dilated ((lhs tracer) (rhs array) &rest keys)
  (apply #'conv-general-dilated lhs (%lift-array rhs lhs) keys))

;;; --- gather: 先頭の軸を整数の添字の配列で引く（issue #240）---

(defgeneric gather (operand indices)
  (:documentation
   "OPERAND [N, rest...] の先頭の軸を整数（:i32 / :u32 / :u64）の添字の配列 INDICES [idx...] で引き、
[idx..., rest...] を返す（JAX の jnp.take(operand, indices, axis=0)）。範囲外の添字（負・N 以上）の行は
0 になる（JAX は NaN）。勾配は、範囲内の添字の行に余接線を足し込んだもの（範囲外の添字は寄与しない）。
OPERAND は :i1 以外の dtype で、先頭の軸の長さは 1 以上 2^31 未満。スカラーの添字は rank 0 の配列にする（Lisp の
整数は TRACING-ERROR）。eager の bf16 / f16 の配列（(UNSIGNED-BYTE 16)）は dtype を区別できないので、トレース中に
使う。片方が配列、もう片方がトレーサなら、配列を自分の dtype の定数にリフトする
（OPERAND と INDICES は dtype が違うので、DOT と違いトレーサの dtype には合わせない）。"))

(defmethod gather ((operand array) (indices array))
  (apply (primitive-eager (find-primitive :gather)) (list operand indices)
         (list (list (array-aval operand) (array-aval indices)))))

(defmethod gather ((operand tracer) (indices tracer))
  (%trace-eqn :gather (list operand indices)))

(defmethod gather ((operand array) (indices tracer))
  (gather (%lift-array-to operand (aval-dtype (array-aval operand))) indices))

(defmethod gather ((operand tracer) (indices array))
  (gather operand (%lift-array-to indices (aval-dtype (array-aval indices)))))

(defmethod gather (operand indices)
  (error 'tracing-error
         :format-control "GATHER の OPERAND と INDICES は配列かトレーサでなければならない（スカラーの添字は rank 0 の配列にする）: ~S, ~S"
         :format-arguments (list operand indices)))
