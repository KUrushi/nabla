;;;; 基本レイヤー: dense / layer-norm / embedding / dropout / sequential と活性化関数（issue #204）。
;;;;
;;;; 初期化と計算は Flax Linen の既定に合わせる（Transformer を Flax と数値で比べる #209 のため）。
;;;; スロット名もチェックポイント（#206）のパス名に使うので Flax のパラメータ名と同じにする
;;;; （kernel / bias / scale / embedding）。決めたこと:
;;;;
;;;; - dense: kernel (in out)、bias (out)。kernel は lecun_normal = fan_in の分散 1/in の切断正規分布
;;;;   （[-2, 2] で切った標準正規分布に 1/(0.87962566 sqrt(in)) をかける。Flax の
;;;;   variance_scaling(1, "fan_in", "truncated_normal")）。bias は 0。y = x kernel + bias で、x は
;;;;   rank 1 以上（最後の軸が in）。use_bias の引数は持たない（必要になったら足す）。
;;;; - layer-norm: 最後の軸で正規化。Flax と同じ「速い分散」 var = max(mean(x^2) - mean(x)^2, 0)、
;;;;   epsilon の既定 1e-6（f64 は 1d-6）、y = (x - mean) * rsqrt(var + epsilon) * scale + bias。
;;;;   scale は 1、bias は 0。
;;;; - embedding: one-hot と dot で書く。gather プリミティブは足さない。理由: gather の逆（勾配）は
;;;;   scatter-add で、プリミティブ・StableHLO 出力・jvp / transpose / バッチ化ルールの5つが要り、この
;;;;   issue の範囲を超える。one-hot + dot なら既存の演算だけで、勾配は dot の転置（one-hot^T ct）
;;;;   として自然に出る。代償は、one-hot が [ids..., 語彙数] の配列になり、計算量とメモリが ids 数 ×
;;;;   語彙数 × features に比例すること（gather は ids 数 × features）。小さな語彙（文字レベル・数千語）
;;;;   では問題にならないが、語彙が大きいモデルでは gather が要る（後続の issue）。範囲外の id は one-hot が
;;;;   全部 0 なので 0 の行になる（Flax の jnp.take は範囲外で NaN を返す）。初期値は
;;;;   N(0, 1/features)（Flax の variance_scaling(1, "fan_in", "normal", out_axis=0)。Flax の
;;;;   サンプルの標準偏差が 1/sqrt(features) になることを tests/fixtures/layers で確かめてある）。
;;;; - dropout: 反転 dropout。keep = 1 - rate、マスクは uniform(key) < keep、保持した要素を 1/keep 倍
;;;;   する（Flax の nn.Dropout と同じ式）。キーと deterministic は呼び出しの引数（モジュールは状態を
;;;;   持たない）。deterministic は Lisp の真偽値（jit では静的引数にする）。
;;;; - sequential: 1引数のモジュールのリストを順に適用する。キーや deterministic を取る dropout は
;;;;   入れられない（Flax の nn.Sequential と同じ制限）。`activation`（:static の関数名）が
;;;;   relu などを層として並べるためのもの。
;;;; - ロジックは `defmodule` の外の `defun` に出す（`:init` / `:call` は薄い）。mutation testing の runner が
;;;;   変異させられるのは defun などの定義だけで、defmodule の中身は変異させられないため。トレースされる
;;;;   部品は `(funcall (load-time-value (nabla:with-tracing ...)) 引数...)` の形で書く（CL の + * max exp
;;;;   などが配列にもトレーサにも使える。本体は loop / setq 不可）。
;;;; - 活性化関数は nabla.nn の普通の関数（relu / gelu / softmax / log-softmax）。コアのプリミティブに
;;;;   せず、max / exp / tanh / reduce の合成で書く（配列にもトレーサにも使え、jvp / transpose は
;;;;   合成の側が持つ）。gelu は Flax の既定の tanh 近似（approximate=True）。

(in-package #:nabla.nn)

;;; --- 部品 ---

(defun %element-type (dtype)
  (unless (member dtype '(:f32 :f64))
    (error "dtype は :f32 か :f64 でなければならない: ~S" dtype))
  (nabla:dtype-element-type dtype))

(defun %filled (shape dtype value)
  (let ((type (%element-type dtype)))
    (make-array shape :element-type type :initial-element (coerce value type))))

(defun %scale-by (x factor)
  "X の各要素に FACTOR（実数）をかける。init で乱数の配列を拡大縮小する。"
  (funcall (load-time-value (nabla:with-tracing (x factor) (* x factor))) x factor))

(defun %range (n)
  "(0 1 ... N-1)。トレースされる本体は loop を使えないので、部品にしてある。"
  (loop for i below n collect i))

(defun %shape (x)
  (nabla:aval-shape (nabla:value-aval x)))

(defun %expand (value shape axis)
  "AXIS を reduce した VALUE を、元の形 SHAPE に（AXIS 以外の軸に対応づけて）ブロードキャストする。"
  (nabla:broadcast-in-dim value shape (loop for i below (length shape) unless (= i axis) collect i)))

(defun %broadcast-last (vector shape)
  "最後の軸の長さの VECTOR を、形 SHAPE（最後の軸が VECTOR の長さ）にブロードキャストする。"
  (nabla:broadcast-in-dim vector shape (list (1- (length shape)))))

(defun %check-last-axis (name x vector)
  "X が rank 1 以上で、最後の軸の長さが VECTOR の長さに等しいことを確かめて X の形を返す。"
  (let ((shape (%shape x)))
    (unless (and shape (= (car (last shape)) (first (%shape vector))))
      (error "~A: 入力の最後の軸の長さが ~D でなければならない（入力の形 ~S）" name (first (%shape vector)) shape))
    shape))

(defun %resolve-axis (axis x)
  "AXIS（負の値は後ろから数える）を 0 以上 rank 未満の整数にする。範囲外は error。"
  (let ((rank (length (%shape x))))
    (unless (and (integerp axis) (<= (- rank) axis) (< axis rank))
      (error "axis は -~D 以上 ~D 未満の整数でなければならない: ~S" rank rank axis))
    (if (minusp axis) (+ axis rank) axis)))

;;; --- 活性化関数 ---

(defun relu (x)
  "max(X, 0)（要素ごと）。X は配列かトレーサ。"
  (funcall (load-time-value (nabla:with-tracing (x) (max x 0))) x))

(defun gelu (x)
  "GELU の tanh 近似（Flax の nn.gelu の既定、approximate=True）: X * 0.5 * (1 + tanh(sqrt(2/pi)
\(X + 0.044715 X^3)))（要素ごと）。X は配列かトレーサ。"
  (funcall (load-time-value
            (nabla:with-tracing (x)
              (* x (* 0.5d0 (+ 1 (tanh (* 0.7978845608028654d0 (+ x (* 0.044715d0 (* x (* x x)))))))))))
           x))

(defun %softmax-shifted (x axis)
  "X から AXIS ごとの最大値を引く（最大値には勾配を流さない）。"
  (funcall (load-time-value
            (nabla:with-tracing (x axis)
              (- x (%expand (nabla:stop-gradient (nabla:reduce-max x :axes (list axis))) (%shape x) axis))))
           x axis))

(defun %softmax (x axis)
  "AXIS に沿った softmax（AXIS は 0 以上）。"
  (funcall (load-time-value
            (nabla:with-tracing (x axis)
              (let ((e (exp (%softmax-shifted x axis))))
                (/ e (%expand (nabla:reduce-sum e :axes (list axis)) (%shape x) axis)))))
           x axis))

(defun %log-softmax (x axis)
  "AXIS に沿った log-softmax（AXIS は 0 以上）。"
  (funcall (load-time-value
            (nabla:with-tracing (x axis)
              (let ((shifted (%softmax-shifted x axis)))
                (- shifted (%expand (log (nabla:reduce-sum (exp shifted) :axes (list axis))) (%shape x) axis)))))
           x axis))

(defun softmax (x &key (axis -1))
  "X の AXIS（既定は最後の軸。負の値は後ろから数える）に沿った softmax。最大値を引いてから exp
するので大きな値でも溢れない（引く最大値には勾配を流さない。softmax はずらしに不変なので結果は同じ）。
X は配列かトレーサ。AXIS が範囲外なら error。"
  (%softmax x (%resolve-axis axis x)))

(defun log-softmax (x &key (axis -1))
  "X の AXIS に沿った log(softmax(X))（既定は最後の軸）。softmax を取ってから log するより数値が
安定する。X は配列かトレーサ。AXIS が範囲外なら error。"
  (%log-softmax x (%resolve-axis axis x)))

(defparameter *activations* '(:relu :gelu :tanh :softmax)
  "ACTIVATION の FN に使える名前。")

(defun %activation-apply (fn x)
  (ecase fn
    (:relu (relu x))
    (:gelu (gelu x))
    (:tanh (funcall (load-time-value (nabla:with-tracing (x) (tanh x))) x))
    (:softmax (softmax x))))

(defmodule activation ((fn :static))
  (:init (key fn)
   (declare (ignore key))
   (unless (member fn *activations*)
     (error "活性化関数は ~S のどれか: ~S" *activations* fn))
   (make-activation :fn fn))
  (:call (self x)
   (%activation-apply (activation-fn self) x)))

;;; --- dense ---

(defparameter *truncated-normal-stddev* 0.87962566103423978d0
  "[-2, 2] に切った標準正規分布の標準偏差（Flax の lecun_normal が割る定数）。")

(defun %lecun-normal-of-shape (key shape fan-in dtype)
  "Flax の lecun_normal: SHAPE の切断正規分布に 1/(0.8796 sqrt(FAN-IN)) をかける。"
  (%scale-by (nabla:truncated-normal key shape :dtype dtype)
             (/ (sqrt (/ 1d0 fan-in)) *truncated-normal-stddev*)))

(defun %lecun-normal (key in-features out-features dtype)
  (%lecun-normal-of-shape key (list in-features out-features) in-features dtype))

(defun %dense-apply (x kernel bias)
  (%check-last-axis "dense" x kernel)
  (funcall (load-time-value
            (nabla:with-tracing (x kernel bias)
              (let ((y (nabla:dot x kernel)))
                (+ y (%broadcast-last bias (%shape y))))))
           x kernel bias))

(defmodule dense (kernel bias)
  (:init (key in-features out-features &key (dtype :f32))
   (make-dense :kernel (%lecun-normal key in-features out-features dtype)
               :bias (%filled (list out-features) dtype 0)))
  (:call (self x)
   (%dense-apply x (dense-kernel self) (dense-bias self))))

;;; --- layer-norm ---

(defun %layer-norm-apply (x scale bias epsilon)
  (let ((shape (%check-last-axis "layer-norm" x scale)))
    (funcall (load-time-value
              (nabla:with-tracing (x scale bias epsilon shape)
                (let* ((axis (1- (length shape)))
                       (n (nth axis shape))
                       (mean (/ (nabla:reduce-sum x :axes (list axis)) n))
                       (mean-of-squares (/ (nabla:reduce-sum (* x x) :axes (list axis)) n))
                       (variance (max (- mean-of-squares (* mean mean)) 0))
                       (multiplier (* (%expand (nabla:rsqrt (+ variance epsilon)) shape axis)
                                      (%broadcast-last scale shape))))
                  (+ (* (- x (%expand mean shape axis)) multiplier)
                     (%broadcast-last bias shape)))))
             x scale bias epsilon shape)))

(defmodule layer-norm (scale bias (epsilon :static))
  (:init (key features &key (dtype :f32) (epsilon (if (eq dtype :f64) 1d-6 1f-6)))
   (declare (ignore key))
   (unless (and (realp epsilon) (plusp epsilon))
     (error "layer-norm の epsilon は正の実数でなければならない: ~S" epsilon))
   (make-layer-norm :scale (%filled (list features) dtype 1) :bias (%filled (list features) dtype 0)
                    :epsilon epsilon))
  (:call (self x)
   (%layer-norm-apply x (layer-norm-scale self) (layer-norm-bias self) (layer-norm-epsilon self))))

;;; --- embedding ---

(defun %check-integer-dtype (name x)
  "X の dtype が整数（:i32 / :u32 / :u64）であることを確かめてその dtype を返す。"
  (let ((dtype (nabla:aval-dtype (nabla:value-aval x))))
    (unless (member dtype '(:i32 :u32 :u64))
      (error "~A: ids は整数の配列でなければならない（dtype ~S）" name dtype))
    dtype))

(defun %iota (n dtype)
  "0 から N-1 の1次元配列（DTYPE は整数）。"
  (let ((array (make-array n :element-type (nabla:dtype-element-type dtype))))
    (dotimes (i n array) (setf (aref array i) i))))

(defun %embedding-apply (ids table)
  (let* ((ids-dtype (%check-integer-dtype "embedding" ids))
         (vocabulary (first (%shape table)))
         (full-shape (append (%shape ids) (list vocabulary)))
         (iota (%iota vocabulary ids-dtype)))
    (funcall (load-time-value
              (nabla:with-tracing (ids table full-shape iota)
                (let* ((ids-rank (1- (length full-shape)))
                       (hits (= (nabla:broadcast-in-dim ids full-shape (%range ids-rank))
                                (nabla:broadcast-in-dim iota full-shape (list ids-rank)))))
                  (nabla:dot (nabla:convert hits (nabla:aval-dtype (nabla:value-aval table))) table))))
             ids table full-shape iota)))

(defmodule embedding (embedding)
  (:init (key num-embeddings features &key (dtype :f32))
   (make-embedding :embedding (%scale-by (nabla:normal key (list num-embeddings features) :dtype dtype)
                                         (/ 1d0 (sqrt features)))))
  (:call (self ids)
   (%embedding-apply ids (embedding-embedding self))))

;;; --- dropout ---

(defun %check-key (key)
  "dropout に deterministic = nil で呼んだとき、キーが渡されていることを確かめて返す。"
  (unless key
    (error "dropout: deterministic が偽のときは PRNG キーが要る（NIL が渡された）"))
  key)

(defun %dropout-apply (x key rate deterministic)
  (cond ((or deterministic (zerop rate)) x)
        ;; keep = 0 の 1/keep は転置で 0/0 になるので別扱い。x < x は常に偽（inf / NaN でも）で、
        ;; 選ばれるのは 0 だけ（勾配も 0）
        ((= rate 1)
         (funcall (load-time-value (nabla:with-tracing (x) (if (< x x) x 0))) x))
        (t (funcall (load-time-value
                     (nabla:with-tracing (x u rate)
                       (let ((keep (- 1 rate)))
                         (if (< u keep) (/ x keep) 0))))
                    x
                    ;; 乱数は常に f32 で引く（Flax と同じ。x の dtype（bf16 / f16 / f64）によらず同じマスクになる）
                    (nabla:uniform (%check-key key) (%shape x) :dtype :f32)
                    rate))))

(defmodule dropout ((rate :static))
  (:init (key rate)
   (declare (ignore key))
   (unless (and (realp rate) (<= 0 rate 1))
     (error "dropout の rate は 0 以上 1 以下の実数でなければならない: ~S" rate))
   (make-dropout :rate rate))
  (:call (self x key deterministic)
   (%dropout-apply x key (dropout-rate self) deterministic)))

;;; --- sequential ---

(defun %sequential-apply (layers x)
  (reduce (lambda (h layer) (call layer h)) layers :initial-value x))

(defmodule sequential (layers)
  (:init (key init-functions)
   (make-sequential :layers (when init-functions
                              (mapcar #'funcall init-functions (split-keys key (length init-functions))))))
  (:call (self x)
   (%sequential-apply (sequential-layers self) x)))
