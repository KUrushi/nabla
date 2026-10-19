;;;; multi-head attention（issue #208）。Flax の nn.MultiHeadDotProductAttention に合わせる。
;;;;
;;;; Transformer を Flax と数値で比べる（#209）ため、スロット名はチェックポイント（#206）のパス名として
;;;; Flax のパラメータ名と同じ: query / key / value / out の各サブモジュール（dense-general.lisp）が
;;;; kernel / bias を持つ（Flax の flatten_dict のパス query.kernel などと一致する）。決めたこと:
;;;;
;;;; - 形のバッチ次元は「先頭の任意個」（Flax と同じ）。入力は [... 長さ 幅]。理由: 学習のコードは
;;;;   バッチ付きの入力を直接渡すのが普通で、Flax のフィクスチャとそのまま比べられる。attention の
;;;;   中身は dot-general のバッチ軸（先頭のバッチ軸とヘッド）で書くので、先頭の軸が何個でも同じコード
;;;;   になる。vmap で外側に軸を足すこともできるが、それ前提にはしない。
;;;; - 呼び出し: (call model inputs-q inputs-kv mask key deterministic)。引数はすべて必須（defmodule の
;;;;   制約）。自己 attention は inputs-q と inputs-kv に同じものを渡す（Flax の inputs_q だけの呼び出し）。
;;;;   Flax には inputs_k と inputs_v を別に渡す形もあるが、key と value は同じ入力から作る
;;;;   inputs_kv の形だけにした（交差 attention に足りる）。mask は NIL か bit（:i1）の配列で、
;;;;   [... ヘッド 長さq 長さkv] に右から揃えてブロードキャストできる形（Flax の mask と同じ）。真が「見える」。
;;;; - スコア: Flax と同じ順で、query を sqrt(head-dim) で割ってから key と内積する。マスクは隠す位置に
;;;;   その dtype の最小値（Flax の jnp.finfo(dtype).min。-inf ではない。f32 / f64 / bf16 / f16）を
;;;;   `if`（= select）で入れ、softmax（layers.lisp の softmax。最大値を引いてから exp する）で正規化する。
;;;;   最小値なので、全部のキーが隠れた行は一様な重みになる（Flax と同じ。NaN にならない）。
;;;; - dropout は重みに layers.lisp の `%dropout-apply` をそのまま使う（rate・key・deterministic の規則は
;;;;   dropout と同じ）。Flax の既定の broadcast_dropout=True（マスクをバッチ軸とヘッドで共有する）とは違い、
;;;;   重みの要素ごとに独立のマスクを引く（Flax の broadcast_dropout=False と同じ）。PRNG が JAX と
;;;;   ビット単位で一致しないので Flax と値を比べられず、独立なマスクのほうが素直なため。
;;;; - ロジックは `defmodule` の外の `defun` に出す。トレースされる部品は
;;;;   `(funcall (load-time-value (nabla:with-tracing ...)) 引数...)` の形（layers.lisp と同じ）。

(in-package #:nabla.nn)

;;; --- multi-head attention ---

(defun %attention-large-negative (dtype)
  "DTYPE の最小の有限値（Flax の jnp.finfo(dtype).min）。"
  (ecase dtype
    (:f32 (coerce most-negative-single-float 'double-float))
    (:f64 most-negative-double-float)
    ;; bf16 は f32 と同じ指数の幅で仮数が 8 ビット: (2 - 2^-7) 2^127。f16 は 65504。
    (:bf16 -3.3895313892515355d38)
    (:f16 -65504d0)))

(defun %check-attention-input (name x features)
  "X が [... 長さ 幅] で最後の軸が FEATURES であることを確かめる。"
  (let ((shape (%shape x)))
    (unless (and (>= (length shape) 2) (= (car (last shape)) features))
      (error "multi-head-attention: ~A は [... 長さ ~D] の形でなければならない（形 ~S）" name features shape))))

(defun %check-attention-mask (mask logits-shape)
  "MASK が bit（:i1）で、LOGITS-SHAPE に右から揃えてブロードキャストできることを確かめて、
ブロードキャストの dims（LOGITS-SHAPE の軸の番号）を返す。"
  (let* ((aval (handler-case (nabla:value-aval mask)
                 (nabla:tracing-error ()
                   (error "multi-head-attention: マスクは bit（:i1）の配列でなければならない: ~S" mask))))
         (shape (nabla:aval-shape aval))
         (rank (length logits-shape)))
    (unless (eq :i1 (nabla:aval-dtype aval))
      (error "multi-head-attention: マスクは bit（:i1）でなければならない（dtype ~S）" (nabla:aval-dtype aval)))
    (unless (and (<= (length shape) rank)
                 (every (lambda (size full) (or (= size 1) (= size full)))
                        shape (last logits-shape (length shape))))
      (error "multi-head-attention: マスクの形 ~S は [... ヘッド 長さq 長さkv] = ~S にブロードキャストできない"
             shape logits-shape))
    (loop for i from (- rank (length shape)) below rank collect i)))

(defun %attention-logits (q k batch-rank denominator)
  "q [B... Q h d]・k [B... K h d] → [B... h Q K]: (q / DENOMINATOR) と k を d で縮約する
\(B... とヘッドがバッチ軸）。"
  (let ((batch (loop for i below batch-rank collect i))
        (heads (1+ batch-rank))
        (depth (+ batch-rank 2)))
    (funcall (load-time-value
              (nabla:with-tracing (q k denominator batch heads depth)
                (nabla:dot-general (/ q denominator) k
                                   :lhs-contracting (list depth) :rhs-contracting (list depth)
                                   :lhs-batch (append batch (list heads)) :rhs-batch (append batch (list heads)))))
             q k denominator batch heads depth)))

(defun %attention-mask-logits (logits mask large-negative dims)
  "MASK（1 が見える）が 0 の位置の LOGITS を LARGE-NEGATIVE に置き換える。"
  (let ((shape (%shape logits)))
    (funcall (load-time-value
              (nabla:with-tracing (logits mask large-negative shape dims)
                (if (nabla:broadcast-in-dim mask shape dims) logits large-negative)))
             logits mask large-negative shape dims)))

(defun %attention-context (weights v batch-rank)
  "weights [B... h Q K]・v [B... K h d] → [B... Q h d]。"
  (let ((batch (loop for i below batch-rank collect i))
        (heads batch-rank)
        (v-heads (1+ batch-rank))
        (keys (+ batch-rank 2))
        (perm (append (loop for i below batch-rank collect i) (list (1+ batch-rank) batch-rank (+ batch-rank 2)))))
    (funcall (load-time-value
              (nabla:with-tracing (weights v batch heads v-heads keys perm)
                (nabla:transpose
                 (nabla:dot-general weights v
                                    :lhs-contracting (list keys) :rhs-contracting (list heads)
                                    :lhs-batch (append batch (list heads)) :rhs-batch (append batch (list v-heads)))
                 perm)))
             weights v batch heads v-heads keys perm)))

(defun %multi-head-attention-apply (model inputs-q inputs-kv mask key deterministic)
  (let* ((q-projection (multi-head-attention-query model))
         (k-projection (multi-head-attention-key model))
         (q-kernel-shape (%shape (dense-general-kernel q-projection)))
         (k-kernel-shape (%shape (dense-general-kernel k-projection))))
    (%check-attention-input "inputs-q" inputs-q (first q-kernel-shape))
    (%check-attention-input "inputs-kv" inputs-kv (first k-kernel-shape))
    (unless (equal (butlast (%shape inputs-q) 2) (butlast (%shape inputs-kv) 2))
      (error "multi-head-attention: inputs-q と inputs-kv の先頭の軸が違う（~S と ~S）"
             (%shape inputs-q) (%shape inputs-kv)))
    (let* ((q (call q-projection inputs-q))
           (k (call k-projection inputs-kv))
           (v (call (multi-head-attention-value model) inputs-kv))
           (q-shape (%shape q))
           (batch-rank (- (length q-shape) 3))
           (dtype (nabla:aval-dtype (nabla:value-aval q)))
           (logits (%attention-logits q k batch-rank (sqrt (coerce (car (last q-shape)) 'double-float))))
           (masked (if mask
                       (%attention-mask-logits logits mask (%attention-large-negative dtype)
                                               (%check-attention-mask mask (%shape logits)))
                       logits))
           (weights (%dropout-apply (softmax masked) key (multi-head-attention-dropout-rate model) deterministic)))
      (call (multi-head-attention-out model) (%attention-context weights v batch-rank)))))

(defmodule multi-head-attention (query key value out (dropout-rate :static))
  (:init (key num-heads in-features &key qkv-features kv-features out-features (dtype :f32) (dropout-rate 0))
   (let* ((qkv-features (or qkv-features in-features))
          (kv-features (or kv-features in-features))
          (out-features (or out-features in-features)))
     (unless (and (integerp num-heads) (plusp num-heads) (integerp qkv-features) (plusp qkv-features)
                  (zerop (mod qkv-features num-heads)))
       (error "multi-head-attention: qkv-features (~S) は num-heads (~S) で割り切れる正の整数でなければならない"
              qkv-features num-heads))
     (unless (and (realp dropout-rate) (<= 0 dropout-rate 1))
       (error "multi-head-attention: dropout-rate は 0 以上 1 以下の実数でなければならない: ~S" dropout-rate))
     (let ((head-dim (floor qkv-features num-heads))
           (keys (split-keys key 4)))
       (make-multi-head-attention
        :query (init-dense-general (first keys) in-features (list num-heads head-dim) :dtype dtype)
        :key (init-dense-general (second keys) kv-features (list num-heads head-dim) :dtype dtype)
        :value (init-dense-general (third keys) kv-features (list num-heads head-dim) :dtype dtype)
        :out (init-dense-general (fourth keys) (list num-heads head-dim) out-features :dtype dtype)
        :dropout-rate dropout-rate))))
  (:call (self inputs-q inputs-kv mask key deterministic)
   (%multi-head-attention-apply self inputs-q inputs-kv mask key deterministic)))
