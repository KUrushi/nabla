(require :asdf)
(asdf:load-system "nabla/nn")
(asdf:load-system "nabla/iree")
(defpackage #:nabla-example-transformer
  (:use #:cl #:nabla.nn))
(in-package #:nabla-example-transformer)

;;;; decoder-only の小さな Transformer を defmodule で書いて、系列のコピーを学習する（issue #209、
;;;; フェーズ4の完了条件）。Flax の次の書き方と、パラメータ名（チェックポイントのパス名）まで揃えてある
;;;; （tests/fixtures/transformer/generate.py が同じモデルを Flax で書く）:
;;;;
;;;;   x = embed(tokens) + pos                              ; pos は (長さ 幅) の学習する位置埋め込み
;;;;   各ブロック（pre-LN）: x = x + attn(ln1(x), 因果マスク); x = x + down(gelu(up(ln2(x))))
;;;;   logits = head(norm(x))                               ; 出力の dense は埋め込みと共有しない
;;;;
;;;; 共有しない理由: 共有（tied）にすると、Flax 側は nn.Embed.attend で書き、パラメータの木にも
;;;; 重みが1つしか現れず、パス名の対応が崩れる。素直な別の dense にして、木の形を単純に保った。
;;;; パス名: embed.embedding, pos, blocks.<i>.{ln1,ln2}.{scale,bias},
;;;; blocks.<i>.attn.{query,key,value,out}.{kernel,bias}, blocks.<i>.{up,down}.{kernel,bias},
;;;; norm.{scale,bias}, head.{kernel,bias}（Flax では blocks_<i>。generate.py が "." に直す）。
;;;;
;;;; 課題: 長さ N のランダムな系列 x1..xN の後ろに区切り SEP を置き、同じ系列をもう一度出す
;;;; （x1..xN SEP x1..xN）。入力は先頭の 2N 個、正解は次のトークンで、損失は SEP 以降の N 位置
;;;; （x1..xN を当てる位置）の平均。コピーは位置から一定の距離だけ戻った場所を引く attention で
;;;; 解けるので、小さなモデルですぐに損失が下がる。

(defparameter *vocab* 12 "語彙の大きさ。0..VOCAB-2 がデータのトークン、VOCAB-1 が区切り。")
(defparameter *copy-length* 4 "コピーする系列の長さ N。入力の長さは 2N。")

;;; --- モデル ---

(defun %scaled-normal (key shape stddev)
  "平均 0・標準偏差 STDDEV の f32 の正規乱数の配列（eager）。"
  (let ((array (nabla:normal key shape :dtype :f32)))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i) (coerce (* stddev (row-major-aref array i)) (quote single-float))))))

(defun %shape-of (x)
  (nabla:aval-shape (nabla:value-aval x)))

(defun %add (a b)
  (funcall (load-time-value (nabla:with-tracing (a b) (+ a b))) a b))

(defun %add-positions (x pos)
  "x (B T d) の各系列に pos (T d) を足す。"
  (funcall (load-time-value
            (nabla:with-tracing (x pos shape) (+ x (nabla:broadcast-in-dim pos shape '(1 2)))))
           x pos (%shape-of x)))

(declaim (ftype function %decoder-block-apply %transformer-apply)) ; 下で定義（defmodule の :call から呼ぶ）

(defmodule decoder-block (ln1 attn ln2 up down)
  (:init (key d-model num-heads mlp-dim)
   (destructuring-bind (ln1-key attn-key ln2-key up-key down-key) (split-keys key 5)
     (make-decoder-block :ln1 (init-layer-norm ln1-key d-model)
                         :attn (init-multi-head-attention attn-key num-heads d-model)
                         :ln2 (init-layer-norm ln2-key d-model)
                         :up (init-dense up-key d-model mlp-dim)
                         :down (init-dense down-key mlp-dim d-model))))
  (:call (self x mask)
   (%decoder-block-apply self x mask)))

(defmodule transformer (embed pos blocks norm head)
  (:init (key vocab max-length d-model num-heads mlp-dim num-layers)
   (destructuring-bind (embed-key pos-key norm-key head-key &rest block-keys) (split-keys key (+ 4 num-layers))
     (make-transformer
      :embed (init-embedding embed-key vocab d-model)
      :pos (%scaled-normal pos-key (list max-length d-model) 0.02d0)
      :blocks (mapcar (lambda (k) (init-decoder-block k d-model num-heads mlp-dim)) block-keys)
      :norm (init-layer-norm norm-key d-model)
      :head (init-dense head-key d-model vocab))))
  (:call (self tokens)
   (%transformer-apply self tokens)))

(defun %transformer-apply (model tokens)
  (let* ((len (car (last (%shape-of tokens))))
         (mask (make-causal-mask len)))
    (unless (= len (first (%shape-of (transformer-pos model))))
      (error "transformer: 系列の長さは ~D でなければならない（~D が渡された）"
             (first (%shape-of (transformer-pos model))) len))
    (let ((x (%add-positions (call (transformer-embed model) tokens) (transformer-pos model))))
      (dolist (blk (transformer-blocks model))
        (setf x (call blk x mask)))
      (call (transformer-head model) (call (transformer-norm model) x)))))

(defun %decoder-block-apply (blk x mask)
  (let* ((attended (let ((y (call (decoder-block-ln1 blk) x)))
                     (call (decoder-block-attn blk) y y mask nil t)))
         (h (%add x attended))
         (mlp (call (decoder-block-down blk)
                    (gelu (call (decoder-block-up blk) (call (decoder-block-ln2 blk) h))))))
    (%add h mlp)))

(defun make-example-transformer (&key (seed 0) (vocab *vocab*) (max-length (* 2 *copy-length*))
                                      (d-model 16) (num-heads 2) (mlp-dim 32) (num-layers 2))
  "例のモデル（語彙 12、長さ 8、幅 16、2 ヘッド、MLP の幅 32、2 ブロック。パラメータ 38 個の葉）。"
  (init-transformer (nabla:prng-key seed) vocab max-length d-model num-heads mlp-dim num-layers))

;;; --- データ ---

(defun make-copy-sequences (count seed &key (vocab *vocab*) (copy-length *copy-length*))
  "COUNT 本の系列 x1..xN SEP x1..xN（i32 の (COUNT 2N+1)）。SEED で再現できる。"
  (let ((state (sb-ext:seed-random-state seed))
        (sequences (make-array (list count (1+ (* 2 copy-length))) :element-type '(signed-byte 32))))
    (dotimes (row count sequences)
      (dotimes (i copy-length)
        (let ((token (random (1- vocab) state)))
          (setf (aref sequences row i) token
                (aref sequences row (+ copy-length 1 i)) token)))
      (setf (aref sequences row copy-length) (1- vocab)))))

(defun sequences->batch (sequences &key (vocab *vocab*) (copy-length *copy-length*))
  "SEQUENCES (B 2N+1) から (values トークン (B 2N) の i32, 正解 (B 2N VOCAB) の f32)。正解は次のトークンの
one-hot に、損失をかける位置（SEP 以降の N 位置）の印を掛けたもの（印の無い位置は全部 0）。"
  (let* ((rows (array-dimension sequences 0))
         (len (* 2 copy-length))
         (tokens (make-array (list rows len) :element-type '(signed-byte 32)))
         (targets (make-array (list rows len vocab) :element-type 'single-float :initial-element 0.0)))
    (dotimes (row rows (values tokens targets))
      (dotimes (p len)
        (setf (aref tokens row p) (aref sequences row p))
        (when (>= p copy-length)
          (setf (aref targets row p (aref sequences row (1+ p))) 1.0))))))

;;; --- 損失と学習 ---

(defun transformer-loss (model tokens targets)
  "交差エントロピー: -sum(TARGETS * log-softmax(logits)) / sum(TARGETS)（印のついた位置の平均）。"
  (funcall (load-time-value
            (nabla:with-tracing (model tokens targets)
              (let ((log-probabilities (log-softmax (call model tokens))))
                (- (/ (nabla:reduce-sum (* targets log-probabilities)) (nabla:reduce-sum targets))))))
           model tokens targets))

(defun make-optimizer (&key (learning-rate 1d-2))
  "大域ノルム 1 のクリップ + Adam（Optax の既定の eps = 1e-8）。"
  (chain (clip-by-global-norm 1d0) (adam learning-rate)))

(defun make-train-step (optimizer &key backend)
  "jit した学習ステップ (model opt-state tokens targets) → (values 損失 新しい model 新しい opt-state)。
入力の葉の数はパラメータ + mu + nu + count + バッチ（このモデルで 38 + 76 + 1 + 2）で、IREE の
local-task の入力数の上限（160）に収まる。ループの外で1回だけ作る（2回目以降はキャッシュを使う）。"
  (let ((value-and-grad (nabla:value-and-grad
                         (nabla:with-tracing (model tokens targets) (transformer-loss model tokens targets)))))
    (nabla:jit (nabla:with-tracing (model opt-state tokens targets)
                 (multiple-value-bind (loss grads) (funcall value-and-grad model tokens targets)
                   (multiple-value-bind (updates new-state) (optimizer-update optimizer grads opt-state model)
                     (values loss (apply-updates model updates) new-state))))
               :backend backend)))

(defun train (&key (steps 100) (batch-size 16) (seed 0) (learning-rate 1d-2))
  "STEPS 回学習して各ステップの損失（single-float）のリストを返す。バッチはステップごとに別の系列。"
  (let* ((optimizer (make-optimizer :learning-rate learning-rate))
         (step (make-train-step optimizer))
         (model (make-example-transformer :seed seed))
         (state (optimizer-init optimizer model)))
    (loop for k below steps
          collect (multiple-value-bind (tokens targets)
                      (sequences->batch (make-copy-sequences batch-size (+ (* 1000 seed) k)))
                    (multiple-value-bind (loss new-model new-state) (funcall step model state tokens targets)
                      (setf model new-model state new-state)
                      (aref loss))))))

;; テストが関数だけを使うために、:nabla-no-demo を *features* に足して load すると学習は走らない。
(unless (member :nabla-no-demo *features*)
  (let ((losses (train)))
    (format t "~&loss[0] = ~,4F~%" (first losses))
    (format t "~&final loss = ~,4F~%" (car (last losses)))))
