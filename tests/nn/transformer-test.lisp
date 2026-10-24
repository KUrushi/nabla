;;;; 小さな Transformer を学習する end-to-end テスト（issue #209、フェーズ4の完了条件）。
;;;;
;;;; モデルと学習ステップは examples/transformer.lisp（defmodule の decoder-only Transformer、jit した
;;;; value-and-grad + Adam）。JAX のフィクスチャは tests/fixtures/transformer/（生成: generate.py。
;;;; Flax + Optax。同じ初期値を init.safetensors で渡し、同じ系列を trajectory.lisp で渡す）。
;;;;
;;;; 守らせる性質:
;;;; - small: モデル全体が因果的（位置 i のロジットは、位置 i より後のトークンを変えても変わらず、位置 i
;;;;   のトークンを変えると変わる）。パス名は Flax と同じ。データの部品（コピーの系列と、損失の印のついた
;;;;   one-hot）。jit する学習ステップの入力の葉の数は IREE local-task の上限（161）に収まる。
;;;; - medium（IREE local）: Flax + Optax と、損失（20 ステップ）、1 ステップ後・5 ステップ後のパラメータが
;;;;   f32 の許容誤差で一致する。損失が下がる（100 ステップで最初の 5% 未満）。同じ入力から同じ損失の
;;;;   列になる（決定的）。学習の途中で保存・復元（モデル + オプティマイザの状態を1つの木で1ファイル）
;;;;   した続きが、止めずに続けた学習と一致する。コンパイル時間とステップ時間はテストの出力に書く。

(in-package #:nabla.nn.tests)

(defvar *transformer-example-loaded* nil)

(defun %transformer-example-fn (name)
  "examples/transformer.lisp（初回のみ、学習を走らせず標準出力を捨てて load する）の関数 NAME。"
  (unless *transformer-example-loaded*
    (let ((*standard-output* (make-broadcast-stream))
          (nabla:*compile-cache-directory* nil))
      (let ((*features* (cons :nabla-no-demo *features*)))
        (load (asdf:system-relative-pathname "nabla" "examples/transformer.lisp"))))
    (setf *transformer-example-loaded* t))
  (symbol-function (find-symbol name "NABLA-EXAMPLE-TRANSFORMER")))

(defun %tf (name &rest arguments)
  (apply (%transformer-example-fn name) arguments))

(defun %tf-var (name)
  "例の特殊変数 NAME（*COPY-LENGTH* など）の値。"
  (%transformer-example-fn "MAKE-OPTIMIZER") ; load させる
  (symbol-value (find-symbol name "NABLA-EXAMPLE-TRANSFORMER")))

(defun %tf-fixture-path (name)
  (asdf:system-relative-pathname "nabla" (concatenate 'string "tests/fixtures/transformer/" name)))

(defvar *transformer-trajectory* nil)

(defun %tf-trajectory ()
  (or *transformer-trajectory*
      (setf *transformer-trajectory*
            (with-open-file (stream (%tf-fixture-path "trajectory.lisp")) (read stream)))))

(defvar *transformer-fixture-tokens* nil
  "フィクスチャの系列を1次元のベクタにしたもの（最初に使うときに1回だけ作る）。")

(defun %tf-fixture-sequences (step)
  "フィクスチャのステップ STEP の系列 (バッチ 2N+1) の i32 の配列。"
  (let* ((rows (getf (%tf-trajectory) :batch-size))
         (width (1+ (* 2 (%tf-var "*COPY-LENGTH*"))))
         (all (or *transformer-fixture-tokens*
                  (setf *transformer-fixture-tokens* (coerce (getf (%tf-trajectory) :sequences) 'vector))))
         (array (make-array (list rows width) :element-type '(signed-byte 32))))
    (dotimes (i (* rows width) array)
      (setf (row-major-aref array i) (aref all (+ (* step rows width) i))))))

(defun %tf-fixture-losses ()
  (map 'vector #'nabla::%make-single-float (getf (%tf-trajectory) :losses)))

(defun %tf-load-params (name)
  "フィクスチャの safetensors NAME をチェックポイントとして読み、例のモデルの木に復元する。"
  (load-checkpoint (%tf-fixture-path name) (%tf "MAKE-EXAMPLE-TRANSFORMER")))

(defun %max-abs-difference (a b)
  "2つの木の対応する葉の差の絶対値の最大。"
  (let ((worst 0d0))
    (loop for x in (nabla:tree-leaves a)
          for y in (nabla:tree-leaves b)
          do (dotimes (i (array-total-size x))
               (setf worst (max worst (abs (- (row-major-aref x i) (row-major-aref y i)))))))
    worst))

(defun %leaves-all-close (a b &key (rtol 1d-4) (atol 1d-5))
  (every (lambda (x y) (allclose x y :dtype :f32 :rtol rtol :atol atol))
         (nabla:tree-leaves a) (nabla:tree-leaves b)))

;;; --- small ---

(test (transformer/paths-match-flax-and-leaf-count-fits-iree :suite :nabla.small)
  "チェックポイントのパス名は Flax の flatten_dict と同じ（blocks_<i> は blocks.<i>）。学習ステップの
jit の入力 = パラメータ + mu + nu + count + バッチは、IREE local-task の上限（161）に収まる。"
  (let* ((model (%tf "MAKE-EXAMPLE-TRANSFORMER"))
         (paths (tree-paths model))
         (optimizer (%tf "MAKE-OPTIMIZER"))
         (state (optimizer-init optimizer model)))
    (is (= 38 (length paths)))
    (dolist (name '("embed.embedding" "pos" "blocks.0.ln1.scale" "blocks.1.attn.query.kernel"
                    "blocks.1.attn.out.bias" "blocks.0.up.kernel" "blocks.1.down.bias" "norm.bias" "head.kernel"))
      (is (member name paths :test #'string=) "~A が無い" name))
    (multiple-value-bind (tokens targets) (%tf "SEQUENCES->BATCH" (%tf-fixture-sequences 0))
      ;; jit の入力の数 = モデルの葉 + 状態の葉 + バッチ（トークンと正解）
      (let ((inputs (length (nabla:tree-leaves (list model state tokens targets)))))
        (is (= (+ (length paths) (* 2 (length paths)) 1 2) inputs))
        (is (< inputs 162) "jit の入力が ~D 個で、IREE local-task の上限（161）を超える" inputs)))))

(test (transformer/copy-batch :suite :nabla.small)
  "コピーの系列は x1..xN SEP x1..xN で SEP は区切りだけに現れる。バッチのトークンは系列の先頭 2N 個、
正解は SEP 以降の N 位置だけに 1 つずつ（次のトークンの one-hot）、ほかは全部 0。"
  (let* ((n (%tf-var "*COPY-LENGTH*"))
         (vocab (%tf-var "*VOCAB*"))
         (sequences (%tf "MAKE-COPY-SEQUENCES" 6 3)))
    (dotimes (row 6)
      (dotimes (i n)
        (is (= (aref sequences row i) (aref sequences row (+ n 1 i))))
        (is (< (aref sequences row i) (1- vocab))))
      (is (= (1- vocab) (aref sequences row n))))
    (is (equalp sequences (%tf "MAKE-COPY-SEQUENCES" 6 3)))
    (is (not (equalp sequences (%tf "MAKE-COPY-SEQUENCES" 6 4))))
    (multiple-value-bind (tokens targets) (%tf "SEQUENCES->BATCH" sequences)
      (is (equal (list 6 (* 2 n)) (array-dimensions tokens)))
      (is (equal (list 6 (* 2 n) vocab) (array-dimensions targets)))
      (dotimes (row 6)
        (dotimes (p (* 2 n))
          (is (= (aref sequences row p) (aref tokens row p)))
          (let ((ones (loop for v below vocab sum (aref targets row p v))))
            (if (< p n)
                (is (= 0 ones))
                (progn (is (= 1 ones))
                       (is (= 1 (aref targets row p (aref sequences row (1+ p)))))))))))))

(defun %copy-array (array)
  (let ((new (make-array (array-dimensions array) :element-type (array-element-type array))))
    (dotimes (i (array-total-size array) new)
      (setf (row-major-aref new i) (row-major-aref array i)))))

(defvar *causal-test-model* nil
  "is-causal の全試行で使うモデル（init は1回だけ。試行ごとに変えるのはトークンだけ）。")

(def-layer-property transformer/is-causal
  "モデル全体（埋め込み・位置埋め込み・2 ブロック・出力）で、位置 I のロジットは、位置 I より後のトークンを
どう変えても変わらない（ちょうど同じ）。位置 I のトークンを別の値にすると位置 I のロジットは変わる
\(性質が空にならないことの対照)。"
  (seed row batch) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 6)
                          (uniform-integer :lo 1 :hi 3))
  (let* ((model (or *causal-test-model* (setf *causal-test-model* (%tf "MAKE-EXAMPLE-TRANSFORMER" :seed 7))))
         (vocab (%tf-var "*VOCAB*"))
         (len (* 2 (%tf-var "*COPY-LENGTH*")))
         (tokens (let ((a (make-array (list batch len) :element-type '(signed-byte 32)))
                       (r (%rand (list batch len) (+ seed 1) :f64)))
                   (dotimes (i (array-total-size a) a)
                     (setf (row-major-aref a i) (mod (floor (* 1000 (abs (row-major-aref r i)))) vocab)))))
         (later (let ((a (%copy-array tokens)))
                  (loop for p from (1+ row) below len
                        do (dotimes (b batch) (setf (aref a b p) (mod (+ (aref a b p) 5) vocab))))
                  a))
         (changed (let ((a (%copy-array tokens)))
                    (dotimes (b batch a) (setf (aref a b row) (mod (+ (aref a b row) 1) vocab)))))
         (logits (call model tokens)))
    (flet ((position-logits (x)
             (let ((out (make-array (list batch vocab) :element-type 'single-float)))
               (dotimes (b batch out)
                 (dotimes (v vocab) (setf (aref out b v) (aref x b row v)))))))
      (and (equalp (position-logits logits) (position-logits (call model later)))
           (not (equalp (position-logits logits) (position-logits (call model changed))))))))

;;; --- medium（IREE local）---

(defun %tf-run-steps (train-step model state sequences-fn from to)
  "ステップ FROM..TO-1 を回して (values 損失の single-float のベクタ モデル 状態)。SEQUENCES-FN は
ステップ番号 → 系列。"
  (let ((losses (make-array (- to from))))
    (loop for k from from below to
          do (multiple-value-bind (tokens targets) (%tf "SEQUENCES->BATCH" (funcall sequences-fn k))
               (multiple-value-bind (loss new-model new-state) (funcall train-step model state tokens targets)
                 (setf (aref losses (- k from)) (aref loss) model new-model state new-state))))
    (values losses model state)))

(defun %tf-training-sequences (k)
  "ステップ K の学習用の系列。バッチの大きさ 8 はフィクスチャと同じ（共有する学習ステップを再コンパイルしない）。"
  (%tf "MAKE-COPY-SEQUENCES" 8 (+ 5000 k)))

(defun %scalar (x)
  (make-array nil :element-type 'single-float :initial-element x))

(defvar *transformer-shared-step* nil
  "(オプティマイザ . jit した学習ステップ)。3つの medium テストが同じステップを使うので、コンパイルは1回で済む
\(テストのプロセスの終わりまで解放しない)。")

(defvar *transformer-first-call-seconds* nil
  "共有のステップの1回目の呼び出し（トレース + コンパイル + 実行）にかかった秒数。")

(defun %seconds-since (start)
  (/ (- (get-internal-real-time) start) (float internal-time-units-per-second 1d0)))

(defun %tf-shared-step ()
  "(values オプティマイザ 学習ステップ)。初回に1回呼んでコンパイルし、その時間を記録する。"
  (unless *transformer-shared-step*
    (let* ((nabla:*compile-cache-directory* nil)
           (optimizer (%tf "MAKE-OPTIMIZER"))
           (train-step (%tf "MAKE-TRAIN-STEP" optimizer :backend (nabla:find-backend :iree)))
           (model (%tf "MAKE-EXAMPLE-TRANSFORMER")))
      (multiple-value-bind (tokens targets) (%tf "SEQUENCES->BATCH" (%tf-fixture-sequences 0))
        (let ((start (get-internal-real-time)))
          (funcall train-step model (optimizer-init optimizer model) tokens targets)
          (setf *transformer-first-call-seconds* (%seconds-since start))))
      (setf *transformer-shared-step* (cons optimizer train-step))))
  (values (car *transformer-shared-step*) (cdr *transformer-shared-step*)))

(defun %key-bias-leaf-p (path)
  (let ((suffix ".attn.key.bias"))
    (and (> (length path) (length suffix)) (string= suffix path :start2 (- (length path) (length suffix))))))

(defun %compare-params (actual expected)
  "ACTUAL と EXPECTED（モデルの木）を葉ごとに比べて (values key.bias 以外の葉が許容誤差に収まるか
  key.bias 以外の最大の差 key.bias の最大の差)。許容誤差は rtol 1e-4 / atol 1e-5。"
  (let ((ok t) (worst 0d0) (worst-key-bias 0d0))
    (loop for path in (tree-paths actual)
          for x in (nabla:tree-leaves actual)
          for y in (nabla:tree-leaves expected)
          do (let ((diff (%max-abs-difference x y)))
               (cond ((%key-bias-leaf-p path) (setf worst-key-bias (max worst-key-bias diff)))
                     (t (setf worst (max worst diff))
                        (unless (allclose x y :dtype :f32 :rtol 1d-4 :atol 1d-5) (setf ok nil))))))
    (values ok worst worst-key-bias)))

(test (transformer/matches-flax-and-optax-on-iree-local :suite :nabla.medium)
  "同じ初期値（init.safetensors）と同じ系列で、IREE local の jit した学習ステップ（value-and-grad + clip +
Adam）は、Flax + Optax と 20 ステップの損失、1 ステップ後・5 ステップ後のパラメータが一致する
\(最大の差は出力に書く)。許容誤差は損失もパラメータも rtol 1e-4 / atol 1e-5（f32 の行列積・総和の順序の
違いのぶん。測った差は 4e-6 以下）。ただし attn.key.bias の葉は別に見る: key の bias は 1 つのクエリの全スコアに
同じ定数を足すだけで softmax で打ち消されるので、損失の勾配は数学的に 0 で、実装ごとの浮動小数点の雑音（1e-9
くらい）しかない。Adam は m / (√v + eps) と勾配の大きさを正規化するため、この雑音を学習率の大きさの更新に
増幅する。だから値は一致せず、どちらも 1 ステップに最大でほぼ学習率しか動かないことだけを使って
|差| <= 2 * 学習率 * ステップ数 を確かめる。"
  (when (%layers-require-iree-or-skip)
    (let ((nabla:*compile-cache-directory* nil)
          (learning-rate 1d-2))
      (multiple-value-bind (optimizer train-step) (%tf-shared-step)
        (let ((init (%tf-load-params "init.safetensors"))
              (expected (%tf-fixture-losses)))
          (flet ((advance (from to model state)
                   (%tf-run-steps train-step model state #'%tf-fixture-sequences from to)))
            (multiple-value-bind (losses-1 model-1 state-1) (advance 0 1 init (optimizer-init optimizer init))
              (multiple-value-bind (losses-5 model-5 state-5) (advance 1 5 model-1 state-1)
                (multiple-value-bind (losses-20) (advance 5 20 model-5 state-5)
                  (let ((losses (concatenate 'vector losses-1 losses-5 losses-20)))
                    (format t "~&transformer: 損失の最大の差 ~,3E~%"
                            (loop for a across losses for b across expected maximize (abs (- a b))))
                    (loop for a across losses for b across expected for k from 0
                          do (is (allclose (%scalar a) (%scalar b) :dtype :f32 :rtol 1d-4 :atol 1d-5)
                                 "ステップ ~D の損失が Flax と一致しない: ~S と ~S" k a b)))
                  (loop for (file steps model) in (list (list "after-1.safetensors" 1 model-1)
                                                              (list "after-5.safetensors" 5 model-5))
                        do (multiple-value-bind (ok worst worst-key-bias)
                               (%compare-params model (%tf-load-params file))
                             (format t "~&transformer: ~D ステップ後のパラメータの最大の差 ~,3E、attn.key.bias は ~,3E（上限 ~,3E）~%"
                                     steps worst worst-key-bias (float (* 2 learning-rate steps) 1f0))
                             (is (not (null ok)) "~D ステップ後のパラメータ（key.bias 以外）が Flax と一致しない（最大の差 ~,3E）" steps worst)
                             (is (<= worst-key-bias (* 2 learning-rate steps))
                                 "~D ステップ後の attn.key.bias の差 ~,3E が上限を超える" steps worst-key-bias))))))))))))

(test (transformer/loss-decreases-and-is-deterministic-on-iree-local :suite :nabla.medium)
  "コピー課題を 100 ステップ学習すると、損失が最初の 5% 未満になる。同じ入力からもう一度回すと損失の列が
\(ビットまで) 同じ。ステップごとに再コンパイルしない（jit のキャッシュに当たる）。1回目の呼び出しの時間と
その後のステップ時間を出力に書く（目安。値は検査しない）。"
  (when (%layers-require-iree-or-skip)
    (let ((nabla:*compile-cache-directory* nil))
      (multiple-value-bind (optimizer train-step) (%tf-shared-step)
        (let* ((model (%tf "MAKE-EXAMPLE-TRANSFORMER"))
               (state (optimizer-init optimizer model))
               (misses nabla::*jit-miss-count*)
               (start (get-internal-real-time))
               (losses (%tf-run-steps train-step model state #'%tf-training-sequences 0 100))
               (per-step (/ (%seconds-since start) 100))
               (rerun (%tf-run-steps train-step model state #'%tf-training-sequences 0 20)))
          (format t "~&transformer: 1回目の呼び出し（トレース + コンパイル + 実行）~,2F 秒、その後 ~,1F ms/ステップ~%"
                  *transformer-first-call-seconds* (* 1000 per-step))
          (format t "~&transformer: 損失 ~{~,3F~^ ~}（ステップ 0, 20, ..., 99）~%"
                  (loop for k in '(0 20 40 60 80 99) collect (aref losses k)))
          (is (< (aref losses 99) (* 0.05 (aref losses 0))) "損失が下がらない: ~S → ~S" (aref losses 0) (aref losses 99))
          (is (equalp rerun (subseq losses 0 20)) "同じ入力なのに損失が違う")
          (is (= misses nabla::*jit-miss-count*) "ステップで再コンパイルしている"))))))

(test (transformer/checkpoint-resume-equals-uninterrupted-training :suite :nabla.medium)
  "6 ステップ学習した時点でモデルとオプティマイザの状態を1つの木 (model state) として1つのファイルに保存し、
復元した木から続きの 6 ステップを回すと、止めずに回した 12 ステップと損失も最終のモデル・状態もちょうど
\(ビットまで) 同じ。同じコンパイル済みの関数を同じ入力で決定的に走らせるので、許容誤差は要らない
\(count も続けて 12 になる)。"
  (when (%layers-require-iree-or-skip)
    (let ((nabla:*compile-cache-directory* nil))
      (multiple-value-bind (optimizer train-step) (%tf-shared-step)
        (let* ((model (%tf "MAKE-EXAMPLE-TRANSFORMER" :seed 3))
               (state (optimizer-init optimizer model)))
          (multiple-value-bind (losses-a model-a state-a)
              (%tf-run-steps train-step model state #'%tf-training-sequences 0 6)
            (with-temporary-directory (dir)
              (let ((file (merge-pathnames "transformer.safetensors" dir)))
                (save-checkpoint file (list model-a state-a))
                (destructuring-bind (model-b state-b) (load-checkpoint file (list model-a state-a))
                  (is (every #'equalp (nabla:tree-leaves (list model-a state-a))
                             (nabla:tree-leaves (list model-b state-b))))
                  (multiple-value-bind (losses-straight model-straight state-straight)
                      (%tf-run-steps train-step model-a state-a #'%tf-training-sequences 6 12)
                    (multiple-value-bind (losses-resumed model-resumed state-resumed)
                        (%tf-run-steps train-step model-b state-b #'%tf-training-sequences 6 12)
                      (is (equalp losses-straight losses-resumed))
                      (is (every #'equalp (nabla:tree-leaves (list model-straight state-straight))
                                 (nabla:tree-leaves (list model-resumed state-resumed))))
                      (is (= 12 (aref (find-if-not (lambda (a) (eq 'single-float (array-element-type a)))
                                                   (nabla:tree-leaves state-resumed)))))
                      (is (< (aref losses-resumed 5) (aref losses-a 0))))))))))))))

(test (transformer/example-script-runs-on-iree-local :suite :nabla.medium)
  "examples/transformer.lisp（README の使用例）をそのまま load すると 100 ステップ学習し、標準出力に最初と最後の
損失が出る。最後の損失は最初より小さい。"
  (when (%layers-require-iree-or-skip)
    (let* ((nabla:*compile-cache-directory* nil)
           (text (with-output-to-string (*standard-output*)
                   (load (asdf:system-relative-pathname "nabla" "examples/transformer.lisp"))))
           (first-at (search "loss[0] = " text))
           (final-at (search "final loss = " text)))
      (is (and first-at final-at) "損失の行が出力に無い: ~S" text)
      (when (and first-at final-at)
        (is (< (let ((*read-default-float-format* 'single-float))
                 (read-from-string text nil nil :start (+ final-at (length "final loss = "))))
               (let ((*read-default-float-format* 'single-float))
                 (read-from-string text nil nil :start (+ first-at (length "loss[0] = "))))))))))

;;; --- 葉が local-task の入力の上限を超える深いモデル（issue #242）---

(defparameter *deep-transformer-layers* 4
  "ブロック 4 つで、パラメータの葉は 70 個。学習ステップの入力は 70 * 3 + 1 + 2 = 213 個で、IREE local-task の
上限（161）を超える（jit が dtype ごとに連結して渡す）。")

(defun %tf-step-function (optimizer)
  "MAKE-TRAIN-STEP と同じ学習ステップの TRACEABLE-FUNCTION（jit しない。eager でも呼べる）。"
  (let ((value-and-grad (nabla:value-and-grad
                         (nabla:with-tracing (model tokens targets) (%tf "TRANSFORMER-LOSS" model tokens targets)))))
    (nabla:with-tracing (model opt-state tokens targets)
      (multiple-value-bind (loss grads) (funcall value-and-grad model tokens targets)
        (multiple-value-bind (updates new-state) (optimizer-update optimizer grads opt-state model)
          (values loss (apply-updates model updates) new-state))))))

(test (transformer/deep-model-exceeds-the-local-task-input-limit :suite :nabla.small)
  "深いモデル（ブロック 4 つ）の学習ステップの入力の葉は 162 個以上（medium のテストが上限を超える形を確かめるための前提）。"
  (let* ((optimizer (%tf "MAKE-OPTIMIZER"))
         (model (%tf "MAKE-EXAMPLE-TRANSFORMER" :num-layers *deep-transformer-layers*)))
    (multiple-value-bind (tokens targets) (%tf "SEQUENCES->BATCH" (%tf-training-sequences 0))
      (is (> (length (nabla:tree-leaves (list model (optimizer-init optimizer model) tokens targets))) 161)))))

(test (transformer/deep-model-past-the-input-limit-trains-on-iree-local-task :suite :nabla.medium)
  "入力の葉が 213 個の学習ステップを IREE の local-task で jit すると（葉は dtype ごとに連結される）、2 ステップの
損失とパラメータが eager と一致する（rtol 1e-4 / atol 1e-5。attn.key.bias は勾配が数学的に 0 なので、Flax との比較と
同じく |差| <= 2 * 学習率 * ステップ数 で見る）。続けて 30 ステップで損失が下がる（最後 < 最初）。ステップ時間は local-sync
（連結しない）と並べて出力に書く（目安。値は検査しない。docs/jit-input-packing.md）。"
  (when (%layers-require-iree-or-skip)
    (let* ((nabla:*compile-cache-directory* nil)
           (learning-rate 1d-2)
           (optimizer (%tf "MAKE-OPTIMIZER" :learning-rate learning-rate))
           (step-fn (%tf-step-function optimizer))
           (backend (nabla:make-backend :iree :target :local))
           (step (nabla:jit step-fn :backend backend))
           (model (%tf "MAKE-EXAMPLE-TRANSFORMER" :num-layers *deep-transformer-layers*))
           (state (optimizer-init optimizer model)))
      (is (eql 161 (nabla:backend-max-inputs backend)))
      (multiple-value-bind (jit-losses jit-model) (%tf-run-steps step model state #'%tf-training-sequences 0 2)
        (multiple-value-bind (eager-losses eager-model)
            (%tf-run-steps step-fn model state #'%tf-training-sequences 0 2)
          (loop for a across jit-losses for b across eager-losses for k from 0
                do (is (allclose (%scalar a) (%scalar b) :dtype :f32 :rtol 1d-4 :atol 1d-5)
                       "ステップ ~D の損失が eager と一致しない: ~S と ~S" k a b))
          (multiple-value-bind (ok worst worst-key-bias) (%compare-params jit-model eager-model)
            (format t "~&deep transformer: 2 ステップ後のパラメータの eager との最大の差 ~,3E、attn.key.bias は ~,3E~%"
                    worst worst-key-bias)
            (is (not (null ok)) "パラメータ（key.bias 以外）が eager と一致しない（最大の差 ~,3E）" worst)
            (is (<= worst-key-bias (* 2 learning-rate 2))))))
      (let* ((start (get-internal-real-time))
             (losses (%tf-run-steps step model state #'%tf-training-sequences 0 30))
             (packed-ms (* 1000 (/ (%seconds-since start) 30)))
             (sync-step (nabla:jit step-fn :backend (nabla:make-backend :iree :target :local :driver :local-sync))))
        (is (< (aref losses 29) (aref losses 0)) "損失が下がらない: ~S → ~S" (aref losses 0) (aref losses 29))
        ;; ホストでの連結だけの時間（1 ステップの入力 213 個を dtype ごとに 1 本にする）
        (multiple-value-bind (tokens targets) (%tf "SEQUENCES->BATCH" (%tf-training-sequences 0))
          (let* ((leaves (nabla:tree-leaves (list model state tokens targets)))
                 (avals (mapcar #'nabla:value-aval leaves))
                 (plan (nabla::%jit-pack-plan avals 161))
                 (start (get-internal-real-time)))
            (dotimes (i 100) (nabla::%jit-pack-leaves plan leaves (nabla::%jit-packed-avals plan avals)))
            (format t "~&deep transformer: ホストでの連結 ~,3F ms/ステップ（入力 ~D 個 → ~D 個）~%"
                    (* 1000 (/ (%seconds-since start) 100)) (length leaves) (length plan))))
        (%tf-run-steps sync-step model state #'%tf-training-sequences 0 1) ; コンパイル
        (let ((start (get-internal-real-time)))
          (%tf-run-steps sync-step model state #'%tf-training-sequences 0 30)
          (format t "~&deep transformer: local-task（連結）~,1F ms/ステップ、local-sync（連結しない）~,1F ms/ステップ~%"
                  packed-ms (* 1000 (/ (%seconds-since start) 30))))))))
