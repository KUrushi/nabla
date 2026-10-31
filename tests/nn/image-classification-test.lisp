;;;; 画像分類の学習の end-to-end テスト（issue #287、フェーズ5の受け入れテスト）。
;;;;
;;;; モデルと学習は examples/image-classification.lisp（合成画像を .npy に書き、npy-source → data-loader
;;;; （シャッフル・f32 への map・反転と切り抜きの random-map）のバッチで、小さな CNN を Adam で学習する）。
;;;;
;;;; 守らせる性質:
;;;; - small: データセットは決定的で、クラスの数が揃い、各クラスの線が描けている（縦線・横線・十字だけが
;;;;   通しの列・行を持つ）。拡張（反転と ±2 の平行移動）の後もクラスの形が残り、正解は変わらない。jit する
;;;;   学習ステップの入力の葉の数は IREE local-task の上限（160）に収まる。
;;;; - medium（eager。ファイルとワーカーのスレッドを使うが IREE は要らない）: ワーカー数 0 と 2 で学習の
;;;;   結果（損失の列・最後のモデルとオプティマイザの状態）がビット単位で同じ。途中で「モデル + オプティマイザの
;;;;   状態 + ローダの状態」を1ファイルに保存・復元して続けた学習（エポックの境界をまたぐ）も同じ。
;;;; - medium（IREE local で jit）: 例の MAIN（途中で保存・復元し、ワーカー 2 で読む）が最後まで動き、損失が
;;;;   下がり、学習データでの正解率が偶然（25%）より明らかに高い。その結果が、ワーカー 0 で中断しない学習と
;;;;   ビット単位で一致する。CI の時間のため jit の学習は 1 本にまとめる（再開とワーカー数のどちらが壊れたかは
;;;;   上の eager のテストで切り分ける）。

(in-package #:nabla.nn.tests)

(defvar *image-classification-example-loaded* nil)

(defun %ic-fn (name)
  "examples/image-classification.lisp（初回のみ、学習を走らせず標準出力を捨てて load する）の関数 NAME。"
  (unless *image-classification-example-loaded*
    (let ((*standard-output* (make-broadcast-stream))
          (nabla:*compile-cache-directory* nil)
          (*features* (cons :nabla-no-demo *features*)))
      (load (asdf:system-relative-pathname "nabla" "examples/image-classification.lisp")))
    (setf *image-classification-example-loaded* t))
  (symbol-function (find-symbol name "NABLA-EXAMPLE-IMAGE-CLASSIFICATION")))

(defun %ic (name &rest arguments)
  (apply (%ic-fn name) arguments))

(defun %ic-long-lines (image threshold)
  "(H W 1) の IMAGE で、明るい画素（0.5 以上か 128 以上）が THRESHOLD 個以上ある (values 列の数 行の数)。"
  (destructuring-bind (h w c) (array-dimensions image)
    (declare (ignore c))
    (flet ((bright-p (i j) (>= (aref image i j 0) (if (floatp (aref image i j 0)) 0.5 128))))
      (values (loop for j below w count (>= (loop for i below h count (bright-p i j)) threshold))
              (loop for i below h count (>= (loop for j below w count (bright-p i j)) threshold))))))

(defun %ic-shape-matches-class-p (image class threshold)
  "縦線（0）と十字（3）だけが長い列を、横線（1）と十字だけが長い行を持つ。"
  (multiple-value-bind (columns rows) (%ic-long-lines image threshold)
    (and (eq (plusp columns) (and (member class '(0 3)) t))
         (eq (plusp rows) (and (member class '(1 3)) t)))))

(defun %ic-record (images classes i)
  "I 枚目のレコード (画像 ラベル)。ローダと同じく array-source（npy-source の中身）で行を取る。"
  (nabla.data:source-item (nabla.data:make-array-source (list images classes)) i))

;;; --- small ---

(test (image-classification/dataset-is-deterministic-balanced-and-drawn :suite :nabla.small)
  "同じシードからは同じ画像、別のシードからは別の画像。i 枚目のクラスは i mod 4 で、画素は背景（64 未満）か
線（192 以上）。縦線・十字だけが通し（16 画素）の列を、横線・十字だけが通しの行を持つ。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 24)))
                (lambda (case)
                  (destructuring-bind (seed count) case
                    (multiple-value-bind (images classes) (%ic "MAKE-IMAGES" count seed)
                      (and (equal (list count 16 16 1) (array-dimensions images))
                           (equal '(unsigned-byte 8) (array-element-type images))
                           (equal '(signed-byte 32) (array-element-type classes))
                           (equalp images (%ic "MAKE-IMAGES" count seed))
                           (not (equalp images (%ic "MAKE-IMAGES" count (1+ seed))))
                           (loop for i below count always (= (aref classes i) (mod i 4)))
                           (loop for k below (array-total-size images)
                                 always (let ((v (row-major-aref images k))) (or (< v 64) (>= v 192))))
                           (loop for i below count
                                 always (%ic-shape-matches-class-p (first (%ic-record images classes i))
                                                                   (aref classes i) 16))))))
                :regression-id image-classification/dataset-is-deterministic-balanced-and-drawn
                :regression-file (regression-path "image-classification-dataset"
                                                  :package "NABLA.NN.TESTS"))))

(test (image-classification/augmentation-keeps-the-class :suite :nabla.small)
  "拡張（反転と ±2 の平行移動）の後の画像も、縦線・十字だけが 14 画素以上の列を、横線・十字だけが 14 画素
以上の行を持ち、正解（one-hot）は変わらない。f32 は u8 / 255。同じキーなら同じ出力。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 0 :hi 100000)))
                (lambda (case)
                  (destructuring-bind (seed position) case
                    (multiple-value-bind (images classes) (%ic "MAKE-IMAGES" 4 seed)
                      (loop for i below 4
                            always (let* ((record (%ic "TO-FLOAT-RECORD" (%ic-record images classes i)))
                                          (key (nabla:fold-in (nabla:prng-key seed) position))
                                          (out (%ic "AUGMENT" record key)))
                                     (and (= (aref (first record) 0 0 0) (/ (aref images i 0 0 0) 255.0))
                                          (equalp (second out) (second record))
                                          (= 1.0 (aref (second record) (aref classes i)))
                                          (%ic-shape-matches-class-p (first out) (aref classes i) 14)
                                          (equalp out (%ic "AUGMENT" record key))))))))
                :regression-id image-classification/augmentation-keeps-the-class
                :regression-file (regression-path "image-classification-augmentation"
                                                  :package "NABLA.NN.TESTS"))))

(test (image-classification/train-step-inputs-fit-the-iree-limit :suite :nabla.small)
  "jit する学習ステップの入力 = パラメータ 6 + mu 6 + nu 6 + count 1 + バッチ 2 = 21 個で、IREE local-task の
上限（160）に収まる。パス名は conv1・conv2・head の kernel と bias。"
  (let* ((model (%ic "MAKE-EXAMPLE-CNN"))
         (state (optimizer-init (%ic "MAKE-OPTIMIZER") model))
         (x (make-array '(8 16 16 1) :element-type 'single-float :initial-element 0.0))
         (y (make-array '(8 4) :element-type 'single-float :initial-element 0.0))
         (inputs (length (nabla:tree-leaves (list model state x y)))))
    (is (equal '("conv1.kernel" "conv1.bias" "conv2.kernel" "conv2.bias" "head.kernel" "head.bias")
               (tree-paths model)))
    (is (= 21 inputs))
    (is (<= inputs 160) "jit の入力が ~D 個で、IREE local-task の上限（160）を超える" inputs)))

;;; --- medium ---

(defun %ic-train (train-step loader steps &key (num-workers 0) checkpoint-at directory)
  "例の初期値から STEPS 回学習して (values 損失のリスト 最後の (model opt-state)) を返す。CHECKPOINT-AT が
あれば、その回数の後で「モデル + オプティマイザの状態 + ローダの状態」を DIRECTORY に保存し、復元した
ものから続ける（前半はワーカー 0、後半は NUM-WORKERS）。"
  (let* ((optimizer (%ic "MAKE-OPTIMIZER"))
         (model (%ic "MAKE-EXAMPLE-CNN"))
         (opt-state (optimizer-init optimizer model)))
    (if (null checkpoint-at)
        (multiple-value-bind (losses model opt-state)
            (%ic "RUN-STEPS" train-step loader model opt-state steps :num-workers num-workers)
          (values losses (list model opt-state)))
        (multiple-value-bind (losses-1 model-1 state-1 loader-state)
            (%ic "RUN-STEPS" train-step loader model opt-state checkpoint-at)
          (let ((file (merge-pathnames "training.safetensors" directory)))
            (%ic "SAVE-TRAINING" file model-1 state-1 loader-state)
            ;; ひな形は初期値（保存した値とは違う）。復元した値だけで続ける
            (multiple-value-bind (model-2 state-2 loader-state-2) (%ic "LOAD-TRAINING" file model opt-state loader)
              (multiple-value-bind (losses-2 model-3 state-3)
                  (%ic "RUN-STEPS" train-step loader model-2 state-2 (- steps checkpoint-at)
                       :state loader-state-2 :num-workers num-workers)
                (values (append losses-1 losses-2) (list model-3 state-3)))))))))

(defun %ic-same-training-p (losses-a tree-a losses-b tree-b)
  (and (equalp losses-a losses-b)
       (every #'equalp (nabla:tree-leaves tree-a) (nabla:tree-leaves tree-b))))

(test (image-classification/resume-and-worker-count-do-not-change-eager-training :suite :nabla.medium)
  "eager の学習（64 枚、バッチ 8 で 1 エポック 8 ステップ、12 ステップ）で、ワーカー数 0 と 2 の結果、
5 ステップ後に保存・復元してワーカー 2 で続けた結果（エポックの境界をまたぐ）が、損失の列も最後のモデルと
オプティマイザの状態もビット単位で同じ。損失は正の有限の single-float。"
  (with-temporary-directory (directory)
    (let* ((paths (%ic "WRITE-DATASET" directory :count 64 :seed 1))
           (loader (%ic "MAKE-LOADER" paths :batch-size 8 :seed 2))
           (step (%ic "MAKE-TRAIN-STEP" (%ic "MAKE-OPTIMIZER") :jit nil)))
      (multiple-value-bind (losses-0 tree-0) (%ic-train step loader 12)
        (is (= 12 (length losses-0)))
        (is (every (lambda (loss) (and (typep loss 'single-float) (< 0 loss 10))) losses-0))
        (multiple-value-bind (losses-2 tree-2) (%ic-train step loader 12 :num-workers 2)
          (is (%ic-same-training-p losses-0 tree-0 losses-2 tree-2) "ワーカー数 0 と 2 で学習が違う"))
        (multiple-value-bind (losses-r tree-r)
            (%ic-train step loader 12 :num-workers 2 :checkpoint-at 5 :directory directory)
          (is (%ic-same-training-p losses-0 tree-0 losses-r tree-r)
              "保存・復元をはさんだ学習が中断しない学習と違う"))))))

(defun %ic-mean (numbers)
  (/ (reduce #'+ numbers) (length numbers)))

(test (image-classification/example-main-learns-and-resumes-bit-exactly-on-iree-local :suite :nabla.medium)
  "例の MAIN（IREE local で jit した学習ステップ、256 枚・バッチ 32 を 48 ステップ = 6 エポック。24 ステップ後に
「モデル + オプティマイザの状態 + ローダの状態」を保存・復元し、ワーカー 2 で読む）が最後まで動き、損失と
正解率の 5 行を出す。最後の 5 ステップの損失の平均が最初の 5 ステップの平均の 3/4 未満になり、学習データ
（拡張なし）での正解率が 50% を超える（偶然は 25%）。同じデータ・初期値をワーカー 0 で中断せずに学習した
結果と、損失の列も最後のモデル・オプティマイザの状態もビット単位で同じ（同じ StableHLO を同じ入力で
決定的に走らせるので許容誤差は要らない）。"
  (when (%layers-require-iree-or-skip)
    (let ((nabla:*compile-cache-directory* nil))
      (multiple-value-bind (losses accuracy model opt-state output)
          (let* ((results '())
                 (output (with-output-to-string (*standard-output*)
                           (setf results (multiple-value-list (%ic "MAIN" :steps 48 :num-workers 2))))))
            (values-list (append results (list output))))
        (format t "~&image-classification: loss ~,4F → ~,4F、正解率 ~,3F~%"
                (first losses) (car (last losses)) accuracy)
        (is (= 5 (count #\Newline output)) "MAIN の出力が 5 行でない: ~S" output)
        (is (= 48 (length losses)))
        (is (< (%ic-mean (last losses 5)) (* 3/4 (%ic-mean (subseq losses 0 5))))
            "損失が下がらない: ~S → ~S" (subseq losses 0 5) (last losses 5))
        (is (> accuracy 1/2) "学習データでの正解率 ~,3F が 50% 以下" accuracy)
        (with-temporary-directory (directory)
          (let ((loader (%ic "MAKE-LOADER" (%ic "WRITE-DATASET" directory)))
                (step (%ic "MAKE-TRAIN-STEP" (%ic "MAKE-OPTIMIZER"))))
            (multiple-value-bind (losses-0 tree-0) (%ic-train step loader 48)
              (is (%ic-same-training-p losses-0 tree-0 losses (list model opt-state))
                  "保存・復元とワーカー数の変更をはさんだ学習が、中断しない学習と違う"))))))))
