;;;; 2層 MLP を手書きのループで学習する end-to-end テスト（issue #88、フェーズ2の完了条件）。
;;;;
;;;; 学習ステップは examples/mlp.lisp の MAKE-MLP-TRAIN-STEP（(jit (value-and-grad loss
;;;; :argnums (0 1 2 3))) を1回だけ作り、SGD の更新は Lisp 側の配列演算）。ここでは
;;;; examples/mlp.lisp を load して、その関数を呼ぶ。README の使用例が壊れていないことの
;;;; 確認（子 SBCL ではなく同じプロセスで load して標準出力を見る方式。example-test.lisp と同じ）も兼ねる。
;;;;
;;;; JAX フィクスチャ: tests/fixtures/train/mlp-sgd.lisp（生成: tests/fixtures/train/generate.py）。
;;;; f32 のみ。bf16 は SGD の更新を f32 で累積する前提でフェーズ1と同じ許容誤差で比べることになるが、
;;;; 学習ループの検証対象は更新の数値であり dtype の丸めではないので、このテストでは f32 だけにする
;;;; （bf16 の jit / grad 自体は tests/iree/jit-test.lisp と grad-test.lisp が確かめている）。

(in-package #:nabla.iree.tests)

(defvar *mlp-example-loaded* nil)

(defun %mlp-example-fn (name)
  "examples/mlp.lisp（初回のみ、標準出力を捨てて load する）の関数 NAME を返す。"
  (unless *mlp-example-loaded*
    (let ((*standard-output* (make-broadcast-stream))
          (nb:*compile-cache-directory* nil))
      (load (asdf:system-relative-pathname "nabla" "examples/mlp.lisp")))
    (setf *mlp-example-loaded* t))
  (symbol-function (find-symbol name "NABLA-EXAMPLE-MLP")))

(defun %train-fixture ()
  (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/train/mlp-sgd.lisp"))
    (read stream)))

(defun %train-fixture-array (entry)
  "(NAME SHAPE BITS) から f32 の配列を作る（f32 は u32 のビットパターン）。"
  (destructuring-bind (name shape bits) entry
    (declare (ignore name))
    (let ((array (make-array shape :element-type 'single-float)))
      (loop for i from 0 for bit in bits
            do (setf (row-major-aref array i) (nb::%make-single-float bit)))
      array)))

(defun %train-fixture-params (entries)
  (mapcar #'%train-fixture-array entries))

(define-iree-test train/mlp-matches-jax-fixture
    "同じ初期値・同じデータから、最初の N ステップの各損失と、各ステップ後のパラメータ
（w1 b1 w2 b2）が、JAX の SGD のフィクスチャと f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (fixture (%train-fixture))
         (inputs (mapcar #'%train-fixture-array (getf fixture :inputs)))
         (x (first inputs))
         (y (second inputs))
         (params (subseq inputs 2))
         (lr (nb::%make-single-float (getf fixture :lr)))
         (step (funcall (%mlp-example-fn "MAKE-MLP-TRAIN-STEP") :lr lr))
         (expected-losses (%train-fixture-array (list "losses" (list (getf fixture :steps)) (getf fixture :losses)))))
    (loop for k from 0 below (getf fixture :steps)
          for expected-params in (getf fixture :params)
          do (multiple-value-bind (loss new-params) (funcall step params x y)
               (is (allclose (make-array nil :element-type 'single-float :initial-element loss)
                             (make-array nil :element-type 'single-float :initial-element (aref expected-losses k))
                             :dtype :f32)
                   "ステップ ~D の損失が JAX と一致しない: ~S と ~S" k loss (aref expected-losses k))
               ;; パラメータは損失（既定の f32 許容誤差）より緩める（rtol 1e-4）。IREE と XLA は
               ;; tanh / exp / log と dot・reduce の総和の順序が違い、その差が勾配を通って
               ;; 最大 N ステップ分たまるため。
               (loop for actual in new-params
                     for expected in (%train-fixture-params expected-params)
                     for name in '(w1 b1 w2 b2)
                     do (is (allclose actual expected :dtype :f32 :rtol 1e-4 :atol 1e-5)
                            "ステップ ~D の ~A が JAX と一致しない" k name))
               (setf params new-params)))))

(define-iree-test train/mlp-loss-decreases-and-compiles-once
    "データと初期値の seed を変えても、K=80 ステップの SGD で最後の損失が最初の損失の
半分未満になる（単調減少は主張しない。学習率が大きいと途中で一時的に増えうる）。
学習ステップの jit は1回だけコンパイルされ、2ステップ目以降は *jit-miss-count* が増えない
（seed を変えて別の入力を与えても増えない）。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (step (funcall (%mlp-example-fn "MAKE-MLP-TRAIN-STEP")))
         (make-blobs (%mlp-example-fn "MAKE-BLOBS"))
         (init-params (%mlp-example-fn "INIT-PARAMS"))
         (misses-after-first nil))
    (is (check-it (generator (integer 0 100000))
                  (lambda (seed)
                    (multiple-value-bind (x y) (funcall make-blobs 16 seed)
                      (let ((params (funcall init-params seed))
                            (first-loss nil)
                            (last-loss nil))
                        (dotimes (k 80)
                          (multiple-value-bind (loss new-params) (funcall step params x y)
                            (when (= k 0)
                              (setf first-loss loss)
                              (unless misses-after-first (setf misses-after-first nb::*jit-miss-count*)))
                            (setf last-loss loss params new-params)))
                        (and (= misses-after-first nb::*jit-miss-count*)
                             (< last-loss (* 0.5 first-loss))))))
                  :regression-id train/mlp-loss-decreases
                  :regression-file (regression-path "iree-mlp-train-loss-decreases"
                                                    :package "NABLA.IREE.TESTS")))))

(define-iree-test example/mlp-lisp/prints-decreasing-loss
    "examples/mlp.lisp（README の使用例）を load でき、標準出力に最初と最後の損失が
\"loss[0] = a\" \"final loss = b\" の形で出て、b < a である。"
  (skip-unless-iree :library :both)
  (let ((output (make-string-output-stream))
        (nb:*compile-cache-directory* nil))
    (let ((*standard-output* output))
      (load (asdf:system-relative-pathname "nabla" "examples/mlp.lisp")))
    (let* ((text (get-output-stream-string output))
           (first-pos (search "loss[0] = " text))
           (last-pos (search "final loss = " text)))
      (is (and first-pos last-pos) "出力に損失の行が無い: ~S" text)
      (when (and first-pos last-pos)
        (let ((first-loss (read-from-string text t nil :start (+ first-pos (length "loss[0] = "))))
              (last-loss (read-from-string text t nil :start (+ last-pos (length "final loss = ")))))
          (is (< last-loss first-loss) "損失が減っていない: ~S -> ~S" first-loss last-loss))))))
