;;;; scan を使った Elman RNN を手書きのループで学習する end-to-end テスト（issue #141、
;;;; フェーズ3の完了条件）。IREE の local で、学習ステップ全体（順伝播・grad・SGD 更新の
;;;; うち勾配まで）を jit して JAX のフィクスチャと比べる。
;;;;
;;;; 学習ステップは examples/rnn.lisp の MAKE-RNN-TRAIN-STEP。JAX フィクスチャ:
;;;; tests/fixtures/rnn/rnn-sgd.lisp（生成: tests/fixtures/rnn/generate.py、f32）。

(in-package #:nabla.iree.tests)

(defvar *rnn-example-loaded* nil)

(defun %rnn-example-fn (name)
  "examples/rnn.lisp（初回のみ、標準出力を捨てて load する）の関数 NAME を返す。"
  (unless *rnn-example-loaded*
    (let ((*standard-output* (make-broadcast-stream))
          (nb:*compile-cache-directory* nil))
      (load (asdf:system-relative-pathname "nabla" "examples/rnn.lisp")))
    (setf *rnn-example-loaded* t))
  (symbol-function (find-symbol name "NABLA-EXAMPLE-RNN")))

(defun %rnn-fixture ()
  (with-open-file (stream (asdf:system-relative-pathname "nabla" "tests/fixtures/rnn/rnn-sgd.lisp"))
    (read stream)))

(defun %rnn-fixture-array (entry)
  "(NAME SHAPE BITS) から f32 の配列を作る（f32 は u32 のビットパターン）。"
  (destructuring-bind (name shape bits) entry
    (declare (ignore name))
    (let ((array (make-array shape :element-type 'single-float)))
      (loop for i from 0 for bit in bits
            do (setf (row-major-aref array i) (nb::%make-single-float bit)))
      array)))

(defun %rnn-fixture-arrays (entries)
  (mapcar #'%rnn-fixture-array entries))

(defun %rnn-fixture-inputs (fixture)
  "(values xs y params)"
  (let ((inputs (%rnn-fixture-arrays (getf fixture :inputs))))
    (values (first inputs) (second inputs) (subseq inputs 2))))

(define-iree-test train/rnn-first-step-matches-jax-fixture
    "同じ初期値・データで、scan の RNN の1ステップ目の損失と5つのパラメータの勾配（jit した
value-and-grad）が JAX（jax.lax.scan）のフィクスチャと f32 の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (fixture (%rnn-fixture))
         (expected-losses (%rnn-fixture-array (list "losses" (list (getf fixture :steps)) (getf fixture :losses))))
         (expected-grads (%rnn-fixture-arrays (getf fixture :grads)))
         (grad-fn (funcall (%rnn-example-fn "MAKE-RNN-GRAD-FN"))))
    (multiple-value-bind (xs y params) (%rnn-fixture-inputs fixture)
      (destructuring-bind (loss &rest grads) (multiple-value-list (apply grad-fn (append params (list xs y))))
        (is (approx= (aref loss) (aref expected-losses 0) :dtype :f32)
            "1ステップ目の損失が JAX と一致しない: ~S と ~S" (aref loss) (aref expected-losses 0))
        (loop for actual in grads
              for expected in expected-grads
              for name in '(wh wx b wo bo)
              do (is (allclose actual expected :dtype :f32)
                     "1ステップ目の ~A の勾配が JAX と一致しない" name))))))

(define-iree-test train/rnn-trajectory-matches-jax-and-compiles-once
    "学習ステップ全体（jit した value-and-grad と SGD の更新）を30ステップ回すと、各ステップの損失・
5ステップ後と30ステップ後のパラメータが JAX の SGD の軌跡と一致し、損失が減る（最後が最初の
1/10 未満）。学習ステップの jit は最初の1回だけコンパイルされ、2ステップ目以降は *jit-miss-count*
が増えない。許容誤差は損失とパラメータの f32 の既定値のまま（総和・tanh の実装差が30ステップで
累積するが、この大きさでは既定の rtol 1e-5 / atol 1e-6 に収まることを確かめてある）。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (fixture (%rnn-fixture))
         (lr (nb::%make-single-float (getf fixture :lr)))
         (steps (getf fixture :steps))
         (expected-losses (%rnn-fixture-array (list "losses" (list steps) (getf fixture :losses))))
         ;; examples/rnn.lisp の load は末尾で train を回して jit を1回使うので、
         ;; *jit-miss-count* を読む前に load を済ませる
         (make-step (%rnn-example-fn "MAKE-RNN-TRAIN-STEP"))
         (before nb::*jit-miss-count*)
         (step (funcall make-step :lr lr)))
    (multiple-value-bind (xs y params) (%rnn-fixture-inputs fixture)
      (dotimes (k steps)
        (multiple-value-bind (loss new-params) (funcall step params xs y)
          (is (approx= loss (aref expected-losses k) :dtype :f32)
              "ステップ ~D の損失が JAX と一致しない: ~S と ~S" k loss (aref expected-losses k))
          (setf params new-params)
          (when (= k 0)
            (is (= (1+ before) nb::*jit-miss-count*)))
          (when (= k 4)
            (loop for actual in params
                  for expected in (%rnn-fixture-arrays (getf fixture :params5))
                  for name in '(wh wx b wo bo)
                  do (is (allclose actual expected :dtype :f32) "5ステップ後の ~A が JAX と一致しない" name)))))
      (loop for actual in params
            for expected in (%rnn-fixture-arrays (getf fixture :params-final))
            for name in '(wh wx b wo bo)
            do (is (allclose actual expected :dtype :f32) "~D ステップ後の ~A が JAX と一致しない" steps name))
      (is (= (1+ before) nb::*jit-miss-count*) "2ステップ目以降で再コンパイルされた")
      (is (< (aref expected-losses (1- steps)) (* 0.1 (aref expected-losses 0))))
      (multiple-value-bind (loss new-params) (funcall step params xs y)
        (declare (ignore new-params))
        (is (< loss (* 0.1 (aref expected-losses 0))) "30ステップ後の損失 ~S が減っていない" loss)))))

(define-iree-test example/rnn-lisp/prints-decreasing-loss
    "examples/rnn.lisp（README の使用例）を load でき、標準出力に最初と最後の損失が
\"loss[0] = a\" \"final loss = b\" の形で出て、b < a である。"
  (skip-unless-iree :library :both)
  (let ((output (make-string-output-stream))
        (nb:*compile-cache-directory* nil))
    (let ((*standard-output* output))
      (load (asdf:system-relative-pathname "nabla" "examples/rnn.lisp")))
    (setf *rnn-example-loaded* t)
    (let* ((text (get-output-stream-string output))
           (first-pos (search "loss[0] = " text))
           (last-pos (search "final loss = " text)))
      (is (and first-pos last-pos) "出力に損失の行が無い: ~S" text)
      (when (and first-pos last-pos)
        (let ((first-loss (read-from-string text t nil :start (+ first-pos (length "loss[0] = "))))
              (last-loss (read-from-string text t nil :start (+ last-pos (length "final loss = ")))))
          (is (< last-loss first-loss) "損失が減っていない: ~S -> ~S" first-loss last-loss))))))
