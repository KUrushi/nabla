;;;; README のチュートリアル（issue #210）が腐らないことを守るテスト。
;;;;
;;;; コードは examples/tutorial.lisp にあり、README のコードブロックは、その「;;;; [[tutorial:名前]]」と
;;;; 「;;;; [[/tutorial]]」で挟んだ行をそのまま貼ったもの（README では直前の行が <!-- tutorial:名前 -->）。
;;;;
;;;; 守らせる性質（README と例の突き合わせとサンプルごとの勾配は small、学習と例の実行は medium で IREE local で jit する）:
;;;; - README のブロックと例の該当部分が一字一句同じで、名前の集合も同じ（片方だけ直して食い違えない）。
;;;; - データローダのバッチで学習して損失が下がる。途中で（ローダの状態も一緒に）保存・復元した続きが、
;;;;   止めずに続けた学習と（ビットまで）同じ。
;;;; - サンプルごとの勾配の平均が、バッチ全体の損失の勾配と一致する（vmap の節）。
;;;; - 例のスクリプトが、そのまま load して最後まで動く。

(in-package #:nabla.nn.tests)

(defvar *tutorial-example-loaded* nil)

(defun %tutorial-example-path ()
  (asdf:system-relative-pathname "nabla" "examples/tutorial.lisp"))

(defun %tutorial-fn (name)
  "examples/tutorial.lisp（初回のみ、学習を走らせず load する）の関数 NAME。"
  (unless *tutorial-example-loaded*
    (let ((*standard-output* (make-broadcast-stream))
          (nabla:*compile-cache-directory* nil)
          (*features* (cons :nabla-no-demo *features*)))
      (load (%tutorial-example-path)))
    (setf *tutorial-example-loaded* t))
  (symbol-function (find-symbol name "NABLA-EXAMPLE-TUTORIAL")))

(defun %tut (name &rest arguments)
  (apply (%tutorial-fn name) arguments))

(defun %line-starts-with-p (prefix line)
  (let ((trimmed (string-left-trim " " line)))
    (and (>= (length trimmed) (length prefix)) (string= prefix trimmed :end2 (length prefix)))))

(defun %join-lines (lines)
  (format nil "~{~A~^~%~}" lines))

(defun %example-sections ()
  "例のファイルの ((名前 . テキスト) ...)。「[[tutorial:名前]]」と「[[/tutorial]]」の間の行を改行でつなぐ。"
  (let ((sections '()) (name nil) (lines '()))
    (dolist (line (uiop:read-file-lines (%tutorial-example-path)) (nreverse sections))
      (cond ((%line-starts-with-p ";;;; [[tutorial:" line)
             (setf name (subseq line (length ";;;; [[tutorial:") (- (length line) 2))
                   lines '()))
            ((%line-starts-with-p ";;;; [[/tutorial]]" line)
             (push (cons name (%join-lines (reverse lines))) sections)
             (setf name nil))
            (name (push line lines))))))

(defun %readme-sections ()
  "README の ((名前 . テキスト) ...)。「<!-- tutorial:名前 -->」の次の ```lisp ブロックの中身。"
  (let ((sections '()) (name nil) (lines '()) (in-block nil))
    (dolist (line (uiop:read-file-lines (asdf:system-relative-pathname "nabla" "README.md"))
                  (nreverse sections))
      (cond (in-block
             (if (%line-starts-with-p "```" line)
                 (progn (push (cons name (%join-lines (reverse lines))) sections)
                        (setf in-block nil name nil))
                 (push line lines)))
            ((and name (%line-starts-with-p "```lisp" line))
             (setf in-block t lines '()))
            ((%line-starts-with-p "<!-- tutorial:" line)
             (setf name (subseq line (length "<!-- tutorial:") (- (length line) (length " -->")))))))))

(test (tutorial/readme-code-blocks-equal-the-example-file :suite :nabla.small)
  "README のチュートリアルのコードブロックが、examples/tutorial.lisp の対応する部分と一字一句同じ。
名前の集合も同じで（README にだけある・例にだけある部分が無い）、順序も同じ。"
  (let ((example (%example-sections))
        (readme (%readme-sections)))
    (is (plusp (length example)) "例のファイルに [[tutorial:...]] の部分が無い")
    (is (equal (mapcar #'car example) (mapcar #'car readme))
        "名前の並びが違う: 例 ~S、README ~S" (mapcar #'car example) (mapcar #'car readme))
    (loop for (name . text) in readme
          do (is (equal text (cdr (assoc name example :test #'string=)))
                 "README の ~A のコードブロックが examples/tutorial.lisp と違う" name))))

(defvar *tutorial-shared-step* nil
  "(オプティマイザ . jit した学習ステップ)。medium テストが同じステップを使うので、コンパイルは1回で済む。")

(defun %tutorial-shared-step ()
  (unless *tutorial-shared-step*
    (let* ((nabla:*compile-cache-directory* nil)
           (optimizer (%tut "MAKE-OPTIMIZER")))
      (setf *tutorial-shared-step* (cons optimizer (%tut "MAKE-TRAIN-STEP" optimizer)))))
  (values (car *tutorial-shared-step*) (cdr *tutorial-shared-step*)))

(defun %tutorial-initial (optimizer)
  "(values model opt-state): README と同じ初期化（キー 0、入力 2、幅 16、クラス 2）。"
  (let ((model (%tut "INIT-MLP" (nabla:prng-key 0) 2 16 2)))
    (values model (optimizer-init optimizer model))))

(defun %mean (numbers)
  (/ (reduce #'+ numbers) (length numbers)))

(defun %tutorial-loader ()
  (%tut "MAKE-LOADER" (%tut "MAKE-DATASET")))

(test (tutorial/loss-decreases-and-checkpoint-resume-equals-uninterrupted-training :suite :nabla.medium)
  "ローダのバッチで 60 ステップ学習すると損失が下がる（最後の 5 ステップの平均が最初の 5 ステップの平均の
半分未満）。30 ステップで「モデル + オプティマイザの状態 + ローダの状態」を保存し、初期値をひな形に復元した
続きの 30 ステップは、同じイテレータで止めずに回した 30 ステップと損失も最終のモデル・オプティマイザの状態も
ちょうど（ビットまで）同じ。同じコンパイル済みの関数を同じ入力で決定的に走らせるので許容誤差は要らない。
復元した木は保存したものと同じで、ローダの状態から再開したバッチの列も続きから始まる。"
  (when (%layers-require-iree-or-skip)
    (let ((nabla:*compile-cache-directory* nil))
      (multiple-value-bind (optimizer train-step) (%tutorial-shared-step)
        (multiple-value-bind (model opt-state) (%tutorial-initial optimizer)
          (let ((loader (%tutorial-loader)))
            (nabla.data:with-data-iterator (iterator loader)
              (multiple-value-bind (losses-first model-a state-a)
                  (%tut "RUN-STEPS" train-step model opt-state iterator 30)
                (with-temporary-directory (dir)
                  (let ((file (merge-pathnames "tutorial.safetensors" dir)))
                    (%tut "SAVE-TRAINING" file model-a state-a iterator)
                    (multiple-value-bind (model-b state-b loader-state)
                        (%tut "LOAD-TRAINING" file model opt-state loader)
                      (is (every #'equalp
                                 (nabla:tree-leaves (list model-a state-a (nabla.data:iterator-state iterator)))
                                 (nabla:tree-leaves (list model-b state-b loader-state)))
                          "復元した木の葉が保存したものと違う")
                      (multiple-value-bind (losses-straight model-straight state-straight)
                          (%tut "RUN-STEPS" train-step model-a state-a iterator 30)
                        (multiple-value-bind (losses-resumed model-resumed state-resumed)
                            (nabla.data:with-data-iterator (resumed loader :state loader-state)
                              (%tut "RUN-STEPS" train-step model-b state-b resumed 30))
                          (let ((all (append losses-first losses-straight)))
                            (is (< (%mean (last all 5)) (* 0.5 (%mean (subseq all 0 5))))
                                "損失が下がらない: ~S → ~S" (subseq all 0 5) (last all 5)))
                          (is (equalp losses-straight losses-resumed) "復元した続きの損失が違う")
                          (is (every #'equalp (nabla:tree-leaves (list model-straight state-straight))
                                     (nabla:tree-leaves (list model-resumed state-resumed)))
                              "復元した続きの最終のモデル・状態が違う"))))))))))))))

(test (tutorial/per-example-grads-average-to-the-batch-gradient :suite :nabla.small)
  "サンプルごとの勾配（vmap (grad 1サンプルの損失)）を軸 0 で足して n で割ったものが、バッチ全体の損失
\(サンプルごとの損失の平均) の勾配と一致する。各葉の形は (n ...パラメータの形)。f32 の総和の順序が
違うぶん、許容誤差は rtol 1e-4 / atol 1e-6。"
  (let* ((n 32)
         (model (%tut "INIT-MLP" (nabla:prng-key 1) 2 16 2))
         (batch-grad (nabla:grad (nabla:with-tracing (model x y)
                                   (- (/ (nabla:reduce-sum (* y (log-softmax (call model x))))
                                         (nabla:reduce-sum y)))))))
    (destructuring-bind (x y) (nabla.data:with-data-iterator (iterator (%tutorial-loader))
                                (nabla.data:next-batch iterator))
      (let ((per-example (%tut "PER-EXAMPLE-GRADS" model x y))
            (whole (funcall batch-grad model x y)))
        (loop for p in (nabla:tree-leaves per-example)
              for w in (nabla:tree-leaves whole)
              do (is (equal (cons n (array-dimensions w)) (array-dimensions p)) "per-example の葉の形が違う")
                 (let ((mean (nabla:reduce-sum p :axes '(0))))
                   (dotimes (i (array-total-size mean))
                     (setf (row-major-aref mean i) (/ (row-major-aref mean i) n)))
                   (is (allclose mean w :dtype :f32 :rtol 1d-4 :atol 1d-6)
                       "サンプルごとの勾配の平均が全体の勾配と違う")))))))

(test (tutorial/example-script-runs :suite :nabla.medium)
  "examples/tutorial.lisp をそのまま load すると最後まで動き、3行（最初の損失・30 ステップ後の損失・
復元して続けた 60 ステップ後の損失）を出す。損失は下がっていく。"
  (when (%layers-require-iree-or-skip)
    (let* ((nabla:*compile-cache-directory* nil)
           (output (with-output-to-string (*standard-output*)
                     (load (%tutorial-example-path))))
           (numbers (let ((found '()) (start 0))
                      (loop for position = (search "= " output :start2 start)
                            while position
                            do (multiple-value-bind (value end)
                                   (read-from-string output t nil :start (+ position 2))
                                 (push value found)
                                 (setf start end)))
                      (nreverse found))))
      (is (= 3 (length numbers)) "損失が3つ出ない: ~S" output)
      (when (= 3 (length numbers))
        (is (< (second numbers) (first numbers)) "30 ステップで損失が下がらない: ~S" output)
        (is (< (third numbers) (second numbers)) "復元した続きで損失が下がらない: ~S" output)))))
