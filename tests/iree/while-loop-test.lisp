;;;; while-loop プリミティブの medium テスト（issue #131）。
;;;;
;;;; stablehlo.while を IREE の local backend でコンパイル・実行した結果が、
;;;; eager（eval-graph）の結果と一致することを確かめる。limit は 0〜6（0回で
;;;; 終わる場合を含む）。graph のコンパイルは1回で、PBT の各試行は module を再利用する。

(in-package #:nabla.iree.tests)

(defun %wl-iree-graph ()
  "limit（cond が閉包で捕まえる）・step（body が閉包で捕まえる）・x を引数に取る while-loop。"
  (nb::trace-to-graph
   (nb:with-tracing (limit step x)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (< (first c) limit))
                    (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (* (second c) 0.5) step)))
                    (list (nb::%scalar-array 0.0 :f32) x))))
       (values (first result) (second result))))
   (list (nb:make-aval '() :f32) (nb:make-aval '(3 5) :f32) (nb:make-aval '(3 5) :f32))))

(defun %wl-iree-matches-eager-p (backend module graph seed)
  (let* ((limit (nb::%scalar-array (float (mod seed 7) 1.0) :f32))
         (step (make-random-array (make-array-spec '(3 5) :f32) :seed seed))
         (x (make-random-array (make-array-spec '(3 5) :f32) :seed (1+ seed)))
         (arrays (list limit step x))
         (device-arrays nil)
         (results nil))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a) (to-device a backend :dtype :f32)) arrays))
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(define-iree-test while-loop/iree-matches-eager
    "stablehlo.while を IREE でコンパイル・実行した結果は、eager の while-loop と一致する
（0回で終わる場合、閉包で捕まえた値を含む）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (graph (%wl-iree-graph))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                       (lambda (seed) (%wl-iree-matches-eager-p backend module graph seed))
                       :regression-id while-loop/iree-matches-eager
                       :regression-file (regression-path "iree-while-loop-matches-eager"
                                                         :package "NABLA.IREE.TESTS"))
             "IREE の実行結果が eager の while-loop と一致しなかった")
      (nabla:backend-unload backend module))))

;;; ---- :i1 の carry。IREE 3.11 のコンパイラは、比較から作ったフラグを carry にした
;;; while の結果が関数の戻り値になるとプロセスごと落ちる（src/while-loop.lisp 冒頭の注）。
;;; 戻り値にしない :i1 の carry は動くので、それを確かめる。落ちる形は子プロセスで守る。

(defun %wl-iree-i1-graph (return-flag)
  "n（f32 の rank 0）から、フラグ (n < 5) を :i1 の carry にした while-loop。
RETURN-FLAG が真ならフラグも関数の戻り値にする（IREE 3.11 でクラッシュしていた形）。"
  (nb::trace-to-graph
   (nb:with-tracing (n)
     (let ((result (nb:while-loop
                    (nb:with-tracing (c) (second c))
                    (nb:with-tracing (c) (list (+ (first c) 1.0) (< (+ (first c) 1.0) 5.0)))
                    (list n (< n 5.0)))))
       (if return-flag
           (values (first result) (second result))
           (first result))))
   (list (nb:make-aval '() :f32))))

(defun %wl-iree-i1-matches-eager-p (backend module graph seed)
  (let* ((n (nb::%scalar-array (float (mod seed 9) 1.0) :f32))
         (device-arrays (list (to-device n backend :dtype :f32)))
         (results nil))
    (unwind-protect
         (progn
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (nb:eval-graph graph n))))
             (and (= (length results) (length expected))
                  (every (lambda (r e)
                           (equalp (to-host r) e))
                         results expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(defmacro %with-wl-iree-i1-check ((return-flag name) message)
  `(let* ((backend (nabla:find-backend :iree))
          (graph (%wl-iree-i1-graph ,return-flag))
          (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
     (unwind-protect
          (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                        (lambda (seed) (%wl-iree-i1-matches-eager-p backend module graph seed))
                        :regression-id ,(intern (string-upcase (format nil "while-loop/iree-~A" name)))
                        :regression-file (regression-path ,(format nil "iree-while-loop-~A" name)
                                                          :package "NABLA.IREE.TESTS"))
              ,message)
       (nabla:backend-unload backend module))))

(define-iree-test while-loop/iree-i1-carry-not-returned-matches-eager
    "戻り値にしない :i1 の carry（ループの継続フラグ）でも、IREE の結果は eager と一致する。"
  (skip-unless-iree :library :both)
  (%with-wl-iree-i1-check (nil "i1-not-returned") "i1 carry を返さない while の IREE 結果が eager と一致しなかった"))

(defparameter *wl-iree-known-crash-text*
  "module {
  func.func @main(%0: tensor<f32>) -> (tensor<f32>, tensor<i1>) {
    %1 = stablehlo.constant dense<5.0> : tensor<f32>
    %2 = stablehlo.compare LT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3, %4 = \"stablehlo.while\"(%0, %2) ({
      ^bb0(%a: tensor<f32>, %b: tensor<i1>):
      stablehlo.return %b : tensor<i1>
    }, {
      ^bb0(%a: tensor<f32>, %b: tensor<i1>):
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %five = stablehlo.constant dense<5.0> : tensor<f32>
      %n = stablehlo.add %a, %one : tensor<f32>
      %f = stablehlo.compare LT, %n, %five : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %n, %f : tensor<f32>, tensor<i1>
    }) : (tensor<f32>, tensor<i1>) -> (tensor<f32>, tensor<i1>)
    func.return %3, %4 : tensor<f32>, tensor<i1>
  }
}
"
  "IREE 3.11 のコンパイラが落ちる StableHLO: 比較から作った :i1 のフラグが carry で、
その while の結果が関数の戻り値になる。")

(defun %wl-iree-compile-in-child (text &key (times 1))
  "TEXT を真っさらな子 SBCL で IREE のコンパイルまで行い（TIMES 回。ディスクキャッシュは
切るので毎回コンパイルする）、(VALUES 終了コード 出力) を返す。全部のコンパイルが最後まで
行けば出力に COMPILED が出る。"
  (let* ((forms (list "(require :asdf)"
                      "(asdf:load-system \"nabla/iree\")"
                      (format nil "(let* ((b (nabla:find-backend :iree)) (text ~S) (nabla:*compile-cache-directory* nil)) (dotimes (i ~D) (nabla:backend-compile b text)) (format t \"COMPILED~~%\") (sb-ext:exit :code 0))"
                              text times)))
         (args (list* "--non-interactive" "--disable-debugger"
                      (loop for form in forms append (list "--eval" form))))
         (env (append (%forward-env-vars (list* "NABLA_IREE_HOME" *child-sbcl-forwarded-env-vars*))
                      (list (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))
         (output (make-string-output-stream))
         (process (sb-ext:run-program "sbcl" args :search t :environment env
                                                   :output output :error output)))
    (values (sb-ext:process-exit-code process) (get-output-stream-string output))))

(define-iree-test while-loop/iree-known-limitation-i1-carry-returned-crashes-the-compiler
    "既知の制限の守り: 比較から作った :i1 のフラグを carry にした while の結果を関数の戻り値に
すると、IREE 3.11 のコンパイラが（子プロセスごと）落ちる。このテストが失敗したら IREE の
バグが直っているので、src/while-loop.lisp の「既知の制限」・while-loop の docstring・
docs/stablehlo-ops.md の注意書きとこのテストを消す。"
  (skip-unless-iree :library :both)
  (multiple-value-bind (code output) (%wl-iree-compile-in-child *wl-iree-known-crash-text*)
    (is (not (and (eql code 0) (search "COMPILED" output)))
        "IREE がクラッシュせずにコンパイルできた（バグが直った？）: ~A" output)))

;;; ---- 定数の carry の初期値を持つ while（IREE 3.11 のコンパイラのもう1つのバグ。
;;; :i1 の carry のバグとは別。docs/stablehlo-ops.md の制御構造の節） ----

(defparameter *wl-iree-const-carry-crash-text*
  "module {
  func.func @main(%0: tensor<f32>, %1: tensor<3xf32>, %2: tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>) {
    %5 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.constant dense<0.0> : tensor<f32>
    %8, %9, %10, %13, %14 = \"stablehlo.while\"(%5, %2, %6, %0, %1) ({
      ^bb0(%s1_0: tensor<f32>, %s1_1: tensor<3xf32>, %s1_2: tensor<f32>, %s1_5: tensor<f32>, %s1_6: tensor<3xf32>):
      %s1_8 = stablehlo.compare LT, %s1_0, %s1_5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %s1_8 : tensor<i1>
    }, {
      ^bb0(%s2_0: tensor<f32>, %s2_1: tensor<3xf32>, %s2_2: tensor<f32>, %s2_5: tensor<f32>, %s2_6: tensor<3xf32>):
      %s2_8 = stablehlo.constant dense<1.0> : tensor<f32>
      %s2_9 = stablehlo.constant dense<0.5> : tensor<f32>
      %s2_10 = stablehlo.constant dense<0.1> : tensor<f32>
      %s2_11 = stablehlo.constant dense<0.9> : tensor<f32>
      %s2_14 = stablehlo.add %s2_0, %s2_8 : tensor<f32>
      %s2_15 = stablehlo.broadcast_in_dim %s2_9, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %s2_16 = stablehlo.multiply %s2_1, %s2_15 : tensor<3xf32>
      %s2_18 = stablehlo.broadcast_in_dim %s2_10, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %s2_19 = stablehlo.multiply %s2_6, %s2_18 : tensor<3xf32>
      %s2_21 = stablehlo.add %s2_16, %s2_19 : tensor<3xf32>
      %s2_23 = stablehlo.multiply %s2_2, %s2_11 : tensor<f32>
      %s2_31 = stablehlo.add %s2_23, %s2_11 : tensor<f32>
      stablehlo.return %s2_14, %s2_21, %s2_31, %s2_5, %s2_6 : tensor<f32>, tensor<3xf32>, tensor<f32>, tensor<f32>, tensor<3xf32>
    }) : (tensor<f32>, tensor<3xf32>, tensor<f32>, tensor<f32>, tensor<3xf32>) -> (tensor<f32>, tensor<3xf32>, tensor<f32>, tensor<f32>, tensor<3xf32>)
    func.return %10, %9 : tensor<f32>, tensor<3xf32>
  }
}
"
  "IREE 3.11 のコンパイラが（ほぼ毎回）落ちる StableHLO: while のオペランドのうち、cond を
駆動する carry（カウンタ）が stablehlo.constant で初期化され、ほかに carry が 2 つ以上あり、
そのうち少なくとも 1 つが rank 1 以上。")

(define-iree-test while-loop/iree-known-bug-constant-carry-crashes-the-compiler
    "既知のバグの守り: 定数で初期化したカウンタ carry を持つ while をそのまま渡すと、IREE 3.11 の
コンパイラが（子プロセスごと）落ちる（AffinityAnalysis の walkTransitiveUses。数回のうちに
落ちる）。このテストが失敗したら IREE のバグが直っているので、src/while-loop.lisp の
%WHILE-BARRIER-LINES と docs/stablehlo-ops.md の注意書きとこのテストを消す。"
  (skip-unless-iree :library :both)
  (multiple-value-bind (code output) (%wl-iree-compile-in-child *wl-iree-const-carry-crash-text* :times 10)
    (is (not (and (eql code 0) (search "COMPILED" output)))
        "IREE がクラッシュせずに 10 回コンパイルできた（バグが直った？）: ~A" output)))

(define-iree-test while-loop/iree-constant-carry-with-barrier-compiles-repeatedly
    "nabla が出す StableHLO は、定数のオペランドを optimization_barrier に通すので、上のバグの形
（定数のカウンタ + 引数の carry 2 つ + rank 1 の carry）の while でも、IREE が 10 回続けて
クラッシュせずにコンパイルする。"
  (skip-unless-iree :library :both)
  (let* ((graph (nb::trace-to-graph
                 (nb:with-tracing (limit w x)
                   (let ((result (nb:while-loop
                                  (nb:with-tracing (c) (< (first c) limit))
                                  (nb:with-tracing (c)
                                    (list (+ (first c) 1.0)
                                          (+ (* (second c) 0.5) (* w 0.1))
                                          (+ (* (third c) 0.9) 0.9)))
                                  (list (nb::%scalar-array 0.0 :f32) x (nb::%scalar-array 0.0 :f32)))))
                     (values (third result) (second result))))
                 (list (nb:make-aval '() :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))
         (text (nb:emit-stablehlo graph)))
    (is (search "stablehlo.optimization_barrier" text))
    (multiple-value-bind (code output) (%wl-iree-compile-in-child text :times 10)
      (is (and (eql code 0) (search "COMPILED" output))
          "barrier 付きの while が IREE でクラッシュした: ~A" output))))

(define-iree-test while-loop/iree-sibling-loops-with-constant-inits-match-eager
    "cond と body が同じで、init が同じ閉包の定数（カウンタ 0 と rank 1 の定数）の while-loop を
1つのモジュールに2つ並べても、IREE と eager で一致する。定数を通す barrier が CSE で1つに
まとめられると、2つのループが同じ SSA 値から始まる（issue #179。直す前でも再現しないかもしれない。
本当の守りは small の while-loop/emits-a-distinct-salt-for-each-constant-barrier）。"
  (skip-unless-iree :library :both)
  (let* ((counter (nb::%scalar-array 0.0 :f32))
         (vec (make-random-array (make-array-spec '(3) :f32) :seed 5))
         (cond-fn (nb:with-tracing (c) (< (first c) 3.0)))
         (body-fn (nb:with-tracing (c)
                    (list (+ (first c) 1.0) (+ (* (second c) 0.5) (third c)) (third c))))
         (graph (nb::trace-to-graph
                 (nb:with-tracing (x)
                   (let ((a (nb:while-loop cond-fn body-fn (list counter vec x)))
                         (b (nb:while-loop cond-fn body-fn (list counter vec (* x 2.0)))))
                     (values (first a) (second a) (first b) (second b))))
                 (list (nb:make-aval '(3) :f32))))
         (backend (nabla:find-backend :iree))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
         (x (make-random-array (make-array-spec '(3) :f32) :seed 6))
         (input nil)
         (results nil))
    (unwind-protect
         (progn
           (setf input (to-device x backend :dtype :f32))
           (setf results (multiple-value-list (nabla:backend-invoke backend module "main" input)))
           (let ((expected (multiple-value-list (nb:eval-graph graph x))))
             (is (= (length expected) (length results)))
             (is (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected)
                 "IREE の並んだ while-loop の結果が eager と一致しなかった")))
      (dolist (r results) (release-device-array r))
      (when input (release-device-array input))
      (nabla:backend-unload backend module))))
