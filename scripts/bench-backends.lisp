;;;; IREE (local) と PJRT (XLA CPU) で、2層 MLP のコンパイル時間と学習ステップ時間を測る（issue #89）。
;;;;
;;;; 直接は呼ばず scripts/bench-backends.sh 経由で使う（backend ごとに別プロセスで測る。
;;;; 同じプロセスに IREE と PJRT を混ぜると、LLVM のシグナルハンドラ（CLAUDE.md の
;;;; with-lisp-signal-handlers-preserved の項）や初回コストが互いに影響するため）。
;;;;
;;;; 出力は1行1レコードの S 式（READ できる plist）。レコードは2種類:
;;;;   (:kind :env :backend B ...)                  測定条件
;;;;   (:kind :result :backend B :config C :metric M :unit "ms" :n N :median X :p10 X :p90 X :min X)
;;;;     未測定のときは :status :unmeasured と :reason を持つ（数値は無い）。
;;;; 時間は全部ミリ秒（浮動小数）。このファイルの上半分（統計・書式・パーサー・表）は
;;;; nabla や IREE / PJRT に依存せず、tests/bench-test.lisp が固定の文字列で検査する。
;;;; 下半分（測定）は MAIN が必要なシステムを asdf:load-system で読む。

(defpackage #:nabla-bench
  (:use #:cl)
  (:export #:summarize #:parse-records #:records-table #:main))
(in-package #:nabla-bench)

;;; --- 統計 ---

(defun percentile (sorted fraction)
  "昇順ソート済みのベクタ SORTED の FRACTION（有理数、0〜1）分位を、隣り合う2点の線形補間で返す。"
  (let* ((n (length sorted))
         (position (* fraction (1- n)))
         (lower (floor position))
         (upper (min (1- n) (1+ lower)))
         (weight (- position lower)))
    (+ (* (- 1 weight) (aref sorted lower)) (* weight (aref sorted upper)))))

(defun summarize (samples)
  "SAMPLES（数のリスト、1個以上）の (:n :median :p10 :p90 :min) の plist。"
  (let ((sorted (sort (coerce samples 'vector) #'<)))
    (list :n (length sorted)
          :median (percentile sorted 1/2)
          :p10 (percentile sorted 1/10)
          :p90 (percentile sorted 9/10)
          :min (aref sorted 0))))

;;; --- レコードの書式とパーサー ---

(defun round-to (x digits)
  (let ((scale (expt 10 digits)))
    (/ (round (* x scale)) (float scale 1d0))))

(defun result-record (backend config metric samples &key (unit "ms"))
  "測定結果のレコード（plist）。SAMPLES はミリ秒のリスト。"
  (let ((summary (summarize samples)))
    (list* :kind :result :backend backend :config config :metric metric :unit unit
           :n (getf summary :n)
           (loop for key in '(:median :p10 :p90 :min)
                 collect key collect (round-to (getf summary key) 4)))))

(defun unmeasured-record (backend reason)
  (list :kind :result :backend backend :config "-" :metric "-" :status :unmeasured :reason reason))

(defun print-record (record &optional (stream *standard-output*))
  "RECORD を1行の S 式として書く（文字列・キーワード・数だけの plist）。"
  (let ((*print-pretty* nil) (*print-readably* nil) (*read-default-float-format* 'double-float))
    (prin1 record stream)
    (terpri stream)))

(defun parse-records (source)
  "SOURCE（文字列またはストリーム）から、1行1レコードの S 式を全部読んで plist のリストで返す。
READ-EVAL は切る。空行と ; で始まる行は読み飛ばす。plist でないものは ERROR。"
  (let ((*read-eval* nil)
        (*package* (find-package '#:nabla-bench))
        (*read-default-float-format* 'double-float))
    (flet ((parse (stream)
             (loop for line = (read-line stream nil nil)
                   while line
                   for trimmed = (string-trim " 	" line)
                   unless (or (zerop (length trimmed)) (char= (char trimmed 0) #\;))
                     collect (let ((record (read-from-string trimmed)))
                               (unless (and (consp record) (keywordp (first record)) (getf record :kind))
                                 (error "bench record without :kind: ~A" trimmed))
                               record))))
      (if (stringp source)
          (with-input-from-string (stream source) (parse stream))
          (parse source)))))

;;; --- 人間向けの表 ---

(defun %unique (list)
  (remove-duplicates list :test #'equal :from-end t))

(defun %cell (record)
  (cond ((null record) "-")
        ((eq (getf record :status) :unmeasured) "未測定")
        ((not (equal (getf record :unit) "ms")) (format nil "~D ~A" (round (getf record :median)) (getf record :unit)))
        ((= (getf record :n) 1) (format nil "~,3F" (getf record :median)))
        (t (format nil "~,3F [~,3F-~,3F]" (getf record :median) (getf record :p10) (getf record :p90)))))

(defun records-table (records)
  "RECORDS（PARSE-RECORDS の結果）から、config ごとに「行 = metric、列 = backend」の
Markdown 表を1つの文字列にして返す。セルは ms で、n > 1 のものは「中央値 [p10-p90]」。
未測定の backend は列の全部が「未測定」になる。"
  (let* ((results (remove-if-not (lambda (r) (eq (getf r :kind) :result)) records))
         (backends (%unique (mapcar (lambda (r) (getf r :backend)) results)))
         (measured (remove-if (lambda (r) (eq (getf r :status) :unmeasured)) results))
         (configs (%unique (mapcar (lambda (r) (getf r :config)) measured))))
    (with-output-to-string (out)
      (dolist (config configs)
        (let ((metrics (%unique (mapcar (lambda (r) (getf r :metric))
                                        (remove-if-not (lambda (r) (equal (getf r :config) config)) measured)))))
          (format out "~%### ~A~%~%| 項目 |~{ ~A |~}~%| --- |~{~* --- |~}~%"
                  (if (equal config "-") "初期化（プロセスにつき1回。ms）" (format nil "~A（ms。n > 1 は 中央値 [p10-p90]）" config))
                  backends backends)
          (dolist (metric metrics)
            (format out "| ~A |" metric)
            (dolist (backend backends)
              (let ((record (find-if (lambda (r) (and (equal (getf r :backend) backend)
                                                      (equal (getf r :config) config)
                                                      (equal (getf r :metric) metric)))
                                     measured))
                    (unmeasured (find-if (lambda (r) (and (equal (getf r :backend) backend)
                                                          (eq (getf r :status) :unmeasured)))
                                         results)))
                (format out " ~A |" (%cell (or record unmeasured)))))
            (terpri out)))))))

;;; ======================================================================
;;; 以下は測定本体（MAIN から呼ばれる。IREE / PJRT / examples/mlp.lisp が要る）
;;; ======================================================================

(defparameter *configs*
  '(("small" :n 16 :d 2 :h 8 :c 2)
    ("medium" :n 256 :d 64 :h 128 :c 10)
    ("large" :n 1024 :d 256 :h 512 :c 10))
  "(名前 :n バッチ :d 入力次元 :h 隠れ層 :c クラス数)。")

(defun now-ms ()
  "単調増加の時計（clock_gettime(CLOCK_MONOTONIC)）のミリ秒。get-internal-real-time は
この SBCL では 4 ms 刻みで、ステップ時間（1 ms 未満）を測れない。"
  (multiple-value-bind (seconds nanoseconds) (sb-unix::clock-gettime 1) ; 1 = CLOCK_MONOTONIC（Linux）
    (+ (* 1000d0 seconds) (/ nanoseconds 1d6))))

(defmacro elapsed-ms (&body body)
  "BODY を評価して (values 経過ミリ秒 BODY の値) を返す。"
  (let ((start (gensym)) (values (gensym)))
    `(let* ((,start (now-ms))
            (,values (multiple-value-list (progn ,@body))))
       (values (- (now-ms) ,start) (first ,values)))))

(defun env-var (name)
  (let ((value (sb-ext:posix-getenv name)))
    (and value (plusp (length value)) value)))

(defun parse-list-env (name default)
  (let ((value (env-var name)))
    (if value
        (loop with start = 0
              for comma = (position #\, value :start start)
              collect (subseq value start comma)
              while comma do (setf start (1+ comma)))
        default)))

(defun cpu-info ()
  "(values CPU 型番 論理コア数)（/proc/cpuinfo）。"
  (let ((model "unknown") (cores 0))
    (ignore-errors
     (with-open-file (stream "/proc/cpuinfo")
       (loop for line = (read-line stream nil nil)
             while line
             do (cond ((and (> (length line) 10) (string= "model name" line :end2 10))
                       (when (equal model "unknown")
                         (setf model (string-trim " 	" (subseq line (1+ (position #\: line)))))))
                      ((and (> (length line) 9) (string= "processor" line :end2 9))
                       (incf cores))))))
    (values model cores)))

(defun lock-value (key)
  "third_party/iree.lock の KEY=値 の値（無ければ NIL）。"
  (ignore-errors
   (with-open-file (stream (asdf:system-relative-pathname "nabla" "third_party/iree.lock"))
     (loop for line = (read-line stream nil nil)
           while line
           when (and (> (length line) (1+ (length key))) (string= key line :end2 (length key))
                     (char= (char line (length key)) #\=))
             return (subseq line (1+ (length key)))))))

(defun sym (package name)
  (symbol-function (or (find-symbol name package) (error "~A::~A not found" package name))))

(defun nabla-call (name &rest args)
  "パッケージ NABLA の関数 NAME を呼ぶ（このファイルを読む時点では nabla がまだ無いので、
nb: の接頭辞は使えない）。"
  (apply (sym "NABLA" name) args))

(defun make-backend-for (name)
  "NAME（iree-local / iree-cuda / pjrt-cpu / pjrt-cuda）の BACKEND とその種類を返す。"
  (cond ((equal name "iree-local") (nabla-call "MAKE-BACKEND" :iree :target :local))
        ((equal name "iree-cuda") (nabla-call "MAKE-BACKEND" :iree :target :cuda))
        ((equal name "pjrt-cpu") (nabla-call "MAKE-BACKEND" :pjrt :target :cpu))
        ((equal name "pjrt-cuda") (nabla-call "MAKE-BACKEND" :pjrt :target :cuda))
        (t (error "unknown backend ~S" name))))

(defun random-f32-array (state dims scale)
  (let ((a (make-array dims :element-type 'single-float)))
    (dotimes (i (array-total-size a) a)
      (setf (row-major-aref a i) (* scale (- (random 2.0 state) 1.0))))))

(defun config-data (config)
  "CONFIG の (values params x y)。x は一様乱数、y は one-hot（クラスは行番号の剰余）。"
  (destructuring-bind (&key n d h c) (rest (assoc config *configs* :test #'string=))
    (let ((state (sb-ext:seed-random-state 42))
          (y (make-array (list n c) :element-type 'single-float :initial-element 0.0)))
      (dotimes (i n) (setf (aref y i (mod i c)) 1.0))
      (values (funcall (sym "NABLA-EXAMPLE-MLP" "INIT-PARAMS") 0 :d d :h h :c c)
              (random-f32-array state (list n d) 1.0)
              y))))

(defun load-mlp-definitions ()
  "examples/mlp.lisp の定義（defun など）だけを評価する。末尾の「100 ステップ学習して表示する」
let フォームは評価しない（学習が初期化の測定に混ざり、別の backend も動かしてしまうため）。
IREE の system は読むが、コンパイラの dlopen は backend を使うまで起きない。"
  (let ((*package* *package*)
        (*read-eval* nil))
    (with-open-file (stream (asdf:system-relative-pathname "nabla" "examples/mlp.lisp"))
      (loop for form = (read stream nil stream)
            until (eq form stream)
            ;; let: 末尾の学習。(require ...) / (asdf:load-system ...): main が backend に合わせて
            ;; 読み済み（pjrt のプロセスに nabla/iree を読み込ませない）。
            unless (and (consp form)
                        (or (eq (first form) 'let)
                            (eq (first form) 'require)
                            (and (symbolp (first form)) (string= (symbol-name (first form)) "LOAD-SYSTEM"))))
              do (eval form)))))

(defun one-line (string)
  "STRING の改行・連続する空白を1つの空白にする（レコードは1行1つなので）。"
  (with-output-to-string (out)
    (let ((space nil))
      (loop for ch across string
            do (if (member ch '(#\Newline #\Return #\Tab #\Space))
                   (setf space t)
                   (progn (when space (write-char #\Space out) (setf space nil))
                          (write-char ch out)))))))

(defun emit (record)
  (print-record record)
  (finish-output))

(defun measure-init (name backend-kind)
  "プロセスにつき1回だけ起きる初期化の時間。BACKEND を作って返す。"
  (let (backend)
    (ecase backend-kind
      (:iree
       (emit (result-record name "-" "init/compiler-load" (list (elapsed-ms (funcall (sym "NABLA.IREE" "ENSURE-COMPILER-LOADED"))))))
       (setf backend (make-backend-for name))
       ;; device は最初の to-device で作られる（遅延）。
       (emit (result-record name "-" "init/device-create"
                            (list (elapsed-ms
                                   (funcall (sym "NABLA.IREE" "RELEASE-DEVICE-ARRAY")
                                            (nabla-call "TO-DEVICE" (make-array 1 :element-type 'single-float) backend)))))))
      (:pjrt
       (let ((target (if (search "cuda" name) :cuda :cpu)))
         (emit (result-record name "-" "init/plugin-load" (list (elapsed-ms (funcall (sym "NABLA.PJRT" "LOAD-PLUGIN") target)))))
         (setf backend nil)
         (emit (result-record name "-" "init/client-create" (list (elapsed-ms (setf backend (make-backend-for name))))))
         (emit (result-record name "-" "init/plugin-sha256" (list (elapsed-ms (funcall (sym "NABLA.PJRT" "%PLUGIN-SHA256") target))))))))
    backend))

(defun measure-compile (name config backend reps)
  "CONFIG の MLP について、jit パイプラインの段ごとの時間（REPS 回）と、新しい jit の初回呼び出し。
ディスクキャッシュは呼び出し側が無効にしている。"
  (let ((make-step (sym "NABLA-EXAMPLE-MLP" "MAKE-MLP-TRAIN-STEP"))
        (destructuring (rest (assoc config *configs* :test #'string=))))
    (multiple-value-bind (params x y) (config-data config)
      (let* ((args (append params (list x y)))
             (stage (make-hash-table :test 'equal)))
        (flet ((note (metric ms) (push ms (gethash metric stage))))
          (dotimes (rep reps)
            ;; 段ごと（trace / emit / backend-compile / backend-load）。新しい jit した関数から取る。
            (multiple-value-bind (step jitted)
                (funcall make-step :n (getf destructuring :n) :h (getf destructuring :h)
                                   :c (getf destructuring :c) :backend backend)
              (declare (ignore step))
              (let* ((fn (funcall (sym "NABLA" "%JITTED-FUNCTION-FN") jitted))
                     (avals (mapcar (sym "NABLA" "%JIT-ARGUMENT-AVAL") args))
                     graph text octets module)
                (note "stage/trace" (elapsed-ms (setf graph (funcall (sym "NABLA" "%JIT-TRACE") fn avals nil nil))))
                (note "stage/emit-stablehlo" (elapsed-ms (setf text (nabla-call "EMIT-STABLEHLO" graph))))
                (note "stage/backend-compile" (elapsed-ms (setf octets (nabla-call "BACKEND-COMPILE" backend text))))
                (note "stage/backend-load" (elapsed-ms (setf module (nabla-call "BACKEND-LOAD" backend octets))))
                (nabla-call "BACKEND-UNLOAD" backend module)
                (when (zerop rep)
                  (emit (result-record name config "size/stablehlo-text" (list (float (length text) 1d0)) :unit "chars"))
                  (emit (result-record name config "size/compiled" (list (float (length octets) 1d0)) :unit "bytes")))))
            ;; 新しい jit の初回呼び出し（trace + emit + compile + load + to-device + invoke + to-host）と2回目。
            (multiple-value-bind (step jitted)
                (funcall make-step :n (getf destructuring :n) :h (getf destructuring :h)
                                   :c (getf destructuring :c) :backend backend)
              (declare (ignore step))
              (note "jit/first-call" (elapsed-ms (apply jitted args)))
              (note "jit/second-call" (elapsed-ms (apply jitted args))))))
        (dolist (metric '("stage/trace" "stage/emit-stablehlo" "stage/backend-compile" "stage/backend-load"
                          "jit/first-call" "jit/second-call"))
          (emit (result-record name config metric (reverse (gethash metric stage)))))))))

(defun measure-steps (name config backend steps warmup)
  "学習ステップ（step-fn 全体）と、jitted の呼び出しだけの時間を STEPS 回ずつ測る。"
  (destructuring-bind (&key n h c &allow-other-keys) (rest (assoc config *configs* :test #'string=))
    (multiple-value-bind (params x y) (config-data config)
      (multiple-value-bind (step jitted)
          (funcall (sym "NABLA-EXAMPLE-MLP" "MAKE-MLP-TRAIN-STEP") :n n :h h :c c :backend backend)
        (let ((full nil) (jit-only nil) (current params))
          (dotimes (i warmup)
            (multiple-value-bind (loss new) (funcall step current x y)
              (declare (ignore loss))
              (setf current new)))
          (sb-ext:gc :full t)
          (dotimes (i steps)
            (push (elapsed-ms (multiple-value-bind (loss new) (funcall step current x y)
                                (declare (ignore loss))
                                (setf current new)))
                  full))
          (let ((args (append params (list x y))))
            (dotimes (i warmup) (apply jitted args))
            (sb-ext:gc :full t)
            (dotimes (i steps)
              (push (elapsed-ms (apply jitted args)) jit-only)))
          (emit (result-record name config "step/full" (reverse full)))
          (emit (result-record name config "step/jitted-call" (reverse jit-only))))))))

(defun emit-env (name backend-kind)
  (multiple-value-bind (model cores) (cpu-info)
    (emit (append
           (list :kind :env :backend name
                 :date (multiple-value-bind (s m h d mo y) (get-decoded-time)
                         (declare (ignore s))
                         (format nil "~D-~2,'0D-~2,'0D ~2,'0D:~2,'0D" y mo d h m))
                 :cpu model :cores cores
                 :sbcl (lisp-implementation-version)
                 :disk-cache "disabled (nabla:*compile-cache-directory* = NIL)"
                 :threads (format nil "nabla sets no thread count (~A default); OMP_NUM_THREADS=~A XLA_FLAGS=~A"
                                  (if (eq backend-kind :iree) "IREE local-task" "XLA CPU")
                                  (or (env-var "OMP_NUM_THREADS") "unset") (or (env-var "XLA_FLAGS") "unset")))
           (ecase backend-kind
             (:iree (list :iree-commit (or (lock-value "commit") "unknown")
                          :iree-compiler-revision (funcall (sym "NABLA.IREE" "COMPILER-REVISION"))))
             (:pjrt (let ((target (if (search "cuda" name) :cuda :cpu)))
                      (list :pjrt-plugin (namestring (funcall (sym "NABLA.PJRT" "PLUGIN-PATH") target))
                            :pjrt-plugin-sha256 (funcall (sym "NABLA.PJRT" "%PLUGIN-SHA256") target)
                            :pjrt-api (multiple-value-bind (major minor)
                                          (funcall (sym "NABLA.PJRT" "PLUGIN-API-VERSION")
                                                   (funcall (sym "NABLA.PJRT" "LOAD-PLUGIN") target))
                                        (format nil "~D.~D" major minor))))))))))

(defun main ()
  "環境変数で設定する: NABLA_BENCH_BACKEND（iree-local / iree-cuda / pjrt-cpu / pjrt-cuda）、
NABLA_BENCH_CONFIGS（既定 small,medium,large）、NABLA_BENCH_STEPS（既定 200）、
NABLA_BENCH_WARMUP（既定 20）、NABLA_BENCH_REPS（コンパイル時間の繰り返し、既定 3）。
レコードを標準出力に書く。"
  (let* ((name (or (env-var "NABLA_BENCH_BACKEND") (error "NABLA_BENCH_BACKEND is required")))
         (kind (if (search "iree" name) :iree :pjrt))
         (configs (parse-list-env "NABLA_BENCH_CONFIGS" (mapcar #'first *configs*)))
         (steps (parse-integer (or (env-var "NABLA_BENCH_STEPS") "200")))
         (warmup (parse-integer (or (env-var "NABLA_BENCH_WARMUP") "20")))
         (reps (parse-integer (or (env-var "NABLA_BENCH_REPS") "3"))))
    (when (and (search "cuda" name) (not (probe-file "/dev/nvidiactl")))
      (emit (unmeasured-record name "no NVIDIA GPU on this machine (issue #12)"))
      (return-from main))
    (asdf:load-system (if (eq kind :iree) "nabla/iree" "nabla/pjrt"))
    (progn
      ;; ディスクキャッシュを無効にする（プロセスの最後まで）。
      (setf (symbol-value (find-symbol "*COMPILE-CACHE-DIRECTORY*" "NABLA")) nil)
      (handler-case
          (progn
            (let ((backend (measure-init name kind)))
              (emit-env name kind)
              (load-mlp-definitions)
              (dolist (config configs)
                (measure-compile name config backend reps)
                (measure-steps name config backend steps warmup))))
        (error (c)
          (emit (unmeasured-record name (one-line (format nil "error: ~A" c)))))))))
