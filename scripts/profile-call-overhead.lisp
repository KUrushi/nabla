;;;; jit した関数の呼び出し1回の時間を段に分けて測る（issue #190）。
;;;;
;;;; Colab の T4 で iree-cuda の step/jitted-call に約 7 ms の固定費があった
;;;; （docs/phase2-report.md §4.4）。その内訳（to-device / invoke / to-host /
;;;; 解放）を backend ごとに測り、pjrt-cuda や iree-local と比べるためのもの。
;;;; scripts/bench-backends.lisp を先に読み、その統計・レコード・表の関数を使う:
;;;;
;;;;   NABLA_BENCH_BACKEND=iree-cuda sbcl --non-interactive --eval '(require :asdf)' \
;;;;     --load scripts/bench-backends.lisp --load scripts/profile-call-overhead.lisp \
;;;;     --eval '(nabla-bench::profile-main)' > profile.sexp
;;;;   scripts/bench-backends.sh --render-only --out profile.sexp   # 表にする
;;;;
;;;; 環境変数:
;;;;   NABLA_BENCH_BACKEND   iree-local / iree-cuda / pjrt-cpu / pjrt-cuda
;;;;   NABLA_PROFILE_LABEL   表の列名（既定は NABLA_BENCH_BACKEND）。フラグ違いを並べるのに使う
;;;;   NABLA_PROFILE_IREE_FLAGS  IREE のランタイムフラグ（空白区切り。例
;;;;                         "--cuda_allow_inline_execution=true"）。デバイスを作る前に
;;;;                         iree_flags_parse に渡す
;;;;   NABLA_PROFILE_ITERATIONS  各項目の繰り返し（既定 200）
;;;;   NABLA_BENCH_CONFIGS   MLP の設定（既定 small,large）

(in-package #:nabla-bench)

;; PARSE-IREE-FLAGS が cffi: のシンボルを読むので、先に読み込む
(asdf:load-system "cffi")

(defparameter *profile-sizes* '(1 4096 262144)
  "h2d / d2h を測る f32 のベクタの要素数（4 B、16 KiB、1 MiB）。")

(defun release-fn (kind)
  (sym (if (eq kind :iree) "NABLA.IREE" "NABLA.PJRT") "RELEASE-DEVICE-ARRAY"))

(defun parse-iree-flags (string)
  "STRING（空白区切りのフラグ）を iree_flags_parse に渡す。未知のフラグはエラー。"
  (let ((flags (loop with start = 0
                     for space = (position #\Space string :start start)
                     for word = (subseq string start space)
                     when (plusp (length word)) collect word
                     while space do (setf start (1+ space)))))
    (when flags
      (funcall (sym "NABLA.IREE" "ENSURE-RUNTIME-LOADED"))
      (let* ((argv (cons "nabla" flags))
             (argc (length argv))
             (strings (mapcar #'cffi:foreign-string-alloc argv)))
        (cffi:with-foreign-objects ((c-argv :pointer argc) (p-argc :int) (p-argv :pointer))
          (loop for s in strings for i from 0 do (setf (cffi:mem-aref c-argv :pointer i) s))
          (setf (cffi:mem-ref p-argc :int) argc
                (cffi:mem-ref p-argv :pointer) c-argv)
          ;; iree_status_t iree_flags_parse(iree_flags_parse_mode_t mode, int* argc, char*** argv)
          ;; （iree/base/tooling/flags.h）。mode 0 = IREE_FLAGS_PARSE_MODE_DEFAULT
          (let ((status (cffi:foreign-funcall "iree_flags_parse" :uint32 0 :pointer p-argc
                                              :pointer p-argv :pointer)))
            (unless (cffi:null-pointer-p status)
              (error "iree_flags_parse ~S: ~A" flags
                     (funcall (sym "NABLA.IREE" "%STATUS-MESSAGE") status)))))))))

(defun repeat-ms (iterations thunk &key setup cleanup)
  "THUNK の時間を ITERATIONS 回測ってミリ秒のリストを返す。SETUP の値を THUNK に渡し、
THUNK の値を CLEANUP に渡す（SETUP と CLEANUP は時間に入れない）。"
  (loop repeat iterations
        collect (let* ((input (and setup (funcall setup)))
                       (start (now-ms))
                       (output (funcall thunk input))
                       (ms (- (now-ms) start)))
                  (when cleanup (funcall cleanup output))
                  ms)))

(defun compile-text (backend text)
  (nabla-call "BACKEND-LOAD" backend (nabla-call "BACKEND-COMPILE" backend text)))

(defun profile-transfers (label backend kind iterations)
  (let ((release (release-fn kind)))
    (dolist (size *profile-sizes*)
      (let ((array (make-array size :element-type 'single-float :initial-element 1.0)))
        (emit (result-record label "transfer" (format nil "h2d/~D" size)
                             (repeat-ms iterations (lambda (_) (declare (ignore _))
                                                     (nabla-call "TO-DEVICE" array backend))
                                        :cleanup release)))
        (let ((device-array (nabla-call "TO-DEVICE" array backend)))
          (emit (result-record label "transfer" (format nil "d2h/~D" size)
                               (repeat-ms iterations (lambda (_) (declare (ignore _))
                                                       (nabla-call "TO-HOST" device-array)))))
          (funcall release device-array))
        (emit (result-record label "transfer" (format nil "release/~D" size)
                             (repeat-ms iterations release
                                        :setup (lambda () (nabla-call "TO-DEVICE" array backend)))))))))

(defun profile-trivial (label backend kind iterations)
  "何もしない（スカラーに 1 を足す）関数の invoke。"
  (let* ((release (release-fn kind))
         (module (compile-text backend "func.func @main(%a: tensor<f32>) -> tensor<f32> {
  %one = stablehlo.constant dense<1.0> : tensor<f32>
  %r = stablehlo.add %a, %one : tensor<f32>
  return %r : tensor<f32>
}"))
         (input (nabla-call "TO-DEVICE" (make-array '() :element-type 'single-float :initial-element 0.0)
                            backend)))
    (dotimes (i 10) (funcall release (nabla-call "BACKEND-INVOKE" backend module "main" input)))
    (emit (result-record label "trivial" "invoke"
                         (repeat-ms iterations (lambda (_) (declare (ignore _))
                                                 (nabla-call "BACKEND-INVOKE" backend module "main" input))
                                    :cleanup release)))
    (emit (result-record label "trivial" "invoke+d2h"
                         (repeat-ms iterations (lambda (_) (declare (ignore _))
                                                 (let ((out (nabla-call "BACKEND-INVOKE" backend module "main" input)))
                                                   (nabla-call "TO-HOST" out)
                                                   out))
                                    :cleanup release)))
    (funcall release input)
    (nabla-call "BACKEND-UNLOAD" backend module)))

(defun profile-mlp (label config backend kind iterations)
  "MLP の学習ステップ（jit した value-and-grad）の呼び出しを段に分ける。"
  (destructuring-bind (&key n h c &allow-other-keys) (rest (assoc config *configs* :test #'string=))
    (multiple-value-bind (params x y) (config-data config)
      (multiple-value-bind (step jitted)
          (funcall (sym "NABLA-EXAMPLE-MLP" "MAKE-MLP-TRAIN-STEP") :n n :h h :c c :backend backend)
        (declare (ignore step))
        (let* ((release (release-fn kind))
               (args (append params (list x y)))
               (avals (mapcar (sym "NABLA" "%JIT-ARGUMENT-AVAL") args))
               (graph (funcall (sym "NABLA" "%JIT-TRACE") (funcall (sym "NABLA" "%JITTED-FUNCTION-FN") jitted)
                               avals nil nil))
               (module (compile-text backend (nabla-call "EMIT-STABLEHLO" graph)))
               (inputs (mapcar (lambda (a) (nabla-call "TO-DEVICE" a backend)) args)))
          (flet ((invoke (_) (declare (ignore _))
                   (multiple-value-list (apply (sym "NABLA" "BACKEND-INVOKE") backend module "main" inputs)))
                 (release-all (arrays) (mapc release arrays)))
            (dotimes (i 10) (release-all (invoke nil)))
            (dotimes (i 10) (apply jitted args))
            (sb-ext:gc :full t)
            (emit (result-record label config "mlp/jitted-call"
                                 (repeat-ms iterations (lambda (_) (declare (ignore _)) (apply jitted args)))))
            (emit (result-record label config "mlp/h2d-inputs"
                                 (repeat-ms iterations (lambda (_) (declare (ignore _))
                                                         (mapcar (lambda (a) (nabla-call "TO-DEVICE" a backend)) args))
                                            :cleanup #'release-all)))
            (emit (result-record label config "mlp/invoke"
                                 (repeat-ms iterations #'invoke :cleanup #'release-all)))
            (emit (result-record label config "mlp/d2h-outputs"
                                 (repeat-ms iterations (lambda (outs) (mapc (sym "NABLA" "TO-HOST") outs) outs)
                                            :setup (lambda () (invoke nil)) :cleanup #'release-all)))
            (emit (result-record label config "mlp/release-outputs"
                                 (repeat-ms iterations #'release-all :setup (lambda () (invoke nil)))))
            (release-all inputs)
            (nabla-call "BACKEND-UNLOAD" backend module)))))))

(defun profile-main ()
  "環境変数はファイル冒頭を参照。レコードを標準出力に書く。"
  (let* ((name (or (env-var "NABLA_BENCH_BACKEND") (error "NABLA_BENCH_BACKEND is required")))
         (label (or (env-var "NABLA_PROFILE_LABEL") name))
         (kind (if (search "iree" name) :iree :pjrt))
         (configs (parse-list-env "NABLA_BENCH_CONFIGS" '("small" "large")))
         (iterations (parse-integer (or (env-var "NABLA_PROFILE_ITERATIONS") "200"))))
    (when (and (search "cuda" name) (not (probe-file "/dev/nvidiactl")))
      (emit (unmeasured-record label "no NVIDIA GPU on this machine (issue #12)"))
      (return-from profile-main))
    (handler-case
        (progn
          (asdf:load-system (if (eq kind :iree) "nabla/iree" "nabla/pjrt"))
          (setf (symbol-value (find-symbol "*COMPILE-CACHE-DIRECTORY*" "NABLA")) nil)
          (when (and (eq kind :iree) (env-var "NABLA_PROFILE_IREE_FLAGS"))
            (parse-iree-flags (env-var "NABLA_PROFILE_IREE_FLAGS")))
          (let ((backend (make-backend-for name)))
            ;; IREE は最初の to-device でデバイスを作るので、測る前に済ませる
            (funcall (release-fn kind)
                     (nabla-call "TO-DEVICE" (make-array 1 :element-type 'single-float) backend))
            (emit-env label kind)
            (load-mlp-definitions)
            (format *error-output* "~&[profile] ~A: transfers~%" label)
            (profile-transfers label backend kind iterations)
            (format *error-output* "~&[profile] ~A: trivial invoke~%" label)
            (profile-trivial label backend kind iterations)
            (dolist (config configs)
              (format *error-output* "~&[profile] ~A ~A: mlp~%" label config)
              (finish-output *error-output*)
              (profile-mlp label config backend kind iterations))))
      (error (c)
        (emit (unmeasured-record label (one-line (format nil "error: ~A" c))))))))
