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
;;;;   NABLA_PROFILE_IREE_COMPILE_FLAGS  IREE のコンパイラに足すフラグ（空白区切り。例
;;;;                         "--iree-opt-level=O3"）。ops と mlp/invoke（の手でコンパイルする
;;;;                         module）にだけ効き、mlp/jitted-call には効かない
;;;;   NABLA_PROFILE_ITERATIONS  各項目の繰り返し（既定 200）
;;;;   NABLA_PROFILE_SECTIONS  測る段（既定 transfers,trivial,ops,mlp）。ops は large の MLP の
;;;;                         演算を1つずつ invoke する（issue #193、#265）
;;;;   NABLA_BENCH_CONFIGS   MLP の設定（既定 small,large）

(in-package #:nabla-bench)

;; PARSE-IREE-FLAGS が cffi: のシンボルを読むので、先に読み込む
(asdf:load-system "cffi")

(defparameter *profile-sizes* '(1 4096 262144)
  "h2d / d2h を測る f32 のベクタの要素数（4 B、16 KiB、1 MiB）。")

(defun release-fn (kind)
  (sym (if (eq kind :iree) "NABLA.IREE" "NABLA.PJRT") "RELEASE-DEVICE-ARRAY"))

(defun split-words (string)
  (loop with start = 0
        for space = (position #\Space string :start start)
        for word = (subseq string start space)
        when (plusp (length word)) collect word
        while space do (setf start (1+ space))))

(defun parse-iree-flags (string)
  "STRING（空白区切りのフラグ）を iree_flags_parse に渡す。未知のフラグはエラー。"
  (let ((flags (split-words string)))
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

(defvar *extra-compile-flags* nil
  "NABLA_PROFILE_IREE_COMPILE_FLAGS を空白で切ったもの。COMPILE-TEXT が IREE の既定のフラグの後ろに足す。")

(defun compile-text (backend text)
  "TEXT をコンパイルして読み込む。*EXTRA-COMPILE-FLAGS* があれば（IREE のみ）、nabla.iree の
COMPILE-FLAGS の後ろに足して COMPILE-STABLEHLO を直接呼ぶ（jit の経路は通らない）。"
  (nabla-call "BACKEND-LOAD" backend
              (if *extra-compile-flags*
                  (funcall (sym "NABLA.IREE" "COMPILE-STABLEHLO") text
                           :flags (append (funcall (sym "NABLA.IREE" "COMPILE-FLAGS")
                                                   (nabla-call "BACKEND-TARGET" backend))
                                          *extra-compile-flags*))
                  (nabla-call "BACKEND-COMPILE" backend text))))

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

(defun %dot-op (lhs rhs contracting out)
  "(LHS) と (RHS) の 2 次元の f32 の dot_general の本体。CONTRACTING は \"[1] x [0]\" の形。"
  (format nil "%r = stablehlo.dot_general %a0, %a1, contracting_dims = ~A : (tensor<~{~D~^x~}xf32>, tensor<~{~D~^x~}xf32>) -> tensor<~{~D~^x~}xf32>
  return %r : tensor<~{~D~^x~}xf32>" contracting lhs rhs out out))

(defun %transpose-dot-op (lhs rhs which out &key barrier)
  "WHICH（:lhs か :rhs）を stablehlo.transpose で並べ替えてから、縮約軸が lhs の最後・rhs の
先頭にある正規形の dot_general にする本体（issue #265）。BARRIER が真なら、並べ替えた配列を
optimization_barrier に通して、コンパイラが transpose を dot に畳み込めないようにする。"
  (let* ((moved (reverse (if (eq which :lhs) lhs rhs)))
         (lhs* (if (eq which :lhs) moved lhs))
         (rhs* (if (eq which :rhs) moved rhs))
         (operand (if barrier "%b" "%t")))
    (format nil "%t = stablehlo.transpose %a~D, dims = [1, 0] : (tensor<~{~D~^x~}xf32>) -> tensor<~{~D~^x~}xf32>
  ~:[~*~;%b = stablehlo.optimization_barrier %t : tensor<~{~D~^x~}xf32>
  ~]%r = stablehlo.dot_general ~:[%a0, ~A~;~A, %a1~], contracting_dims = [1] x [0] : (tensor<~{~D~^x~}xf32>, tensor<~{~D~^x~}xf32>) -> tensor<~{~D~^x~}xf32>
  return %r : tensor<~{~D~^x~}xf32>"
            (if (eq which :lhs) 0 1) (if (eq which :lhs) lhs rhs) moved
            barrier moved
            (eq which :lhs) operand lhs* rhs* out out)))

(defparameter *profile-ops*
  ;; (名前 入力の形のリスト StableHLO の本体)。large の MLP（N1024 D256 H512 C10）の
  ;; 学習ステップに出てくる演算を1つずつ切り出したもの（issue #193）と、縮約軸が正規形
  ;; （lhs の最後・rhs の先頭）でない dot_general を transpose + 正規形の dot にした形（issue #265）。
  ;; dg[0]x[0] は dot の転置ルールが重みの勾配に作る形、dg[1]x[1] は入力の勾配に作る形。
  `(("op/dot-1024x256x512" ((1024 256) (256 512)) ,(%dot-op '(1024 256) '(256 512) "[1] x [0]" '(1024 512)))
    ("op/dot-256x1024x512" ((256 1024) (1024 512)) ,(%dot-op '(256 1024) '(1024 512) "[1] x [0]" '(256 512)))
    ("op/dg[0]x[0]-256x1024x512" ((1024 256) (1024 512)) ,(%dot-op '(1024 256) '(1024 512) "[0] x [0]" '(256 512)))
    ("op/transpose+dot-256x1024x512" ((1024 256) (1024 512)) ,(%transpose-dot-op '(1024 256) '(1024 512) :lhs '(256 512)))
    ("op/transpose+barrier+dot-256x1024x512" ((1024 256) (1024 512))
     ,(%transpose-dot-op '(1024 256) '(1024 512) :lhs '(256 512) :barrier t))
    ("op/dot-1024x512x256" ((1024 512) (512 256)) ,(%dot-op '(1024 512) '(512 256) "[1] x [0]" '(1024 256)))
    ("op/dg[1]x[1]-1024x512x256" ((1024 512) (256 512)) ,(%dot-op '(1024 512) '(256 512) "[1] x [1]" '(1024 256)))
    ("op/transpose+dot-1024x512x256" ((1024 512) (256 512)) ,(%transpose-dot-op '(1024 512) '(256 512) :rhs '(1024 256)))
    ("op/transpose+barrier+dot-1024x512x256" ((1024 512) (256 512))
     ,(%transpose-dot-op '(1024 512) '(256 512) :rhs '(1024 256) :barrier t))
    ("op/transpose-1024x256" ((1024 256))
     "%r = stablehlo.transpose %a0, dims = [1, 0] : (tensor<1024x256xf32>) -> tensor<256x1024xf32>
  return %r : tensor<256x1024xf32>")
    ("op/tanh-1024x512" ((1024 512))
     "%r = stablehlo.tanh %a0 : tensor<1024x512xf32>
  return %r : tensor<1024x512xf32>")
    ("op/reduce-sum-1024x512" ((1024 512))
     "%z = stablehlo.constant dense<0.0> : tensor<f32>
  %r = stablehlo.reduce(%a0 init: %z) applies stablehlo.add across dimensions = [1] : (tensor<1024x512xf32>, tensor<f32>) -> tensor<1024xf32>
  return %r : tensor<1024xf32>"))
  "PROFILE-OPS が1つずつ invoke の時間を測る演算。")

(defun op-module-text (shapes body)
  (format nil "func.func @main(~{~A~^, ~}) -> ~A {~%  ~A~%}"
          (loop for shape in shapes for i from 0
                collect (format nil "%a~D: tensor<~{~D~^x~}xf32>" i shape))
          (subseq body (+ (search "return %r : " body) (length "return %r : ")))
          body))

(defun profile-ops (label backend kind iterations)
  "*PROFILE-OPS* の各演算を、入力を送った後の invoke だけで測る（出力の読み出しは含めない）。"
  (let ((release (release-fn kind))
        (state (sb-ext:seed-random-state 7)))
    (dolist (op *profile-ops*)
      (destructuring-bind (name shapes body) op
        (let* ((module (compile-text backend (op-module-text shapes body)))
               (inputs (mapcar (lambda (shape)
                                 (nabla-call "TO-DEVICE" (random-f32-array state shape 1.0) backend))
                               shapes)))
          (flet ((invoke (_) (declare (ignore _))
                   (apply (sym "NABLA" "BACKEND-INVOKE") backend module "main" inputs)))
            (dotimes (i 10) (funcall release (invoke nil)))
            (emit (result-record label "ops" name
                                 (repeat-ms iterations #'invoke :cleanup release))))
          (mapc release inputs)
          (nabla-call "BACKEND-UNLOAD" backend module))))))

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
                               avals nil nil (nabla-call "TREE-STRUCTURE" args)))
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
         (sections (parse-list-env "NABLA_PROFILE_SECTIONS" '("transfers" "trivial" "ops" "mlp")))
         (*extra-compile-flags* (and (eq kind :iree)
                                     (split-words (or (env-var "NABLA_PROFILE_IREE_COMPILE_FLAGS") ""))))
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
            (flet ((section (name thunk)
                     (when (member name sections :test #'string=)
                       (format *error-output* "~&[profile] ~A: ~A~%" label name)
                       (finish-output *error-output*)
                       (funcall thunk))))
              (section "transfers" (lambda () (profile-transfers label backend kind iterations)))
              (section "trivial" (lambda () (profile-trivial label backend kind iterations)))
              (section "ops" (lambda () (profile-ops label backend kind iterations)))
              (section "mlp" (lambda ()
                               (dolist (config configs)
                                 (profile-mlp label config backend kind iterations)))))))
      (error (c)
        (emit (unmeasured-record label (one-line (format nil "error: ~A" c))))))))
