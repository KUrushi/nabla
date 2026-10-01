;;;; NABLA_PJRT_HOME の解決、プラグインの dlopen、GetPjrtApi の呼び出し。
;;;;
;;;; プラグインは scripts/fetch-pjrt.sh が NABLA_PJRT_HOME の下に置く
;;;; （cpu/xla_cpu_pjrt.so、cuda/xla_cuda_plugin.so）。ロードは明示的に
;;;; load-plugin を呼んだときだけ行うので、プラグインが無い環境でも
;;;; nabla/pjrt のロード自体は失敗しない。
;;;;
;;;; GetPjrtApi は「PJRT_Api 構造体へのポインタを返すだけ」の関数で、
;;;; LLVM 初期化（シグナルハンドラの登録）は走らないことを CPU プラグインで
;;;; 確かめた。ただし念のため、dlopen（プラグインの静的初期化子が動く）と
;;;; GetPjrtApi は with-all-float-traps-masked で包む。クライアント作成など
;;;; LLVM / XLA を実際に動かす入口は、後続の issue（#85）で
;;;; with-lisp-signal-handlers-preserved も使って包む。
;;;;
;;;; ヘッダは third_party/pjrt/pjrt_c_api.h。構造体の配置は PJRT_Api の先頭
;;;; （struct_size / extension_start / pjrt_api_version）をそこから写した。

(in-package #:nabla.pjrt)

(defparameter *default-pjrt-home-name* ".local/share/nabla/pjrt-0.0.1/"
  "NABLA_PJRT_HOME が未設定のときに使う、$HOME からの相対パス。")

(define-condition pjrt-plugin-not-found (error)
  ((path :initarg :path :reader pjrt-plugin-not-found-path))
  (:report (lambda (condition stream)
             (format stream "PJRT plugin not found: ~A (run scripts/fetch-pjrt.sh or set NABLA_PJRT_HOME)"
                     (pjrt-plugin-not-found-path condition))))
  (:documentation "プラグインの .so が見つからないときに load-plugin が通知する。"))

(defun %ensure-trailing-slash (string)
  (if (and (plusp (length string)) (char= (char string (1- (length string))) #\/))
      string
      (concatenate 'string string "/")))

(defun pjrt-home ()
  "PJRT プラグインのインストール先ディレクトリを pathname で返す。
NABLA_PJRT_HOME 環境変数があればそれを、無ければ
~/.local/share/nabla/pjrt-0.0.1/ を使う。"
  (let ((env (sb-ext:posix-getenv "NABLA_PJRT_HOME")))
    (if (and env (plusp (length env)))
        (pathname (%ensure-trailing-slash env))
        (merge-pathnames *default-pjrt-home-name* (user-homedir-pathname)))))

(defun plugin-path (kind)
  "KIND（:cpu または :cuda）に対応するプラグインの .so のパスを返す。
存在するかどうかは調べない。"
  (merge-pathnames
   (ecase kind
     (:cpu "cpu/xla_cpu_pjrt.so")
     (:cuda "cuda/xla_cuda_plugin.so"))
   (pjrt-home)))

(defun pjrt-available-p (&key (kind :cpu))
  "KIND（:cpu / :cuda、既定 :cpu）のプラグインの .so が NABLA_PJRT_HOME の下に
存在するかを probe-file で調べる。ロードは行わない（副作用がない）。"
  (and (probe-file (plugin-path kind)) t))

(cffi:defcstruct %api-version
  (struct-size :size)
  (extension-start :pointer)
  (major :int)
  (minor :int))

(cffi:defcstruct %api-head
  (struct-size :size)
  (extension-start :pointer)
  (api-version (:struct %api-version)))

(defvar *plugin-apis* (make-hash-table)
  "kind -> ロード済みプラグインの PJRT_Api ポインタ。")

(defvar *plugin-lock* (sb-thread:make-mutex :name "nabla-pjrt-plugin-load")
  "プラグインのロードをプロセスにつき KIND ごとに1回だけにするロック。")

(defun load-plugin (kind)
  "KIND のプラグインを dlopen し、GetPjrtApi を呼んで得た PJRT_Api への
ポインタを返す。プロセスにつき KIND ごとに1回だけロードし、2回目以降は
同じポインタを返す（cffi は同じパスをもう一度ロードすると前のハンドルを
dlclose してしまい、前の API ポインタが無効になるため）。ロードした
プラグインはアンロードしない。プラグインが無ければ
PJRT-PLUGIN-NOT-FOUND を通知する。"
  (sb-thread:with-mutex (*plugin-lock*)
    (or (gethash kind *plugin-apis*)
        (let ((path (probe-file (plugin-path kind))))
          (unless path
            (error 'pjrt-plugin-not-found :path (plugin-path kind)))
          (setf (gethash kind *plugin-apis*)
                (nabla.ffi-support:with-all-float-traps-masked
                  (let* ((library (cffi:load-foreign-library path))
                         (get-api (cffi:foreign-symbol-pointer
                                   "GetPjrtApi"
                                   :library (cffi::foreign-library-name library))))
                    (unless get-api
                      (error "~A does not export GetPjrtApi" path))
                    (cffi:foreign-funcall-pointer get-api () :pointer))))))))

(defun plugin-api-version (api)
  "load-plugin が返した API（PJRT_Api へのポインタ）の PJRT API の版を、
(values major minor) で返す。"
  (let ((version (cffi:foreign-slot-pointer api '(:struct %api-head) 'api-version)))
    (values (cffi:foreign-slot-value version '(:struct %api-version) 'major)
            (cffi:foreign-slot-value version '(:struct %api-version) 'minor))))
