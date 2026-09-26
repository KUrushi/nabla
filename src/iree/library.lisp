;;;; NABLA_IREE_HOME の解決と、IREE の共有ライブラリの遅延ロード。
;;;;
;;;; define-foreign-library はトップレベルで書いてよいが、実際に
;;;; load-foreign-library するのは ensure-compiler-loaded / (後続の)
;;;; ensure-runtime-loaded を最初に呼んだときだけにする。こうすることで、
;;;; 共有ライブラリが1つも無い環境でも nabla/iree のロード自体は失敗しない。

(in-package #:nabla.iree)

(defparameter *default-iree-home-name* ".local/share/nabla/iree-3.11.0/"
  "NABLA_IREE_HOME が未設定のときに使う、$HOME からの相対パス。")

(defun %ensure-trailing-slash (string)
  (if (and (plusp (length string)) (char= (char string (1- (length string))) #\/))
      string
      (concatenate 'string string "/")))

(defun iree-home ()
  "IREE のインストール先ディレクトリを pathname で返す。
NABLA_IREE_HOME 環境変数があればそれを、無ければ
~/.local/share/nabla/iree-3.11.0/ を使う。"
  (let ((env (sb-ext:posix-getenv "NABLA_IREE_HOME")))
    (if (and env (plusp (length env)))
        (pathname (%ensure-trailing-slash env))
        (merge-pathnames *default-iree-home-name* (user-homedir-pathname)))))

(defun %library-path (home library)
  "HOME（iree-home の返り値のようなディレクトリ pathname）配下の、
LIBRARY（:compiler または :runtime）に対応する共有ライブラリのパスを返す。"
  (merge-pathnames
   (ecase library
     (:compiler "lib/libIREECompiler.so")
     (:runtime "lib/libnabla_iree_runtime.so"))
   home))

(defun iree-available-p (&key (library :both))
  "LIBRARY（:compiler / :runtime / :both、既定 :both）に対応する共有
ライブラリが NABLA_IREE_HOME の下に存在するかを probe-file で調べる。
ライブラリのロードは行わない（副作用がない）。"
  (let ((home (iree-home)))
    (ecase library
      (:compiler (and (probe-file (%library-path home :compiler)) t))
      (:runtime (and (probe-file (%library-path home :runtime)) t))
      (:both (and (probe-file (%library-path home :compiler))
                  (probe-file (%library-path home :runtime))
                  t)))))

(cffi:define-foreign-library nabla-iree-compiler
  (:unix "libIREECompiler.so"))

(defvar *compiler-load-lock* (sb-thread:make-mutex :name "nabla-iree-compiler-load")
  "ensure-compiler-loaded を複数スレッドから呼んでも、ロードとグローバル
初期化がちょうど1回だけ起きるようにするロック。")

(defvar *compiler-loaded-p* nil
  "ireeCompilerGlobalInitialize まで済んでいれば真。")

(defun ensure-compiler-loaded ()
  "libIREECompiler.so を（まだなら）ロードし、ireeCompilerGlobalInitialize
をプロセスにつき1回だけ呼ぶ。共有ライブラリが見つからなければ
IREE-LIBRARY-NOT-FOUND を signal する。

このマシンに入っている cffi（apt の cl-cffi 0.24.1）の load-foreign-library
は、define-foreign-library で名前を付けたライブラリ（今回の
nabla-iree-compiler のようなシンボル）に対しては、呼び出し時に渡した
:search-path キーワード引数を見ない（cffi::%do-load-foreign-library の
symbol 分岐が (foreign-library-search-path lib) だけを見て、渡された
search-path 引数を無視するため。CFFI 自体のこの版の挙動で、nabla のバグ
ではない）。そのため、探索先を確実に効かせるにはグローバルな
cffi:*foreign-library-directories* に push しておく必要がある。"
  (sb-thread:with-mutex (*compiler-load-lock*)
    (unless *compiler-loaded-p*
      (let ((home (iree-home)))
        (pushnew (merge-pathnames "lib/" home) cffi:*foreign-library-directories*
                  :test #'equal)
        (handler-case
            (cffi:load-foreign-library 'nabla-iree-compiler)
          (cffi:load-foreign-library-error ()
            (error 'iree-library-not-found
                   :path (%library-path home :compiler)
                   :home home
                   :library :compiler))))
      (%compiler-global-initialize)
      (setf *compiler-loaded-p* t)))
  (values))
