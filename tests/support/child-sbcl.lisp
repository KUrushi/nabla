;;;; 子 SBCL プロセスを立てるテストが共有する、環境の組み立て。
;;;;
;;;; SB-EXT:RUN-PROGRAM は :environment を渡さないと親の環境を引き継ぐ
;;;; （実行環境の CL_SOURCE_REGISTRY に暗黙に依存してしまう）ので、子プロセスを
;;;; 立てるテストは、ここの関数で組み立てた環境を明示的に渡す。
;;;; （issue #79 で tests/iree/support.lisp から移動。）

(in-package #:nabla.tests.support)

(defun %child-source-registry ()
  "子プロセスの ASDF に、このリポジトリと依存の置き場所を見せる
CL_SOURCE_REGISTRY の値。scripts/run-tests.sh と同じ組み立て方
（リポジトリは非再帰、依存は再帰）にする。"
  (let* ((repo (namestring (asdf:system-source-directory "nabla")))
         (deps (or (sb-ext:posix-getenv "NABLA_LISP_DEPS")
                   (namestring (merge-pathnames ".local/share/nabla/lisp-deps/"
                                                 (user-homedir-pathname))))))
    (format nil "~A:~A//:" repo deps)))

(defparameter *child-sbcl-forwarded-env-vars*
  '("PATH" "HOME" "LD_LIBRARY_PATH" "LANG" "LC_ALL"
    "TMPDIR" "XDG_CACHE_HOME" "NABLA_LISP_DEPS" "SBCL_HOME")
  "子 SBCL プロセスへ、その値があれば転送する環境変数名の共通リスト
（sbcl・ASDF・CFFI の動作に関わるもの）。子プロセスを立てるテストヘルパー
（このファイルの %CHILD-SBCL-ENVIRONMENT、compiler-test.lisp の
%RUN-WITH-MISSING-IREE-HOME・%RUN-SIGNAL-REGISTRATION-CHECK-CHILD）は
このリストを共有し、転送する変数の組をここ1箇所だけで管理する。
NABLA_IREE_HOME・CL_SOURCE_REGISTRY は呼び出し側ごとに要件が違う
（転送するだけでよいか、明示的に上書き・組み立てるか）ため、ここには
含めない。")

(defun %forward-env-vars (names)
  "NAMES の各環境変数について、このプロセスに値が設定されていれば
\"NAME=VALUE\" の文字列を、無ければ何も作らずに、SB-EXT:RUN-PROGRAM の
:environment にそのまま渡せるリストにして返す。"
  (remove nil
          (mapcar (lambda (name)
                    (let ((v (sb-ext:posix-getenv name)))
                      (and v (format nil "~A=~A" name v))))
                  names)))
