;;;; nabla.iree のコンディション階層。
;;;;
;;;; IREE 連携で起きるエラーは、Lisp の他のエラーと区別できるよう、
;;;; すべて IREE-ERROR の下にぶら下げる。

(in-package #:nabla.iree)

(define-condition iree-error (error)
  ()
  (:documentation
   "nabla.iree が signal するすべてのエラーの root コンディション。"))

(define-condition iree-library-not-found (iree-error)
  ((path :initarg :path :reader iree-library-not-found-path)
   (home :initarg :home :reader iree-library-not-found-home)
   (library :initarg :library :reader iree-library-not-found-library))
  (:report
   (lambda (condition stream)
     (format stream
             "IREE の共有ライブラリ ~A（~A）が見つからなかった。~
NABLA_IREE_HOME（現在: ~A）が正しいか確認するか、scripts/build-iree.sh で~
ビルドすること。"
             (iree-library-not-found-path condition)
             (iree-library-not-found-library condition)
             (iree-library-not-found-home condition))))
  (:documentation
   "PATH（試した共有ライブラリのパス）が見つからないときに signal する。
LIBRARY は :compiler または :runtime、HOME は探索に使った NABLA_IREE_HOME
の値（未設定なら既定値）。"))

(define-condition iree-compile-error (iree-error)
  ((phase :initarg :phase :reader iree-compile-error-phase)
   (diagnostics :initarg :diagnostics :initform nil :reader iree-compile-error-diagnostics)
   (message :initarg :message :initform nil :reader iree-compile-error-message))
  (:report
   (lambda (condition stream)
     (format stream "IREE のコンパイルが phase ~A で失敗した。"
             (iree-compile-error-phase condition))
     (when (iree-compile-error-message condition)
       (format stream "~%  ~A" (iree-compile-error-message condition)))
     (dolist (diagnostic (iree-compile-error-diagnostics condition))
       (format stream "~%  [~A] ~A" (car diagnostic) (cdr diagnostic)))))
  (:documentation
   "StableHLO のコンパイルが失敗したときに signal する。PHASE は
:flags / :parse / :compile / :output のどれか。DIAGNOSTICS は
(severity . text) のリスト（severity は :note :warning :error :remark の
どれかで、MLIR が出した順）。MESSAGE は iree_compiler_error_t から得た
テキスト（無ければ NIL）。"))
