;;;; nabla.iree のコンディション階層。
;;;;
;;;; IREE 連携で起きるエラーは、Lisp の他のエラーと区別できるよう、
;;;; すべて IREE-ERROR の下にぶら下げる。

(in-package #:nabla.iree)

(define-condition iree-error (nabla:backend-error)
  ()
  (:documentation
   "nabla.iree が signal するすべてのエラーの root コンディション。
NABLA:BACKEND-ERROR の subtype（issue #9 の backend プロトコル）。"))

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

(define-condition iree-object-released (iree-error)
  ((kind :initarg :kind :reader iree-object-released-kind)
   (context :initarg :context :reader iree-object-released-context))
  (:report
   (lambda (condition stream)
     (format stream "~A に解放済みの ~A を渡した。"
             (iree-object-released-context condition)
             (iree-object-released-kind condition))))
  (:documentation
   "release-device / release-session / release-device-array で解放済みの
オブジェクトを、それを必要とするラッパー関数に渡したときに signal する。
KIND は :device / :session / :device-array、CONTEXT は呼び出した
nabla.iree 側の関数の名前（文字列）。C 側に解放済みポインタを渡すと
メモリ不正アクセスになるため、渡す前にここで検出する。"))

(define-condition iree-status-error (iree-error)
  ((code :initarg :code :reader iree-status-error-code)
   (message :initarg :message :reader iree-status-error-message)
   (context :initarg :context :reader iree-status-error-context))
  (:report
   (lambda (condition stream)
     (format stream "IREE ランタイムの呼び出し ~A が ~A で失敗した: ~A"
             (iree-status-error-context condition)
             (iree-status-error-code condition)
             (iree-status-error-message condition))))
  (:documentation
   "IREE ランタイムの iree_status_t が非OKだったときに signal する。CODE は
iree_status_code_e から得たキーワード（例: :not-found）。MESSAGE は
iree_status_to_string のテキスト（\"file.c:line: CODE; msg\" の形式を含む）。
CONTEXT は失敗した nabla.iree 側のラッパー関数の名前。"))
