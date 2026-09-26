;;;; iree_status_t を検査し、非OKなら IREE-STATUS-ERROR に変換する共通の仕組み。
;;;;
;;;; iree_status_t は成功なら NULL ポインタ、失敗なら下位 IREE_STATUS_CODE_MASK
;;;; (0x1F) ビットにコードを埋め込んだポインタ（base/status.h の
;;;; IREE_STATUS_CODE マクロ、status.h:163,186）。アロケータで確保された
;;;; メッセージを持つこともあり、iree_status_to_string で文字列化してから
;;;; iree_allocator_free で解放し、最後に iree_status_free で本体を解放する
;;;; （status.h:566-576）。

(in-package #:nabla.iree)

;; iree_status_code_e（base/status.h:78-163）。
(defparameter *status-code-table*
  '((0 . :ok)
    (1 . :cancelled)
    (2 . :unknown)
    (3 . :invalid-argument)
    (4 . :deadline-exceeded)
    (5 . :not-found)
    (6 . :already-exists)
    (7 . :permission-denied)
    (8 . :resource-exhausted)
    (9 . :failed-precondition)
    (10 . :aborted)
    (11 . :out-of-range)
    (12 . :unimplemented)
    (13 . :internal)
    (14 . :unavailable)
    (15 . :data-loss)
    (16 . :unauthenticated)
    (17 . :deferred)
    (18 . :incompatible))
  "iree_status_code_e の整数値から Lisp のキーワードへの対応表。")

;; IREE_STATUS_CODE_MASK（base/status.h:163）。
(defconstant +status-code-mask+ #x1F)

(defun status-code (status)
  "STATUS（iree_status_t、foreign pointer）から iree_status_code_t を取り出し、
*STATUS-CODE-TABLE* に無ければ :unknown-status-code を、あれば対応する
キーワードを返す（base/status.h の IREE_STATUS_CODE マクロと同じ計算）。"
  (let ((code (logand (cffi:pointer-address status) +status-code-mask+)))
    (or (cdr (assoc code *status-code-table*))
        :unknown-status-code)))

(defun system-allocator ()
  "iree_allocator_system() 相当のプロパティリスト。このビルドは
IREE_ALLOCATOR_SYSTEM=libc で構成されているため（third_party/iree ビルド設定。
iree_allocator_system() 自体は base/allocator.h の static inline で
共有ライブラリからは export されていない）、libnabla_iree_runtime.so が
export している iree_allocator_libc_ctl を ctl 関数として直接使う。"
  (list 'self (cffi:null-pointer)
        'ctl (cffi:foreign-symbol-pointer "iree_allocator_libc_ctl")))

(defun null-allocator ()
  "iree_allocator_null() 相当のプロパティリスト（base/allocator.h:524-526）。"
  (list 'self (cffi:null-pointer) 'ctl (cffi:null-pointer)))

(defun %fill-system-allocator (allocator-pointer)
  "ALLOCATOR-POINTER が指す (:struct %allocator-t) を、(system-allocator) と
同じ内容（{self=NULL, ctl=iree_allocator_libc_ctl}）で埋める。system-allocator
が値渡し用に返すプロパティリストと、ここでポインタ渡しが必要な場面
（iree_status_to_string の allocator 引数など）とで、どのシンボルを ctl に
使うかの判断を1箇所にまとめるためのヘルパー。"
  (let ((filled (system-allocator)))
    (setf (cffi:foreign-slot-value allocator-pointer '(:struct %allocator-t) 'self)
          (getf filled 'self))
    (setf (cffi:foreign-slot-value allocator-pointer '(:struct %allocator-t) 'ctl)
          (getf filled 'ctl))
    allocator-pointer))

(defun %status-message (status)
  "STATUS（非NULL）を iree_status_to_string で文字列化して返す。"
  (cffi:with-foreign-object (allocator '(:struct %allocator-t))
    (%fill-system-allocator allocator)
    (cffi:with-foreign-objects ((out-buffer :pointer) (out-length :size))
      (if (%status-to-string status allocator out-buffer out-length)
          (let ((buffer (cffi:mem-ref out-buffer :pointer))
                (length (cffi:mem-ref out-length :size)))
            (unwind-protect
                 (cffi:foreign-string-to-lisp buffer :count length :encoding :utf-8)
              (%allocator-free (system-allocator) buffer)))
          "(iree_status_to_string に失敗した)"))))

(defmacro check-status (form context)
  "FORM（iree_status_t を返す式）を評価する。返り値が NULL ポインタなら
そのまま NIL を返す。非NULLなら status-code / %status-message でメッセージを
組み立て、iree_status_free で解放してから IREE-STATUS-ERROR を signal する。
CONTEXT は失敗した nabla.iree 側の呼び出しの名前（文字列）。"
  (let ((status (gensym "STATUS")))
    `(let ((,status ,form))
       (unless (cffi:null-pointer-p ,status)
         (let ((code (status-code ,status))
               (message (%status-message ,status)))
           (%status-free ,status)
           (error 'iree-status-error :code code :message message :context ,context))))))

(defmacro with-string-view ((var string) &body body)
  "STRING（Lisp 文字列）の UTF-8 バイト列を動的エクステントで確保し、
VAR に iree_string_view_t 相当のプロパティリスト（'data 'size）を束縛して
BODY を評価する。cffi:with-foreign-string が返す長さは終端 NUL を含むため、
1引いた値を size にする（with-foreign-string の length は NUL 込み。これを
引かずに iree_string_view_t へ渡すと、ドライバ名の末尾に余計な1バイトが
付き \"no driver 'local-task\\0' registered\" のような失敗になる）。"
  (let ((buffer (gensym "BUFFER")) (length (gensym "LENGTH")))
    `(cffi:with-foreign-string ((,buffer ,length) ,string :encoding :utf-8)
       (let ((,var (list 'data ,buffer 'size (1- ,length))))
         ,@body))))
