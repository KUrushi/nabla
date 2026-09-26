;;;; IREE ランタイムでの実行（instance / device / session / call /
;;;; buffer_view）の、nabla.iree の公開 API。
;;;;
;;;; オブジェクトの対応（設計タブの対応表）:
;;;;   instance ... プロセスに1つ（iree-instance、mutex で保護）。解放しない
;;;;   device   ... (make-device :local 0) のように driver + index で作る。
;;;;                index は v1（フェーズ0）では 0 だけを受け付ける
;;;;   session  ... device 1つに対して作る。vmfb（バイトコードモジュール）を
;;;;                ロードする器
;;;;   call     ... 呼び出しごとに新しく作り、スレッド間で共有しない
;;;;                （with-call）
;;;;
;;;; アルゴリズム（すべて iree/runtime/*.h の高水準 API どおり。関数名は
;;;; runtime-ffi.lisp のコメントに挙げたヘッダから写した）。

(in-package #:nabla.iree)

;;; ------------------------------------------------------------------------
;;; instance
;;; ------------------------------------------------------------------------

(defstruct (instance (:constructor %make-instance (pointer))
                      (:predicate nil)
                      (:copier nil))
  "プロセスに1つの iree_runtime_instance_t を包む。POINTER は foreign
pointer。解放は行わない（プロセスの寿命まで生きる。設計タブの対応表）。"
  pointer)

(defvar *instance-lock* (sb-thread:make-mutex :name "nabla-iree-instance"))
(defvar *instance* nil
  "iree-instance が一度作った INSTANCE。プロセスにつき1つだけ作る。")

(defun %create-instance ()
  (cffi:with-foreign-object (options '(:struct %runtime-instance-options-t))
    (%runtime-instance-options-initialize options)
    (%runtime-instance-options-use-all-available-drivers options)
    (cffi:with-foreign-object (out-instance :pointer)
      (check-status
       (%runtime-instance-create options (system-allocator) out-instance)
       "iree-instance")
      (%make-instance (cffi:mem-ref out-instance :pointer)))))

(defun iree-instance ()
  "プロセス全体で共有する iree_runtime_instance_t を返す（無ければ作る）。
iree_runtime_instance_options_use_all_available_drivers でこのバイナリに
組み込まれている全ドライバ（local-sync / local-task、ビルドに含まれていれば
cuda）を使えるようにする。共有ライブラリが無ければ IREE-LIBRARY-NOT-FOUND を
signal する。"
  (ensure-runtime-loaded)
  (sb-thread:with-mutex (*instance-lock*)
    (or *instance* (setf *instance* (%create-instance)))))

;;; ------------------------------------------------------------------------
;;; iree_string_view_t のヘルパー
;;; ------------------------------------------------------------------------

(defun %string-view-plist-to-lisp (plist)
  "PLIST（'data 'size の iree_string_view_t 相当。libffi が構造体の値返しから
組み立てたもの）を Lisp 文字列にする。"
  (let ((data (getf plist 'data)) (size (getf plist 'size)))
    (if (or (cffi:null-pointer-p data) (zerop size))
        ""
        (cffi:foreign-string-to-lisp data :count size :encoding :utf-8))))

(defun %string-view-field-to-lisp (parent-pointer struct-type field)
  "PARENT-POINTER が指す STRUCT-TYPE の中の、iree_string_view_t 型の FIELD を
Lisp 文字列にする（配列に並んだ構造体からフィールドを読むときに使う。
foreign-slot-value ではなく foreign-slot-pointer を使うのは、STRUCT 型の
スロットは値ではなくその場所へのポインタとして扱う必要があるため）。"
  (let ((view-pointer (cffi:foreign-slot-pointer parent-pointer struct-type field)))
    (let ((data (cffi:foreign-slot-value view-pointer '(:struct %string-view-t) 'data))
          (size (cffi:foreign-slot-value view-pointer '(:struct %string-view-t) 'size)))
      (if (or (cffi:null-pointer-p data) (zerop size))
          ""
          (cffi:foreign-string-to-lisp data :count size :encoding :utf-8)))))

;;; ------------------------------------------------------------------------
;;; driver-names
;;; ------------------------------------------------------------------------

(defun driver-names ()
  "(iree-instance) のドライバレジストリに登録されている全ドライバの
canonical name（\"local-task\" など）をリストで返す
（iree_hal_driver_registry_enumerate、driver_registry.h:136-139）。返り値の
配列は system-allocator で確保されているので、読んだ後 iree_allocator_free
で解放する。"
  (let ((registry (%runtime-instance-driver-registry (instance-pointer (iree-instance)))))
    (cffi:with-foreign-objects ((count :size) (infos :pointer))
      (check-status
       (%hal-driver-registry-enumerate registry (system-allocator) count infos)
       "driver-names")
      (let ((n (cffi:mem-ref count :size))
            (array (cffi:mem-ref infos :pointer)))
        (unwind-protect
             (loop for i below n
                   for element = (cffi:mem-aptr array '(:struct %hal-driver-info-t) i)
                   collect (%string-view-field-to-lisp element '(:struct %hal-driver-info-t)
                                                        'driver-name))
          (unless (cffi:null-pointer-p array)
            (%allocator-free (system-allocator) array)))))))

;;; ------------------------------------------------------------------------
;;; device
;;; ------------------------------------------------------------------------

(defclass device ()
  ((pointer :accessor %device-pointer :initarg :pointer)
   (driver :reader device-driver :initarg :driver
           :documentation ":local-task / :local-sync / :cuda のどれか（:local は make-device で :local-task に正規化される）。")
   (name :reader device-name :initarg :name
         :documentation "device の作成に使ったドライバの canonical name（文字列）。"))
  (:documentation "iree_hal_device_t を包む。POINTER は release-device の後 null-pointer になる。"))

(defun %canonical-driver (driver)
  (case driver
    (:local :local-task)
    (t driver)))

(defun %driver-name-string (driver)
  "DRIVER を iree_hal_driver_registry に渡す canonical name（文字列）にする。
文字列ならそのまま使う。既知のキーワード（:local-task / :local-sync /
:cuda）は対応する名前に、それ以外のキーワードは symbol-name を小文字化した
ものにする（driver_registry には無い名前を渡して NOT_FOUND を確かめる
テストのため、ここでは検証しない。実際の存在チェックは
iree_runtime_instance_try_create_default_device が行う）。"
  (etypecase driver
    (string driver)
    (keyword (case driver
               (:local-task "local-task")
               (:local-sync "local-sync")
               (:cuda "cuda")
               (t (string-downcase (symbol-name driver)))))))

(defun make-device (driver &key (index 0))
  "DRIVER（:local-task / :local-sync / :cuda、その別名 :local、または任意の
ドライバ名の文字列）に対応する iree_hal_device_t を、
iree_runtime_instance_try_create_default_device で作る（instance.h の
TODO(#5724) コメントのとおり、v1 ではこの高水準 API を使う）。

INDEX は既定 0 のみを受け付ける（v1 は各ドライバの既定デバイス1つだけを
扱う。0 以外は未対応としてエラーにする）。

登録されていないドライバ名を渡すと、IREE-STATUS-ERROR（code :not-found）が
signal される。"
  (unless (zerop index)
    (error "make-device: index ~S はまだサポートされていない（0 だけ受け付ける）" index))
  (let* ((canonical (%canonical-driver driver))
         (name (%driver-name-string canonical))
         (instance (iree-instance)))
    (with-string-view (view name)
      (cffi:with-foreign-object (out-device :pointer)
        (check-status
         (%runtime-instance-try-create-default-device
          (instance-pointer instance) view out-device)
         "make-device")
        (make-instance 'device
                        :pointer (cffi:mem-ref out-device :pointer)
                        :driver canonical
                        :name name)))))

(defun device-released-p (device)
  "DEVICE が release-device 済みなら真を返す。"
  (cffi:null-pointer-p (%device-pointer device)))

(defun %live-device-pointer (device context)
  "DEVICE の foreign pointer（iree_hal_device_t*）を返す。DEVICE が
release-device 済みなら、解放済みの NULL ポインタを C へ渡してクラッシュ
させる前に IREE-OBJECT-RELEASED を signal する。CONTEXT は呼び出し元の
nabla.iree 側の関数名（文字列）。"
  (when (device-released-p device)
    (error 'iree-object-released :kind :device :context context))
  (%device-pointer device))

(defun release-device (device)
  "DEVICE を解放する。二重解放しても何もしない（idempotent）。"
  (unless (device-released-p device)
    (%hal-device-release (%device-pointer device))
    (setf (%device-pointer device) (cffi:null-pointer))))

(defmacro with-device ((var driver) &body body)
  "(make-device DRIVER) を VAR に束縛して BODY を評価し、終わったら
release-device する。"
  `(let ((,var (make-device ,driver)))
     (unwind-protect (progn ,@body)
       (release-device ,var))))

;;; ------------------------------------------------------------------------
;;; session
;;; ------------------------------------------------------------------------

(defclass session ()
  ((pointer :accessor %session-pointer :initarg :pointer)
   (device :reader session-device :initarg :device
           :documentation "この session を作った device オブジェクト。")
   (module-blocks :accessor %session-module-blocks :initform nil
                  :documentation "session-append-module が foreign-alloc した、解放前に生かしておくメモリブロックのリスト。"))
  (:documentation "iree_runtime_session_t を包む。"))

(defun make-session (device)
  "DEVICE に固定した iree_runtime_session_t を作る
（iree_runtime_session_create_with_device）。"
  (let ((instance (iree-instance)))
    (cffi:with-foreign-object (options '(:struct %runtime-session-options-t))
      (%runtime-session-options-initialize options)
      (cffi:with-foreign-object (out-session :pointer)
        (check-status
         (%runtime-session-create-with-device
          (instance-pointer instance) options (%live-device-pointer device "make-session")
          (system-allocator) out-session)
         "make-session")
        (make-instance 'session :pointer (cffi:mem-ref out-session :pointer) :device device)))))

(defun session-released-p (session)
  "SESSION が release-session 済みなら真を返す。"
  (cffi:null-pointer-p (%session-pointer session)))

(defun %live-session-pointer (session context)
  "SESSION の foreign pointer（iree_runtime_session_t*）を返す。SESSION が
release-session 済みなら、解放済みの NULL ポインタを C へ渡してクラッシュ
させる前に IREE-OBJECT-RELEASED を signal する。CONTEXT は呼び出し元の
nabla.iree 側の関数名（文字列）。"
  (when (session-released-p session)
    (error 'iree-object-released :kind :session :context context))
  (%session-pointer session))

(defun release-session (session)
  "SESSION を解放し、session-append-module が確保したモジュールのメモリ
ブロックも解放する。二重解放は何もしない。"
  (unless (session-released-p session)
    (%runtime-session-release (%session-pointer session))
    (setf (%session-pointer session) (cffi:null-pointer))
    (dolist (block (%session-module-blocks session))
      (cffi:foreign-free block))
    (setf (%session-module-blocks session) nil)))

(defmacro with-session ((var device) &body body)
  "(make-session DEVICE) を VAR に束縛して BODY を評価し、終わったら
release-session する。"
  `(let ((,var (make-session ,device)))
     (unwind-protect (progn ,@body)
       (release-session ,var))))

(defun session-append-module (session bytes)
  "BYTES（(unsigned-byte 8) の simple-array、vmfb の内容）のコピーを
foreign-alloc したメモリに作り、
iree_runtime_session_append_bytecode_module_from_memory でモジュールとして
追加する。コピーした先のメモリは SESSION が解放されるまで（あるいは
このモジュールの追加が失敗した場合はここで即座に）解放する。
flatbuffer_allocator には iree_allocator_null を渡す（データの所有権は
このメモリブロックを追跡している nabla.iree 側にあるため。
session.h:150-164 の doc コメントのとおり、失敗時も含めてこの引数が呼ばれる
だけで、null アロケータなので何も起きない）。"
  (check-type bytes (simple-array (unsigned-byte 8) (*)))
  (let* ((length (length bytes))
         (block (cffi:foreign-alloc :uint8 :count (max length 1))))
    (when (plusp length)
      (sb-sys:with-pinned-objects (bytes)
        (cffi:foreign-funcall "memcpy"
                               :pointer block
                               :pointer (sb-sys:vector-sap bytes)
                               :size length
                               :pointer)))
    (handler-case
        (progn
          (check-status
           (%runtime-session-append-bytecode-module-from-memory
            (%live-session-pointer session "session-append-module")
            (list 'data block 'data-length length)
            (null-allocator))
           "session-append-module")
          (push block (%session-module-blocks session))
          (values))
      (error (condition)
        (cffi:foreign-free block)
        (error condition)))))

(defun session-append-module-from-file (session path)
  "PATH（文字列または pathname）にある vmfb ファイルを、メモリマップ経由で
モジュールとして追加する（iree_runtime_session_append_bytecode_module_from_file）。"
  (check-status
   (%runtime-session-append-bytecode-module-from-file
    (%live-session-pointer session "session-append-module-from-file") (namestring path))
   "session-append-module-from-file"))

;;; ------------------------------------------------------------------------
;;; vm-function
;;; ------------------------------------------------------------------------

(defstruct (vm-function (:constructor %make-vm-function (module linkage ordinal)))
  "iree_vm_function_t から必要な値だけコピーした Lisp 側の表現。MODULE は
foreign pointer（iree_vm_module_t*）、LINKAGE は iree_vm_function_linkage_t
の整数値（IREE_VM_FUNCTION_LINKAGE_EXPORT なら2）、ORDINAL はそのリンケージ
内での番号。"
  module linkage ordinal)

(defun session-lookup-function (session full-name)
  "SESSION から FULL-NAME（\"module.main\" のような完全修飾名）の関数を探し、
VM-FUNCTION を返す。見つからなければ IREE-STATUS-ERROR（code :not-found）が
signal される（session.h:186-198）。"
  (with-string-view (view full-name)
    (cffi:with-foreign-object (out-function '(:struct %vm-function-t))
      (check-status
       (%runtime-session-lookup-function
        (%live-session-pointer session "session-lookup-function") view out-function)
       "session-lookup-function")
      (%make-vm-function
       (cffi:foreign-slot-value out-function '(:struct %vm-function-t) 'module)
       (cffi:foreign-slot-value out-function '(:struct %vm-function-t) 'linkage)
       (cffi:foreign-slot-value out-function '(:struct %vm-function-t) 'ordinal)))))

(defun session-function-names (session)
  "SESSION の VM context に登録されているモジュールのうち、名前が \"hal\"
以外のものについて、export されている関数名を集めてリストで返す
（context は既定で組み込みの hal モジュールを最初に持つため、それは除く。
vm/context.h の module_count / module_at と vm/module.h の
module_name / module_signature / lookup_function_by_ordinal /
function_name を使う）。コンパイルしたモジュールに `__init` のような
IREE が自動生成する初期化関数があれば、それもここに含まれる（利用者定義の
関数とは限らない）。"
  (let ((context (%runtime-session-context
                  (%live-session-pointer session "session-function-names")))
        (names nil))
    (dotimes (i (%vm-context-module-count context))
      (let ((module (%vm-context-module-at context i)))
        (unless (string= (%string-view-plist-to-lisp (%vm-module-name module)) "hal")
          (let* ((signature (%vm-module-signature module))
                 (export-count (getf signature 'export-function-count)))
            (dotimes (j export-count)
              (cffi:with-foreign-object (out-function '(:struct %vm-function-t))
                (check-status
                 (%vm-module-lookup-function-by-ordinal
                  module +vm-function-linkage-export+ j out-function)
                 "session-function-names")
                (push (%string-view-plist-to-lisp (%vm-function-name out-function)) names)))))))
    (nreverse names)))

;;; ------------------------------------------------------------------------
;;; call
;;; ------------------------------------------------------------------------

(defmacro with-call ((var session full-name) &body body)
  "SESSION の FULL-NAME 関数に対する iree_runtime_call_t を foreign メモリに
確保して初期化し、VAR（foreign pointer）に束縛して BODY を評価する。BODY を
抜けるとき（正常終了・非局所脱出のどちらでも）必ず
iree_runtime_call_deinitialize する。呼び出しごとに新しく作り、スレッド間で
共有しない（call.h の doc コメントどおり）。"
  (let ((call-pointer (gensym "CALL")))
    `(cffi:with-foreign-object (,call-pointer '(:struct %runtime-call-t))
       (with-string-view (view ,full-name)
         (check-status
          (%runtime-call-initialize-by-name
           (%live-session-pointer ,session "with-call") view ,call-pointer)
          "with-call"))
       (unwind-protect
            (let ((,var ,call-pointer))
              ,@body)
         (%runtime-call-deinitialize ,call-pointer)))))

(defun call-push-buffer-view (call buffer-view)
  "BUFFER-VIEW（foreign pointer、iree_hal_buffer_view_t*）を CALL の入力
リストの末尾へ push する。BUFFER-VIEW は呼び出し側が引き続き所有する
（call.h:110-113 のとおりリストが retain する）。"
  (check-status (%runtime-call-inputs-push-back-buffer-view call buffer-view)
                "call-push-buffer-view"))

(defun call-invoke (call)
  "CALL を実行する（フラグは常に0。IREE_RUNTIME_CALL_FLAG_RESERVED しか
定義されていない、call.h:28-31）。"
  (check-status (%runtime-call-invoke call +runtime-call-flags-none+) "call-invoke"))

(defun call-pop-buffer-view (call)
  "CALL の出力リストの先頭から buffer view を pop して、その foreign
pointer を返す。所有権は呼び出し側に移るので、使い終わったら
buffer-view-release で解放すること（call.h:115-117）。"
  (cffi:with-foreign-object (out-buffer-view :pointer)
    (check-status
     (%runtime-call-outputs-pop-front-buffer-view call out-buffer-view)
     "call-pop-buffer-view")
    (cffi:mem-ref out-buffer-view :pointer)))

;;; ------------------------------------------------------------------------
;;; buffer_view
;;; ------------------------------------------------------------------------

;; iree_hal_element_type_t の値は
;; IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count) =
;;   (numerical_type << 24) | bit_count （buffer_view.h:65-66）で計算する。
;;   IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE  = FLOAT(0x20) | 0x01 = #x21 (buffer_view.h:44,46)
;;   IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN = FLOAT(0x20) | 0x02 = #x22 (buffer_view.h:44,48)
(defparameter *element-types*
  (list (cons :f32 (logior (ash #x21 24) 32))
        (cons :bf16 (logior (ash #x22 24) 16))
        (cons :f16 (logior (ash #x21 24) 16)))
  "nabla.iree で扱う dtype キーワードと iree_hal_element_type_t の整数値の対応表。")

(defun %element-type-code (keyword)
  (or (cdr (assoc keyword *element-types*))
      (error "buffer-view-allocate-copy: 未知の element-type ~S（~{~S~^ ~} のどれか）"
             keyword (mapcar #'car *element-types*))))

(defun %element-type-keyword (code)
  (or (car (rassoc code *element-types*)) code))

(defun %element-type-bit-width (element-type)
  "ELEMENT-TYPE（*element-types* のキー）1要素あたりのビット数を返す
（IREE_HAL_ELEMENT_TYPE_VALUE の下位24ビットがビット数そのものになる）。"
  (logand (%element-type-code element-type) #xFFFFFF))

(defun buffer-view-allocate-copy (device shape element-type bytes-pointer byte-length)
  "DEVICE のアロケータに、SHAPE（次元のリスト）・ELEMENT-TYPE（:f32 など、
*element-types* のキー）の buffer view を作り、BYTES-POINTER が指す
BYTE-LENGTH バイトを初期値としてコピーする
（iree_hal_buffer_view_allocate_buffer_copy、buffer_view_util.h:63-69）。
BYTE-LENGTH が SHAPE と ELEMENT-TYPE から計算される長さ（要素数 ×
要素あたりのバイト数）と一致しないと、コピー元の一部が未初期化のまま
buffer view に残ってしまうため error を signal する。
返り値は呼び出し側が buffer-view-release で解放する foreign pointer。"
  (let ((expected-byte-length (* (reduce #'* shape :initial-value 1)
                                  (/ (%element-type-bit-width element-type) 8))))
    (unless (= byte-length expected-byte-length)
      (error "buffer-view-allocate-copy: BYTE-LENGTH ~S は SHAPE ~S と ~
ELEMENT-TYPE ~S から期待される長さ ~S と一致しない"
             byte-length shape element-type expected-byte-length)))
  (let* ((rank (length shape))
         (device-pointer (%live-device-pointer device "buffer-view-allocate-copy"))
         (allocator (%hal-device-allocator device-pointer)))
    (cffi:with-foreign-object (dims :size rank)
      (loop for i from 0 for dim in shape
            do (setf (cffi:mem-aref dims :size i) dim))
      (cffi:with-foreign-object (out-buffer-view :pointer)
        (check-status
         (%hal-buffer-view-allocate-buffer-copy
          device-pointer allocator rank dims
          (%element-type-code element-type) +hal-encoding-type-dense-row-major+
          (list 'usage +hal-buffer-usage-default+
                'access +hal-memory-access-all+
                'type +hal-memory-type-device-local+
                'queue-affinity +hal-queue-affinity-any+
                'min-alignment 0)
          (list 'data bytes-pointer 'data-length byte-length)
          out-buffer-view)
         "buffer-view-allocate-copy")
        (cffi:mem-ref out-buffer-view :pointer)))))

(defun buffer-view-release (buffer-view)
  "BUFFER-VIEW（foreign pointer）を解放する（iree_hal_buffer_view_release）。"
  (%hal-buffer-view-release buffer-view))

(defun buffer-view-shape (buffer-view)
  "BUFFER-VIEW の形をリストで返す。"
  (let ((rank (%hal-buffer-view-shape-rank buffer-view)))
    (loop for i below rank
          collect (%hal-buffer-view-shape-dim buffer-view i))))

(defun buffer-view-element-type (buffer-view)
  "BUFFER-VIEW の要素型を *element-types* のキーワードにして返す
（未知の値ならヘッダの enum 整数値そのものを返す）。"
  (%element-type-keyword (%hal-buffer-view-element-type buffer-view)))

(defun buffer-view-byte-length (buffer-view)
  "BUFFER-VIEW の内容の合計バイト数を返す（iree_hal_buffer_view_byte_length）。"
  (%hal-buffer-view-byte-length buffer-view))

(defun %buffer-view-read-into-sap (device-pointer buffer-view sap byte-length)
  "BUFFER-VIEW（DEVICE-POINTER が指す device 上にある）の先頭 BYTE-LENGTH
バイトを SAP（foreign pointer、あらかじめ BYTE-LENGTH バイト以上をピン留め
済みであること）へ同期的に読み出す（iree_hal_device_transfer_d2h、
buffer_transfer.h:88-92）。DEVICE-POINTER が解放済みでないかは呼び出し側の
責任（released チェックはしない）。buffer-view-read-into（#6）と
device-array の to-host（#7）が共有する内部実装。"
  (let ((buffer (%hal-buffer-view-buffer buffer-view)))
    (check-status
     (%hal-device-transfer-d2h
      device-pointer buffer 0 sap byte-length
      +hal-transfer-buffer-flag-default+
      (list 'type +timeout-absolute+ 'nanos +time-infinite-future+))
     "buffer-view-read-into")))

(defun buffer-view-read-into (device buffer-view octets)
  "BUFFER-VIEW の内容を DEVICE から同期的に読み出し、OCTETS
（(unsigned-byte 8) の simple-array）へ書き込む
（iree_hal_device_transfer_d2h、buffer_transfer.h:88-92）。OCTETS の長さが
読み出す量になる。"
  (check-type octets (simple-array (unsigned-byte 8) (*)))
  (sb-sys:with-pinned-objects (octets)
    (%buffer-view-read-into-sap
     (%live-device-pointer device "buffer-view-read-into")
     buffer-view (sb-sys:vector-sap octets) (length octets)))
  octets)
