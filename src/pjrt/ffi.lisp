;;;; PJRT C API の関数ポインタ表（PJRT_Api）と、引数構造体（*_Args）の
;;;; CFFI 定義、PJRT_Error* のコンディションへの変換（issue #85）。
;;;;
;;;; PJRT の関数はどれも「*_Args 構造体へのポインタ」を1つ受け取り、
;;;; PJRT_Error*（成功なら NULL）を返す。引数構造体は値渡しではない
;;;; ので cffi-libffi は要らず、素の CFFI の foreign-funcall-pointer で呼べる。
;;;;
;;;; 構造体の配置は third_party/pjrt/pjrt_c_api.h（API 0.116）から写した
;;;; （記憶では書かない）。固定したプラグイン（API 0.81）はこのヘッダより
;;;; 古いが、PJRT の構造体は「末尾に足すだけ」で既存のフィールドの位置は
;;;; 変わらないので、ヘッダの struct_size で渡しても動く（プラグインは自分が
;;;; 知っている分だけを読む）。PJRT_Api の関数ポインタ表も同じ理由で、
;;;; 古いプラグインは表の末尾が短いだけ。%api-function は表の struct_size を
;;;; 見て、プラグインが持たない関数を呼ぼうとしたら PJRT-ERROR にする。
;;;;
;;;; 生の CFFI（% 接頭辞のシンボル）はこのパッケージの中だけで使い、
;;;; export しない。

(in-package #:nabla.pjrt)

(define-condition pjrt-error (nabla:backend-error)
  ((message :initarg :message :reader pjrt-error-message)
   (context :initarg :context :initform nil :reader pjrt-error-context))
  (:report (lambda (condition stream)
             (format stream "PJRT~@[ ~A~]: ~A"
                     (pjrt-error-context condition)
                     (pjrt-error-message condition))))
  (:documentation
   "PJRT の関数が PJRT_Error を返したときに signal する。MESSAGE は
PJRT_Error_Message が返した文字列、CONTEXT は呼んだ関数の名前（PJRT_ 接頭辞付き）。
プラグインのクラッシュではなく、PJRT が「失敗」を報告した場合がこれになる。"))

(define-condition pjrt-object-released (pjrt-error)
  ((kind :initarg :kind :reader pjrt-object-released-kind))
  (:documentation
   "解放済みの device-array などを使おうとしたときに signal する。"))

(defparameter *api-function-names*
  '("PJRT_Error_Destroy" "PJRT_Error_Message" "PJRT_Error_GetCode" "PJRT_Plugin_Initialize"
    "PJRT_Plugin_Attributes" "PJRT_Event_Destroy" "PJRT_Event_IsReady" "PJRT_Event_Error"
    "PJRT_Event_Await" "PJRT_Event_OnReady" "PJRT_Client_Create" "PJRT_Client_Destroy"
    "PJRT_Client_PlatformName" "PJRT_Client_ProcessIndex" "PJRT_Client_PlatformVersion"
    "PJRT_Client_Devices" "PJRT_Client_AddressableDevices" "PJRT_Client_LookupDevice"
    "PJRT_Client_LookupAddressableDevice" "PJRT_Client_AddressableMemories"
    "PJRT_Client_Compile" "PJRT_Client_DefaultDeviceAssignment"
    "PJRT_Client_BufferFromHostBuffer" "PJRT_DeviceDescription_Id"
    "PJRT_DeviceDescription_ProcessIndex" "PJRT_DeviceDescription_Attributes"
    "PJRT_DeviceDescription_Kind" "PJRT_DeviceDescription_DebugString"
    "PJRT_DeviceDescription_ToString" "PJRT_Device_GetDescription"
    "PJRT_Device_IsAddressable" "PJRT_Device_LocalHardwareId"
    "PJRT_Device_AddressableMemories" "PJRT_Device_DefaultMemory" "PJRT_Device_MemoryStats"
    "PJRT_Memory_Id" "PJRT_Memory_Kind" "PJRT_Memory_DebugString" "PJRT_Memory_ToString"
    "PJRT_Memory_AddressableByDevices" "PJRT_Executable_Destroy" "PJRT_Executable_Name"
    "PJRT_Executable_NumReplicas" "PJRT_Executable_NumPartitions"
    "PJRT_Executable_NumOutputs" "PJRT_Executable_SizeOfGeneratedCodeInBytes"
    "PJRT_Executable_GetCostAnalysis" "PJRT_Executable_OutputMemoryKinds"
    "PJRT_Executable_OptimizedProgram" "PJRT_Executable_Serialize"
    "PJRT_LoadedExecutable_Destroy" "PJRT_LoadedExecutable_GetExecutable"
    "PJRT_LoadedExecutable_AddressableDevices" "PJRT_LoadedExecutable_Delete"
    "PJRT_LoadedExecutable_IsDeleted" "PJRT_LoadedExecutable_Execute"
    "PJRT_Executable_DeserializeAndLoad" "PJRT_LoadedExecutable_Fingerprint"
    "PJRT_Buffer_Destroy" "PJRT_Buffer_ElementType" "PJRT_Buffer_Dimensions"
    "PJRT_Buffer_UnpaddedDimensions" "PJRT_Buffer_DynamicDimensionIndices"
    "PJRT_Buffer_GetMemoryLayout" "PJRT_Buffer_OnDeviceSizeInBytes" "PJRT_Buffer_Device"
    "PJRT_Buffer_Memory" "PJRT_Buffer_Delete" "PJRT_Buffer_IsDeleted"
    "PJRT_Buffer_CopyToDevice" "PJRT_Buffer_ToHostBuffer" "PJRT_Buffer_IsOnCpu"
    "PJRT_Buffer_ReadyEvent" "PJRT_Buffer_UnsafePointer"
    "PJRT_Buffer_IncreaseExternalReferenceCount" "PJRT_Buffer_DecreaseExternalReferenceCount"
    "PJRT_Buffer_OpaqueDeviceMemoryDataPointer" "PJRT_CopyToDeviceStream_Destroy"
    "PJRT_CopyToDeviceStream_AddChunk" "PJRT_CopyToDeviceStream_TotalBytes"
    "PJRT_CopyToDeviceStream_GranuleSize" "PJRT_CopyToDeviceStream_CurrentBytes"
    "PJRT_TopologyDescription_Create" "PJRT_TopologyDescription_Destroy"
    "PJRT_TopologyDescription_PlatformName" "PJRT_TopologyDescription_PlatformVersion"
    "PJRT_TopologyDescription_GetDeviceDescriptions" "PJRT_TopologyDescription_Serialize"
    "PJRT_TopologyDescription_Attributes" "PJRT_Compile" "PJRT_Executable_OutputElementTypes"
    "PJRT_Executable_OutputDimensions" "PJRT_Buffer_CopyToMemory"
    "PJRT_Client_CreateViewOfDeviceBuffer" "PJRT_Executable_Fingerprint"
    "PJRT_Client_TopologyDescription" "PJRT_Executable_GetCompiledMemoryStats"
    "PJRT_Memory_Kind_Id" "PJRT_ExecuteContext_Create" "PJRT_ExecuteContext_Destroy"
    "PJRT_Buffer_CopyRawToHost" "PJRT_AsyncHostToDeviceTransferManager_Destroy"
    "PJRT_AsyncHostToDeviceTransferManager_TransferData"
    "PJRT_Client_CreateBuffersForAsyncHostToDevice"
    "PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer"
    "PJRT_AsyncHostToDeviceTransferManager_Device"
    "PJRT_AsyncHostToDeviceTransferManager_BufferCount"
    "PJRT_AsyncHostToDeviceTransferManager_BufferSize"
    "PJRT_AsyncHostToDeviceTransferManager_SetBufferError"
    "PJRT_AsyncHostToDeviceTransferManager_AddMetadata" "PJRT_Client_DmaMap"
    "PJRT_Client_DmaUnmap" "PJRT_Client_CreateUninitializedBuffer"
    "PJRT_Client_UpdateGlobalProcessInfo" "PJRT_TopologyDescription_Deserialize"
    "PJRT_Client_CreateAliasBuffer" "PJRT_Client_FulfillAliasBuffer"
    "PJRT_LoadedExecutable_GetDeviceAssignment" "PJRT_Client_CreateErrorBuffer"
    "PJRT_AsyncHostToDeviceTransferManager_TransferLiteral" "PJRT_Buffer_CopyRawToHostFuture"
    "PJRT_Device_PoisonExecution" "PJRT_Device_CreateAsyncTrackingEvent"
    "PJRT_AsyncTrackingEvent_Destroy" "PJRT_Executable_GetCompileOptions"
    "PJRT_Buffer_DonateWithControlDependency" "PJRT_Event_Create" "PJRT_Event_Set"
    "PJRT_Device_GetAttributes" "PJRT_Client_Load"
    "PJRT_LoadedExecutable_AddressableDeviceLogicalIds" "PJRT_Buffer_Bitcast"
    "PJRT_Error_ForEachPayload" "PJRT_TopologyDescription_Fingerprint"
    "PJRT_Executable_ParameterMemoryKinds" "PJRT_Device_ClearMemoryStats"
    "PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace"
    "PJRT_TopologyDescription_GetMemorySpaceKindIds")
  "PJRT_Api の関数ポインタ表のフィールド名を、ヘッダの宣言順に並べたもの
（pjrt_c_api.h の struct PJRT_Api の _PJRT_API_STRUCT_FIELD の並び）。
先頭の3つ組（struct_size / extension_start / pjrt_api_version）の後ろに
ポインタが1つずつ並ぶので、i 番目の関数は PJRT_Api の先頭から
+API-FUNCTIONS-OFFSET+ + 8 * i バイトの位置にある。ヘッダを更新したら、
この並びも写し直す（small テストが先頭と末尾を検査する）。")

(defconstant +api-functions-offset+ 40
  "PJRT_Api の最初の関数ポインタの位置: struct_size (8) + extension_start (8) +
PJRT_Api_Version (struct_size 8 + extension_start 8 + major 4 + minor 4 = 24)。")

(defun %api-function (api name)
  "API（PJRT_Api*）の関数表から、NAME（\"PJRT_Client_Create\" のような文字列）の
関数ポインタを返す。プラグインの表が古くて NAME を持たない（struct_size が
足りない、またはポインタが NULL）ときは PJRT-ERROR。"
  (let* ((index (or (position name *api-function-names* :test #'string=)
                    (error "unknown PJRT API function ~A" name)))
         (offset (+ +api-functions-offset+ (* 8 index)))
         (struct-size (cffi:foreign-slot-value api '(:struct %api-head) 'struct-size)))
    (when (> (+ offset 8) struct-size)
      (error 'pjrt-error :context name
                         :message "the plugin's PJRT_Api does not provide this function"))
    (let ((pointer (cffi:mem-ref api :pointer offset)))
      (when (cffi:null-pointer-p pointer)
        (error 'pjrt-error :context name
                           :message "the plugin's PJRT_Api has a NULL entry for this function"))
      pointer)))

;;; 引数構造体（pjrt_c_api.h の同名の構造体から、フィールドの順に写した）

(cffi:defcstruct %error-destroy-args
  (struct-size :size) (extension-start :pointer) (error :pointer))

(cffi:defcstruct %error-message-args
  (struct-size :size) (extension-start :pointer) (error :pointer)
  (message :pointer) (message-size :size))

(cffi:defcstruct %plugin-initialize-args
  (struct-size :size) (extension-start :pointer))

(cffi:defcstruct %client-create-args
  (struct-size :size) (extension-start :pointer)
  (create-options :pointer) (num-options :size)
  (kv-get-callback :pointer) (kv-get-user-arg :pointer)
  (kv-put-callback :pointer) (kv-put-user-arg :pointer)
  (client :pointer)
  (kv-try-get-callback :pointer) (kv-try-get-user-arg :pointer))

(cffi:defcstruct %client-destroy-args
  (struct-size :size) (extension-start :pointer) (client :pointer))

(cffi:defcstruct %client-platform-name-args
  (struct-size :size) (extension-start :pointer) (client :pointer)
  (platform-name :pointer) (platform-name-size :size))

(cffi:defcstruct %client-addressable-devices-args
  (struct-size :size) (extension-start :pointer) (client :pointer)
  (addressable-devices :pointer) (num-addressable-devices :size))

(cffi:defcstruct %client-buffer-from-host-buffer-args
  (struct-size :size) (extension-start :pointer) (client :pointer)
  (data :pointer) (type :int)
  (dims :pointer) (num-dims :size)
  (byte-strides :pointer) (num-byte-strides :size)
  (host-buffer-semantics :int)
  (device :pointer) (memory :pointer) (device-layout :pointer)
  (done-with-host-buffer :pointer)
  (buffer :pointer))

(cffi:defcstruct %event-await-args
  (struct-size :size) (extension-start :pointer) (event :pointer))

(cffi:defcstruct %event-destroy-args
  (struct-size :size) (extension-start :pointer) (event :pointer))

(cffi:defcstruct %buffer-destroy-args
  (struct-size :size) (extension-start :pointer) (buffer :pointer))

(cffi:defcstruct %buffer-to-host-buffer-args
  (struct-size :size) (extension-start :pointer) (src :pointer)
  (host-layout :pointer) (dst :pointer) (dst-size :size)
  (event :pointer))

(defparameter *buffer-types*
  '((:f16 . 10) (:f32 . 11) (:f64 . 12) (:bf16 . 13))
  "dtype キーワードと PJRT_Buffer_Type の値の対応。値は pjrt_c_api.h の
PJRT_Buffer_Type の enum を INVALID = 0 から数えた位置（PRED 1、S8 2、S16 3、
S32 4、S64 5、U8 6、U16 7、U32 8、U64 9、F16 10、F32 11、F64 12、BF16 13）。
small テストがこの値をヘッダの enum と突き合わせる。:i1（PRED）はまだ
対応しない。")

(defconstant +host-buffer-immutable-until-transfer-completes+ 1
  "PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes
（kImmutableOnlyDuringCall = 0 の次）。")

(defun %zero-args (args type)
  "ARGS（TYPE の foreign 構造体）をゼロ埋めして struct_size を設定する。"
  (cffi:foreign-funcall "memset" :pointer args :int 0
                                 :size (cffi:foreign-type-size type) :pointer)
  (setf (cffi:foreign-slot-value args type 'struct-size) (cffi:foreign-type-size type))
  args)

(defmacro %with-args ((var type &rest inits) &body body)
  "TYPE（(:struct ...)）の引数構造体をゼロ埋めで確保して struct_size を設定し、
INITS（(スロット名 値) の並び）を書き込んで VAR に束縛し、BODY を評価する。"
  `(cffi:with-foreign-object (,var ',type)
     (%zero-args ,var ',type)
     ,@(loop for (slot value) in inits
             collect `(setf (cffi:foreign-slot-value ,var ',type ',slot) ,value))
     ,@body))

(defun %error-message-and-destroy (api error)
  "ERROR（PJRT_Error*）のメッセージを Lisp の文字列にして取り出し、
PJRT_Error_Destroy で解放してから、そのメッセージを返す。"
  (let ((message
          (%with-args (args (:struct %error-message-args) (error error))
            (cffi:foreign-funcall-pointer (%api-function api "PJRT_Error_Message") ()
                                          :pointer args :void)
            (cffi:foreign-string-to-lisp
             (cffi:foreign-slot-value args '(:struct %error-message-args) 'message)
             :count (cffi:foreign-slot-value args '(:struct %error-message-args) 'message-size)
             :encoding :utf-8))))
    (%with-args (args (:struct %error-destroy-args) (error error))
      (cffi:foreign-funcall-pointer (%api-function api "PJRT_Error_Destroy") ()
                                    :pointer args :void))
    message))

(defmacro %pjrt-call ((api name var type &rest inits) &body body)
  "API の関数 NAME（文字列）を、TYPE の引数構造体（INITS で初期化）を VAR に
束縛して呼ぶ。PJRT_Error が返ったら、メッセージを取り出して解放してから
PJRT-ERROR を signal する。成功したら、引数構造体がまだ有効な間に BODY
（out フィールドの読み出しなど）を評価してその値を返す。"
  (let ((error (gensym "ERROR")) (api-var (gensym "API")))
    `(let ((,api-var ,api))
       (%with-args (,var ,type ,@inits)
         (let ((,error (cffi:foreign-funcall-pointer (%api-function ,api-var ,name) ()
                                                      :pointer ,var :pointer)))
           (unless (cffi:null-pointer-p ,error)
             (error 'pjrt-error :context ,name
                                :message (%error-message-and-destroy ,api-var ,error)))
           ,@body)))))

(defun %await-and-destroy-event (api event)
  "EVENT（PJRT_Event*）の完了を PJRT_Event_Await で待ち、成否にかかわらず
PJRT_Event_Destroy で解放する。EVENT が NULL なら何もしない（転送がすでに
終わっていて、イベントを返さないプラグインがありうる）。"
  (unless (cffi:null-pointer-p event)
    (unwind-protect
         (%pjrt-call (api "PJRT_Event_Await" args (:struct %event-await-args) (event event)))
      (%pjrt-call (api "PJRT_Event_Destroy" args (:struct %event-destroy-args) (event event))))))
