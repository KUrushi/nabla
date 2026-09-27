;;;; IREE ランタイムの高水準 C API（iree/runtime/*.h）と、そこから参照する
;;;; HAL / VM / base の一部の、生の CFFI バインディング。
;;;;
;;;; 関数名・構造体のレイアウトは $NABLA_IREE_HOME/include/iree/ 配下の
;;;; ヘッダ（IREE v3.11.0, commit e4a3b0405d7d23554da26403658d0e8c3c5ecf25。
;;;; third_party/iree.lock で固定）から書き写した:
;;;;   runtime/instance.h, runtime/session.h, runtime/call.h
;;;;   base/allocator.h, base/string_view.h, base/status.h, base/time.h
;;;;   hal/driver_registry.h, hal/driver.h, hal/device.h
;;;;   hal/buffer.h, hal/buffer_view.h, hal/buffer_view_util.h,
;;;;   hal/buffer_transfer.h
;;;;   vm/module.h, vm/context.h
;;;;
;;;; このランタイム API は iree_allocator_t / iree_string_view_t /
;;;; iree_const_byte_span_t / iree_hal_buffer_params_t / iree_timeout_t を
;;;; 値渡しし、iree_vm_module_signature や iree_vm_function_name のように
;;;; 構造体を値で返す関数もある。素の CFFI（SBCL の FFI）は構造体の値渡し・
;;;; 値返しに対応していないため、このファイルをロードする前に
;;;; cffi-libffi をロードしておく必要がある（nabla/iree の :depends-on に
;;;; 含めてある）。cffi-libffi は cffi:*foreign-structures-by-value* を
;;;; 差し替えることで、ここの defcfun / defcstruct をそのまま libffi 経由の
;;;; 呼び出しに切り替える。C 側のヘルパーは書かない
;;;; （contract 0.3、PR 本文にも理由を書く）。
;;;;
;;;; ここのシンボルは % 接頭辞を付け、export しない。呼び出し側は
;;;; runtime.lisp / status.lisp のラッパー（% の付かない関数）だけを使う。

(in-package #:nabla.iree)

;;; ------------------------------------------------------------------------
;;; base/allocator.h, base/string_view.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %allocator-t
  (self :pointer)
  (ctl :pointer))

(cffi:defcstruct %string-view-t
  (data :pointer)
  (size :size))

(cffi:defcstruct %const-byte-span-t
  (data :pointer)
  (data-length :size))

(cffi:defcfun ("iree_allocator_free" %allocator-free) :void
  (allocator (:struct %allocator-t))
  (ptr :pointer))

;;; ------------------------------------------------------------------------
;;; base/status.h
;;; ------------------------------------------------------------------------

;; iree_status_to_string は allocator をポインタで受け取る（値渡しではない。
;; base/status.h:574）。テキストは allocator で確保されるので、
;; iree_allocator_free（上記、値渡し）で解放する。
(cffi:defcfun ("iree_status_to_string" %status-to-string) %bool
  (status :pointer)
  (allocator (:pointer (:struct %allocator-t)))
  (out-buffer :pointer)
  (out-length :pointer))

(cffi:defcfun ("iree_status_free" %status-free) :void
  (status :pointer))

;;; ------------------------------------------------------------------------
;;; base/time.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %timeout-t
  (type :int32)
  (nanos :int64))

;; iree_timeout_type_e（time.h:90-95）。
(defconstant +timeout-absolute+ 0)

;; IREE_TIME_INFINITE_FUTURE（time.h:33、int64_t の最大値）。
(defconstant +time-infinite-future+ #x7FFFFFFFFFFFFFFF)

;;; ------------------------------------------------------------------------
;;; runtime/instance.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %runtime-instance-options-t
  (driver-registry :pointer))

(cffi:defcfun ("iree_runtime_instance_options_initialize" %runtime-instance-options-initialize) :void
  (out-options :pointer))

(cffi:defcfun ("iree_runtime_instance_options_use_all_available_drivers"
               %runtime-instance-options-use-all-available-drivers)
    :void
  (options :pointer))

(cffi:defcfun ("iree_runtime_instance_create" %runtime-instance-create) :pointer
  (options :pointer)
  (host-allocator (:struct %allocator-t))
  (out-instance :pointer))

(cffi:defcfun ("iree_runtime_instance_release" %runtime-instance-release) :void
  (instance :pointer))

(cffi:defcfun ("iree_runtime_instance_driver_registry" %runtime-instance-driver-registry) :pointer
  (instance :pointer))

(cffi:defcfun ("iree_runtime_instance_try_create_default_device"
               %runtime-instance-try-create-default-device)
    :pointer
  (instance :pointer)
  (driver-name (:struct %string-view-t))
  (out-device :pointer))

;;; ------------------------------------------------------------------------
;;; hal/driver_registry.h, hal/driver.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %hal-driver-info-t
  (driver-name (:struct %string-view-t))
  (full-name (:struct %string-view-t)))

(cffi:defcfun ("iree_hal_driver_registry_enumerate" %hal-driver-registry-enumerate) :pointer
  (registry :pointer)
  (host-allocator (:struct %allocator-t))
  (out-driver-info-count :pointer)
  (out-driver-infos :pointer))

;;; ------------------------------------------------------------------------
;;; hal/device.h
;;; ------------------------------------------------------------------------

(cffi:defcfun ("iree_hal_device_release" %hal-device-release) :void
  (device :pointer))

;; iree_hal_device_retain（hal/device.h:375）。device-array が生成時に device を
;; retain するために使う（#7 の設計。iree_hal_buffer_heap_t が allocator の
;; 統計ブロックへの生ポインタを持ち、その allocator は device が所有するため、
;; device-array が生きている間は device を生かしておく必要がある）。
(cffi:defcfun ("iree_hal_device_retain" %hal-device-retain) :void
  (device :pointer))

(cffi:defcfun ("iree_hal_device_allocator" %hal-device-allocator) :pointer
  (device :pointer))

;;; ------------------------------------------------------------------------
;;; runtime/session.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %runtime-session-options-t
  (context-flags :uint32)
  (builtin-modules :uint64))

(cffi:defcfun ("iree_runtime_session_options_initialize" %runtime-session-options-initialize) :void
  (out-options :pointer))

(cffi:defcfun ("iree_runtime_session_create_with_device" %runtime-session-create-with-device) :pointer
  (instance :pointer)
  (options :pointer)
  (device :pointer)
  (host-allocator (:struct %allocator-t))
  (out-session :pointer))

(cffi:defcfun ("iree_runtime_session_release" %runtime-session-release) :void
  (session :pointer))

(cffi:defcfun ("iree_runtime_session_context" %runtime-session-context) :pointer
  (session :pointer))

(cffi:defcfun ("iree_runtime_session_device" %runtime-session-device) :pointer
  (session :pointer))

(cffi:defcfun ("iree_runtime_session_append_bytecode_module_from_memory"
               %runtime-session-append-bytecode-module-from-memory)
    :pointer
  (session :pointer)
  (flatbuffer-data (:struct %const-byte-span-t))
  (flatbuffer-allocator (:struct %allocator-t)))

(cffi:defcfun ("iree_runtime_session_append_bytecode_module_from_file"
               %runtime-session-append-bytecode-module-from-file)
    :pointer
  (session :pointer)
  (file-path :string))

(cffi:defcfun ("iree_runtime_session_lookup_function" %runtime-session-lookup-function) :pointer
  (session :pointer)
  (full-name (:struct %string-view-t))
  (out-function :pointer))

;;; ------------------------------------------------------------------------
;;; runtime/call.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %vm-function-t
  (module :pointer)
  (linkage :uint16)
  (ordinal :uint16))

(cffi:defcstruct %runtime-call-t
  (session :pointer)
  (function (:struct %vm-function-t))
  (inputs :pointer)
  (outputs :pointer))

(cffi:defcfun ("iree_runtime_call_initialize_by_name" %runtime-call-initialize-by-name) :pointer
  (session :pointer)
  (full-name (:struct %string-view-t))
  (out-call :pointer))

(cffi:defcfun ("iree_runtime_call_deinitialize" %runtime-call-deinitialize) :void
  (call :pointer))

(cffi:defcfun ("iree_runtime_call_invoke" %runtime-call-invoke) :pointer
  (call :pointer)
  (flags :uint32))

(cffi:defcfun ("iree_runtime_call_inputs_push_back_buffer_view"
               %runtime-call-inputs-push-back-buffer-view)
    :pointer
  (call :pointer)
  (buffer-view :pointer))

(cffi:defcfun ("iree_runtime_call_outputs_pop_front_buffer_view"
               %runtime-call-outputs-pop-front-buffer-view)
    :pointer
  (call :pointer)
  (out-buffer-view :pointer))

;; iree_runtime_call_flag_bits_t（call.h:28-31）に定義済みのフラグは無い。
(defconstant +runtime-call-flags-none+ 0)

;;; ------------------------------------------------------------------------
;;; hal/buffer_view.h, hal/buffer_view_util.h, hal/buffer.h,
;;; hal/buffer_transfer.h
;;; ------------------------------------------------------------------------

(cffi:defcstruct %hal-buffer-params-t
  (usage :uint32)
  (access :uint16)
  (type :uint32)
  (queue-affinity :uint64)
  (min-alignment :size))

(cffi:defcfun ("iree_hal_buffer_view_allocate_buffer_copy" %hal-buffer-view-allocate-buffer-copy) :pointer
  (device :pointer)
  (allocator :pointer)
  (shape-rank :size)
  (shape :pointer)
  (element-type :uint32)
  (encoding-type :uint32)
  (buffer-params (:struct %hal-buffer-params-t))
  (initial-data (:struct %const-byte-span-t))
  (out-buffer-view :pointer))

(cffi:defcfun ("iree_hal_buffer_view_release" %hal-buffer-view-release) :void
  (buffer-view :pointer))

(cffi:defcfun ("iree_hal_buffer_view_buffer" %hal-buffer-view-buffer) :pointer
  (buffer-view :pointer))

(cffi:defcfun ("iree_hal_buffer_view_shape_rank" %hal-buffer-view-shape-rank) :size
  (buffer-view :pointer))

(cffi:defcfun ("iree_hal_buffer_view_shape_dim" %hal-buffer-view-shape-dim) :size
  (buffer-view :pointer)
  (index :size))

(cffi:defcfun ("iree_hal_buffer_view_element_type" %hal-buffer-view-element-type) :uint32
  (buffer-view :pointer))

(cffi:defcfun ("iree_hal_buffer_view_byte_length" %hal-buffer-view-byte-length) :size
  (buffer-view :pointer))

(cffi:defcfun ("iree_hal_device_transfer_d2h" %hal-device-transfer-d2h) :pointer
  (device :pointer)
  (source :pointer)
  (source-offset :size)
  (target :pointer)
  (data-length :size)
  (flags :uint32)
  (timeout (:struct %timeout-t)))

;; hal/buffer_view.h の enum（IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR = 1）。
(defconstant +hal-encoding-type-dense-row-major+ 1)

;; hal/buffer_transfer.h の iree_hal_transfer_buffer_flags_t（既定値は 0）。
(defconstant +hal-transfer-buffer-flag-default+ 0)

;; hal/buffer.h の enum 値。ビット演算はすべてヘッダのコメントどおり
;; 計算し直す（記憶や以前のバージョンから持ち込まない、CLAUDE.md の約束）。
;;   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE      = 1u << 0   (buffer.h:195)
;;   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET      = 1u << 1   (buffer.h:207)
;;   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ  = 1u << 10 (buffer.h:257)
;;   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE = 1u << 11 (buffer.h:261)
(defconstant +hal-buffer-usage-default+
  (logior (ash 1 0) (ash 1 1) (ash 1 10) (ash 1 11)))

;; IREE_HAL_MEMORY_ACCESS_READ/WRITE/DISCARD = 1u<<0 / 1u<<1 / 1u<<2
;; (buffer.h:131,136,140)。
(defconstant +hal-memory-access-all+
  (logior (ash 1 0) (ash 1 1) (ash 1 2)))

;; IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE = 1u<<4、それに 1u<<5 を足したものが
;; IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL（buffer.h:88,95）。
(defconstant +hal-memory-type-device-local+
  (logior (ash 1 4) (ash 1 5)))

;; IREE_HAL_QUEUE_AFFINITY_ANY = (iree_hal_queue_affinity_t)(-1)、
;; iree_hal_queue_affinity_t は uint64_t（hal/queue.h:33,36）。
(defconstant +hal-queue-affinity-any+ #xFFFFFFFFFFFFFFFF)

;;; ------------------------------------------------------------------------
;;; vm/module.h, vm/context.h
;;; ------------------------------------------------------------------------

;; iree_vm_function_linkage_t（vm/module.h:33-44）。
(defconstant +vm-function-linkage-export+ 2)

(cffi:defcfun ("iree_vm_module_name" %vm-module-name) (:struct %string-view-t)
  (module :pointer))

(cffi:defcstruct %vm-module-signature-t
  (version :uint32)
  (attr-count :size)
  (import-function-count :size)
  (export-function-count :size)
  (internal-function-count :size))

(cffi:defcfun ("iree_vm_module_signature" %vm-module-signature) (:struct %vm-module-signature-t)
  (module :pointer))

(cffi:defcfun ("iree_vm_module_lookup_function_by_ordinal" %vm-module-lookup-function-by-ordinal) :pointer
  (module :pointer)
  (linkage :int)
  (ordinal :size)
  (out-function :pointer))

(cffi:defcfun ("iree_vm_function_name" %vm-function-name) (:struct %string-view-t)
  (function :pointer))

(cffi:defcfun ("iree_vm_context_module_count" %vm-context-module-count) :size
  (context :pointer))

(cffi:defcfun ("iree_vm_context_module_at" %vm-context-module-at) :pointer
  (context :pointer)
  (index :size))
