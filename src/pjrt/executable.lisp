;;;; PJRT のコンパイル・シリアライズ・ロード・実行（issue #87）。
;;;;
;;;; 流れ（backend プロトコルとの対応は backend.lisp）:
;;;;   StableHLO テキスト --PJRT_Client_Compile--> PJRT_LoadedExecutable
;;;;     --PJRT_LoadedExecutable_GetExecutable--> PJRT_Executable
;;;;     --PJRT_Executable_Serialize--> バイト列（backend-compile の戻り値）
;;;;   バイト列 --PJRT_Executable_DeserializeAndLoad--> PJRT_LoadedExecutable（backend-load）
;;;;   PJRT_LoadedExecutable_Execute（backend-invoke）
;;;;
;;;; 構造体の配置は third_party/pjrt/pjrt_c_api.h（API 0.116）から写した。固定した
;;;; プラグインは API 0.81 で、使う関数はすべて 0.81 にある（実地に確かめた。
;;;; ffi.lisp の %api-function がプラグインの struct_size を見て保証する）。
;;;;
;;;; シグナルハンドラ（issue #5 の LLVM 問題）: XLA の CPU コンパイルは LLVM を
;;;; 動かす。実地に調べた結果（固定した CPU プラグイン xla-cpu-pjrt 0.0.1、API 0.81）:
;;;; with-lisp-signal-handlers-preserved を外した素の状態で、クライアント作成後に
;;;; PJRT_Client_Compile（tanh を含む StableHLO）・PJRT_Executable_Serialize・
;;;; PJRT_Executable_DeserializeAndLoad・PJRT_LoadedExecutable_Execute を呼び、
;;;; 各段階の後で全シグナル（1..64）の処分
;;;; （nabla.ffi-support::%signal-handler-address）を最初と比べたが、1つも
;;;; 変わらなかった（SIGUSR2 も同じ）。XLA は LLVM の RegisterHandlers
;;;; （llvm::sys::AddSignalHandler / PrintStackTraceOnErrorSignal）を呼ばない
;;;; ビルドらしい。よって IREE の ensure-compiler-loaded のような「世界を止めた
;;;; 1点で最初の登録を済ませる」手順は要らない。ただしプラグインのビルドが
;;;; 変わると前提が崩れうるので、コンパイルとロードは
;;;; with-lisp-signal-handlers-preserved（処分が変わっていたら元に戻す多重防御。
;;;; 世界を止めない分、戻すまでの窓に別スレッドが GC を始める競合は残るが、
;;;; 今のプラグインでは登録自体が起きないので実害はない）と
;;;; with-all-float-traps-masked（XLA / LLVM が作るスレッドへ MXCSR を継承
;;;; させず、実行中の FP 例外が Lisp のハンドラに乗り込まないようにする）で
;;;; 包む。Execute も with-all-float-traps-masked で包む。この性質は
;;;; tests/pjrt/executable-test.lisp の子プロセス検査（別スレッドが GC を回し続ける
;;;; 中での compile / load / invoke の前後でシグナルの処分と FP モードが
;;;; 変わらない）が毎回確かめる。もしそれが落ちるようになったら、
;;;; src/iree/library.lisp の ensure-compiler-loaded を参考に、初回の
;;;; PJRT_Client_Compile を %call-with-world-stopped の中で行う。

(in-package #:nabla.pjrt)

(cffi:defcstruct %program
  (struct-size :size) (extension-start :pointer)
  (code :pointer) (code-size :size)
  (format :pointer) (format-size :size))

(cffi:defcstruct %client-compile-args
  (struct-size :size) (extension-start :pointer) (client :pointer)
  (program :pointer)
  (compile-options :pointer) (compile-options-size :size)
  (executable :pointer))

(cffi:defcstruct %loaded-executable-get-executable-args
  (struct-size :size) (extension-start :pointer)
  (loaded-executable :pointer) (executable :pointer))

(cffi:defcstruct %executable-serialize-args
  (struct-size :size) (extension-start :pointer) (executable :pointer)
  (serialized-bytes :pointer) (serialized-bytes-size :size)
  (serialized-executable :pointer) (serialized-executable-deleter :pointer))

(cffi:defcstruct %executable-deserialize-and-load-args
  (struct-size :size) (extension-start :pointer) (client :pointer)
  (serialized-executable :pointer) (serialized-executable-size :size)
  (loaded-executable :pointer)
  (overridden-compile-options :pointer) (overridden-compile-options-size :size)
  (load-options :pointer))

(cffi:defcstruct %executable-destroy-args
  (struct-size :size) (extension-start :pointer) (executable :pointer))

(cffi:defcstruct %executable-num-outputs-args
  (struct-size :size) (extension-start :pointer) (executable :pointer)
  (num-outputs :size))

(cffi:defcstruct %execute-options
  (struct-size :size) (extension-start :pointer)
  (send-callbacks :pointer) (recv-callbacks :pointer)
  (num-send-ops :size) (num-recv-ops :size)
  (launch-id :int)
  (non-donatable-input-indices :pointer) (num-non-donatable-input-indices :size)
  (context :pointer) (call-location :pointer)
  (num-tasks :size) (task-ids :pointer) (incarnation-ids :pointer)
  (multi-slice-config :pointer)
  (use-major-to-minor-data-layout-for-callbacks :bool)
  (hlo-output-callbacks :pointer) (num-hlo-output-callbacks :size)
  (custom-options :pointer) (num-custom-options :size))

(cffi:defcstruct %loaded-executable-execute-args
  (struct-size :size) (extension-start :pointer) (executable :pointer)
  (options :pointer) (argument-lists :pointer)
  (num-devices :size) (num-args :size)
  (output-lists :pointer) (device-complete-events :pointer)
  (execute-device :pointer))

(cffi:defcstruct %buffer-element-type-args
  (struct-size :size) (extension-start :pointer) (buffer :pointer)
  (type :int))

(cffi:defcstruct %buffer-dimensions-args
  (struct-size :size) (extension-start :pointer) (buffer :pointer)
  (dims :pointer) (num-dims :size))

(defparameter *compile-options*
  (coerce '(#x20 #x01 #x1a #x04 #x20 #x01 #x28 #x01) '(simple-array (unsigned-byte 8) (*)))
  "PJRT_Client_Compile に渡す、シリアライズした xla.CompileOptionsProto の最小形。
protobuf の手書きエンコードで、中身は
  compile_portable_executable (field 4, varint: タグ 0x20) = 1
  executable_build_options (field 3, 長さ区切り: タグ 0x1a, 長さ 4) {
    num_replicas   (ExecutableBuildOptionsProto field 4, タグ 0x20) = 1
    num_partitions (同 field 5, タグ 0x28) = 1 }
だけ。field 番号は jaxlib の xla_client.CompileOptions に num_replicas =
num_partitions = 1 を設定して SerializeAsString した出力で確かめた（確認できたのは
field 番号だけで、jaxlib の出力にはこの他にも多数のフィールドがある。他の項目は
XLA の既定に任せる）。空のオプション（0バイト）を渡すと、プラグインが
PJRT_Client_Compile の CHECK で SIGABRT するので、この値は必須。
compile_portable_executable=1 にするのは、実行体にデバイス割り当てを焼き込ませない
ためで、PJRT_LoadedExecutable_Execute の execute_device（backend が選んだデバイス）と
焼き込みの割り当てが食い違わなくなる（CUDA で device-index が 0 でないとき
問題になる）。ポータブルな実行体は execute_device を指定して実行する。")

(defun %destroy-executable (api name pointer)
  "NAME（PJRT_Executable_Destroy / PJRT_LoadedExecutable_Destroy）で POINTER を破棄する。"
  (%pjrt-call (api name args (:struct %executable-destroy-args) (executable pointer))))

(defun %get-executable (api loaded)
  "LOADED（PJRT_LoadedExecutable*）から PJRT_Executable* を取り出す。
呼び出し側が PJRT_Executable_Destroy する。"
  (%pjrt-call (api "PJRT_LoadedExecutable_GetExecutable" args
                   (:struct %loaded-executable-get-executable-args)
                   (loaded-executable loaded))
    (cffi:foreign-slot-value args '(:struct %loaded-executable-get-executable-args) 'executable)))

(defun %loaded-executable-num-outputs (api loaded)
  "LOADED の出力の数（GetExecutable で得た PJRT_Executable に NumOutputs を問う）。"
  (let ((executable (%get-executable api loaded)))
    (unwind-protect
         (%pjrt-call (api "PJRT_Executable_NumOutputs" args (:struct %executable-num-outputs-args)
                          (executable executable))
           (cffi:foreign-slot-value args '(:struct %executable-num-outputs-args) 'num-outputs))
      (%destroy-executable api "PJRT_Executable_Destroy" executable))))

(defun %serialize-loaded-executable (api loaded)
  "LOADED をシリアライズしたバイト列（(unsigned-byte 8) の simple-array）を返す。
PJRT が持つバッファは、Lisp へコピーした後で deleter により解放する。"
  (let ((executable (%get-executable api loaded)))
    (unwind-protect
         (%pjrt-call (api "PJRT_Executable_Serialize" args (:struct %executable-serialize-args)
                          (executable executable))
           (let* ((size (cffi:foreign-slot-value args '(:struct %executable-serialize-args)
                                                 'serialized-bytes-size))
                  (bytes (cffi:foreign-slot-value args '(:struct %executable-serialize-args)
                                                  'serialized-bytes))
                  (backing (cffi:foreign-slot-value args '(:struct %executable-serialize-args)
                                                    'serialized-executable))
                  (deleter (cffi:foreign-slot-value args '(:struct %executable-serialize-args)
                                                    'serialized-executable-deleter))
                  (octets (make-array size :element-type '(unsigned-byte 8))))
             (unwind-protect
                  (dotimes (i size) (setf (aref octets i) (cffi:mem-aref bytes :uint8 i)))
               (unless (cffi:null-pointer-p deleter)
                 (cffi:foreign-funcall-pointer deleter () :pointer backing :void)))
             octets))
      (%destroy-executable api "PJRT_Executable_Destroy" executable))))

(defun %client-compile (client text)
  "TEXT（StableHLO のテキスト）を CLIENT でコンパイルし、シリアライズした
実行体のバイト列を返す。コンパイルで作った PJRT_LoadedExecutable はここで破棄する
（ロードは BACKEND-LOAD が改めて行う）。"
  (let* ((api (%pjrt-client-api client))
         (code (sb-ext:string-to-octets text :external-format :utf-8))
         (format-octets (sb-ext:string-to-octets "mlir" :external-format :ascii))
         (options-octets *compile-options*))
    (cffi:with-foreign-object (program '(:struct %program))
      (%zero-args program '(:struct %program))
      (sb-sys:with-pinned-objects (code format-octets options-octets)
        (setf (cffi:foreign-slot-value program '(:struct %program) 'code) (sb-sys:vector-sap code)
              (cffi:foreign-slot-value program '(:struct %program) 'code-size) (length code)
              (cffi:foreign-slot-value program '(:struct %program) 'format)
              (sb-sys:vector-sap format-octets)
              (cffi:foreign-slot-value program '(:struct %program) 'format-size)
              (length format-octets))
        (let ((loaded
                (with-lisp-signal-handlers-preserved
                  (with-all-float-traps-masked
                    (%pjrt-call (api "PJRT_Client_Compile" args (:struct %client-compile-args)
                                     (client (%pjrt-client-pointer client))
                                     (program program)
                                     (compile-options (sb-sys:vector-sap options-octets))
                                     (compile-options-size (length options-octets)))
                      (cffi:foreign-slot-value args '(:struct %client-compile-args)
                                               'executable))))))
          (unwind-protect (%serialize-loaded-executable api loaded)
            (%destroy-executable api "PJRT_LoadedExecutable_Destroy" loaded)))))))

(defstruct (pjrt-module (:constructor %make-pjrt-module (client address num-outputs)))
  "BACKEND-LOAD が返す不透明な module。ADDRESS は PJRT_LoadedExecutable* の
整数アドレス。RELEASED-P は BACKEND-UNLOAD の CAS で立てる。CLIENT を強い参照で
持つので、client はこの module より先には破棄されない。finalizer は持たない
ので、使い終わったら BACKEND-UNLOAD すること（jit のキャッシュは行う）。"
  client
  (address 0 :type (unsigned-byte 64) :read-only t)
  (num-outputs 0 :type fixnum :read-only t)
  (released-p nil))

(defun %client-load (client octets)
  "OCTETS（%client-compile が返したバイト列）を CLIENT へロードして PJRT-MODULE を返す。"
  (let ((api (%pjrt-client-api client))
        (octets (coerce octets '(simple-array (unsigned-byte 8) (*)))))
    (sb-sys:with-pinned-objects (octets)
      (let ((loaded
              (with-lisp-signal-handlers-preserved
                (with-all-float-traps-masked
                  (%pjrt-call (api "PJRT_Executable_DeserializeAndLoad" args
                                   (:struct %executable-deserialize-and-load-args)
                                   (client (%pjrt-client-pointer client))
                                   (serialized-executable (sb-sys:vector-sap octets))
                                   (serialized-executable-size (length octets)))
                    (cffi:foreign-slot-value args '(:struct %executable-deserialize-and-load-args)
                                             'loaded-executable)))))
            (done nil))
        (unwind-protect
             (prog1 (%make-pjrt-module client (cffi:pointer-address loaded)
                                       (%loaded-executable-num-outputs api loaded))
               (setf done t))
          (unless done
            (%destroy-executable api "PJRT_LoadedExecutable_Destroy" loaded)))))))

(defun %module-unload (module)
  "MODULE の PJRT_LoadedExecutable を破棄する。2回目以降は何もしない。"
  (when (null (sb-ext:compare-and-swap (pjrt-module-released-p module) nil t))
    (%destroy-executable (%pjrt-client-api (pjrt-module-client module))
                         "PJRT_LoadedExecutable_Destroy"
                         (cffi:make-pointer (pjrt-module-address module))))
  nil)

(defun %buffer-aval (api buffer)
  "BUFFER（PJRT_Buffer*）の ElementType と Dimensions から NABLA:AVAL を作る。"
  (let* ((type (%pjrt-call (api "PJRT_Buffer_ElementType" args (:struct %buffer-element-type-args)
                                (buffer buffer))
                 (cffi:foreign-slot-value args '(:struct %buffer-element-type-args) 'type)))
         (shape (%pjrt-call (api "PJRT_Buffer_Dimensions" args (:struct %buffer-dimensions-args)
                                 (buffer buffer))
                  (let ((dims (cffi:foreign-slot-value args '(:struct %buffer-dimensions-args)
                                                       'dims))
                        (count (cffi:foreign-slot-value args '(:struct %buffer-dimensions-args)
                                                        'num-dims)))
                    (loop for i below count collect (cffi:mem-aref dims :int64 i)))))
         (dtype (car (rassoc type *buffer-types*))))
    (unless dtype (error 'nabla:unsupported-dtype :dtype (list :pjrt-buffer-type type)))
    (nabla:make-aval shape dtype)))

(defun %wrap-output-buffers (client buffers)
  "BUFFERS（実行が返した PJRT_Buffer* のリスト）の所有権を device-array へ移して
リストで返す。途中で失敗したら、既に包んだ分は release-device-array、まだ包んで
いない分は PJRT_Buffer_Destroy で解放してからエラーを伝える。"
  (let ((api (%pjrt-client-api client))
        (wrapped nil)
        (pending buffers)
        (done nil))
    (unwind-protect
         (progn
           (loop while pending
                 do (push (%wrap-buffer (first pending) client (%buffer-aval api (first pending)))
                          wrapped)
                    (pop pending))
           (setf done t)
           (nreverse wrapped))
      (unless done
        (mapc #'release-device-array wrapped)
        (dolist (buffer pending)
          (ignore-errors
           (%pjrt-call (api "PJRT_Buffer_Destroy" args (:struct %buffer-destroy-args)
                            (buffer buffer)))))))))

(defun %module-invoke (module device arrays)
  "MODULE を、ARRAYS（device-array のリスト）を引数に DEVICE（PJRT_Device*）
1台で実行し、出力の device-array を多値で返す。実行の完了イベントを待ってから返す。
同じ MODULE に対する BACKEND-UNLOAD と本関数を並行させてはならない（unload が
実行中の PJRT_LoadedExecutable を破棄しうる。呼び出し側が直列化する）。"
  (when (pjrt-module-released-p module)
    (error 'pjrt-object-released :kind :module :context "invoke"
                                 :message "the module was already unloaded"))
  (let* ((client (pjrt-module-client module))
         (api (%pjrt-client-api client))
         (num-args (length arrays))
         (num-outputs (pjrt-module-num-outputs module)))
    (dolist (array arrays)
      (unless (eq (device-array-client array) client)
        (error "the device-array belongs to a different PJRT client"))
      (%live-device-array-pointer array "invoke"))
    (cffi:with-foreign-objects ((options '(:struct %execute-options))
                                (arg-list :pointer (max num-args 1))
                                (arg-lists :pointer 1)
                                (out-list :pointer (max num-outputs 1))
                                (out-lists :pointer 1)
                                (events :pointer 1))
      (%zero-args options '(:struct %execute-options))
      (loop for array in arrays for i from 0
            do (setf (cffi:mem-aref arg-list :pointer i) (%device-array-pointer array)))
      (setf (cffi:mem-aref arg-lists :pointer 0) arg-list
            (cffi:mem-aref out-lists :pointer 0) out-list
            (cffi:mem-aref events :pointer 0) (cffi:null-pointer))
      (dotimes (i num-outputs) (setf (cffi:mem-aref out-list :pointer i) (cffi:null-pointer)))
      (with-all-float-traps-masked
        (%pjrt-call (api "PJRT_LoadedExecutable_Execute" args
                         (:struct %loaded-executable-execute-args)
                         (executable (cffi:make-pointer (pjrt-module-address module)))
                         (options options)
                         (argument-lists arg-lists)
                         (num-devices 1)
                         (num-args num-args)
                         (output-lists out-lists)
                         (device-complete-events events)
                         (execute-device device))))
      ;; 出力の包みに失敗しても、完了イベントは必ず待って破棄する
      ;; （実行は既に始まっている）。
      (let ((outputs nil) (done nil))
        (unwind-protect
             (progn
               (setf outputs (%wrap-output-buffers
                              client (loop for i below num-outputs
                                           collect (cffi:mem-aref out-list :pointer i))))
               (%await-and-destroy-event api (cffi:mem-aref events :pointer 0))
               (setf done t)
               (values-list outputs))
          (unless done
            (mapc #'release-device-array outputs)
            (unless outputs
              (ignore-errors
               (%await-and-destroy-event api (cffi:mem-aref events :pointer 0))))))))))
