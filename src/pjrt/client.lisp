;;;; PJRT のクライアントとデバイス（issue #85）。
;;;;
;;;; クライアント（PJRT_Client*）は PJRT_Client_Create で作り、プラグインの
;;;; 計算資源（XLA の CPU なら Eigen のスレッドプールなど）を持つ。
;;;;
;;;; シグナルハンドラ（issue #5 の LLVM 問題）について、実地に調べた結果:
;;;; 固定した CPU プラグイン（xla-cpu-pjrt 0.0.1、PJRT API 0.81）で、
;;;; dlopen + GetPjrtApi、PJRT_Plugin_Initialize、PJRT_Client_Create、
;;;; PJRT_Client_PlatformName / AddressableDevices、PJRT_Client_BufferFromHostBuffer、
;;;; PJRT_Buffer_ToHostBuffer の前後で、全シグナル（1..64）の処分
;;;; （nabla.ffi-support::%signal-handler-address。特に SBCL が GC の
;;;; stop-the-world に使う SIGUSR2）が1つも変わらないことを確かめた
;;;; （tests/pjrt/executable-test.lisp が毎回検査する）。つまりこの段階の PJRT
;;;; には、IREE の ensure-compiler-loaded のような「世界を止めた1点で LLVM の
;;;; シグナルハンドラ登録を済ませる」手順は要らない（LLVM の
;;;; RegisterHandlers を呼ぶ経路が無い）。
;;;; XLA の CPU はコンパイル（PJRT_Client_Compile、#87）でも LLVM を動かすが、
;;;; そこでも処分は変わらないことを確かめた（src/pjrt/executable.lisp 冒頭）。
;;;; 保険として、クライアントの作成は
;;;; with-lisp-signal-handlers-preserved で包む（処分が変わっていたら元に戻す。
;;;; 多重防御）。また XLA の CPU は Eigen のスレッドプールを作るので、
;;;; スレッド生成の瞬間の MXCSR を継承させないため、作成を
;;;; with-all-float-traps-masked で包む（docs/float-traps-experiments.md）。
;;;;
;;;; 既知の制限: 別スレッドが sleep なしの tight loop で (sb-ext:gc :full t) を
;;;; 回し続けると、PJRT の呼び出し（Client_Create や転送）が極端に遅くなり、
;;;; 固まったように見える。調べた結果、nabla/pjrt のバグでもデッドロックでも
;;;; なく、SBCL の GC ロックが公平でないために他のスレッドが進めなくなる
;;;; 餓死（src/ffi-support/signals.lisp の「残る課題」3）で、IREE でも同程度に
;;;; 起きる。GC の間に眠れば問題ない（tests/pjrt/executable-test.lisp は 10ms 眠る）。
;;;;
;;;; 寿命: クライアントはバッファより先に破棄してはならない。バッファの
;;;; finalizer（device-array.lisp）はオブジェクト本体を捕まえられないので、
;;;; クライアントの foreign 側の状態を CLIENT-STATE（device-array とは別の
;;;; 構造体）に分け、「生きているバッファの数」を数える。PJRT_Client_Destroy は
;;;; 「所有者（PJRT-CLIENT オブジェクト）が消えた」かつ「生きているバッファが
;;;; 0」になった時点で、最後に行った方が実行する。

(in-package #:nabla.pjrt)

(defstruct (client-state (:constructor %make-client-state (api-address client-address)))
  "PJRT_Client* の foreign 側の状態。finalizer が捕まえてよいのは、整数と
この構造体だけ（PJRT-CLIENT や device-array を捕まえてはいけない）。
ロックは使わず、アトミック操作だけで更新する: finalizer は SBCL の
finalizer スレッドで動き、GC の stop-the-world で止められうる。Lisp の
ロックを持ったまま止められると、別スレッドがそのロックを待って GC と
噛み合わなくなりうるため（子プロセスの試験で、別スレッドが GC を回し続ける
中でハングした）。"
  (api-address 0 :type (unsigned-byte 64) :read-only t)
  (client-address 0 :type (unsigned-byte 64) :read-only t)
  (live-buffers 0 :type sb-ext:word)
  (owner-alive-p t)
  (destroyed-p nil))

(defun %client-state-api (state)
  (cffi:make-pointer (client-state-api-address state)))

(defun %client-state-destroy-if-unused (state)
  "所有者がいなくなり、生きているバッファも無ければ PJRT_Client_Destroy する。
destroyed-p を CAS で立てた1つのスレッドだけが実行する（所有者が消える側と
最後のバッファを解放する側が同時に到達しても1回だけ）。"
  (when (and (not (client-state-owner-alive-p state))
             (zerop (client-state-live-buffers state))
             (null (sb-ext:compare-and-swap (client-state-destroyed-p state) nil t)))
    (let ((api (%client-state-api state)))
      (%pjrt-call (api "PJRT_Client_Destroy" args (:struct %client-destroy-args)
                       (client (cffi:make-pointer (client-state-client-address state))))))))

(defun %client-state-buffer-created (state)
  "生きているバッファを1つ数える。"
  (sb-ext:atomic-incf (client-state-live-buffers state)))

(defun %client-state-destroy-buffer (state buffer-address)
  "BUFFER-ADDRESS（PJRT_Buffer* の整数アドレス）を PJRT_Buffer_Destroy し、
生きているバッファの数を1つ減らす。所有者が既にいなくて、これが最後の
バッファなら、クライアントも破棄する。finalizer からも呼ばれるので、
エラーは握りつぶす（finalizer スレッドで例外を出しても誰も受けられない）。"
  (unwind-protect
       (ignore-errors
        (let ((api (%client-state-api state)))
          (%pjrt-call (api "PJRT_Buffer_Destroy" args (:struct %buffer-destroy-args)
                           (buffer (cffi:make-pointer buffer-address))))))
    (sb-ext:atomic-decf (client-state-live-buffers state))
    (ignore-errors (%client-state-destroy-if-unused state))))

(defun %client-state-owner-gone (state)
  "所有者（PJRT-CLIENT）が消えたことを記録し、生きているバッファが無ければ
クライアントを破棄する。"
  (setf (client-state-owner-alive-p state) nil)
  (sb-thread:barrier (:memory))
  (ignore-errors (%client-state-destroy-if-unused state)))

(defclass pjrt-client ()
  ((state :reader %pjrt-client-state :initarg :state)
   (platform-name :reader pjrt-client-platform-name :initarg :platform-name)
   (devices :reader %pjrt-client-devices :initarg :devices
            :documentation "addressable なデバイス（PJRT_Device*）のリスト。
寿命はクライアントと同じ。"))
  (:documentation
   "PJRT_Client を包む。所有者としての寿命は、このオブジェクトが GC されるまで
（そのとき、生きているバッファが残っていれば、最後のバッファが解放された後に
クライアントを破棄する。ファイル冒頭参照）。"))

(defun %pjrt-client-api (client)
  (%client-state-api (%pjrt-client-state client)))

(defun %pjrt-client-pointer (client)
  (cffi:make-pointer (client-state-client-address (%pjrt-client-state client))))

(defun %client-platform-name (api client-pointer)
  (%pjrt-call (api "PJRT_Client_PlatformName" args (:struct %client-platform-name-args)
                   (client client-pointer))
    (cffi:foreign-string-to-lisp
     (cffi:foreign-slot-value args '(:struct %client-platform-name-args) 'platform-name)
     :count (cffi:foreign-slot-value args '(:struct %client-platform-name-args)
                                     'platform-name-size)
     :encoding :utf-8)))

(defun %client-addressable-devices (api client-pointer)
  (%pjrt-call (api "PJRT_Client_AddressableDevices" args
                   (:struct %client-addressable-devices-args) (client client-pointer))
    (let ((devices (cffi:foreign-slot-value args '(:struct %client-addressable-devices-args)
                                            'addressable-devices))
          (count (cffi:foreign-slot-value args '(:struct %client-addressable-devices-args)
                                          'num-addressable-devices)))
      (loop for i below count collect (cffi:mem-aref devices :pointer i)))))

(defvar *plugin-initialized* (make-hash-table)
  "kind -> PJRT_Plugin_Initialize を呼んだかどうか。")

(defun %initialize-plugin (kind api)
  "PJRT_Plugin_Initialize を KIND ごとにプロセスにつき1回だけ呼ぶ（ヘッダが
「他の関数より先に呼ぶこと」と定めている）。*plugin-lock* の外から呼ぶので、
自前のロックで1回だけにする。"
  (sb-thread:with-mutex (*plugin-lock*)
    (unless (gethash kind *plugin-initialized*)
      (with-lisp-signal-handlers-preserved
        (with-all-float-traps-masked
          (%pjrt-call (api "PJRT_Plugin_Initialize" args (:struct %plugin-initialize-args)))))
      (setf (gethash kind *plugin-initialized*) t))))

(defun make-pjrt-client (kind)
  "KIND（:cpu / :cuda）のプラグインをロードし（まだなら）、PJRT_Client を作って
PJRT-CLIENT を返す。プラグインが無ければ PJRT-PLUGIN-NOT-FOUND。
LLVM / XLA を動かしうる入口なので、with-lisp-signal-handlers-preserved と
with-all-float-traps-masked で包む（ファイル冒頭参照）。"
  (let ((api (load-plugin kind)))
    (%initialize-plugin kind api)
    (let* ((client-pointer
             (with-lisp-signal-handlers-preserved
               (with-all-float-traps-masked
                 (%pjrt-call (api "PJRT_Client_Create" args (:struct %client-create-args))
                   (cffi:foreign-slot-value args '(:struct %client-create-args) 'client)))))
           (state (%make-client-state (cffi:pointer-address api)
                                      (cffi:pointer-address client-pointer)))
           (client nil))
      ;; クライアントを包むのに失敗したら、foreign 側のクライアントを破棄する。
      (let ((done nil))
        (unwind-protect
             (progn
               (setf client (make-instance 'pjrt-client
                                           :state state
                                           :platform-name (%client-platform-name api client-pointer)
                                           :devices (%client-addressable-devices api client-pointer)))
               (setf done t))
          (unless done (%client-state-owner-gone state))))
      (tg:finalize client (lambda () (%client-state-owner-gone state)))
      client)))
