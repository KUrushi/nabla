;;;; device-array: Lisp の配列と PJRT_Buffer を橋渡しするクラス（issue #85）。
;;;; nabla.iree の device-array（src/iree/device-array.lisp）と同じ役割・
;;;; 同じ解放の方針。
;;;;
;;;; メモリの所有権: device-array は生成時に PJRT_Buffer* の所有権を引き取り、
;;;; trivial-garbage:finalize で自動解放を登録する。finalizer のクロージャは
;;;; device-array 本体ではなく、バッファの整数アドレスと CLIENT-STATE
;;;; （client.lisp。device-array とは別の構造体）だけを捕まえる（本体を
;;;; 捕まえると GC がそのオブジェクトを回収できなくなる。CLAUDE.md の約束）。
;;;; release-device-array は tg:cancel-finalization を先に呼んでから自分で
;;;; 解放するので、明示的に解放した後に GC が走っても二重解放にはならない。
;;;;
;;;; クライアントとの順序: device-array は PJRT-CLIENT を強い参照で持つほか、
;;;; 生きているバッファの数を CLIENT-STATE に数えさせる。バッファの解放は
;;;; 必ず PJRT_Buffer_Destroy → （最後なら）PJRT_Client_Destroy の順になる
;;;; （client.lisp 冒頭）。

(in-package #:nabla.pjrt)

(defclass device-array ()
  ((pointer :accessor %device-array-pointer :initarg :pointer
            :documentation "PJRT_Buffer*。release-device-array の後は null-pointer。")
   (client :reader device-array-client :initarg :client
           :documentation "このバッファを作った PJRT-CLIENT（強い参照）。")
   (aval :reader device-array-aval :initarg :aval
         :documentation "形状と dtype（NABLA:AVAL）。core の
NABLA:DEVICE-ARRAY-AVAL にメソッドを足す。"))
  (:documentation
   "PJRT のデバイス上のバッファ（PJRT_Buffer）を包む、JAX の jax.Array に
相当するクラス。TO-DEVICE の出力としてのみ作られ、直接 MAKE-INSTANCE する
ことは想定していない。"))

(defun %wrap-buffer (buffer-pointer client aval)
  "BUFFER-POINTER（PJRT_Buffer*）の所有権を引き取って DEVICE-ARRAY に包む。
finalizer は整数アドレスと CLIENT-STATE だけを捕まえる。"
  (let* ((state (%pjrt-client-state client))
         (address (cffi:pointer-address buffer-pointer))
         (array (make-instance 'device-array :pointer buffer-pointer
                                             :client client :aval aval)))
    (%client-state-buffer-created state)
    (tg:finalize array (lambda () (%client-state-destroy-buffer state address)))
    array))

(defun device-array-released-p (device-array)
  "DEVICE-ARRAY が release-device-array 済みなら真を返す。"
  (cffi:null-pointer-p (%device-array-pointer device-array)))

(defun release-device-array (device-array)
  "DEVICE-ARRAY のバッファを解放する。先に tg:cancel-finalization で finalizer を
取り消し（二重解放を避ける）、PJRT_Buffer_Destroy する。2回目以降は何もしない
（idempotent）。"
  (unless (device-array-released-p device-array)
    (tg:cancel-finalization device-array)
    (let ((address (cffi:pointer-address (%device-array-pointer device-array))))
      (setf (%device-array-pointer device-array) (cffi:null-pointer))
      (%client-state-destroy-buffer
       (%pjrt-client-state (device-array-client device-array)) address))
    nil))

(defun %live-device-array-pointer (device-array &optional (context "to-host"))
  (when (device-array-released-p device-array)
    (error 'pjrt-object-released :kind :device-array :context context
                                 :message "the device-array was already released"))
  (%device-array-pointer device-array))

(defun %buffer-type (dtype)
  "DTYPE に対応する PJRT_Buffer_Type の値。無ければ NABLA:UNSUPPORTED-DTYPE。"
  (or (cdr (assoc dtype *buffer-types*))
      (error 'nabla:unsupported-dtype :dtype dtype)))

(defun %client-to-device (client device array dtype)
  "ARRAY（simple-array）を CLIENT の DEVICE（PJRT_Device*）へコピーした
DEVICE-ARRAY を返す。ホストの実体を pin したまま、転送の完了
（done_with_host_buffer の PJRT_Event）を待ってから返すので、返った時点で
ARRAY は自由に書き換えてよい。"
  (check-type array simple-array)
  (let* ((aval (nabla:array-aval array dtype))
         (element-dtype (nabla:aval-dtype aval))
         (type (%buffer-type element-dtype))
         (shape (nabla:aval-shape aval))
         (rank (length shape))
         (storage (sb-ext:array-storage-vector array))
         (api (%pjrt-client-api client)))
    (cffi:with-foreign-object (dims :int64 (max rank 1))
      (loop for dim in shape for i from 0 do (setf (cffi:mem-aref dims :int64 i) dim))
      (sb-sys:with-pinned-objects (storage)
        (multiple-value-bind (buffer event)
            (%pjrt-call (api "PJRT_Client_BufferFromHostBuffer" args
                             (:struct %client-buffer-from-host-buffer-args)
                             (client (%pjrt-client-pointer client))
                             (data (sb-sys:vector-sap storage))
                             (type type)
                             (dims dims)
                             (num-dims rank)
                             (host-buffer-semantics +host-buffer-immutable-until-transfer-completes+)
                             (device device))
              (values (cffi:foreign-slot-value
                       args '(:struct %client-buffer-from-host-buffer-args) 'buffer)
                      (cffi:foreign-slot-value
                       args '(:struct %client-buffer-from-host-buffer-args)
                       'done-with-host-buffer)))
          ;; バッファの所有権は先に device-array へ移し、転送の待ちが失敗したら
          ;; 解放してから伝える（バッファを漏らさない）。
          (let ((result (%wrap-buffer buffer client aval))
                (done nil))
            (unwind-protect
                 (progn (%await-and-destroy-event api event)
                        (setf done t))
              (unless done (release-device-array result)))
            result))))))

(defmethod to-host ((device-array device-array))
  "DEVICE-ARRAY の内容を、その aval と同じ shape・要素型を持つ新しい多次元
simple-array にコピーして返す。PJRT_Buffer_ToHostBuffer で結果配列の実体
（array-storage-vector）へ直接書かせ、PJRT_Event の完了を待つ。release-device-array
済みなら PJRT-OBJECT-RELEASED。"
  (let* ((buffer (%live-device-array-pointer device-array))
         (client (device-array-client device-array))
         (api (%pjrt-client-api client))
         (aval (device-array-aval device-array))
         (array (make-array (nabla:aval-shape aval)
                            :element-type (nabla:dtype-element-type (nabla:aval-dtype aval))))
         (storage (sb-ext:array-storage-vector array)))
    (sb-sys:with-pinned-objects (storage)
      (let ((event (%pjrt-call (api "PJRT_Buffer_ToHostBuffer" args
                                    (:struct %buffer-to-host-buffer-args)
                                    (src buffer)
                                    (dst (sb-sys:vector-sap storage))
                                    (dst-size (nabla:aval-byte-length aval)))
                     (cffi:foreign-slot-value args '(:struct %buffer-to-host-buffer-args)
                                              'event))))
        (%await-and-destroy-event api event)))
    array))

(defmethod print-object ((device-array device-array) stream)
  (print-unreadable-object (device-array stream :type t)
    (if (device-array-released-p device-array)
        (format stream "released")
        (let ((aval (device-array-aval device-array)))
          (format stream "~(~A~)[~{~D~^ ~}] ~A"
                  (nabla:aval-dtype aval)
                  (nabla:aval-shape aval)
                  (pjrt-client-platform-name (device-array-client device-array)))))))
