;;;; PJRT-BACKEND: core の backend プロトコル（src/backend.lisp）を実装する、
;;;; PJRT 向けの NABLA:BACKEND（issue #85。この段階は to-device / to-host まで。
;;;; backend-compile / backend-load / backend-invoke は #87）。
;;;;
;;;; クライアントは MAKE-BACKEND の時点で作る（IREE と違って遅延しない）。
;;;; プラグインが無ければ、PJRT-PLUGIN-NOT-FOUND ではなく
;;;; NABLA:BACKEND-NOT-AVAILABLE を signal する（core の契約）。

(in-package #:nabla.pjrt)

(defclass pjrt-backend (nabla:backend)
  ((target :initarg :target :reader nabla:backend-target
           :documentation ":cpu または :cuda。")
   (client :initarg :client :reader %pjrt-backend-client)
   (device :initarg :device :reader %pjrt-backend-device
           :documentation "to-device の送り先の PJRT_Device*（addressable な
デバイスのうち DEVICE-INDEX 番目）。"))
  (:documentation
   "NABLA:BACKEND の PJRT 実装。MAKE-BACKEND :PJRT で作る。"))

(defun pjrt-backend-platform-name (backend)
  "BACKEND のプラグインが PJRT_Client_PlatformName で返した文字列（\"cpu\" など）。"
  (pjrt-client-platform-name (%pjrt-backend-client backend)))

(defmethod nabla:make-backend ((kind (eql :pjrt)) &key (target :cpu) (device-index 0))
  "TARGET（:cpu / :cuda）のプラグインで PJRT のクライアントを作り、その
addressable なデバイスの DEVICE-INDEX 番目を送り先にする PJRT-BACKEND を返す。
プラグインの .so が無ければ NABLA:BACKEND-NOT-AVAILABLE。DEVICE-INDEX が
範囲外なら ERROR。"
  (check-type target (member :cpu :cuda))
  (unless (pjrt-available-p :kind target)
    (error 'nabla:backend-not-available :kind kind))
  (let* ((client (make-pjrt-client target))
         (devices (%pjrt-client-devices client)))
    (unless (< -1 device-index (length devices))
      (error "device-index ~D is out of range: the client has ~D addressable device~:P"
             device-index (length devices)))
    (make-instance 'pjrt-backend :target target :client client
                                 :device (nth device-index devices))))

(defmethod to-device (array (backend pjrt-backend) &key dtype)
  "ARRAY（simple-array）を BACKEND のデバイスへコピーし、DEVICE-ARRAY を返す。
f32 / f64 / bf16 / f16 に対応する（:i1 などは NABLA:UNSUPPORTED-DTYPE）。
ARRAY が simple-array でなければ TYPE-ERROR、要素型と DTYPE が矛盾すれば
NABLA:DTYPE-MISMATCH。"
  (%client-to-device (%pjrt-backend-client backend) (%pjrt-backend-device backend)
                     array dtype))
