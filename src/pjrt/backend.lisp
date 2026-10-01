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
      ;; 作ったばかりのクライアントは finalizer 任せにせず、すぐ手放す。
      (%client-state-owner-gone (%pjrt-client-state client))
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

;;; --- コンパイル・ロード・実行（issue #87） ---

(defvar *plugin-sha256-cache* (make-hash-table :test 'equal)
  "プラグインの絶対パス（文字列） -> sha256 の16進文字列。")

(defvar *plugin-sha256-lock* (sb-thread:make-mutex :name "nabla-pjrt-plugin-sha256"))

(defun %plugin-sha256 (kind)
  "KIND のプラグインの .so のファイルの中身の SHA-256（16進、小文字）。
数十 MB を読むので、パスごとにプロセスにつき1回だけ計算してキャッシュする。
シリアライズした実行体はプラグインのビルドごとに互換性がないので、
backend-fingerprint に入れる（ビルドのバージョン文字列ではなく中身で区別する）。"
  (let ((path (namestring (truename (plugin-path kind)))))
    (sb-thread:with-mutex (*plugin-sha256-lock*)
      (or (gethash path *plugin-sha256-cache*)
          (setf (gethash path *plugin-sha256-cache*)
                (ironclad:byte-array-to-hex-string
                 (ironclad:digest-file :sha256 path)))))))

(defmethod nabla:backend-fingerprint ((backend pjrt-backend))
  "BACKEND-COMPILE の出力（シリアライズした実行体）を左右するもの: キャッシュ
キーの版、実装名 \"pjrt\"、プラグインの target・プラットフォーム名、PJRT API の版、
プラグインの .so の sha256。IREE の fingerprint とは先頭の実装名で必ず異なる。"
  (let ((target (nabla:backend-target backend)))
    (multiple-value-bind (major minor) (plugin-api-version (load-plugin target))
      (list "nabla-module-cache-v1" "pjrt"
            (format nil "target=~(~A~)" target)
            (format nil "platform=~A" (pjrt-backend-platform-name backend))
            (format nil "pjrt-api=~D.~D" major minor)
            (format nil "plugin-sha256=~A" (%plugin-sha256 target))))))

(defmethod nabla:backend-compile ((backend pjrt-backend) text)
  "TEXT（StableHLO）を PJRT_Client_Compile でコンパイルし、
PJRT_Executable_Serialize したバイト列を返す。失敗は PJRT-ERROR。"
  (%client-compile (%pjrt-backend-client backend) text))

(defmethod nabla:backend-load ((backend pjrt-backend) octets)
  "OCTETS（BACKEND-COMPILE の戻り値）を PJRT_Executable_DeserializeAndLoad で
ロードし、不透明な PJRT-MODULE を返す。"
  (%client-load (%pjrt-backend-client backend) octets))

(defmethod nabla:backend-unload ((backend pjrt-backend) module)
  "MODULE の PJRT_LoadedExecutable を破棄する。2回目以降は何もしない（冪等）。"
  (declare (ignore backend))
  (%module-unload module))

(defmethod nabla:backend-invoke ((backend pjrt-backend) module function-name &rest arrays)
  "MODULE を ARRAYS（BACKEND の device-array）に適用し、出力の device-array を
多値で返す。PJRT の実行体はエントリ関数を1つだけ持つので、FUNCTION-NAME は
\"main\" でなければ ERROR。"
  (unless (equal function-name "main")
    (error "a PJRT module has only the entry function \"main\", not ~S" function-name))
  (%module-invoke module (%pjrt-backend-device backend) arrays))
