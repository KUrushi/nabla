;;;; nabla/pjrt のパッケージ（issue #78）。
;;;;
;;;; PJRT（XLA の外部向け C API）のプラグイン（.so）を CFFI で dlopen して
;;;; 使う層。core（nabla）は PJRT の名前を知らない。この段階（#78）は
;;;; プラグインの場所の解決・ロード・API の版の読み出しまで。#85 でクライアント・
;;;; デバイス・device-array（to-device / to-host）を足した。コンパイル・実行は
;;;; 後続の issue（#87）。

(defpackage #:nabla.pjrt
  (:use #:cl)
  ;; TO-DEVICE / TO-HOST / DEVICE-ARRAY-AVAL は core (nabla) の backend
  ;; プロトコルの総称関数。nabla.pjrt の device-array がこれらのメソッドに
  ;; なるよう import-from する（nabla.iree と同じ。#:nabla は :use しない）。
  (:import-from #:nabla #:to-device #:to-host #:device-array-aval)
  (:import-from #:nabla.ffi-support
                #:with-lisp-signal-handlers-preserved
                #:with-all-float-traps-masked)
  (:export
   #:pjrt-home
   #:pjrt-available-p
   #:plugin-path
   #:load-plugin
   #:plugin-api-version
   #:pjrt-plugin-not-found
   ;; バックエンド・device-array（issue #85）
   #:pjrt-backend
   #:pjrt-backend-platform-name
   #:device-array
   #:device-array-released-p
   #:release-device-array
   ;; コンディション
   #:pjrt-error
   #:pjrt-error-message
   #:pjrt-error-context
   #:pjrt-object-released
   #:pjrt-object-released-kind)
  (:documentation
   "PJRT C API プラグインのロードと、API の版の読み出し（nabla/pjrt）。"))
