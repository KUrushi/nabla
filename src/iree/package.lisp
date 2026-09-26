;;;; nabla/iree のパッケージ。
;;;;
;;;; IREE の埋め込み C API（コンパイラ・ランタイム）を CFFI で呼ぶ層を
;;;; ここに置く。共有ライブラリ（libIREECompiler.so /
;;;; libnabla_iree_runtime.so）の探索と読み込みは遅延させ、見つからない
;;;; 環境でもこのシステムのロード自体は失敗しない
;;;; （設計タブ「全体アーキテクチャ」。詳しくは src/iree/library.lisp）。
;;;;
;;;; 生の CFFI バインディング（`%` 接頭辞のシンボル）はこのパッケージの
;;;; 中だけで使い、export しない。

(defpackage #:nabla.iree
  (:use #:cl)
  (:export
   ;; コンパイラ
   #:compile-stablehlo
   #:compile-flags
   #:compiler-revision
   #:compiler-api-version
   #:iree-available-p
   ;; コンディション
   #:iree-error
   #:iree-library-not-found
   #:iree-library-not-found-path
   #:iree-library-not-found-home
   #:iree-library-not-found-library
   #:iree-compile-error
   #:iree-compile-error-phase
   #:iree-compile-error-diagnostics
   #:iree-compile-error-message
   #:iree-object-released
   #:iree-object-released-kind
   #:iree-object-released-context
   #:iree-status-error
   #:iree-status-error-code
   #:iree-status-error-message
   #:iree-status-error-context
   ;; ランタイム
   #:iree-instance
   #:driver-names
   #:make-device
   #:release-device
   #:device-released-p
   #:device-driver
   #:device-name
   #:with-device
   #:allocator-statistics
   #:allocator-statistics-host-bytes-peak
   #:allocator-statistics-host-bytes-allocated
   #:allocator-statistics-host-bytes-freed
   #:allocator-statistics-device-bytes-peak
   #:allocator-statistics-device-bytes-allocated
   #:allocator-statistics-device-bytes-freed
   #:device-allocator-statistics
   #:make-session
   #:release-session
   #:with-session
   #:session-append-module
   #:session-append-module-from-file
   #:session-lookup-function
   #:session-function-names
   #:vm-function
   #:vm-function-module
   #:vm-function-linkage
   #:vm-function-ordinal
   #:with-call
   #:call-push-buffer-view
   #:call-invoke
   #:call-pop-buffer-view
   #:buffer-view-allocate-copy
   #:buffer-view-release
   #:buffer-view-shape
   #:buffer-view-element-type
   #:buffer-view-byte-length
   #:buffer-view-read-into
   ;; device-array
   #:device-array
   #:to-device
   #:to-host
   #:release-device-array
   #:device-array-released-p
   #:device-array-aval
   #:device-array-device
   ;; execute
   #:invoke)
  (:documentation
   "IREE 連携用のパッケージ。埋め込み C API を CFFI で呼び、StableHLO の
テキストから vmfb のバイト列を得るコンパイラのバインディングと、
IREE ランタイム（instance / device / session / call / buffer_view）で
その vmfb を実行するバインディングを持つ。"))
