;;;; nabla/ffi-support のパッケージ（issue #79）。
;;;;
;;;; C ライブラリを FFI で呼ぶ層（IREE、将来の PJRT）が共有する、SBCL の
;;;; シグナルハンドラと浮動小数点トラップの保護。core（nabla）にも
;;;; 特定のライブラリにも依存しない。export するのは、呼び出し側が
;;;; 使うマクロと関数だけ。

(defpackage #:nabla.ffi-support
  (:use #:cl)
  (:export
   #:with-lisp-signal-handlers-preserved
   #:with-all-float-traps-masked)
  (:documentation
   "外部ライブラリ呼び出しから SBCL のシグナルハンドラと浮動小数点トラップの
設定を守る内部用の道具。他の % 接頭辞のシンボルは、呼び出し側（IREE など）が
:import-from で明示的に取り込む。"))
