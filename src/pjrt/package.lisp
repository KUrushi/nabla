;;;; nabla/pjrt のパッケージ（issue #78）。
;;;;
;;;; PJRT（XLA の外部向け C API）のプラグイン（.so）を CFFI で dlopen して
;;;; 使う層。core（nabla）は PJRT の名前を知らない。この段階（#78）は
;;;; プラグインの場所の解決・ロード・API の版の読み出しまで。クライアントの
;;;; 作成やコンパイル・実行は後続の issue で足す。

(defpackage #:nabla.pjrt
  (:use #:cl)
  (:export
   #:pjrt-home
   #:pjrt-available-p
   #:plugin-path
   #:load-plugin
   #:plugin-api-version
   #:pjrt-plugin-not-found)
  (:documentation
   "PJRT C API プラグインのロードと、API の版の読み出し（nabla/pjrt）。"))
