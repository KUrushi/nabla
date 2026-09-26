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
   #:iree-compile-error-message)
  (:documentation
   "IREE 連携用のパッケージ。埋め込み C API を CFFI で呼び、StableHLO の
テキストから vmfb のバイト列を得るコンパイラのバインディング（このフェーズ）
と、IREE ランタイムでの実行のバインディング（後続のフェーズ）を持つ。"))
