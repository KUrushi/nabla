;;;; nabla/iree のプレースホルダーパッケージ。
;;;;
;;;; ここではまだ CFFI の define-foreign-library / use-foreign-library を
;;;; 書かない。共有ライブラリ（libIREECompiler.so / libnabla_iree_runtime.so）
;;;; の探索と読み込みは、backend を生成するときに行い、見つからなければ
;;;; ロードエラーではなくコンディションを出す方針にする
;;;; （設計タブ「全体アーキテクチャ」）。そのため、このシステムは
;;;; 共有ライブラリが1つも無い環境でも問題なくロードできる。

(defpackage #:nabla.iree
  (:use #:cl)
  (:documentation
   "IREE 連携用のパッケージ。フェーズ0では骨格のみで、公開シンボルはまだない。"))
