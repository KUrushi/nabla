;;;; FiveAM のスイート定義。
;;;;
;;;; スイートはここで1度だけ定義する。他の全システムのテストファイルは
;;;; (in-suite :nabla.small) のように、このキーワードで参加する。
;;;; システムごとの親スイートは、このフェーズでは作らない。

(in-package #:nabla.tests.support)

(def-suite :nabla.small
  :description "1プロセス内で完結し、FFI・ファイル・スレッドを使わないテスト。既定で実行する。")

(def-suite :nabla.medium
  :description "1台のマシン内で完結するテスト（IREE の local バックエンド、ファイル I/O など）。既定で実行する。")

(def-suite :nabla.large
  :description "GPU での実行や JAX フィクスチャの再生成を伴うテスト。手動または定期実行のみ。")
