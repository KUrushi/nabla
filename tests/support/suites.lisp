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

(def-suite :nabla.isolated-medium
  :description ":NABLA.MEDIUM と同じ意味論（1台のマシン内で完結する）だが、
他の medium テストと同じ SBCL プロセスで実行すると in-process の
libIREECompiler.so が壊れることが分かっているテスト専用のスイート
（issue #68、tests/iree/support.lisp の DEFINE-IREE-TEST/ISOLATED-MEDIUM）。
既定で実行するが、scripts/run-tests.sh が :NABLA.MEDIUM とは別の SBCL
プロセスで実行する。")
