;;;; CUDA を使うランタイムのテスト（issue #6）。GPU が無い環境ではスキップする
;;;; （:nabla.large。既定のスイート small + medium には含まれない）。

(in-package #:nabla.iree.tests)

(fiveam:test (runtime/make-device/cuda-creates-and-releases :suite :nabla.large)
  "\"cuda\" ドライバが (driver-names) に含まれる環境でだけ、cuda の device を
作成・解放できることを確かめる。含まれない環境（GPU 無し）ではスキップする。"
  (block iree-test
    (skip-unless-iree :library :runtime)
    (unless (member "cuda" (driver-names) :test #'string=)
      (fiveam:skip "\"cuda\" ドライバが登録されていないのでスキップする")
      (return-from iree-test))
    (let ((device (make-device :cuda)))
      (is (not (device-released-p device)))
      (release-device device)
      (is (device-released-p device)))))
