;;;; skip-unless-pjrt と plugin-path の挙動（プラグインが無くても動く）。

(in-package #:nabla.pjrt.tests)

(test (plugin-path-follows-nabla-pjrt-home :suite :nabla.small)
  "plugin-path は NABLA_PJRT_HOME の下の cpu/ と cuda/ を指す。"
  (let ((old (sb-ext:posix-getenv "NABLA_PJRT_HOME")))
    (unwind-protect
         (progn
           (sb-posix:setenv "NABLA_PJRT_HOME" "/nonexistent-pjrt-home" 1)
           (is (string= "/nonexistent-pjrt-home/cpu/xla_cpu_pjrt.so"
                        (namestring (plugin-path :cpu))))
           (is (string= "/nonexistent-pjrt-home/cuda/xla_cuda_plugin.so"
                        (namestring (plugin-path :cuda))))
           (is (not (pjrt-available-p :kind :cpu)))
           (signals pjrt-plugin-not-found (load-plugin :cpu)))
      (if old
          (sb-posix:setenv "NABLA_PJRT_HOME" old 1)
          (sb-posix:unsetenv "NABLA_PJRT_HOME")))))
