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
           ;; load-plugin はロード済みの API をキャッシュするので、他のテストの
           ;; 結果に左右されないよう空のキャッシュで呼ぶ。
           (let ((nabla.pjrt::*plugin-apis* (make-hash-table)))
             (signals pjrt-plugin-not-found (load-plugin :cpu))))
      (if old
          (sb-posix:setenv "NABLA_PJRT_HOME" old 1)
          (sb-posix:unsetenv "NABLA_PJRT_HOME")))))

(test (default-home-matches-lock :suite :nabla.small)
  "NABLA_PJRT_HOME の既定（pjrt-<版>）が third_party/pjrt.lock の cpu_version と一致する。"
  (let* ((lock (asdf:system-relative-pathname "nabla/pjrt" "third_party/pjrt.lock"))
         (version (with-open-file (in lock)
                    (loop for line = (read-line in nil)
                          while line
                          when (and (> (length line) 12) (string= "cpu_version=" line :end2 12))
                            return (subseq line 12)))))
    (is (stringp version))
    (is (string= (format nil ".local/share/nabla/pjrt-~A/" version)
                 nabla.pjrt::*default-pjrt-home-name*))))
