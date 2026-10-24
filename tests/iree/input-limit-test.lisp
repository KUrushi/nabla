;;;; IREE 3.11 の local-task device は、入力が 162 個以上の関数を呼ぶと NULL を参照して
;;;; プロセスごと落ちる（issue #199、docs/iree-upstream-bugs.md の 6）。nabla.iree はその呼び出しを
;;;; 拒否する（iree-too-many-inputs）。守らせる性質:
;;;; - local-task で入力 162 個の関数を invoke すると、落ちずに iree-too-many-inputs を signal する。
;;;; - jit は葉が 161 個を超えると dtype ごとに連結して渡す（issue #242）: local-task で葉 170 個・dtype と
;;;;   device array の混ざった葉 200 個の木が動き、eager と一致する。backend が上限を申告するのは local-task だけ。
;;;; - 境界の 161 個は（連結せずに）動く。
;;;; - 出力は 162 個以上でも落ちない（連結を入力だけにする根拠。子プロセス）。
;;;; - local-sync device（make-backend :iree :driver :local-sync）なら 170 個でも動き、結果は eager と一致する。
;;;; - バグが IREE に残っていること（直ったら失敗する。検査と回避策を外す合図）: 最小の再現
;;;;   （docs/iree-repros/）を iree-run-module で local-task に流すと 162 個で落ち、161 個は通る。
;;;;   落ちる呼び出しはテストプロセスの外（子プロセス）で行う。

(in-package #:nabla.iree.tests)

(defun %input-limit-arrays (n)
  (loop for i below n collect (make-random-array (make-array-spec '(3) :f32) :seed i)))

(defun %input-limit-jit-call (backend n)
  "N 個の rank 1 の配列のリストを、葉ごとに tanh(x+x) する関数を BACKEND で jit して呼んだ結果。"
  (let* ((f (pytree-map-function))
         (jf (nb:jit f :backend backend))
         (arrays (%input-limit-arrays n)))
    (unwind-protect
         (values (funcall jf arrays) (funcall f arrays))
      (nb::%jit-cache-forget f))))

(defun %read-repro (file)
  (uiop:read-file-string (asdf:system-relative-pathname "nabla" (format nil "docs/iree-repros/~A" file))))

(define-iree-test input-limit/invoke-on-local-task-refuses-162-inputs-instead-of-crashing
    "local-task device で入力 162 個の関数を invoke すると、iree-too-many-inputs を signal する
（プロセスは落ちない）。エラーは個数と上限を持つ。jit はこの手前で葉を連結するので、ここに来るのは
invoke を直接呼んだときだけ。"
  (skip-unless-iree :library :both)
  (with-device (device :local-task)
    (with-session (session device)
      (session-append-module session (compile-stablehlo (%read-repro "local-task-162-inputs.mlir")))
      (let ((arguments (loop repeat 162 collect (to-device (make-random-array (make-array-spec '(3) :f32)) device))))
        (unwind-protect
             (handler-case (progn (apply #'invoke session "module.main" arguments)
                                  (fail "iree-too-many-inputs にならなかった"))
               (nabla.iree:iree-too-many-inputs (condition)
                 (is (= 162 (nabla.iree:iree-too-many-inputs-count condition)))
                 (is (= nabla.iree:+local-task-max-inputs+ (nabla.iree:iree-too-many-inputs-limit condition)))
                 (is (search "local-sync" (princ-to-string condition)))))
          (mapc #'release-device-array arguments))))))

(define-iree-test input-limit/backend-max-inputs-is-161-only-on-local-task
    "IREE の backend は local-task（:local の既定、:local-task、文字列 \"local-task\"）のときだけ入力の上限 161 を
申告し、local-sync では上限なし（NIL）。jit はこの値を超える葉を連結する（issue #242）。"
  (skip-unless-iree :library :both)
  (dolist (driver '(nil :local :local-task "local-task"))
    (is (eql nabla.iree:+local-task-max-inputs+
             (nabla:backend-max-inputs (nabla:make-backend :iree :target :local :driver driver)))
        "driver ~S" driver))
  (is (null (nabla:backend-max-inputs (nabla:make-backend :iree :target :local :driver :local-sync)))))

(define-iree-test input-limit/jit-on-local-task-concatenates-170-leaves
    "local-task で葉 170 個の木を jit すると、葉を dtype ごとに連結して渡すので落ちずに動き、eager と一致する
（issue #242。連結しなければ入力 170 個で iree-too-many-inputs になる形）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local))
        (nb:*compile-cache-directory* nil))
    (multiple-value-bind (result expected) (%input-limit-jit-call backend 170)
      (is (= 170 (length result)))
      (is (every (lambda (r e) (allclose r e :dtype :f32)) result expected)))
    (gc-and-run-finalizers)))

(define-iree-test input-limit/jit-on-local-task-concatenates-mixed-dtypes-and-device-arrays
    "local-task で、f32・f64・i32 の葉と device array（bf16 を含む）が混ざった 200 個の葉の木を jit すると、
dtype ごとに連結されて動き、各葉が eager と一致する（葉ごとに x + x）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:make-backend :iree :target :local))
         (nb:*compile-cache-directory* nil)
         (dtypes '(:f32 :f64 :i32 :bf16))
         (hosts (loop for i below 200
                      collect (make-random-array (make-array-spec (subseq '(2 3 1) 0 (mod i 3))
                                                                  (nth (mod i 4) dtypes))
                                                 :seed i)))
         (leaves (loop for host in hosts for i from 0
                       collect (if (or (eq :bf16 (nth (mod i 4) dtypes)) (zerop (mod i 5)))
                                   (nabla:to-device host backend :dtype (nth (mod i 4) dtypes))
                                   host)))
         (f (nb:with-tracing (xs) (nb:tree-map (lambda (x) (+ x x)) xs)))
         (result (funcall (nb:jit f :backend backend) leaves)))
    (is (= 200 (length result)))
    (loop for r in result for host in hosts for i from 0
          for dtype = (nth (mod i 4) dtypes)
          for expected = (if (eq dtype :bf16)
                             (nabla:to-host (funcall (nb:jit (nb:with-tracing (x) (+ x x)) :backend backend)
                                                     (nabla:to-device host backend :dtype :bf16)))
                             (funcall f host))
          do (is (if (eq dtype :i32) (equalp r expected) (allclose r expected :dtype dtype))
                 "葉 ~D（~S）が一致しない" i dtype))
    (gc-and-run-finalizers)))

(define-iree-test input-limit/local-task-runs-161-inputs
    "境界の入力 161 個は local-task でも動き、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local))
        (nb:*compile-cache-directory* nil))
    (multiple-value-bind (result expected) (%input-limit-jit-call backend 161)
      (is (= 161 (length result)))
      (is (every (lambda (r e) (allclose r e :dtype :f32)) result expected)))
    (gc-and-run-finalizers)))

(define-iree-test input-limit/local-sync-runs-170-inputs
    "local-sync device なら入力 170 個の関数も動き、eager と一致する（回避策）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local :driver :local-sync))
        (nb:*compile-cache-directory* nil))
    (multiple-value-bind (result expected) (%input-limit-jit-call backend 170)
      (is (= 170 (length result)))
      (is (every (lambda (r e) (allclose r e :dtype :f32)) result expected)))
    (gc-and-run-finalizers)))

(defun %iree-bin (name)
  (namestring (merge-pathnames (format nil "bin/~A" name) (nabla.iree::iree-home))))

(defun %run-repro-on-local-task (file)
  "docs/iree-repros/FILE を iree-compile してから iree-run-module --device=local-task で
子プロセスとして実行し、終了コードを返す。入力の個数は main の引数の個数から数える。"
  (let* ((source (namestring (asdf:system-relative-pathname "nabla" (format nil "docs/iree-repros/~A" file))))
         (vmfb (namestring (uiop:tmpize-pathname #p"/tmp/nabla-input-limit.vmfb")))
         (inputs (with-open-file (stream source)
                   (loop for line = (read-line stream)
                         when (search "func.func @main" line)
                           return (count #\: line :end (position #\) line)))))
         (compile-code
           (nth-value 2 (uiop:run-program
                         (list (%iree-bin "iree-compile")
                               "--iree-hal-target-device=local"
                               "--iree-hal-local-target-device-backends=llvm-cpu"
                               source "-o" vmfb)
                         :ignore-error-status t :output nil :error-output nil))))
    (unwind-protect
         (progn
           (unless (zerop compile-code) (return-from %run-repro-on-local-task :compile-failed))
           (nth-value 2 (uiop:run-program
                         (append (list (%iree-bin "iree-run-module")
                                       "--device=local-task" (format nil "--module=~A" vmfb)
                                       "--function=main")
                                 (loop repeat inputs collect "--input=3xf32=1,2,3"))
                         :ignore-error-status t :output nil :error-output nil)))
      (ignore-errors (delete-file vmfb)))))

(define-iree-test input-limit/iree-known-bug-local-task-162-inputs-crashes
    "IREE 3.11 のバグが残っていること: 入力 162 個の最小の再現は local-task で落ち（終了コード 128
超）、161 個は通る。直ると失敗する——そのときは +local-task-max-inputs+ の検査
（src/iree/execute.lisp）と docs/iree-upstream-bugs.md の 6 を外す。"
  (skip-unless-iree :library :both)
  (unless (and (probe-file (%iree-bin "iree-run-module"))
               (probe-file (%iree-bin "iree-compile")))
    (skip "iree-compile / iree-run-module が無い")
    (return-from iree-test))
  (is (eql 0 (%run-repro-on-local-task "local-task-161-inputs.mlir")))
  (is (< 128 (%run-repro-on-local-task "local-task-162-inputs.mlir"))))

;;; --- 出力の個数には 162 の制限が効かない（#242。連結を入力だけにする根拠）---

(defun %generated-io-mlir (inputs outputs)
  "INPUTS 個の tensor<3xf32> を取り、OUTPUTS 個の tensor<3xf32> を返す main の MLIR。
出力 i は a0 に定数 i を足したもの（CSE で畳まれないように）。使わない入力は捨てる。"
  (with-output-to-string (s)
    (format s "module {~%func.func @main(~{%a~D: tensor<3xf32>~^, ~}) -> (~{~*tensor<3xf32>~^, ~}) {~%"
            (loop for i below inputs collect i) (loop repeat outputs collect 0))
    (dotimes (i outputs)
      (format s "  %c~D = stablehlo.constant dense<~D.0> : tensor<3xf32>~%" i i)
      (format s "  %r~D = stablehlo.add %a~D, %c~D : tensor<3xf32>~%" i (mod i inputs) i))
    (format s "  return ~{%r~D~^, ~} : ~{~*tensor<3xf32>~^, ~}~%}~%}~%"
            (loop for i below outputs collect i) (loop repeat outputs collect 0))))

(defun %run-generated-on-local-task (inputs outputs)
  (let ((file (uiop:tmpize-pathname #p"/tmp/nabla-io-probe.mlir")))
    (with-open-file (s file :direction :output :if-exists :supersede)
      (write-string (%generated-io-mlir inputs outputs) s))
    (let* ((vmfb (namestring (uiop:tmpize-pathname #p"/tmp/nabla-io-probe.vmfb")))
           (compile-code
             (nth-value 2 (uiop:run-program
                           (list (%iree-bin "iree-compile")
                                 "--iree-hal-target-device=local"
                                 "--iree-hal-local-target-device-backends=llvm-cpu"
                                 (namestring file) "-o" vmfb)
                           :ignore-error-status t :output nil :error-output nil))))
      (unwind-protect
           (if (zerop compile-code)
               (nth-value 2 (uiop:run-program
                             (append (list (%iree-bin "iree-run-module")
                                           "--device=local-task" (format nil "--module=~A" vmfb)
                                           "--function=main")
                                     (loop repeat inputs collect "--input=3xf32=1,2,3"))
                             :ignore-error-status t :output nil :error-output nil))
               :compile-failed)
        (ignore-errors (delete-file vmfb))
        (ignore-errors (delete-file file))))))

(define-iree-test input-limit/local-task-runs-many-outputs
    "local-task は出力が 162 個以上でも落ちない（入力 1 個・出力 170 個、入力 2 個・出力 300 個。子プロセスの
iree-run-module）。jit が連結するのは入力だけでよい根拠（issue #242。学習ステップは出力も 200 個を超える）。
落ちるようになったら、出力も連結する。"
  (skip-unless-iree :library :both)
  (unless (and (probe-file (%iree-bin "iree-run-module"))
               (probe-file (%iree-bin "iree-compile")))
    (skip "iree-compile / iree-run-module が無い")
    (return-from iree-test))
  (is (eql 0 (%run-generated-on-local-task 1 170)))
  (is (eql 0 (%run-generated-on-local-task 2 300))))
