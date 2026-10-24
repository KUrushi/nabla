;;;; 1 つの dispatch の binding の上限（32）に当たる IREE 3.11 の制限と、その回避策
;;;; （issue #264、docs/iree-upstream-bugs.md の 10、medium）。
;;;;
;;;; 守らせる性質:
;;;; - 制限が IREE に残っていること（直ったら失敗する。compile-flags の
;;;;   --iree-dispatch-creation-fuse-multi-use=false を外せるかの合図）: 最小の再現
;;;;   docs/iree-repros/elementwise-chain-32-inputs.mlir（32 個の入力を逐次に足す連鎖）をフラグなしで
;;;;   iree-compile して iree-run-module に流すと、local-task でも local-sync でも
;;;;   "binding count 33 > 32" で失敗し、31 個の連鎖は通る。同じ 32 個の連鎖をフラグ付きで
;;;;   コンパイルすると通る（フラグが効いていること）。呼び出しは子プロセスで行う。
;;;; - 回避策: 葉が 32〜161 個（161 は local-task の入力の上限、docs の 6）の要素ごとの加算の連鎖を
;;;;   local-task の backend で jit した結果は eager と一致する。

(in-package #:nabla.iree.tests)

;;; ---- 今も落ちること ----

(defun %run-chain-repro (file device &rest compile-flags)
  "docs/iree-repros/FILE を COMPILE-FLAGS を足して iree-compile し、iree-run-module --device=DEVICE で
子プロセスとして実行して (values 終了コード 出力) を返す。入力は main の引数の個数ぶんの
--input=3xf32=1,2,3。コンパイルに失敗したら :compile-failed。"
  (let* ((source (%repro-path file))
         (vmfb (namestring (uiop:tmpize-pathname #p"/tmp/nabla-binding-limit.vmfb")))
         (inputs (with-open-file (stream source)
                   (loop for line = (read-line stream)
                         when (search "func.func @main" line)
                           return (count #\: line :end (position #\) line)))))
         (compile-code
           (nth-value 2 (uiop:run-program
                         (append (list (%iree-bin "iree-compile")
                                       "--iree-hal-target-device=local"
                                       "--iree-hal-local-target-device-backends=llvm-cpu")
                                 compile-flags
                                 (list source "-o" vmfb))
                         :ignore-error-status t :output nil :error-output nil))))
    (unwind-protect
         (if (zerop compile-code)
             (multiple-value-bind (out err code)
                 (uiop:run-program
                  (append (list (%iree-bin "iree-run-module") (format nil "--device=~A" device)
                                (format nil "--module=~A" vmfb) "--function=main")
                          (loop repeat inputs collect "--input=3xf32=1,2,3"))
                  :ignore-error-status t :output :string :error-output :string)
               (values code (concatenate 'string out err)))
             :compile-failed)
      (ignore-errors (delete-file vmfb)))))

(define-iree-test binding-limit/iree-known-limit-32-input-chain-fails-at-load
    "IREE 3.11 の制限が残っていること: 32 個の入力を逐次に足す連鎖は、フラグなしでコンパイルすると
local-task でも local-sync でも \"binding count 33 > 32\" で失敗し、31 個の連鎖は通る。
--iree-dispatch-creation-fuse-multi-use=false を付けると 32 個の連鎖も通る。直ると失敗する——
そのときは compile-flags（src/iree/compiler.lisp）のそのフラグと docs/iree-upstream-bugs.md の 10 を
外せるか判断する。"
  (skip-unless-iree :library :both)
  (unless (and (probe-file (%iree-bin "iree-run-module"))
               (probe-file (%iree-bin "iree-compile")))
    (skip "iree-compile / iree-run-module が無い")
    (return-from iree-test))
  (dolist (device '("local-task" "local-sync"))
    (is (eql 0 (%run-chain-repro "elementwise-chain-31-inputs.mlir" device))
        "~A: 31 個の連鎖は通るはず" device)
    (multiple-value-bind (code output) (%run-chain-repro "elementwise-chain-32-inputs.mlir" device)
      (is (and (integerp code) (/= 0 code) (search "binding count 33 > 32" output))
          "~A: 32 個の連鎖が \"binding count 33 > 32\" にならなかった（制限が直った？）: ~S ~A"
          device code output))
    (multiple-value-bind (code output)
        (%run-chain-repro "elementwise-chain-32-inputs.mlir" device
                          "--iree-dispatch-creation-fuse-multi-use=false")
      (is (eql 0 code) "~A: フラグ付きの 32 個の連鎖が失敗した: ~S ~A" device code output))))

;;; ---- 回避策 ----

(defparameter *bl-add* (nb:with-tracing (a b) (+ a b)))

(defun %bl-chain-function ()
  "配列のリストを受け取り、先頭から逐次に足した和を返す TRACEABLE-FUNCTION（(reduce #'+ xs) の形）。"
  (nb::%make-traceable-function
   (list (gensym "XS"))
   (lambda (xs) (reduce (lambda (acc x) (funcall *bl-add* acc x)) xs))))

(defun %bl-chain-matches-eager-p (backend n)
  "N 個の f32 の (3) の配列の逐次の和を BACKEND で jit した結果が eager と一致するか。"
  (let* ((f (%bl-chain-function))
         (jf (nb:jit f :backend backend))
         (xs (loop for i below n collect (make-random-array (make-array-spec '(3) :f32) :seed (+ 300 i)))))
    (unwind-protect
         (allclose (funcall jf xs) (funcall f xs) :dtype :f32)
      (nb::%jit-cache-forget f))))

(define-iree-test binding-limit/local-task-runs-elementwise-chains-of-32-to-161-leaves
    "葉が 32〜161 個（境界の 32・161 と、その間のランダムな個数）の要素ごとの加算の連鎖を、local-task の
backend で jit した結果は eager と一致する（回避策の --iree-dispatch-creation-fuse-multi-use=false が
無いと、32 個以上はロード時に binding count の上限で失敗する）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:make-backend :iree :target :local))
        (*num-trials* 6)
        (nb:*compile-cache-directory* nil))
    (dolist (n '(32 161))
      (is (%bl-chain-matches-eager-p backend n) "葉 ~D 個の連鎖" n))
    (is (check-it (generator (uniform-integer :lo 33 :hi 160))
                  (lambda (n) (%bl-chain-matches-eager-p backend n))))
    (gc-and-run-finalizers)))
