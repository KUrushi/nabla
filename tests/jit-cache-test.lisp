;;;; jit-cache-test: jit キャッシュのモジュール解放・並行性・リスタート・
;;;; defjit の :STATIC-ARGS（issue #71）。フェイク backend だけを使う。
;;;; スレッドや GC を使うテストは :NABLA.MEDIUM、それ以外は :NABLA.SMALL。
;;;; IREE 経由の end-to-end は tests/iree/jit-cache-test.lisp にある。

(in-package #:nabla.tests)

(defclass %recording-fake-backend (fake-backend)
  ((lock :initform (sb-thread:make-mutex :name "recording-fake-backend") :reader %recording-lock)
   (loaded :initform nil :accessor %recording-loaded)
   (unloaded :initform nil :accessor %recording-unloaded))
  (:documentation
   "BACKEND-LOAD が返した module と、BACKEND-UNLOAD に渡された module を
記録する FAKE-BACKEND（テスト専用）。finalizer は別スレッドで走りうるので、
記録はロックで守る。"))

(defmethod nabla:backend-load :around ((backend %recording-fake-backend) octets)
  (declare (ignore octets))
  (let ((module (call-next-method)))
    (sb-thread:with-mutex ((%recording-lock backend))
      (push module (%recording-loaded backend)))
    module))

(defmethod nabla:backend-unload :after ((backend %recording-fake-backend) module)
  (sb-thread:with-mutex ((%recording-lock backend))
    (pushnew module (%recording-unloaded backend))))

(defun %unloaded-count (backend)
  (sb-thread:with-mutex ((%recording-lock backend))
    (length (%recording-unloaded backend))))

(defun %all-loaded-unloaded-p (backend)
  "BACKEND が読み込んだ module がすべて（EQ で）BACKEND-UNLOAD されていれば真。"
  (sb-thread:with-mutex ((%recording-lock backend))
    (null (set-difference (%recording-loaded backend) (%recording-unloaded backend)))))

(in-suite :nabla.small)

;;; --- エントリが消えるときに module を解放する ---

(test jit-cache/forget-unloads-every-module
  "形の違う2回の呼び出しで2つの module を読み込んだ後、%JIT-CACHE-FORGET は
その2つをどちらも BACKEND-UNLOAD する。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (make-instance '%recording-fake-backend))
         (f (nb:with-tracing (a b) (+ a b)))
         (jf (nb:jit f :backend backend)))
    (funcall jf (make-random-array (make-array-spec '(2 3) :f32) :seed 1)
             (make-random-array (make-array-spec '(2 3) :f32) :seed 2))
    (funcall jf (make-random-array (make-array-spec '(3 2) :f32) :seed 3)
             (make-random-array (make-array-spec '(3 2) :f32) :seed 4))
    (is (= 2 (length (%recording-loaded backend))))
    (is (= 0 (%unloaded-count backend)))
    (is (= 2 (nb::%jit-cache-forget f)))
    (is (= 2 (%unloaded-count backend)))
    (is (%all-loaded-unloaded-p backend))))

(test jit-cache/defjit-redefinition-unloads-old-modules
  "同じ DEFJIT を再評価すると、前の定義が読み込んだ module は BACKEND-UNLOAD
される。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (make-instance '%recording-fake-backend))
         (nb:*default-backend* backend)
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 5)))
    (eval '(nb:defjit %jit-cache-test-add (a b) (+ a b)))
    (funcall '%jit-cache-test-add a a)
    (is (= 0 (%unloaded-count backend)))
    (eval '(nb:defjit %jit-cache-test-add (a b) (+ a b)))
    (is (= 1 (%unloaded-count backend)))
    (is (%all-loaded-unloaded-p backend))))

;;; --- RECOMPILE はループ、USE-EAGER は失敗した graph を使い回す ---

(defclass %failing-fake-backend (fake-backend)
  ((failures-left :initarg :failures :accessor %failures-left))
  (:documentation
   "最初の FAILURES 回の BACKEND-COMPILE だけ BACKEND-ERROR を signal し、
その後は FAKE-BACKEND と同じに戻る backend（テスト専用）。"))

(defmethod nabla:backend-compile ((backend %failing-fake-backend) text)
  (if (plusp (%failures-left backend))
      (progn
        (decf (%failures-left backend))
        (error 'nabla.tests.support::%fake-backend-unsupported-op
               :format-control "failing: わざと失敗する" :format-arguments nil))
      (call-next-method)))

(test jit-cache/recompile-restart-does-not-nest
  "BACKEND-COMPILE が N 回（1..30）続けて失敗し、ハンドラが毎回 RECOMPILE を
選んでも、各失敗の時点のスタックの深さ（バックトレースのフレーム数）は
変わらない（%JIT-CALL を再帰で呼び直すと、失敗のたびに深くなり、常に
RECOMPILE を選ぶハンドラでは際限なく深くなる）。最後は成功して eager と
一致する。"
  (is (check-it (generator (integer 1 30))
                (lambda (failures)
                  (let* ((nb:*compile-cache-directory* nil)
                         (backend (make-instance '%failing-fake-backend :failures failures))
                         (f (nb:with-tracing (a b) (+ a b)))
                         (jf (nb:jit f :backend backend))
                         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 6))
                         (depths nil)
                         (result nil))
                    (handler-bind ((nb:jit-compile-error
                                     (lambda (c)
                                       (declare (ignore c))
                                       (push (length (sb-debug:list-backtrace)) depths)
                                       (invoke-restart 'nb:recompile))))
                      (setf result (funcall jf a a)))
                    (and (= failures (length depths))
                         (every (lambda (depth) (= depth (first depths))) depths)
                         (allclose result (funcall f a a) :dtype :f32)))))))

(defvar *%jit-cache-trace-count* 0
  "jit-cache/use-eager-does-not-retrace が、トレースされる本体から数える
トレース回数。")

(defun %jit-cache-note-trace ()
  (incf *%jit-cache-trace-count*))

(test jit-cache/use-eager-does-not-retrace
  "コンパイルに失敗して USE-EAGER を選んだ呼び出しは、本体を1回しか
トレースしない（JIT-COMPILE-ERROR が運ぶ graph を使い回す）。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (f (nb:with-tracing (a b) (progn (%jit-cache-note-trace) (- a b))))
         (jf (nb:jit f :backend backend))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 7))
         (b (make-random-array (make-array-spec '(2 3) :f32) :seed 8))
         (*%jit-cache-trace-count* 0)
         (result nil))
    (handler-bind ((nb:jit-compile-error (lambda (c) (declare (ignore c)) (invoke-restart 'nb:use-eager))))
      (setf result (funcall jf a b)))
    (is (= 1 *%jit-cache-trace-count*))
    (is (allclose result (funcall f a b) :dtype :f32))))

;;; --- jit の入れ子 ---

(test jit-cache/nested-jit-on-tracers-is-inlined
  "jit した関数を、別の jit した関数の本体からトレーサを引数に呼ぶと、内側は
外側の graph に展開される（内側を別にコンパイルしない。compile-count は1）。
結果は eager と一致する（eager の OUTER は内側を具体的な配列で呼ぶので、
compile-count はその前に確かめる）。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (inner (nb:jit (nb:with-tracing (a b) (+ a b)) :backend backend))
         (outer (nb:with-tracing (a b) (funcall inner a b)))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 9))
         (b (make-random-array (make-array-spec '(2 3) :f32) :seed 10))
         (result (funcall (nb:jit outer :backend backend) a b)))
    (is (= 1 (fake-backend-compile-count backend)))
    (is (allclose result (funcall outer a b) :dtype :f32))))

(test jit-cache/nested-jit-on-concrete-values-compiles-both
  "外側のトレース中に、内側の jit した関数を（静的引数の）具体的な配列で呼ぶと、
内側も普通に jit され、ロックの再帰エラーにならない（compile-count は2）。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (inner (nb:jit (nb:with-tracing (a b) (+ a b)) :backend backend))
         (outer (nb:with-tracing (a b c) (progn (funcall inner c c) (+ a b))))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 11))
         (b (make-random-array (make-array-spec '(2 3) :f32) :seed 12)))
    (is (allclose (funcall (nb:jit outer :static-args '(2) :backend backend) a b a)
                  (funcall outer a b a)
                  :dtype :f32))
    (is (= 2 (fake-backend-compile-count backend)))))

;;; --- defjit の :STATIC-ARGS ---

(test jit-cache/defjit-accepts-static-args
  "(DEFJIT (NAME :STATIC-ARGS '(2)) (A W MODE) ...) の MODE は静的引数になる:
同じ形の引数でも MODE ごとに1回だけコンパイルし、結果は eager と一致する。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (nb:*default-backend* backend)
         (a (make-random-array (make-array-spec '(3 3) :f32) :seed 13))
         (w (make-random-array (make-array-spec '(3 3) :f32) :seed 14)))
    (eval '(nb:defjit (%jit-cache-test-op :static-args '(2)) (a w mode)
            (if (eq mode :add) (+ a w) (nb:dot a w))))
    (let ((eager (get '%jit-cache-test-op 'nb::%defjit-traceable)))
      (is (allclose (%jit-cache-test-op a w :add) (funcall eager a w :add) :dtype :f32))
      (is (allclose (%jit-cache-test-op a w :dot) (funcall eager a w :dot) :dtype :f32))
      (%jit-cache-test-op a w :dot)
      (is (= 2 (fake-backend-compile-count backend))))))

(test jit-cache/defjit-rejects-invalid-static-args
  "DEFJIT の :STATIC-ARGS が範囲外なら、JIT と同じく JIT-ERROR になる。"
  (signals nb:jit-error
    (eval '(nb:defjit (%jit-cache-test-bad :static-args '(5)) (a) a))))

;;; --- medium: GC と並行性 ---

(in-suite :nabla.medium)

(defun %jit-and-drop (backend seed)
  "新しい TRACEABLE-FUNCTION を jit して1回呼び、参照を残さずに戻る。"
  (let ((a (make-random-array (make-array-spec '(2 3) :f32) :seed seed)))
    (funcall (nb:jit (nb:with-tracing (a b) (+ a b)) :backend backend) a a)
    nil))

(test jit-cache/gc-unloads-modules
  "jit した関数への参照が無くなり GC されると、その module は BACKEND-UNLOAD
される。保守的なスタックルートでごく少数が回収されずに残ることがあるので
（tests/iree/support.lisp の GC-AND-RUN-FINALIZERS 参照）、50個中45個以上で
確かめる。"
  (let ((nb:*compile-cache-directory* nil)
        (backend (make-instance '%recording-fake-backend)))
    (dotimes (i 50) (%jit-and-drop backend i))
    (is (= 50 (length (%recording-loaded backend))))
    (sb-ext:gc :full t)
    (sb-kernel:run-pending-finalizers)
    (is (<= 45 (%unloaded-count backend)))))

(defun %make-jit-thread (function)
  "FUNCTION を新しいスレッドで呼ぶ。新しいスレッドは呼び出し元の動的束縛を
引き継がないので、スレッドの中で NB:*COMPILE-CACHE-DIRECTORY* を NIL に
束縛し直す（ディスクキャッシュがコンパイルを隠さないように）。FUNCTION が
ERROR を signal したら、プロセスごと落とさずにそのコンディションを
スレッドの戻り値にする（テストの比較が失敗として報告する）。"
  (sb-thread:make-thread
   (lambda ()
     (handler-case (let ((nb:*compile-cache-directory* nil)) (funcall function))
       (error (condition) condition)))))

(defclass %rendezvous-fake-backend (fake-backend)
  ((inside :initform (list 0) :reader %rendezvous-inside
           :documentation "CAR が BACKEND-COMPILE に入ったスレッドの数（ATOMIC-INCF で数える）。")
   (met :initform nil :accessor %rendezvous-met))
  (:documentation
   "BACKEND-COMPILE の中で、別のスレッドの BACKEND-COMPILE がもう1つ入って
くるのを最大5秒待つ FAKE-BACKEND（テスト専用）。先に入ったスレッドが
待っている間に2つ目が入ってきたら MET を真にする。コンパイルが直列化されて
いれば、待ち合わせは成立しない。"))

(defmethod nabla:backend-compile ((backend %rendezvous-fake-backend) text)
  (declare (ignore text))
  (let* ((cell (%rendezvous-inside backend))
         (first (= 0 (sb-ext:atomic-incf (car cell))))
         (deadline (+ (get-internal-real-time) (* 5 internal-time-units-per-second))))
    (loop until (or (<= 2 (car cell)) (>= (get-internal-real-time) deadline))
          do (sleep 0.01))
    (when (and first (<= 2 (car cell)))
      (setf (%rendezvous-met backend) t)))
  (call-next-method))

(test jit-cache/different-functions-compile-concurrently
  "別々の関数の jit を2つのスレッドで同時に呼ぶと、2つのコンパイルは同時に
進む（片方のコンパイル中にもう片方が BACKEND-COMPILE に入れる）。どちらの
結果も eager と一致する。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (make-instance '%rendezvous-fake-backend))
         (f1 (nb:with-tracing (a b) (+ a b)))
         (f2 (nb:with-tracing (a w) (nb:dot a w)))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 14))
         (w (make-random-array (make-array-spec '(3 4) :f32) :seed 15))
         (t1 (%make-jit-thread (lambda () (funcall (nb:jit f1 :backend backend) a a))))
         (t2 (%make-jit-thread (lambda () (funcall (nb:jit f2 :backend backend) a w)))))
    (let ((r1 (sb-thread:join-thread t1 :timeout 30 :default :timeout))
          (r2 (sb-thread:join-thread t2 :timeout 30 :default :timeout)))
      (is (%rendezvous-met backend))
      (is (allclose r1 (funcall f1 a a) :dtype :f32))
      (is (allclose r2 (funcall f2 a w) :dtype :f32)))))

(defclass %slow-fake-backend (fake-backend)
  ((compile-lock :initform (sb-thread:make-mutex :name "slow-fake") :reader %slow-compile-lock))
  (:documentation
   "BACKEND-COMPILE に 0.2 秒かかる FAKE-BACKEND（テスト専用）。
COMPILE-COUNT の更新をロックで守る。"))

(defmethod nabla:backend-compile ((backend %slow-fake-backend) text)
  (declare (ignore text))
  (sleep 0.2)
  (sb-thread:with-mutex ((%slow-compile-lock backend))
    (call-next-method)))

(test jit-cache/same-key-concurrent-calls-compile-once
  "同じ関数を同じ引数の形で4つのスレッドから同時に呼んでも、コンパイルは
1回だけで、4つの結果はどれも eager と一致する。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (make-instance '%slow-fake-backend))
         (f (nb:with-tracing (a b) (+ a b)))
         (jf (nb:jit f :backend backend))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 16))
         (threads (loop repeat 4 collect (%make-jit-thread (lambda () (funcall jf a a)))))
         (results (mapcar (lambda (thread) (sb-thread:join-thread thread :timeout 30 :default :timeout)) threads)))
    (is (= 1 (fake-backend-compile-count backend)))
    (is (every (lambda (result) (allclose result (funcall f a a) :dtype :f32)) results))))
