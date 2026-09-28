;;;; jit キャッシュのモジュール解放・並行性・入れ子の IREE 経由 end-to-end
;;;; テスト（issue #71）。フェイク backend で確かめる性質は
;;;; tests/jit-cache-test.lisp にある。

(in-package #:nabla.iree.tests)

(defun %jit-cache-sessions-of (fn)
  "FN（jit に渡した TRACEABLE-FUNCTION）のキャッシュにある module の
session のリストを返す（内部の表を覗く。テスト専用）。"
  (let ((cache (nb::%jit-function-cache fn :create nil))
        (sessions nil))
    (when cache
      (maphash (lambda (key entry)
                 (declare (ignore key))
                 (push (nabla.iree::iree-module-session (nb::%jit-entry-module entry)) sessions))
               (nb::%jit-function-cache-entries cache)))
    sessions))

(defun %jit-and-drop-sessions (backend seed)
  "新しい TRACEABLE-FUNCTION を BACKEND で jit して1回呼び、その module の
session のリストだけを返す（FN・jit した関数への参照は残さない）。"
  (let* ((f (nb:with-tracing (a b) (+ a b)))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed seed)))
    (funcall (nb:jit f :backend backend) a a)
    (%jit-cache-sessions-of f)))

(define-iree-test jit-cache/gc-releases-sessions
  "jit した関数への参照が無くなり GC されると、その module の session は
解放される。保守的なスタックルートでごく少数が残ることがあるので
（GC-AND-RUN-FINALIZERS 参照）、10個中8個以上で確かめる。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nabla:find-backend :iree))
         (sessions (loop for seed below 10 append (%jit-and-drop-sessions backend seed))))
    (is (= 10 (length sessions)))
    (is (notany #'nabla.iree::session-released-p sessions))
    (gc-and-run-finalizers)
    (is (<= 8 (count-if #'nabla.iree::session-released-p sessions)))))

(define-iree-test jit-cache/redefinition-and-forget-release-sessions
  "DEFJIT の再定義と %JIT-CACHE-FORGET は、どちらもその場で session を
解放する。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (nb:*default-backend* (nabla:find-backend :iree))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 1)))
    (eval '(nb:defjit %jit-cache-iree-add (a b) (+ a b)))
    (%jit-cache-iree-add a a)
    (let* ((old (get '%jit-cache-iree-add 'nb::%defjit-traceable))
           (old-sessions (%jit-cache-sessions-of old)))
      (eval '(nb:defjit %jit-cache-iree-add (a b) (+ a b)))
      (is (= 1 (length old-sessions)))
      (is (every #'nabla.iree::session-released-p old-sessions)))
    (%jit-cache-iree-add a a)
    (let* ((new (get '%jit-cache-iree-add 'nb::%defjit-traceable))
           (new-sessions (%jit-cache-sessions-of new)))
      (nb::%jit-cache-forget new)
      (is (= 1 (length new-sessions)))
      (is (every #'nabla.iree::session-released-p new-sessions)))))

(define-iree-test jit-cache/concurrent-jit-of-different-functions
  "別々の関数の jit を2つのスレッドで同時に呼ぶと、どちらも（デッドロック
せずに）終わり、結果は eager と一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (f1 (nb:with-tracing (a b) (tanh (+ a b))))
         (f2 (nb:with-tracing (a w) (nb:dot a w)))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 2))
         (w (make-random-array (make-array-spec '(3 4) :f32) :seed 3))
         (threads (list (sb-thread:make-thread
                         (lambda () (let ((nb:*compile-cache-directory* nil))
                                      (funcall (nb:jit f1 :backend backend) a a))))
                        (sb-thread:make-thread
                         (lambda () (let ((nb:*compile-cache-directory* nil))
                                      (funcall (nb:jit f2 :backend backend) a w))))))
         (results (mapcar (lambda (thread) (sb-thread:join-thread thread :timeout 120 :default :timeout))
                          threads)))
    (is (allclose (first results) (funcall f1 a a) :dtype :f32))
    (is (allclose (second results) (funcall f2 a w) :dtype :f32))
    (nb::%jit-cache-forget f1)
    (nb::%jit-cache-forget f2)))

(define-iree-test jit-cache/nested-jit
  "jit した関数の本体から別の jit した関数を、トレーサ（外側に展開される）と
静的引数の具体的な配列（内側も jit される）の両方で呼べ、結果は eager と
一致する。"
  (skip-unless-iree :library :both)
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nabla:find-backend :iree))
         (inner-f (nb:with-tracing (a b) (* a b)))
         (inner (nb:jit inner-f :backend backend))
         (outer (nb:with-tracing (a b c) (+ (funcall inner a b) (funcall inner c c))))
         (a (make-random-array (make-array-spec '(2 3) :f32) :seed 4))
         (b (make-random-array (make-array-spec '(2 3) :f32) :seed 5))
         (c (make-random-array (make-array-spec '(2 3) :f32) :seed 6)))
    (is (allclose (funcall (nb:jit outer :static-args '(2) :backend backend) a b c)
                  (funcall outer a b c)
                  :dtype :f32))
    (nb::%jit-cache-forget inner-f)
    (nb::%jit-cache-forget outer)))
