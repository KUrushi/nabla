;;;; nabla/pjrt/tests が共有するテスト部品。
;;;;
;;;; skip-unless-pjrt: PJRT プラグインが無い環境ではテストをスキップし
;;;; (CI では NABLA_REQUIRE_PJRT=1 で失敗にする)、あるときは何もしない。
;;;; tests/iree/support.lisp の skip-unless-iree と同じ方式。

(in-package #:nabla.pjrt.tests)

(defmacro define-pjrt-test (name docstring &body body)
  "fiveam:test 相当だが、本体を (block pjrt-test ...) でくるみ、常に
:nabla.medium スイートに登録する。skip-unless-pjrt はこの block から
return-from する。スイートを固定している理由は define-iree-test と同じ。"
  `(fiveam:test (,name :suite :nabla.medium) ,docstring
     (block pjrt-test
       ,@body)))

(defmacro skip-unless-pjrt (&key (kind :cpu))
  "(pjrt-available-p :kind KIND) が偽なら、NABLA_REQUIRE_PJRT 環境変数が
空でなければ fiveam:fail で失敗させ、無ければ fiveam:skip でスキップし、
どちらの場合も呼び出し元の define-pjrt-test の block から return-from する。
真ならなにもしない。"
  `(unless (pjrt-available-p :kind ,kind)
     (if (let ((value (sb-ext:posix-getenv "NABLA_REQUIRE_PJRT")))
           (and value (plusp (length value))))
         (progn
           (fiveam:fail "PJRT ~A plugin required but not found: ~A"
                        ,kind (plugin-path ,kind))
           (return-from pjrt-test))
         (progn
           (fiveam:skip "PJRT ~A plugin not found: ~A (set NABLA_PJRT_HOME or run scripts/fetch-pjrt.sh)"
                        ,kind (plugin-path ,kind))
           (return-from pjrt-test)))))

(defun gc-and-run-finalizers ()
  "(sb-ext:gc :full t) してから (sb-kernel:run-pending-finalizers) する。
SBCL は finalizer を別スレッドで非同期に実行するので、gc の直後に確認しても
ほとんど実行されていない。run-pending-finalizers で溜まった分を呼び出し
スレッドで同期的に実行する（tests/iree/support.lisp の同名の関数と同じ。
保守的なスタックルートのせいで少数のオブジェクトは残りうる）。"
  (sb-ext:gc :full t)
  (sb-kernel:run-pending-finalizers))

(defmacro with-pjrt-arrays ((&rest bindings) &body body)
  "BINDINGS の各 (VAR FORM) を順に評価して束縛し、BODY の後、逆順に
release-device-array する（非局所脱出でも、束縛済みの分は解放する）。"
  (let ((vars (mapcar #'first bindings)))
    `(let ,(mapcar (lambda (var) (list var nil)) vars)
       (unwind-protect
            (progn
              ,@(mapcar (lambda (binding) `(setf ,(first binding) ,(second binding)))
                        bindings)
              ,@body)
         ,@(mapcar (lambda (var) `(when ,var (release-device-array ,var)))
                   (reverse vars))))))
