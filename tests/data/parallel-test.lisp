;;;; 並列ローダ（lparallel のワーカー、issue #286）。
;;;;
;;;; スレッドを使うので medium。タイミングに頼らない（sleep で待たない。順序のテストはセマフォで
;;;; 完了の順を決め、待ちには上限を付けて、外れたら固まる代わりに失敗する）。
;;;;
;;;; 守らせる性質:
;;;; - ワーカー数 0（逐次）・1・N で、バッチの列がビット単位で一致する（filter あり、エポックの境界、
;;;;   drop-remainder nil の端数、:empty-epoch も同じ）。
;;;; - 並列のイテレータの状態から再開した残りの列は、逐次の列の続きと一致する（ワーカー数を
;;;;   取ったときと変えてもよい）。
;;;; - 後の区間のほうが先に終わっても、バッチは位置の順に並ぶ。
;;;; - ワーカーの中の error（source-item・変換）は、元のコンディションのまま呼び出し側に届き、
;;;;   閉じた後にワーカーのスレッドが残らない。同じ位置からやり直すと同じ error になる（逐次と同じ）。
;;;; - 閉じた後の next-batch は :closed。close-iterator は何度呼んでもよい。with-data-iterator は
;;;;   本体を抜けると（非局所脱出でも）閉じる。不正なワーカー数・先読みは :bad-num-workers / :bad-prefetch。
;;;; - 先読み: バッチを 1 個返して閉じると、読んだ位置はちょうどバッチ 1 + prefetch 個ぶん（列の終わりで
;;;;   止まる）。呼び出し側が next-batch を呼ばない間にも、ワーカーが次のバッチを読む。filter で
;;;;   頼まれる区間が先読みの区間の切れ目とずれても、各位置は 1 回だけ読む。

(in-package #:nabla.data.tests)

(defmacro def-parallel-property (name docstring (&rest vars) generator &body body)
  "DEF-SOURCE-PROPERTY の medium 版（スレッドを使う性質）。"
  `(test (,name :suite :nabla.medium)
     ,docstring
     (is (check-it ,generator
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.DATA.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

(defun live-worker-count ()
  "生きている、ローダのワーカーのスレッドの数。"
  (count-if (lambda (thread)
              (and (sb-thread:thread-alive-p thread)
                   (equal (sb-thread:thread-name thread) nabla.data::*worker-thread-name*)))
            (sb-thread:list-all-threads)))

(defun parallel-batches-or-reason (loader num-workers &optional (prefetch 2))
  (handler-case (with-data-iterator (it loader :num-workers num-workers :prefetch prefetch)
                  (loader-all-batches it))
    (data-loader-error (c) (data-loader-error-reason c))))

(defun same-batches-p (a b)
  (if (listp a)
      (and (listp b) (= (length a) (length b)) (every #'same-tree-of-arrays-p a b))
      (eq a b)))

;;; --- ワーカー数に依らない ---

(def-parallel-property parallel-loader/batches-do-not-depend-on-the-number-of-workers
  "ワーカー数 0（逐次）・1・N（2..8）のバッチの列は同じ（filter・エポックの境界・端数を含む。
先読みのバッチ数 0..3 はシードから決める）。"
  (seed num-workers)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 2 :hi 8)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (flet ((batches (workers)
             (parallel-batches-or-reason (apply #'data-loader (loader-index-source (* 3 n)) arguments)
                                         workers (mod seed 4))))
      (let ((sequential (batches 0)))
        (and (same-batches-p sequential (batches 1))
             (same-batches-p sequential (batches num-workers)))))))

(def-parallel-property parallel-loader/large-batches-do-not-depend-on-the-number-of-workers
  "大きなバッチ（1 区間が何位置にもなる）でも、ワーカー数 0 と N（2..8）のバッチの列は同じ。"
  (n batch-size num-workers keep drop-remainder)
  (generator (tuple (uniform-integer :lo 1 :hi 300) (uniform-integer :lo 1 :hi 64)
                    (uniform-integer :lo 2 :hi 8) (uniform-integer :lo 1 :hi 4)
                    (uniform-integer :lo 0 :hi 1)))
  (flet ((batches (workers)
           (parallel-batches-or-reason
            (data-loader (loader-index-source n) :batch-size batch-size :num-epochs 2
                                                 :shuffle t :seed (sampler-test-seed n batch-size)
                                                 :drop-remainder (= 1 drop-remainder)
                                                 :transforms (list (filter-transform
                                                                    (lambda (record)
                                                                      (zerop (mod (loader-record-index record)
                                                                                  keep))))))
            workers)))
    (same-batches-p (batches 0) (batches num-workers))))

(def-parallel-property parallel-loader/restoring-continues-the-sequential-sequence
  "並列のイテレータで m 個取った状態から、別のワーカー数で再開した残りは逐次の列の続き。"
  (seed m workers-before workers-after)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 0 :hi 20)
                    (uniform-integer :lo 1 :hi 6) (uniform-integer :lo 0 :hi 6)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (flet ((make-loader () (apply #'data-loader (loader-index-source (* 3 n)) arguments)))
      (let ((all (parallel-batches-or-reason (make-loader) 0)))
        (or (eq all :empty-epoch)
            (let ((state (with-data-iterator (it (make-loader) :num-workers workers-before
                                                               :prefetch (mod seed 4))
                           (loader-take it m)
                           (iterator-state it))))
              (same-batches-p (nthcdr (min m (length all)) all)
                              (with-data-iterator (it (make-loader) :state (loader-copy-state state)
                                                                    :num-workers workers-after)
                                (loader-all-batches it)))))))))

(define-condition parallel-test-failure (error)
  ((thread :initarg :thread :reader parallel-test-failure-thread)))

(define-condition parallel-test-serious (serious-condition)
  ((thread :initarg :thread :reader parallel-test-failure-thread))
  (:documentation "error でない SERIOUS-CONDITION（lparallel は error しか呼び出し側へ渡さない）。"))

(defclass rendezvous-source (data-source)
  ((length :initarg :length :reader source-length)
   (fail :initarg :fail :initform nil :reader rendezvous-fail)
   (semaphore :initform (sb-thread:make-semaphore) :reader rendezvous-semaphore))
  (:documentation "添字 0 のレコードは、最後の添字に取りかかるまで待つ（後の区間が先に終わる）。
FAIL（コンディションの型）が nil でなければ、最後の添字はそれを signal する。添字 0 を読むスレッドが
待っている間に最後の添字を読むのは、呼び出し側でなくワーカーのスレッド（呼び出し側は先頭の区間の
future を待つか、自分で読んで待っている）。"))

(defmethod source-item ((source rendezvous-source) index)
  (cond ((zerop index)
         (unless (sb-thread:wait-on-semaphore (rendezvous-semaphore source) :timeout 30)
           (error "最後の添字に 30 秒経っても取りかからない")))
        ((= index (1- (source-length source)))
         (sb-thread:signal-semaphore (rendezvous-semaphore source))
         (when (rendezvous-fail source)
           (error (rendezvous-fail source) :thread sb-thread:*current-thread*))))
  (list (make-array '() :element-type 'double-float :initial-element (float index 1d0))))

(test (parallel-loader/orders-batches-by-position-not-by-completion :suite :nabla.medium)
  "先頭の区間が最後の区間より後に終わっても、バッチのレコードは位置の順。"
  (let ((loader (data-loader (make-instance 'rendezvous-source :length 8) :batch-size 8 :num-epochs 1)))
    (with-data-iterator (it loader :num-workers 2)
      (is (equalp (coerce (loop for i below 8 collect (float i 1d0)) 'vector)
                  (first (next-batch it)))))))

;;; --- エラーと片付け ---

(defclass failing-source (data-source)
  ((length :initarg :length :reader source-length)
   (bad-index :initarg :bad-index :reader failing-source-bad-index)))

(defmethod source-item ((source failing-source) index)
  (when (= index (failing-source-bad-index source))
    (error 'parallel-test-failure :thread sb-thread:*current-thread*))
  (list (make-array '() :element-type 'double-float :initial-element (float index 1d0))))

(defun caught-failure (thunk)
  "THUNK が signal した PARALLEL-TEST-FAILURE か PARALLEL-TEST-SERIOUS（無ければ nil）。"
  (handler-case (progn (funcall thunk) nil)
    ((or parallel-test-failure parallel-test-serious) (c) c)))

(test (parallel-loader/worker-errors-reach-the-caller-and-leave-no-threads :suite :nabla.medium)
  "source-item・変換の error はワーカーのスレッドから元のコンディションのまま届き、閉じた後に
ワーカーのスレッドが残らない。"
  (let ((before (live-worker-count))
        (source (make-instance 'failing-source :length 20 :bad-index 13)))
    ;; with-data-iterator の本体から error で抜けても閉じる
    (is (typep (caught-failure
                (lambda ()
                  (with-data-iterator (it (data-loader source :batch-size 4 :num-epochs 1)
                                          :num-workers 3)
                    (is (= (+ before 3) (live-worker-count)))
                    (loader-all-batches it))))
               'parallel-test-failure))
    (is (= before (live-worker-count)))
    ;; ワーカーのスレッドで起きた error と、error でない serious-condition が、呼び出し側のスレッドに
    ;; 元のコンディションのまま届く
    (dolist (type '(parallel-test-failure parallel-test-serious))
      (let ((failure (caught-failure
                      (lambda ()
                        (with-data-iterator (it (data-loader (make-instance 'rendezvous-source :length 8
                                                                                               :fail type)
                                                             :batch-size 8 :num-epochs 1)
                                                :num-workers 2)
                          (next-batch it))))))
        (is (typep failure type))
        (is (not (eq sb-thread:*current-thread* (parallel-test-failure-thread failure))))))
    (is (= before (live-worker-count)))
    ;; 明示のイテレータ: 同じ位置からやり直すと同じ error になり、close-iterator で片付く
    (let* ((failing-map (map-transform (lambda (record)
                                         (if (= 5d0 (aref (first record)))
                                             (error 'parallel-test-failure
                                                    :thread sb-thread:*current-thread*)
                                             record))))
           (parallel (make-data-iterator (data-loader (make-instance 'failing-source :length 20
                                                                                     :bad-index -1)
                                                      :batch-size 4 :num-epochs 1
                                                      :transforms (list failing-map))
                                         :num-workers 2)))
      (is (= 1 (length (loader-take parallel 1))))
      (dotimes (i 2)
        (is (typep (caught-failure (lambda () (next-batch parallel))) 'parallel-test-failure)))
      (is (equal '(0 4) (loader-state-values (iterator-state parallel))))
      (close-iterator parallel)
      (is (= before (live-worker-count))))))

(test (parallel-loader/closed-iterators-refuse-to-continue :suite :nabla.medium)
  "閉じた後の next-batch は :closed（列の終わりでも）、close-iterator は何度呼んでもよい。"
  (let ((before (live-worker-count)))
    (dolist (num-workers '(0 1 4))
      (let ((it (make-data-iterator (data-loader (loader-index-source 6) :batch-size 2 :num-epochs 1)
                                    :num-workers num-workers)))
        (is (= 3 (length (loader-all-batches it))))
        (close-iterator it)
        (close-iterator it)
        (is (eq :closed (loader-failure-reason (lambda () (next-batch it)))))
        ;; 状態は閉じた後も取れる
        (is (equal '(1 0) (loader-state-values (iterator-state it))))))
    ;; 本体から return-from で抜けても閉じる
    (let ((escaped (block body
                     (with-data-iterator (it (data-loader (loader-index-source 6) :batch-size 2)
                                             :num-workers 2)
                       (return-from body it)))))
      (is (eq :closed (loader-failure-reason (lambda () (next-batch escaped))))))
    (is (= before (live-worker-count)))))

(defparameter *bad-num-workers* (list -1 1.5 nil "2")
  "ワーカー数（と先読みのバッチ数）にならない値。")

(test (parallel-loader/rejects-bad-numbers-of-workers :suite :nabla.medium)
  "ワーカー数が 0 以上の整数でなければ :bad-num-workers（スレッドを作らない）。"
  (let ((before (live-worker-count))
        (loader (data-loader (loader-index-source 6) :batch-size 2)))
    (dolist (num-workers *bad-num-workers*)
      (is (eq :bad-num-workers
              (loader-failure-reason (lambda () (make-data-iterator loader :num-workers num-workers)))))
      (is (eq :bad-num-workers
              (loader-failure-reason
               (lambda () (restore-iterator loader (iterator-state (make-data-iterator loader))
                                            :num-workers num-workers))))))
    (dolist (prefetch *bad-num-workers*)
      (is (eq :bad-prefetch
              (loader-failure-reason (lambda () (make-data-iterator loader :num-workers 2
                                                                           :prefetch prefetch))))))
    (is (= before (live-worker-count)))
    ;; 既定のワーカー数は 0（スレッドを作らない）
    (make-data-iterator loader)
    (restore-iterator loader (iterator-state (make-data-iterator loader)))
    (is (= before (live-worker-count)))))

;;; --- 先読み ---

(defclass recording-source (data-source)
  ((length :initarg :length :reader source-length)
   (lock :initform (sb-thread:make-mutex) :reader recording-lock)
   (indices :initform '() :accessor recording-indices)
   (watched :initarg :watched :initform nil :reader recording-watched)
   (semaphore :initform (sb-thread:make-semaphore) :reader recording-semaphore))
  (:documentation "読まれた添字を記録する。WATCHED の添字を読んだら SEMAPHORE を signal する。"))

(defmethod source-item ((source recording-source) index)
  (sb-thread:with-mutex ((recording-lock source))
    (push index (recording-indices source)))
  (when (eql index (recording-watched source))
    (sb-thread:signal-semaphore (recording-semaphore source)))
  (list (make-array '() :element-type 'double-float :initial-element (float index 1d0))))

(defun read-indices (source)
  (sort (copy-list (recording-indices source)) #'<))

(test (parallel-loader/prefetch-reads-the-next-batches-ahead :suite :nabla.medium)
  "1 個のバッチを返して閉じると、読んだ位置はちょうどバッチ 1 + prefetch 個ぶん（列の終わりで止まる）。"
  (loop for (n prefetch num-epochs expected) in '((40 0 1 4) (40 1 1 8) (40 2 nil 12) (10 5 1 10) (6 3 2 12))
        do (let ((source (make-instance 'recording-source :length n)))
             (with-data-iterator (it (data-loader source :batch-size 4 :num-epochs num-epochs)
                                     :num-workers 2 :prefetch prefetch)
               (next-batch it))
             ;; 閉じると投げた区間が終わるのを待つので、記録はもう増えない。シャッフルしないので
             ;; 位置 k の添字は k mod n
             (is (equal (sort (loop for k below expected collect (mod k n)) #'<) (read-indices source))
                 "n ~D prefetch ~D で読んだ添字 ~S" n prefetch (read-indices source))))
  ;; 既定の先読みはバッチ 2 個（make-data-iterator と restore-iterator）
  (dolist (restore '(nil t))
    (let* ((source (make-instance 'recording-source :length 40))
           (loader (data-loader source :batch-size 4 :num-epochs 1))
           (it (if restore
                   (restore-iterator loader (iterator-state (make-data-iterator loader)) :num-workers 2)
                   (make-data-iterator loader :num-workers 2))))
      (next-batch it)
      (close-iterator it)
      (is (equal (loop for i below 12 collect i) (read-indices source))))))

(test (parallel-loader/workers-read-ahead-while-the-caller-uses-a-batch :suite :nabla.medium)
  "next-batch が返った後、呼び出し側が next-batch を呼ばなくても、ワーカーが次のバッチを読む。"
  (let ((source (make-instance 'recording-source :length 16 :watched 6)))
    (with-data-iterator (it (data-loader source :batch-size 4 :num-epochs 1) :num-workers 2 :prefetch 1)
      (next-batch it)
      (is-true (sb-thread:wait-on-semaphore (recording-semaphore source) :timeout 30)))))

(test (parallel-loader/reads-each-position-once :suite :nabla.medium)
  "filter で頼まれる区間が区間の切れ目とずれても、先読みした区間の残りを使い、各位置を1回だけ読む。"
  (let ((source (make-instance 'recording-source :length 24)))
    (with-data-iterator (it (data-loader source :batch-size 4 :num-epochs 1
                                                :transforms (list (filter-transform
                                                                   (lambda (record)
                                                                     (zerop (mod (aref (first record)) 3))))))
                            :num-workers 2 :prefetch 1)
      (is (= 2 (length (loader-all-batches it)))))
    (is (equal (loop for i below 24 collect i) (read-indices source)))))
