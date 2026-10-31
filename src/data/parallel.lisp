;;;; 並列ローダ（issue #286）。lparallel のワーカーでサンプラーの位置の区間を並列に読む。
;;;;
;;;; イテレータは自分の lparallel のカーネル（ワーカーのスレッドの組）を持つ（利用者のカーネルとは
;;;; 共有しない。ローダの片付けで利用者のカーネルを止めないため）。組み立ては逐次のイテレータと
;;;; 同じ %ASSEMBLE-BATCH で、渡す FETCH だけが違う。FETCH は頼まれた区間をワーカー数個の
;;;; 連続した区間に分けて、区間ごとの future でワーカーに %RANGE-RECORDS を投げ、future を作った
;;;; 順に FORCE して位置の順につなぐ（終わった順に受ける RECEIVE-RESULT は使わない）。
;;;; %RANGE-RECORDS は位置の純粋関数なので、バッチの列はワーカー数に依らない。
;;;;
;;;; ワーカーが返すのはホストの Lisp の配列（レコード）だけで、device-array は作らない
;;;; （finalizer が別スレッドで走るので、デバイスへの転送は呼び出し側のスレッドで行う）。
;;;; ワーカーの中のコンディション（error と、ほかの SERIOUS-CONDITION）は、区間を読む future の中で
;;;; 捕まえて値にし、FORCE した呼び出し側で元のコンディションのまま signal し直す。カーネルの
;;;; *DEBUG-TASKS-P* も nil にしてある（t のままだと、捕まえ損ねた error がワーカーのスレッドで
;;;; デバッガを起動して止まる。nil なら FORCE した側に届く）。まだどのワーカーも始めていない区間は、
;;;; FORCE した呼び出し側のスレッドが自分で読む（lparallel の future の性質）が、コンディションの
;;;; 扱いは同じ（どちらでも、読んだ側のスタックを巻き戻してから呼び出し側で signal する）。

(in-package #:nabla.data)

(defparameter *worker-thread-name* "nabla.data loader worker"
  "ローダのワーカーのスレッドの名前（テストが、閉じた後にスレッドが残らないことを確かめる）。")

(defstruct (%worker-pool (:constructor %make-worker-pool (kernel num-workers)))
  ;; lparallel のカーネル。ワーカー数 0（逐次）なら nil。
  kernel
  (num-workers 0 :type (integer 0) :read-only t)
  (closed nil))

(defun %open-worker-pool (num-workers)
  "NUM-WORKERS（0 以上の整数）のワーカーの組を作る。0 なら逐次でスレッドを作らない。"
  (unless (typep num-workers '(integer 0))
    (%data-loader-error :bad-num-workers "num-workers ~S が 0 以上の整数でない" num-workers))
  (%make-worker-pool (and (plusp num-workers)
                          (lparallel:make-kernel num-workers
                                                 :name *worker-thread-name*
                                                 :bindings `((lparallel:*debug-tasks-p* . nil)
                                                             (*standard-output* . ,*standard-output*)
                                                             (*error-output* . ,*error-output*))))
                     num-workers))

(defun %chunk-future (pool loader start end)
  "区間 [START, END) の %RANGE-RECORDS を読む future。読む中で signal されて誰も処理しなかった
SERIOUS-CONDITION（error に限らない）は、その場で捕まえて値にし、%FORCE-CHUNK が呼び出し側で
signal し直す（lparallel が呼び出し側へ渡すのは error だけで、ほかはワーカーのスレッドで
デバッガになる。ワーカーが読んでも呼び出し側が自分で読んでも、同じ振る舞いにする）。"
  (let ((lparallel:*kernel* (%worker-pool-kernel pool)))
    (lparallel:future
      (handler-case (%range-records loader start end)
        (serious-condition (condition) condition)))))

(defun %force-chunk (future)
  (let ((result (lparallel:force future)))
    (if (typep result 'condition)
        (error result)
        result)))

(defun %pool-fetch (pool loader start end)
  "%RANGE-RECORDS と同じ結果を、区間 [START, END) をワーカー数個の連続区間に分けてワーカーに
読ませ、位置の順につないで返す。"
  (let* ((chunk-size (ceiling (- end start) (%worker-pool-num-workers pool)))
         (futures (loop for chunk-start from start below end by chunk-size
                        collect (%chunk-future pool loader chunk-start
                                               (min end (+ chunk-start chunk-size))))))
    (loop for future in futures append (%force-chunk future))))

(defun %iterator-fetcher (iterator)
  "ITERATOR の %ASSEMBLE-BATCH に渡す FETCH。閉じたイテレータなら :closed（NEXT-BATCH が
組み立ての前に呼ぶので、列の終わりでも閉じていれば error になる）。"
  (let ((pool (data-iterator-pool iterator))
        (loader (data-iterator-loader iterator)))
    (when (%worker-pool-closed pool)
      (%data-loader-error :closed "閉じたイテレータからはバッチを取り出せない"))
    (if (%worker-pool-kernel pool)
        (lambda (start end) (%pool-fetch pool loader start end))
        (%range-fetcher loader))))

(defun close-iterator (iterator)
  "ITERATOR のワーカーを止めて片付ける（投げてある区間が終わるのを待つ）。閉じた後の
NEXT-BATCH は :closed の DATA-LOADER-ERROR。ITERATOR-STATE は閉じた後も取れる。何度呼んでも
よい。ワーカー数 0 のイテレータも閉じられる。nil を返す。"
  (let* ((pool (data-iterator-pool iterator))
         (kernel (%worker-pool-kernel pool)))
    (setf (%worker-pool-closed pool) t)
    (when kernel
      ;; END-KERNEL は各ワーカーの終わりの知らせを待つだけなので、スレッドが消えるまで join する。
      ;; 片付けが終わってからカーネルを外す（途中で抜けたら、もう一度呼ぶと片付け直す）。
      (let ((lparallel:*kernel* kernel))
        (dolist (thread (lparallel:end-kernel :wait t))
          (sb-thread:join-thread thread :default nil)))
      (setf (%worker-pool-kernel pool) nil))
    nil))

(defun %open-iterator (loader &rest options &key state &allow-other-keys)
  (let ((options (loop for (key value) on options by #'cddr
                       unless (eq key :state) append (list key value))))
    (if state
        (apply #'restore-iterator loader state options)
        (apply #'make-data-iterator loader options))))

(defmacro with-data-iterator ((var loader &rest options &key state &allow-other-keys) &body body)
  "LOADER のイテレータを VAR に束縛して BODY を実行し、抜けるとき（非局所脱出でも）
CLOSE-ITERATOR で閉じる。STATE（評価して nil でなければ）から RESTORE-ITERATOR で再開し、
nil なら MAKE-DATA-ITERATOR で先頭から始める。ほかの OPTIONS（:num-workers など）はそのまま渡す
（フォームは書いた順に評価する）。BODY の値を返す。"
  (declare (ignore state))
  `(let ((,var (%open-iterator ,loader ,@options)))
     (unwind-protect (progn ,@body)
       (close-iterator ,var))))
