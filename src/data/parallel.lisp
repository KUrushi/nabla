;;;; 並列ローダ（issue #286）。lparallel のワーカーでサンプラーの位置の区間を並列に読む。
;;;;
;;;; イテレータは自分の lparallel のカーネル（ワーカーのスレッドの組）を持つ（利用者のカーネルとは
;;;; 共有しない。ローダの片付けで利用者のカーネルを止めないため）。組み立ては逐次のイテレータと
;;;; 同じ %ASSEMBLE-BATCH で、渡す FETCH だけが違う。FETCH は頼まれた区間を、1 バッチの位置を
;;;; ワーカー数個に分けた大きさの連続した区間に分けて、区間ごとの future でワーカーに
;;;; %RANGE-RECORDS を投げ、future を作った順に FORCE して位置の順につなぐ（終わった順に受ける
;;;; RECEIVE-RESULT は使わない）。
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
;;;;
;;;; 先読み（prefetch）: FETCH は頼まれた区間の先、バッチ PREFETCH 個ぶんの位置もワーカーに投げて
;;;; おき、投げた区間を待ち行列（位置の順）に持つ。NEXT-BATCH が返った後、呼び出し側がバッチを
;;;; デバイスへ送り学習ステップを走らせる間に、ワーカーが次のバッチを読む。再開状態は返したバッチの
;;;; 位置なので、先読みは状態に影響しない（RESTORE-ITERATOR は先読みを持たない新しいイテレータ）。

(in-package #:nabla.data)

(defparameter *worker-thread-name* "nabla.data loader worker"
  "ローダのワーカーのスレッドの名前（テストが、閉じた後にスレッドが残らないことを確かめる）。")

(defstruct (%chunk (:constructor %make-chunk (start end result)))
  ;; 位置の区間 [start, end) と、%RANGE-RECORDS の結果（future か、分けた残りのリスト）。
  (start 0 :type (integer 0) :read-only t)
  (end 0 :type (integer 0) :read-only t)
  (result nil :read-only t))

(defstruct (%worker-pool (:constructor %make-worker-pool (kernel num-workers prefetch)))
  ;; lparallel のカーネル。ワーカー数 0（逐次）なら nil。
  kernel
  (num-workers 0 :type (integer 0) :read-only t)
  ;; 頼まれた区間の先に、バッチ何個ぶんの位置を投げておくか。
  (prefetch 0 :type (integer 0) :read-only t)
  ;; 投げた区間の待ち行列（位置の順で、隣り合う区間は連続する）と、その次の位置。
  (chunks '() :type list)
  (frontier 0 :type (integer 0))
  (closed nil))

(defun %open-worker-pool (num-workers prefetch)
  "NUM-WORKERS（0 以上の整数）のワーカーの組を作る。0 なら逐次でスレッドを作らない。
PREFETCH（0 以上の整数）はワーカーが先に読んでおくバッチの数。"
  (unless (typep num-workers '(integer 0))
    (%data-loader-error :bad-num-workers "num-workers ~S が 0 以上の整数でない" num-workers))
  (unless (typep prefetch '(integer 0))
    (%data-loader-error :bad-prefetch "prefetch ~S が 0 以上の整数でない" prefetch))
  (%make-worker-pool (and (plusp num-workers)
                          (lparallel:make-kernel num-workers
                                                 :name *worker-thread-name*
                                                 :bindings `((lparallel:*debug-tasks-p* . nil)
                                                             (*standard-output* . ,*standard-output*)
                                                             (*error-output* . ,*error-output*))))
                     num-workers prefetch))

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
  "FUTURE（%CHUNK-FUTURE。分けた残りのリストでもよい）の結果。読む中で起きたコンディションなら
呼び出し側で signal し直す。"
  (let ((result (lparallel:force future)))
    (if (typep result 'condition)
        (error result)
        result)))

(defun %submit-chunks (pool loader target)
  "待ち行列の末尾から位置 TARGET まで（列の終わりで止める）を、1 バッチの位置をワーカー数個に
分けた大きさの区間にしてワーカーに投げる。"
  (let* ((n (data-loader-length loader))
         (num-epochs (data-loader-num-epochs loader))
         (target (if num-epochs (min target (* num-epochs n)) target))
         (size (ceiling (data-loader-batch-size loader) (%worker-pool-num-workers pool)))
         (new (loop for start from (%worker-pool-frontier pool) below target by size
                    collect (let ((end (min target (+ start size))))
                              (%make-chunk start end (%chunk-future pool loader start end))))))
    (setf (%worker-pool-frontier pool) (max target (%worker-pool-frontier pool))
          (%worker-pool-chunks pool) (append (%worker-pool-chunks pool) new))))

(defun %pool-fetch (pool loader start end)
  "%RANGE-RECORDS と同じ結果を、ワーカーに投げた区間から位置の順に集めて返す。返す前に、
END の先 PREFETCH バッチぶんの位置もワーカーに投げておく（呼び出し側がバッチを使う間に読む）。
START が待ち行列の先頭でなければ（error で位置が進まなかった、filter と drop-remainder nil で
エポック末の位置を読み越した）、待ち行列を捨てて START から投げ直す（捨てた区間は終わるまで走る）。"
  (let ((head (first (%worker-pool-chunks pool))))
    (unless (and head (= start (%chunk-start head)))
      (setf (%worker-pool-chunks pool) '()
            (%worker-pool-frontier pool) start)))
  (%submit-chunks pool loader (+ end (* (%worker-pool-prefetch pool) (data-loader-batch-size loader))))
  (loop while (and (%worker-pool-chunks pool)
                   (< (%chunk-start (first (%worker-pool-chunks pool))) end))
        append (let* ((chunk (pop (%worker-pool-chunks pool)))
                      ;; 先に待ち行列から外すので、FORCE が error で抜けると、次の FETCH は
                      ;; 待ち行列を捨てて投げ直す（同じ位置からのやり直しは、逐次と同じく読み直す）。
                      (records (%force-chunk (%chunk-result chunk))))
                 (if (<= (%chunk-end chunk) end)
                     records
                     ;; 区間が END を越える: END より前を返し、残りを先頭の区間に戻す。
                     (let ((tail (member-if (lambda (pair) (>= (car pair) end)) records)))
                       (push (%make-chunk end (%chunk-end chunk) tail) (%worker-pool-chunks pool))
                       (ldiff records tail))))))

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
  "ITERATOR のワーカーを止めて片付ける（先読みを含め、投げてある区間が終わるのを待つ）。閉じた後の
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
nil なら MAKE-DATA-ITERATOR で先頭から始める。ほかの OPTIONS（:num-workers・:prefetch）はそのまま渡す
（フォームは書いた順に評価する）。BODY の値を返す。"
  (declare (ignore state))
  `(let ((,var (%open-iterator ,loader ,@options)))
     (unwind-protect (progn ,@body)
       (close-iterator ,var))))
