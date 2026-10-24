;;;; embedding の引きを one-hot + dot（#204 の実装）と gather（#240）で比べる。結果は docs/embedding-gather.md。
;;;;
;;;;   export CL_SOURCE_REGISTRY="$PWD/:${NABLA_LISP_DEPS:-$HOME/.local/share/nabla/lisp-deps}//:"
;;;;   sbcl --dynamic-space-size 8GB --non-interactive --load scripts/bench-embedding.lisp
;;;;
;;;; 語彙数 V を変え、ids [8 32]（256 個）・features 32 の表で、順伝播（引き）と勾配（表についての grad）を測る。
;;;; eager（SBCL の上の CPU 実装）は常に測り、IREE のライブラリ（NABLA_IREE_HOME）があれば jit（local、
;;;; コンパイル後の呼び出し）も測る。時間は中央値（ミリ秒。eager の one-hot は遅いので1回、ほかは20回）、メモリは1回の呼び出しで確保したバイト数
;;;; （SB-EXT:GET-BYTES-CONSED の差。eager の中間配列の大きさの目安）。1行1レコードの S 式で出す。
;;;; NABLA_EMBED_VOCABS="1000 10000" で語彙数を絞る。

(require :asdf)
(asdf:load-system "nabla/nn")

(defpackage #:nabla-bench-embedding (:use #:cl))
(in-package #:nabla-bench-embedding)

(defun one-hot-lookup (table ids)
  "#204 の embedding の引き（ids を one-hot [ids... V] にして表と dot）を、:i32 の ids と f32 の表に特化して書き直したもの。"
  (let* ((vocabulary (first (nabla:aval-shape (nabla:value-aval table))))
         (id-shape (nabla:aval-shape (nabla:value-aval ids)))
         (full-shape (append id-shape (list vocabulary)))
         (ids-rank (length id-shape))
         (id-axes (loop for i below ids-rank collect i))
         (iota (make-array vocabulary :element-type '(signed-byte 32))))
    (dotimes (i vocabulary) (setf (aref iota i) i))
    (funcall (nabla:with-tracing (ids table full-shape iota ids-rank id-axes)
               (let ((hits (= (nabla:broadcast-in-dim ids full-shape id-axes)
                              (nabla:broadcast-in-dim iota full-shape (list ids-rank)))))
                 (nabla:dot (nabla:convert hits :f32) table)))
             ids table full-shape iota ids-rank id-axes)))

(defun gather-lookup (table ids)
  (nabla.nn:call (nabla.nn:make-embedding :embedding table) ids))

(defun loss-function (lookup weights)
  "(table ids) → sum(lookup(table, ids) * weights) の traceable function（grad は table について取る）。"
  (nabla:with-tracing (table ids) (nabla:reduce-sum (* (funcall lookup table ids) weights))))

(defun median-ms (thunk repeats)
  (let ((samples (loop repeat repeats
                       collect (let ((start (get-internal-real-time)))
                                 (funcall thunk)
                                 (/ (- (get-internal-real-time) start) (/ internal-time-units-per-second 1000d0))))))
    (nth (floor repeats 2) (sort samples #'<))))

(defun bytes-consed (thunk)
  (sb-ext:gc :full t)
  (let ((before (sb-ext:get-bytes-consed)))
    (funcall thunk)
    (- (sb-ext:get-bytes-consed) before)))

(defun report (&rest plist)
  (let ((*print-pretty* nil)) (prin1 plist) (terpri) (finish-output)))

(defun random-array (shape seed)
  (let ((state (sb-ext:seed-random-state seed))
        (array (make-array shape :element-type 'single-float)))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i) (- (random 2.0 state) 1.0)))))

(defun measure (backend vocabulary)
  "BACKEND（NIL なら eager）で、語彙数 VOCABULARY の引きと勾配を one-hot / gather それぞれ測って出力する。
jit では表と ids を jit の引数にし（ids を閉包の定数にすると one-hot の比較がコンパイル時に畳まれ、実運用の形と
違う）、表は先に device array にしておく（呼び出しごとのホストからの転送を測らない）。"
  (let* ((features 32)
         (table (random-array (list vocabulary features) vocabulary))
         (ids (let ((state (sb-ext:seed-random-state 7)) (a (make-array '(8 32) :element-type '(signed-byte 32))))
                (dotimes (i (array-total-size a) a) (setf (row-major-aref a i) (random vocabulary state)))))
         (weights (random-array '(8 32 32) 3))
         (table-arg (if backend (nabla:to-device table backend) table))
         (ids-arg (if backend (nabla:to-device ids backend) ids)))
    (dolist (method '(:one-hot :gather))
      (let ((lookup (if (eq method :one-hot) #'one-hot-lookup #'gather-lookup))
            ;; eager の one-hot は1回で数秒〜数十秒かかるので1回、それ以外は20回の中央値。
            (repeats (if (and (null backend) (eq method :one-hot)) 1 20)))
        (dolist (pass '(:forward :grad))
          (let* ((f (if (eq pass :forward)
                        (nabla:with-tracing (table ids) (funcall lookup table ids))
                        (nabla:grad (loss-function lookup weights))))
                 (run (if backend
                          (let ((jf (nabla:jit f :backend backend)))
                            (funcall jf table-arg ids-arg) ; コンパイル
                            (lambda () (funcall jf table-arg ids-arg)))
                          (lambda () (funcall f table-arg ids-arg)))))
            ;; jit の戻り値はホストの配列（to-host まで含めて測る）。確保したバイト数は Lisp のヒープの分だけなので、
            ;; IREE の中の確保を数えない jit では出さない。
            (let ((bytes (and (null backend) (bytes-consed run))))
              (apply #'report :backend (if backend :iree :eager) :vocab vocabulary :features features :ids 256
                     :method method :pass pass :repeats repeats
                     :median-ms (/ (round (* 1000 (median-ms run repeats))) 1000d0)
                     (and bytes (list :bytes-consed bytes))))))))))

(let ((vocabs (let ((env (sb-ext:posix-getenv "NABLA_EMBED_VOCABS")))
                (if env
                    (mapcar #'parse-integer (uiop:split-string (string-trim " " env) :separator " "))
                    '(1000 10000 50000)))))
  (report :kind :env :lisp (lisp-implementation-version) :machine (machine-type) :os (software-type))
  (dolist (v vocabs) (measure nil v))
  (asdf:load-system "nabla/iree")
  (if (funcall (intern "IREE-AVAILABLE-P" "NABLA.IREE"))
      (let ((backend (nabla:find-backend :iree))
            (nabla:*compile-cache-directory* nil))
        (dolist (v vocabs) (measure backend v)))
      (report :backend :iree :status :unmeasured :reason "IREE のライブラリが無い（NABLA_IREE_HOME）")))
