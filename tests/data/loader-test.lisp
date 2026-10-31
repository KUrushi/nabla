;;;; data-loader と data-iterator、再開状態（issue #285）。
;;;;
;;;; 守らせる性質（small）:
;;;; - オラクル: バッチの列は、エポックごとに「サンプラーの順にレコードを引いて変換し、残ったものを
;;;;   batch-size 個ずつに切る（端数は drop-remainder なら捨て、nil なら短いバッチ）」を素朴に
;;;;   書いたものと、エポックの順に並べて一致する（バッチはエポックをまたがない。filter あり）。
;;;; - 1 エポックで全レコードがちょうど1回ずつ出る（前提: filter なし。drop-remainder なら
;;;;   n mod batch-size 個が欠け、nil なら欠けない）。
;;;; - エポック単位の取り出し: (epoch loader e) を e = 0.. の順につなぐと、イテレータの列と同じ。
;;;; - 終わり: num-epochs のエポックを出し切ると next-batch は何度呼んでも (values nil nil)。
;;;; - 1 エポックからバッチが1つも作れない（filter が全部落とす・batch-size > n で端数を捨てる・
;;;;   レコードが 0 個）は :empty-epoch（num-epochs nil で戻らなくなるのを防ぐ）。
;;;; - 不正な引数は理由つきの data-loader-error。
;;;; - 再開: 任意の個数のバッチを取った後の状態から別に作ったローダで再開すると、残りのバッチの列は
;;;;   中断しなかった場合とビット単位で一致する（エポックの境界をまたぐ・filter あり・
;;;;   num-epochs nil を含む）。状態は取った時点の写しで、元のイテレータを進めても変わらない。
;;;; - 状態の木: 子の名前は key / epoch / position / version で、葉は :u32 の (2)（ローダのシードの
;;;;   写し、シードが無ければ 0）と rank 0 の (signed-byte 32) の配列。version はシャッフルの算法の版。
;;;; - 範囲外・形の違う状態・別のシードや別の版の状態は、理由つきの data-loader-error。

(in-package #:nabla.data.tests)

;;; --- 生成とオラクル ---

(defun loader-index-source (n)
  "INDEX 番目のレコードが (list 添字の :i32 の rank 0 配列, 添字の 2 倍の f64 の (2) 配列) の
array-source。"
  (make-array-source
   (list (make-array n :element-type '(signed-byte 32)
                       :initial-contents (loop for i below n collect i))
         (make-array (list n 2) :element-type 'double-float
                                :initial-contents (loop for i below n
                                                        collect (list (* 2d0 i) (- (* 2d0 i))))))))

(defun loader-record-index (record)
  "LOADER-INDEX-SOURCE のレコード（変換の後も添字の葉は残す）の添字。"
  (aref (first record)))

(defun loader-random-spec (state)
  "変換の仕様（:shift k は f64 の葉に k を足す map、:keep-mod m r は添字 mod m = r の filter）。"
  (if (zerop (random 2 state))
      (list :shift (- (random 9 state) 4))
      (let ((m (1+ (random 3 state)))) (list :keep-mod m (random m state)))))

(defun loader-spec-transform (spec)
  (destructuring-bind (kind a &optional b) spec
    (ecase kind
      (:shift (map-transform
               (lambda (record)
                 (let ((v (copy-seq (second record))))
                   (dotimes (i 2) (incf (aref v i) a))
                   (list (first record) v)))))
      (:keep-mod (filter-transform (lambda (record) (= b (mod (loader-record-index record) a))))))))

(defun loader-random-case (seed)
  "SEED から決まる data-loader の引数の plist と、レコードの個数。"
  (let* ((state (sb-ext:seed-random-state seed))
         (n (random 12 state))
         (shuffle (zerop (random 2 state))))
    (values (list :batch-size (1+ (random 5 state))
                  :shuffle shuffle
                  :seed (and (or shuffle (zerop (random 2 state)))
                             (make-array 2 :element-type '(unsigned-byte 32)
                                           :initial-contents (list (random (ash 1 32) state)
                                                                   (random (ash 1 32) state))))
                  :num-epochs (1+ (random 3 state))
                  :transforms (mapcar #'loader-spec-transform
                                      (loop repeat (random 3 state) collect (loader-random-spec state)))
                  :drop-remainder (zerop (random 2 state)))
            n)))

(defun loader-oracle (n &key batch-size shuffle seed num-epochs transforms drop-remainder)
  "バッチ（レコードのリスト）の列を素朴に作る。エポックごとに全レコードを引いて変換し、
BATCH-SIZE 個ずつに切る。1 エポックからバッチが作れなければ :empty-epoch を返す。"
  (let ((sampler (make-index-sampler n :shuffle shuffle :seed seed :num-epochs num-epochs))
        (source (loader-index-source n))
        (batches '()))
    (dotimes (epoch num-epochs (nreverse batches))
      (let* ((kept (loop for i below n
                         for (record keep) = (multiple-value-list
                                              (apply-transforms
                                               transforms
                                               (source-item source
                                                            (nth-value 1 (sampler-index sampler (+ (* epoch n) i))))
                                               :seed seed :position (+ (* epoch n) i)))
                         when keep collect record))
             (epoch-batches (loop while (or (>= (length kept) batch-size)
                                            (and kept (not drop-remainder)))
                                  collect (loop repeat batch-size while kept collect (pop kept)))))
        (when (null epoch-batches)
          (return :empty-epoch))
        (dolist (batch epoch-batches) (push batch batches))))))

(defparameter *loader-max-batches* 1000
  "テストのローダが返すバッチの数の上限（テストのケースはどれもこれより十分少ない）。終わらない列を
集め続けてヒープを使い切る代わりに失敗させる（mutation testing の変異体が列を終わらせないとき）。")

(defun loader-all-batches (iterator)
  "ITERATOR の残りのバッチをすべて集めたリスト。*LOADER-MAX-BATCHES* 個を超えたら error。"
  (loop for count from 0
        for (batch more) = (multiple-value-list (next-batch iterator))
        while more
        when (>= count *loader-max-batches*)
          do (error "バッチの列が ~D 個を超えても終わらない" *loader-max-batches*)
        collect batch))

(defun loader-failure-reason (thunk)
  "THUNK が signal した data-loader-error の reason。signal しなければ :no-error。"
  (handler-case (progn (funcall thunk) :no-error)
    (data-loader-error (c) (data-loader-error-reason c))))

(defun loader-batches-or-reason (loader)
  (handler-case (loader-all-batches (make-data-iterator loader))
    (data-loader-error (c) (data-loader-error-reason c))))

;;; --- オラクルとの一致 ---

(def-source-property data-loader/batches-follow-the-per-epoch-oracle
  "バッチの列は、エポックごとに引いて変換し batch-size 個ずつに切ったものと一致する。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (let ((expected (apply #'loader-oracle n arguments))
          (actual (loader-batches-or-reason (apply #'data-loader (loader-index-source n) arguments))))
      (if (eq expected :empty-epoch)
          (eq actual :empty-epoch)
          (and (listp actual)
               (= (length expected) (length actual))
               (every (lambda (records batch) (same-tree-of-arrays-p (batch-records records) batch))
                      expected actual))))))

(def-source-property data-loader/each-record-appears-once-per-epoch
  "前提: filter なし。1 エポックのバッチには各レコードが高々1回ずつ出て、欠けるのは drop-remainder
で捨てた n mod batch-size 個だけ（drop-remainder nil なら欠けない）。"
  (n batch-size shuffle hi drop-remainder)
  (generator (tuple (uniform-integer :lo 1 :hi 40) (uniform-integer :lo 1 :hi 7)
                    (uniform-integer :lo 0 :hi 1) (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 1)))
  (let* ((drop (= 1 drop-remainder))
         (loader (data-loader (loader-index-source n)
                              :batch-size batch-size :shuffle (= 1 shuffle) :num-epochs 1
                              :seed (sampler-test-seed hi 7) :drop-remainder drop))
         (indices (loop for batch in (if (and drop (< n batch-size))
                                         '()
                                         (loader-all-batches (make-data-iterator loader)))
                        append (coerce (first batch) 'list))))
    (and (= (length indices) (length (remove-duplicates indices)))
         (every (lambda (i) (< -1 i n)) indices)
         (= (length indices) (if drop (- n (mod n batch-size)) n)))))

(def-source-property data-loader/epochs-concatenate-to-the-iterator
  "(epoch loader e) を e = 0, 1, ... の順につなぐと、イテレータのバッチの列と同じ。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (let ((loader (apply #'data-loader (loader-index-source n) arguments)))
      (handler-case
          (every #'same-tree-of-arrays-p
                 (loop for e below (getf arguments :num-epochs) append (epoch loader e))
                 (loader-all-batches (make-data-iterator loader)))
        (data-loader-error (c) (eq :empty-epoch (data-loader-error-reason c)))))))

(test (data-loader/epoch-reads-only-its-own-records :suite :nabla.small)
  "(epoch loader e) は、エポック e のレコードだけを引く（次のエポックを先読みしない）。"
  (let* ((calls 0)
         (loader (data-loader (loader-index-source 6) :batch-size 3 :num-epochs 3
                                                      :transforms (list (map-transform
                                                                         (lambda (record)
                                                                           (incf calls)
                                                                           record))))))
    (is (= 2 (length (epoch loader 1))))
    (is (= 6 calls))))

;;; --- 終わりと空のエポック ---

(test (data-loader/next-batch-returns-nil-nil-after-the-last-epoch :suite :nabla.small)
  "num-epochs のエポックを出し切ると、next-batch は何度呼んでも (values nil nil)。"
  (let ((it (make-data-iterator (data-loader (loader-index-source 5) :batch-size 2 :num-epochs 2
                                                                     :drop-remainder nil))))
    (is (= 6 (length (loader-all-batches it))))
    (is (equal '(nil nil) (multiple-value-list (next-batch it))))
    (is (equal '(nil nil) (multiple-value-list (next-batch it)))))
  (dolist (n '(5 0))
    (is (null (loader-all-batches
               (make-data-iterator (data-loader (loader-index-source n) :batch-size 2 :num-epochs 0)))))))

(test (data-loader/unbounded-epochs-keep-going :suite :nabla.small)
  "num-epochs nil の列は終わらず、エポック e のバッチは (epoch loader e) と同じ。"
  (let* ((loader (data-loader (loader-index-source 5) :batch-size 2 :shuffle t
                                                      :seed (sampler-test-seed 1 2)))
         (it (make-data-iterator loader))
         (batches (loop repeat 10 collect (next-batch it))))
    (is (every #'same-tree-of-arrays-p batches
               (loop for e below 5 append (epoch loader e))))))

(test (data-loader/an-epoch-without-batches-is-an-error :suite :nabla.small)
  "1 エポックからバッチが1つも作れないときは :empty-epoch（num-epochs nil でも戻る）。"
  (flet ((reason (&rest arguments)
           (loader-failure-reason
            (lambda () (next-batch (make-data-iterator (apply #'data-loader arguments)))))))
    (let ((drop-all (list (filter-transform (constantly nil)))))
      (is (eq :empty-epoch (reason (loader-index-source 5) :batch-size 2 :transforms drop-all)))
      (is (eq :empty-epoch (reason (loader-index-source 5) :batch-size 2 :transforms drop-all
                                                           :num-epochs 3 :drop-remainder nil))))
    (is (eq :empty-epoch (reason (loader-index-source 3) :batch-size 4)))
    (is (eq :no-error (reason (loader-index-source 3) :batch-size 4 :drop-remainder nil)))
    (is (eq :empty-epoch (reason (loader-index-source 0) :batch-size 1)))
    (is (eq :empty-epoch (reason (loader-index-source 0) :batch-size 1 :num-epochs 1)))
    ;; エポックが 0 個なら、レコードが 0 個でも空の列
    (is (eq :no-error (reason (loader-index-source 0) :batch-size 1 :num-epochs 0))))
  (is (eq :empty-epoch (loader-failure-reason
                        (lambda () (epoch (data-loader (loader-index-source 3) :batch-size 4) 0))))))

;;; --- 不正な引数 ---

(defparameter *bad-batch-sizes* (list 0 -1 1.5 nil "2")
  "batch-size にならない値。定数を直接渡すとコンパイル時に型の WARNING になるので、変数から渡す。")

(test (data-loader/rejects-bad-arguments :suite :nabla.small)
  "batch-size が正の整数でない・変換でないもの・範囲外のエポックは、理由つきの data-loader-error。"
  (let ((source (loader-index-source 4)))
    (dolist (batch-size *bad-batch-sizes*)
      (is (eq :bad-batch-size
              (loader-failure-reason (lambda () (data-loader source :batch-size batch-size))))))
    (is (eq :bad-transform
            (loader-failure-reason
             (lambda () (data-loader source :batch-size 1 :transforms (list #'identity))))))
    (let ((loader (data-loader source :batch-size 1 :num-epochs 2)))
      (dolist (e (list -1 2 1.0 nil))
        (is (eq :bad-epoch (loader-failure-reason (lambda () (epoch loader e))))))
      (is (= 4 (length (epoch loader 1)))))
    (is (typep (handler-case (data-loader source :batch-size 0) (error (c) c)) 'data-error))))

;;; --- 再開状態（iterator-state / restore-iterator） ---

(defun loader-copy-state (state)
  "STATE の葉を写した新しい状態（チェックポイントを往復した状態の代わり）。"
  (multiple-value-bind (leaves treedef) (nabla:tree-flatten state)
    (nabla:tree-unflatten treedef (mapcar (lambda (leaf)
                                            (let ((copy (make-array (array-dimensions leaf)
                                                                    :element-type (array-element-type leaf))))
                                              (replace (make-array (array-total-size copy)
                                                                   :element-type (array-element-type leaf)
                                                                   :displaced-to copy)
                                                       (make-array (array-total-size leaf)
                                                                   :element-type (array-element-type leaf)
                                                                   :displaced-to leaf))
                                              copy))
                                          leaves))))

(defun loader-take (iterator count)
  "ITERATOR から高々 COUNT 個のバッチを取ったリスト。"
  (loop repeat count
        for (batch more) = (multiple-value-list (next-batch iterator))
        while more collect batch))

(def-source-property iterator-state/restoring-continues-the-uninterrupted-sequence
  "m 個のバッチを取った後の状態から再開した残りの列は、中断しなかった列の m 番目以降と一致する。"
  (seed m)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 0 :hi 30)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (let ((all (loader-batches-or-reason (apply #'data-loader (loader-index-source n) arguments))))
      ;; 中断しない列が :empty-epoch なら、性質の対象外（その場合はオラクルの性質が見る）。
      (or (eq all :empty-epoch)
          (let* ((it (make-data-iterator (apply #'data-loader (loader-index-source n) arguments)))
                 (taken (loader-take it m))
                 (state (iterator-state it))
                 (rest-of-original (loader-all-batches it))
                 (restored (restore-iterator (apply #'data-loader (loader-index-source n) arguments)
                                             (loader-copy-state state)))
                 (rest-of-restored (loader-all-batches restored)))
            (and (every #'same-tree-of-arrays-p taken all)
                 (= (length all) (+ (length taken) (length rest-of-original)))
                 (every #'same-tree-of-arrays-p rest-of-original (nthcdr (length taken) all))
                 (= (length rest-of-restored) (length rest-of-original))
                 (every #'same-tree-of-arrays-p rest-of-restored rest-of-original)))))))

(def-source-property iterator-state/restoring-an-unbounded-loader-continues-the-sequence
  "num-epochs nil でも、m 個取った後の状態から再開した次の 6 個は中断しなかった列と同じ。"
  (n batch-size m hi drop-remainder)
  (generator (tuple (uniform-integer :lo 1 :hi 9) (uniform-integer :lo 1 :hi 4)
                    (uniform-integer :lo 0 :hi 12) (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 1)))
  (flet ((make-loader ()
           (data-loader (loader-index-source n) :batch-size batch-size :shuffle t
                                                :seed (sampler-test-seed hi 3)
                                                :transforms (list (filter-transform
                                                                   (lambda (r) (/= 1 (loader-record-index r)))))
                                                :drop-remainder (= 1 drop-remainder))))
    ;; 1 エポックに残るレコードは n - 1 個（n = 1 なら添字 1 が無いので 1 個）。端数を捨てて
    ;; バッチが作れなければ :empty-epoch。
    (if (and (= 1 drop-remainder) (< (if (= n 1) 1 (1- n)) batch-size))
        (eq :empty-epoch (loader-failure-reason (lambda () (next-batch (make-data-iterator (make-loader))))))
        (let* ((all (loader-take (make-data-iterator (make-loader)) (+ m 6)))
               (it (make-data-iterator (make-loader))))
          (loader-take it m)
          (let ((rest (loader-take (restore-iterator (make-loader) (loader-copy-state (iterator-state it)))
                                   6)))
            (and (= (+ m 6) (length all))
                 (= 6 (length rest))
                 (every #'same-tree-of-arrays-p rest (nthcdr m all))))))))

(test (iterator-state/is-a-tree-of-named-arrays :suite :nabla.small)
  "状態の子の名前は key / epoch / position / version で、葉はシードの写しと rank 0 の :i32 の配列。"
  (let* ((seed (sampler-test-seed 11 22))
         (it (make-data-iterator (data-loader (loader-index-source 5) :batch-size 2 :shuffle t
                                                                      :seed seed)))
         (fresh (iterator-state it)))
    (multiple-value-bind (children aux node-p names) (nabla:tree-children fresh)
      (declare (ignore aux))
      (is-true node-p)
      (is (equal '("key" "epoch" "position" "version") names))
      (is (equal '((unsigned-byte 32) (signed-byte 32) (signed-byte 32) (signed-byte 32))
                 (mapcar #'array-element-type children)))
      (is (equal '((2) () () ()) (mapcar #'array-dimensions children))))
    (is (= nabla.data::+index-sampler-algorithm-version+ (aref (iterator-state-version fresh))))
    (is (equalp seed (iterator-state-key fresh)))
    (is (not (eq seed (iterator-state-key fresh))))
    (is (equal '(0 0) (list (aref (iterator-state-epoch fresh)) (aref (iterator-state-position fresh)))))
    ;; 5 個から 2 個ずつ（端数を捨てる）: 2 個目のバッチの後はエポック 0 の位置 4
    (loader-take it 2)
    (let ((state (iterator-state it)))
      (is (equal '(0 4) (list (aref (iterator-state-epoch state)) (aref (iterator-state-position state)))))
      (loader-take it 1)
      (is (equal '(0 4) (list (aref (iterator-state-epoch state)) (aref (iterator-state-position state)))))
      (is (equal '(1 2) (let ((s (iterator-state it)))
                          (list (aref (iterator-state-epoch s)) (aref (iterator-state-position s))))))))
  (is (equalp (make-array 2 :element-type '(unsigned-byte 32) :initial-element 0)
              (iterator-state-key
               (iterator-state (make-data-iterator (data-loader (loader-index-source 3) :batch-size 1)))))))

(defun loader-state-with (state &key (key (iterator-state-key state)) epoch position version)
  "STATE の葉を KEY・EPOCH・POSITION・VERSION（整数なら rank 0 の :i32、配列ならそのまま）に
替えた状態。"
  (flet ((leaf (value default)
           (cond ((null value) default)
                 ((integerp value) (make-array '() :element-type '(signed-byte 32) :initial-element value))
                 (t value))))
    (nabla:tree-unflatten (nabla:tree-structure state)
                          (list key
                                (leaf epoch (iterator-state-epoch state))
                                (leaf position (iterator-state-position state))
                                (leaf version (iterator-state-version state))))))

(test (iterator-state/restore-rejects-states-that-do-not-fit-the-loader :suite :nabla.small)
  "別のシード・別の版・範囲外のエポックや位置・形や型の違う葉は、理由つきの data-loader-error。"
  (let* ((loader (data-loader (loader-index-source 5) :batch-size 2 :shuffle t
                                                      :seed (sampler-test-seed 1 2) :num-epochs 2))
         (state (iterator-state (make-data-iterator loader))))
    (flet ((reason (state) (loader-failure-reason (lambda () (restore-iterator loader state)))))
      (is (eq :no-error (reason state)))
      (is (eq :no-error (reason (loader-state-with state :epoch 1 :position 5))))
      (is (eq :no-error (reason (loader-state-with state :epoch 2 :position 0))))
      (is (eq :key-mismatch (reason (loader-state-with state :key (sampler-test-seed 1 3)))))
      ;; 別の版のシャッフルの算法の状態は、黙って順序を変える代わりに :version-mismatch
      (dolist (version (list (1+ nabla.data::+index-sampler-algorithm-version+)
                             (1- nabla.data::+index-sampler-algorithm-version+)))
        (is (eq :version-mismatch (reason (loader-state-with state :version version)))))
      (is (eq :bad-state (reason (loader-state-with state :version (make-array '() :initial-element 1)))))
      (dolist (bad '((-1 0) (0 -1) (0 6) (2 1) (3 0)))
        (is (eq :state-out-of-range (reason (loader-state-with state :epoch (first bad)
                                                                     :position (second bad))))))
      (is (eq :bad-state (reason (loader-state-with state :key (make-array 2 :initial-element 0)))))
      (is (eq :bad-state (reason (loader-state-with state :epoch (make-array '() :initial-element 0)))))
      (is (eq :bad-state (reason (loader-state-with state :position (make-array 1 :element-type '(signed-byte 32))))))
      (is (eq :bad-state (reason (list 0 0 0)))))
    ;; num-epochs nil ならエポックの上限は無い
    (is (eq :no-error
            (loader-failure-reason
             (lambda ()
               (restore-iterator (data-loader (loader-index-source 5) :batch-size 2)
                                 (loader-state-with (iterator-state (make-data-iterator
                                                                     (data-loader (loader-index-source 5)
                                                                                  :batch-size 2)))
                                                    :epoch 1000 :position 3))))))))

(defclass loader-counting-source (data-source)
  ((length :initarg :length :reader source-length))
  (:documentation "INDEX 番目のレコードが (INDEX の f64 の rank 0 配列) の、配列を持たない長いデータソース。"))

(defmethod source-item ((source loader-counting-source) index)
  (list (make-array '() :element-type 'double-float :initial-element (float index 1d0))))

(defun loader-state-values (state)
  (list (aref (iterator-state-epoch state)) (aref (iterator-state-position state))))

(test (iterator-state/epoch-and-position-must-fit-in-i32 :suite :nabla.small)
  "状態のエポックと位置は :i32 の上限 2^31 - 1 まで取れ、超えたら :state-out-of-range。"
  (let ((max (1- (expt 2 31))))
    (flet ((state-after-one-batch (n epoch position)
             (let* ((loader (data-loader (make-instance 'loader-counting-source :length n) :batch-size 1))
                    (it (restore-iterator loader (loader-state-with
                                                  (iterator-state (make-data-iterator loader))
                                                  :epoch epoch :position position))))
               (loader-state-values (iterator-state it))
               (next-batch it)
               (iterator-state it))))
      ;; 位置: 長さ 3 × 2^30 のエポックで、位置 2^31 - 2 の次は上限、その次は収まらない
      (is (equal (list 0 max) (loader-state-values (state-after-one-batch (* 3 (expt 2 30)) 0 (1- max)))))
      (is (eq :state-out-of-range
              (loader-failure-reason (lambda () (state-after-one-batch (* 3 (expt 2 30)) 0 max)))))
      ;; エポック: 長さ 1 で、エポック 2^31 - 2 の次は上限、その次は収まらない
      (is (equal (list max 0) (loader-state-values (state-after-one-batch 1 (1- max) 0))))
      (is (eq :state-out-of-range
              (loader-failure-reason (lambda () (state-after-one-batch 1 max 0)))))))
  ;; レコードが 0 個のローダの状態はエポック 0・位置 0
  (is (equal '(0 0) (loader-state-values
                     (iterator-state (make-data-iterator (data-loader (loader-index-source 0)
                                                                      :batch-size 1 :num-epochs 0)))))))
