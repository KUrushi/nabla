;;;; data-loader と data-iterator（issue #285。逐次版）。
;;;;
;;;; ローダはデータソース・サンプラー・変換・バッチの大きさを束ねたもの（Grain の DataLoader）、
;;;; イテレータはその上の「次に調べるサンプラーの位置」だけを持つカーソル（DataLoaderIterator）。
;;;; 組み立ては次の 2 段に分ける（後続の並列ローダ #286 が同じ 2 段を共有する）。
;;;;   1. %RANGE-RECORDS: サンプラーの位置の区間 [start, end) → 変換で残ったレコードと位置の組。
;;;;      位置ごとに独立な純粋関数なので、区間をワーカーに分けて投げられる。
;;;;   2. %ASSEMBLE-BATCH: 位置 k から 1 の結果だけを順に読み、バッチ 1 つ分のレコードと次の位置を
;;;;      決める。バッチはエポックをまたがず、エポック末の端数は drop-remainder なら捨てる。
;;;; 次の位置は「返したバッチに取り込んだ最後の位置 + 1」（入力側の位置）なので、filter を
;;;; はさんでも、その位置から再開すれば残りの列は中断しなかった場合と一致する。

(in-package #:nabla.data)

(define-condition data-loader-error (data-error)
  ((reason :initarg :reason :reader data-loader-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader data-loader-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "data-loader: ~A (~S)"
                     (data-loader-error-detail c) (data-loader-error-reason c))))
  (:documentation
   "ローダを作れない、またはバッチを取り出せないときに signal する。REASON は次のどれか。
  :bad-batch-size  BATCH-SIZE が正の整数でない
  :bad-transform   TRANSFORMS に MAP-TRANSFORM / FILTER-TRANSFORM / RANDOM-MAP-TRANSFORM でないものがある
  :bad-epoch       EPOCH のエポックが 0 以上 NUM-EPOCHS 未満の整数でない
  :empty-epoch     1 エポックの全位置を調べてもバッチが1つも作れなかった（filter が全部
                   落とした、端数を捨てて何も残らなかった、レコードが 0 個）
  :bad-state       RESTORE-ITERATOR の状態が ITERATOR-STATE でないか、葉の型・形が違う
  :version-mismatch  状態のシャッフルの算法の版がこのサンプラーの版と違う
  :key-mismatch    状態のキーがローダのシード（無ければ 0）と違う
  :state-out-of-range  状態のエポック・位置がローダの範囲の外か、ITERATOR-STATE で :i32 に
                   収まらない
  :bad-num-workers NUM-WORKERS が 0 以上の整数でない
  :bad-prefetch    PREFETCH が 0 以上の整数でない
  :closed          CLOSE-ITERATOR で閉じたイテレータから NEXT-BATCH した"))

(defun %data-loader-error (reason control &rest arguments)
  (error 'data-loader-error :reason reason :detail (apply #'format nil control arguments)))

(defstruct (data-loader (:constructor %make-data-loader
                            (source sampler length transforms batch-size drop-remainder num-epochs
                             key))
                        (:copier nil) (:predicate nil))
  "データソース・サンプラー・変換・バッチの大きさを束ねたもの。DATA-LOADER で作り、
MAKE-DATA-ITERATOR か EPOCH でバッチを取り出す。作った後は変わらない。"
  (source nil :read-only t)
  (sampler nil :read-only t)
  (length 0 :type (integer 0) :read-only t)
  (transforms '() :type list :read-only t)
  (batch-size 1 :type (integer 1) :read-only t)
  (drop-remainder t :read-only t)
  (num-epochs nil :type (or null (integer 0)) :read-only t)
  ;; シード（無ければ 0）の写し。:u32 の (2)。再開状態の key になる。
  (key nil :type (simple-array (unsigned-byte 32) (2)) :read-only t))

(defun data-loader (source &key batch-size shuffle seed num-epochs transforms (drop-remainder t))
  "データソース SOURCE からバッチを作るローダを作る（Grain の DataLoader）。
サンプラー（MAKE-INDEX-SAMPLER に SHUFFLE・SEED・NUM-EPOCHS を渡したもの）の位置の順に
レコードを SOURCE-ITEM で引き、TRANSFORMS（MAP-TRANSFORM / FILTER-TRANSFORM / RANDOM-MAP-TRANSFORM
のリスト）を APPLY-TRANSFORMS でかけ（RANDOM-MAP-TRANSFORM のキーは SEED（無ければ 0）と通し位置で
決まる）、残ったレコードを BATCH-SIZE（正の整数。必須）個ずつ BATCH-RECORDS で
積む。バッチはエポックをまたがない。エポック末の端数は DROP-REMAINDER（既定 t）なら捨て、nil なら
短いバッチにする（pad はしない）。NUM-EPOCHS が nil（既定）ならバッチの列は終わらない。
SEED（無ければ 0）の写しを持ち、ITERATOR-STATE の key にする（シャッフルしなくても、別の
シードのローダには再開状態を渡せない）。
不正な引数は DATA-LOADER-ERROR か SAMPLER-ERROR。SOURCE の長さは作るときに読む。"
  (unless (typep batch-size '(integer 1))
    (%data-loader-error :bad-batch-size "batch-size ~S が正の整数でない" batch-size))
  (unless (and (listp transforms)
               (every (lambda (transform)
                    (typep transform '(or map-transform filter-transform random-map-transform)))
                      transforms))
    (%data-loader-error :bad-transform
                        "transforms ~S が変換（map / filter / random-map）のリストでない" transforms))
  (let* ((n (source-length source))
         (sampler (make-index-sampler n :shuffle shuffle :seed seed :num-epochs num-epochs)))
    ;; SEED の形は MAKE-INDEX-SAMPLER が検査済み。
    (%make-data-loader source sampler n transforms batch-size (and drop-remainder t) num-epochs
                       (make-array 2 :element-type '(unsigned-byte 32)
                                     :initial-contents (if seed (list (aref seed 0) (aref seed 1)) '(0 0))))))

;;; --- 組み立て（#286 の並列ローダと共有する） ---

(defun %range-records (loader start end)
  "サンプラーの位置の区間 [START, END) のうち、変換で残ったレコードを (位置 . レコード) の組の
リストにして位置の昇順で返す。位置ごとに独立な純粋関数（SOURCE-ITEM と変換が純粋である限り）
なので、区間を分けて別々のスレッドで呼び、結果を位置の順につないでもよい。"
  (loop for k from start below end
        for (record keep) = (multiple-value-list
                             (apply-transforms (data-loader-transforms loader)
                                               (source-item (data-loader-source loader)
                                                            (nth-value 1 (sampler-index
                                                                          (data-loader-sampler loader)
                                                                          k)))
                                               :seed (data-loader-key loader) :position k))
        when keep collect (cons k record)))

(defun %assemble-batch (loader k fetch)
  "位置 K（エポック内の位置ではなく通しの位置）から、次のバッチに入るレコードを集める。
FETCH は (FETCH start end) → %RANGE-RECORDS と同じ結果を返す関数で、区間はいつも 1 エポックの
中に収まり、前の区間の直後から始まる。(values レコードのリスト 次の位置 バッチのエポック) を返す。
次の位置は、バッチの最後のレコードの位置 + 1。列の終わり（NUM-EPOCHS のエポックを出し切った）
なら (values nil K nil)。エポックの先頭から末尾まで調べてバッチが作れなければ :empty-epoch。"
  (let ((n (data-loader-length loader))
        (batch-size (data-loader-batch-size loader))
        (num-epochs (data-loader-num-epochs loader)))
    (when (zerop n)
      (if (eql num-epochs 0)
          (return-from %assemble-batch (values nil k nil))
          (%data-loader-error :empty-epoch "レコードが 0 個なので、エポックからバッチを作れない")))
    (loop
      (multiple-value-bind (epoch position) (floor k n)
        (when (and num-epochs (>= epoch num-epochs))
          (return (values nil k nil)))
        (let ((epoch-end (* (1+ epoch) n))
              (records '())
              (count 0)
              (last nil))
          ;; 足りない個数ずつ区間を読むので、バッチより先の位置は読まない。
          (loop for start = k then end
                for end = (min epoch-end (+ start (- batch-size count)))
                while (and (< count batch-size) (< start epoch-end))
                do (loop for (k1 . record) in (funcall fetch start end)
                         do (push record records) (incf count) (setf last k1)))
          (when (or (= count batch-size)
                    (and (plusp count) (not (data-loader-drop-remainder loader))))
            (return (values (nreverse records) (1+ last) epoch)))
          (when (zerop position)
            (%data-loader-error :empty-epoch
                                "エポック ~D の全 ~D 位置を調べても、batch-size ~D のバッチを作れなかった"
                                epoch n batch-size))
          ;; エポックの途中から調べて残りが端数だけなら、次のエポックの先頭へ。
          (setf k epoch-end))))))

;;; --- イテレータ ---

(defstruct (data-iterator (:constructor %make-data-iterator (loader position pool))
                          (:copier nil) (:predicate nil))
  "LOADER のバッチを 1 つずつ返すカーソル。MAKE-DATA-ITERATOR で作り、NEXT-BATCH で進める。"
  (loader nil :read-only t)
  (position 0 :type (integer 0))
  ;; ワーカーの組（#286、src/data/parallel.lisp）。逐次でも閉じたかどうかを持つ。
  (pool nil :read-only t))

(defun %range-fetcher (loader)
  (lambda (start end) (%range-records loader start end)))

(defun make-data-iterator (loader &key (num-workers 0) (prefetch 2))
  "LOADER のバッチの列を先頭（エポック 0 の最初）から返すイテレータを作る
（Grain の DataLoaderIterator）。NEXT-BATCH で 1 つずつ取り出す。NUM-WORKERS（0 以上の整数、
既定 0 = 逐次）が正なら、そのイテレータだけのワーカーのスレッドでレコードを並列に読む
（バッチの列はワーカー数に依らない）。ワーカーは、返したバッチの先 PREFETCH（0 以上の整数、
既定 2）個ぶんのバッチの位置も先に読んでおく（呼び出し側がバッチを使う間に読む。ワーカー数 0 では
効かない）。使い終わったら CLOSE-ITERATOR で閉じる（WITH-DATA-ITERATOR なら自動で閉じる）。"
  (%make-data-iterator loader 0 (%open-worker-pool num-workers prefetch)))

(defun next-batch (iterator)
  "ITERATOR の次のバッチを (values バッチ t) で返して進める。バッチはレコードの木の各葉を
新しい先頭の軸で積んだ木（BATCH-RECORDS）。列が終わっていれば (values nil nil) を返し、
何度呼んでも同じ。エポックからバッチが作れなければ :empty-epoch、閉じたイテレータなら :closed の
DATA-LOADER-ERROR。ワーカーの中の error（SOURCE-ITEM や変換の中）は元のコンディションのまま
届き、位置は進まない（もう一度呼ぶと同じ位置から読み直す）。"
  (let ((loader (data-iterator-loader iterator)))
    (multiple-value-bind (records next)
        (%assemble-batch loader (data-iterator-position iterator) (%iterator-fetcher iterator))
      (cond (records
             (setf (data-iterator-position iterator) next)
             (values (batch-records records) t))
            (t (values nil nil))))))

(defun epoch (loader epoch)
  "LOADER のエポック EPOCH（0 から数える）のバッチを、イテレータと同じ順のリストで返す。
エポックを順につなぐとイテレータの列になる。EPOCH が 0 以上 NUM-EPOCHS（nil なら無限）未満の
整数でなければ :bad-epoch。1 エポック分のバッチをすべてメモリに持つので、大きなデータでは
MAKE-DATA-ITERATOR を使う。"
  (let ((num-epochs (data-loader-num-epochs loader)))
    (unless (and (integerp epoch) (<= 0 epoch) (or (null num-epochs) (< epoch num-epochs)))
      (%data-loader-error :bad-epoch "エポック ~S が 0 以上~:[の~;~:* ~D 未満の~]整数でない"
                          epoch num-epochs)))
  (let ((fetch (%range-fetcher loader))
        (end (* (1+ epoch) (data-loader-length loader))))
    ;; 最初の 1 回は必ず呼ぶ（バッチが作れなければ :empty-epoch）。次の位置がエポックの末尾に
    ;; 達したら、次のエポックのレコードを読まずに止める。
    (loop with k = (* epoch (data-loader-length loader))
          for (records next batch-epoch) = (multiple-value-list (%assemble-batch loader k fetch))
          while (eql batch-epoch epoch)
          collect (batch-records records)
          do (setf k next)
          until (>= k end))))
