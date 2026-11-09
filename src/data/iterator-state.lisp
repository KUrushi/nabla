;;;; イテレータの再開状態（issue #285）。
;;;;
;;;; 状態は (key, epoch, position) の 3 つ（Grain の DataLoaderIterator.get_state）と、サンプラーの
;;;; シャッフルの算法の版 version（+INDEX-SAMPLER-ALGORITHM-VERSION+。版が違う状態から再開すると
;;;; 黙って順序が変わるので、RESTORE-ITERATOR が :version-mismatch にする）。position は
;;;; 「エポック内で、次に調べるサンプラーの位置」で、イテレータが返したバッチに取り込んだ最後の
;;;; 位置 + 1。エポックとエポック内の位置に分けるので、レコードの個数とエポックの数が 2^31 未満なら
;;;; :i32 に収まる。状態はチェックポイントの木に入れるので、葉はどれも配列で（Lisp の整数は葉に
;;;; できない）、子に名前を付けて defpytree に登録する（パス名が "<i>.key" のようになり、子の
;;;; 並びに依らない）。nabla/nn の defmodule は使わない（nabla/data は core にだけ依存する）。
;;;;
;;;; ローダの指紋 fingerprint（issue #322）: 状態は位置しか持たないので、別の引数のローダで再開すると
;;;; error にならずに別のバッチの列が出る。そこで、バッチの列を変えるローダの設定のうちプロセスをまたいで
;;;; 比べられるもの（レコードの個数・シャードが引く位置の数・batch-size・シャードの番号と数・shuffle・
;;;; drop-remainder・pad-remainder・変換の数と種類の並び）を :u32 の (9) に詰めて状態に入れ、
;;;; RESTORE-ITERATOR が比べる（違えば :loader-mismatch）。変換の関数とデータソースの中身は比べられない
;;;; （関数の同一性はプロセスをまたげない）。シードは key、シャッフルの算法は version が守るので入れない。
;;;; num-epochs も入れない（同じ列を長くするだけなので、学習を延ばすために変えてよい）。指紋は算法の版とは
;;;; 独立で、指紋の詰め方を変えたときは *LOADER-FINGERPRINT-FIELDS* の説明と一緒に変える（版の数ではなく
;;;; 指紋の値そのものが変わるので、古い状態は :loader-mismatch になる）。

(in-package #:nabla.data)

(defstruct (iterator-state (:constructor %make-iterator-state (key epoch position version fingerprint))
                           (:copier nil))
  "DATA-ITERATOR の再開状態。ITERATOR-STATE で取り、RESTORE-ITERATOR で再開する。葉は KEY
（ローダのシードの写し。:u32 の (2)）、EPOCH・POSITION・VERSION（rank 0 の (signed-byte 32) の配列）、
FINGERPRINT（ローダの設定の指紋。:u32 の (9)）。"
  (key nil :read-only t)
  (epoch nil :read-only t)
  (position nil :read-only t)
  (version nil :read-only t)
  (fingerprint nil :read-only t))

(setf (documentation 'iterator-state-key 'function)
      "ITERATOR-STATE のキー（ローダのシードの写し。シードが無ければ 0。:u32 の (2) の配列）。"
      (documentation 'iterator-state-epoch 'function)
      "ITERATOR-STATE のエポック（次に調べるサンプラーの位置のエポック。rank 0 の :i32 の配列）。"
      (documentation 'iterator-state-position 'function)
      "ITERATOR-STATE のエポック内の位置（次に調べるサンプラーの位置。rank 0 の :i32 の配列）。"
      (documentation 'iterator-state-version 'function)
      "ITERATOR-STATE を作ったサンプラーのシャッフルの算法の版（rank 0 の :i32 の配列）。"
      (documentation 'iterator-state-fingerprint 'function)
      "ITERATOR-STATE を取ったローダの設定の指紋（:u32 の (9) の配列。RESTORE-ITERATOR が比べる）。")

(nabla:defpytree iterator-state
  :flatten (lambda (state)
             (values (list (iterator-state-key state)
                           (iterator-state-epoch state)
                           (iterator-state-position state)
                           (iterator-state-version state)
                           (iterator-state-fingerprint state))
                     nil
                     '("key" "epoch" "position" "version" "fingerprint")))
  :unflatten (lambda (aux children)
               (declare (ignore aux))
               (destructuring-bind (key epoch position version fingerprint) children
                 (%make-iterator-state key epoch position version fingerprint))))

(defparameter *loader-fingerprint-fields*
  '("レコードの個数（上位 32 ビット）" "レコードの個数（下位 32 ビット）"
    "シャードが 1 エポックに引く位置の数（上位 32 ビット）" "シャードが 1 エポックに引く位置の数（下位 32 ビット）"
    "batch-size" "shard-index" "shard-count" "shuffle・drop-remainder・pad-remainder のビット"
    "変換の数と種類の並び")
  "ITERATOR-STATE の fingerprint の各要素の意味（:loader-mismatch の説明に使う）。")

(defun %transforms-signature (transforms)
  "TRANSFORMS の数と種類の並びの 32 ビットの要約（変換の関数は見ない）。"
  (let ((hash (length transforms)))
    (loop for transform in transforms
          for i from 1
          do (setf hash (%mix64 (logxor hash
                                        (+ (* 4 i)
                                           (etypecase transform
                                             (map-transform 1)
                                             (filter-transform 2)
                                             (random-map-transform 3)))))))
    (ldb (byte 32 0) hash)))

(defun %loader-fingerprint (loader)
  "LOADER の設定の指紋（:u32 の (9) の新しい配列。要素の意味は *LOADER-FINGERPRINT-FIELDS*）。"
  (let ((n (data-loader-record-count loader))
        (m (data-loader-length loader)))
    (make-array 9 :element-type '(unsigned-byte 32)
                  :initial-contents
                  (list (ldb (byte 32 32) n) (ldb (byte 32 0) n)
                        (ldb (byte 32 32) m) (ldb (byte 32 0) m)
                        (ldb (byte 32 0) (data-loader-batch-size loader))
                        (ldb (byte 32 0) (data-loader-shard-index loader))
                        (ldb (byte 32 0) (data-loader-shard-count loader))
                        (logior (if (index-sampler-shuffle (data-loader-sampler loader)) 1 0)
                                (if (data-loader-drop-remainder loader) 2 0)
                                (if (data-loader-pad-remainder loader) 4 0))
                        (%transforms-signature (data-loader-transforms loader))))))

(defun %i32-scalar (value)
  (make-array '() :element-type '(signed-byte 32) :initial-element value))

(defun %loader-state (loader k)
  "LOADER の通しの位置 K を指す、新しい配列の ITERATOR-STATE。"
  (let ((n (data-loader-length loader)))
    (multiple-value-bind (epoch position) (if (zerop n) (values 0 0) (floor k n))
      (unless (and (typep epoch '(signed-byte 32)) (typep position '(signed-byte 32)))
        (%data-loader-error :state-out-of-range
                            "エポック ~D・位置 ~D が :i32 に収まらない" epoch position))
      (%make-iterator-state (copy-seq (data-loader-key loader)) (%i32-scalar epoch)
                            (%i32-scalar position) (%i32-scalar +index-sampler-algorithm-version+)
                            (%loader-fingerprint loader)))))

(defun iterator-state (iterator)
  "ITERATOR の今の再開状態を、新しい配列の ITERATOR-STATE で返す（後で ITERATOR を進めても
変わらない）。key はローダのシード（無ければ 0）の写し、epoch と position は次に調べる
サンプラーの位置のエポックとエポック内の位置、version はシャッフルの算法の版。エポックか位置が
:i32 に収まらなければ :state-out-of-range の DATA-LOADER-ERROR。"
  (%loader-state (data-iterator-loader iterator) (data-iterator-position iterator)))

(defun initial-iterator-state (loader)
  "LOADER の列の先頭（エポック 0・位置 0）の再開状態を、新しい配列の ITERATOR-STATE で返す。
開いたばかりのイテレータの ITERATOR-STATE と同じ木だが、イテレータを開かない（ワーカーの
スレッドを作らず、レコードも読まない）。チェックポイントから再開するとき LOAD-CHECKPOINT に
渡すひな形に使う（ひな形の値は使われず、葉の名前・形・dtype だけが照合される）。
RESTORE-ITERATOR に渡せば先頭から始まる。"
  (%loader-state loader 0))

(defun %state-leaf-p (leaf element-type dimensions)
  (and (arrayp leaf)
       (equal (array-element-type leaf) element-type)
       (equal (array-dimensions leaf) dimensions)))

(defun restore-iterator (loader state &key (num-workers 0) (prefetch 2))
  "LOADER のバッチの列を、STATE（ITERATOR-STATE。チェックポイントから復元したものでよい）の
位置から続けるイテレータを返す。残りのバッチの列は、STATE を取ったイテレータの続きと同じ。
STATE が ITERATOR-STATE でないか葉の型・形が違えば :bad-state、version がこの版のサンプラーの
シャッフルの算法の版と違えば（順序が変わるので）:version-mismatch、key が LOADER のシード（無ければ 0）
と違えば :key-mismatch、fingerprint（レコードの個数・シャードが引く位置の数・BATCH-SIZE・
シャードの番号と数・SHUFFLE・DROP-REMAINDER・PAD-REMAINDER・変換の数と種類）が LOADER と違えば
:loader-mismatch、position が 0 以上 LOADER のレコードの個数以下でないか、epoch が
0 以上でないか、NUM-EPOCHS を超える（epoch = NUM-EPOCHS なら position は 0 だけ）なら
:state-out-of-range の DATA-LOADER-ERROR。状態は変換の関数とデータソースの中身を比べないので、
LOADER は状態を取ったローダと同じデータソース・変換で作ること（その違いは検査できない）。NUM-EPOCHS は違ってよい（範囲に収まる限り）。
NUM-WORKERS と PREFETCH は MAKE-DATA-ITERATOR と同じ（状態を取ったイテレータと違ってよい）。"
  (unless (and (iterator-state-p state)
               (%state-leaf-p (iterator-state-key state) '(unsigned-byte 32) '(2))
               (%state-leaf-p (iterator-state-epoch state) '(signed-byte 32) '())
               (%state-leaf-p (iterator-state-position state) '(signed-byte 32) '())
               (%state-leaf-p (iterator-state-version state) '(signed-byte 32) '())
               (%state-leaf-p (iterator-state-fingerprint state) '(unsigned-byte 32) '(9)))
    (%data-loader-error :bad-state
                        "~S が、:u32 の (2) と rank 0 の :i32 3 つと :u32 の (9) の iterator-state でない"
                        state))
  (unless (= (aref (iterator-state-version state)) +index-sampler-algorithm-version+)
    (%data-loader-error :version-mismatch
                        "状態のシャッフルの算法の版 ~D が、このサンプラーの版 ~D と違う"
                        (aref (iterator-state-version state)) +index-sampler-algorithm-version+))
  (unless (equalp (iterator-state-key state) (data-loader-key loader))
    (%data-loader-error :key-mismatch "状態の key ~S がローダのシード ~S と違う"
                        (iterator-state-key state) (data-loader-key loader)))
  (let* ((expected (%loader-fingerprint loader))
         (actual (iterator-state-fingerprint state))
         (i (mismatch actual expected)))
    (when i
      (%data-loader-error :loader-mismatch
                          "状態を取ったローダと設定が違う（~A が状態では ~D、このローダでは ~D）。~
状態を取ったときと同じデータソース・引数でローダを作ること"
                          (nth i *loader-fingerprint-fields*) (aref actual i) (aref expected i))))
  (let* ((n (data-loader-length loader))
         (num-epochs (data-loader-num-epochs loader))
         (epoch (aref (iterator-state-epoch state)))
         (position (aref (iterator-state-position state)))
         (k (+ (* epoch n) position)))
    (unless (and (<= 0 epoch) (<= 0 position n)
                 (or (null num-epochs) (and (<= epoch num-epochs) (<= k (* num-epochs n)))))
      (%data-loader-error :state-out-of-range
                          "エポック ~D・位置 ~D が、レコード ~D 個・~:[終わらない~;~:*~D エポック~]の範囲の外"
                          epoch position n num-epochs))
    (%watch-iterator (%make-data-iterator loader k (%open-worker-pool num-workers prefetch)))))
