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

(in-package #:nabla.data)

(defstruct (iterator-state (:constructor %make-iterator-state (key epoch position version))
                           (:copier nil))
  "DATA-ITERATOR の再開状態。ITERATOR-STATE で取り、RESTORE-ITERATOR で再開する。葉は KEY
（ローダのシードの写し。:u32 の (2)）、EPOCH・POSITION・VERSION（rank 0 の (signed-byte 32) の配列）。"
  (key nil :read-only t)
  (epoch nil :read-only t)
  (position nil :read-only t)
  (version nil :read-only t))

(setf (documentation 'iterator-state-key 'function)
      "ITERATOR-STATE のキー（ローダのシードの写し。シードが無ければ 0。:u32 の (2) の配列）。"
      (documentation 'iterator-state-epoch 'function)
      "ITERATOR-STATE のエポック（次に調べるサンプラーの位置のエポック。rank 0 の :i32 の配列）。"
      (documentation 'iterator-state-position 'function)
      "ITERATOR-STATE のエポック内の位置（次に調べるサンプラーの位置。rank 0 の :i32 の配列）。"
      (documentation 'iterator-state-version 'function)
      "ITERATOR-STATE を作ったサンプラーのシャッフルの算法の版（rank 0 の :i32 の配列）。")

(nabla:defpytree iterator-state
  :flatten (lambda (state)
             (values (list (iterator-state-key state)
                           (iterator-state-epoch state)
                           (iterator-state-position state)
                           (iterator-state-version state))
                     nil
                     '("key" "epoch" "position" "version")))
  :unflatten (lambda (aux children)
               (declare (ignore aux))
               (destructuring-bind (key epoch position version) children
                 (%make-iterator-state key epoch position version))))

(defun %i32-scalar (value)
  (make-array '() :element-type '(signed-byte 32) :initial-element value))

(defun iterator-state (iterator)
  "ITERATOR の今の再開状態を、新しい配列の ITERATOR-STATE で返す（後で ITERATOR を進めても
変わらない）。key はローダのシード（無ければ 0）の写し、epoch と position は次に調べる
サンプラーの位置のエポックとエポック内の位置、version はシャッフルの算法の版。エポックか位置が
:i32 に収まらなければ :state-out-of-range の DATA-LOADER-ERROR。"
  (let* ((loader (data-iterator-loader iterator))
         (n (data-loader-length loader)))
    (multiple-value-bind (epoch position)
        (if (zerop n) (values 0 0) (floor (data-iterator-position iterator) n))
      (unless (and (typep epoch '(signed-byte 32)) (typep position '(signed-byte 32)))
        (%data-loader-error :state-out-of-range
                            "エポック ~D・位置 ~D が :i32 に収まらない" epoch position))
      (%make-iterator-state (copy-seq (data-loader-key loader)) (%i32-scalar epoch)
                            (%i32-scalar position) (%i32-scalar +index-sampler-algorithm-version+)))))

(defun %state-leaf-p (leaf element-type dimensions)
  (and (arrayp leaf)
       (equal (array-element-type leaf) element-type)
       (equal (array-dimensions leaf) dimensions)))

(defun restore-iterator (loader state &key (num-workers 0) (prefetch 2))
  "LOADER のバッチの列を、STATE（ITERATOR-STATE。チェックポイントから復元したものでよい）の
位置から続けるイテレータを返す。残りのバッチの列は、STATE を取ったイテレータの続きと同じ。
STATE が ITERATOR-STATE でないか葉の型・形が違えば :bad-state、version がこの版のサンプラーの
シャッフルの算法の版と違えば（順序が変わるので）:version-mismatch、key が LOADER のシード（無ければ 0）
と違えば :key-mismatch、position が 0 以上 LOADER のレコードの個数以下でないか、epoch が
0 以上でないか、NUM-EPOCHS を超える（epoch = NUM-EPOCHS なら position は 0 だけ）なら
:state-out-of-range の DATA-LOADER-ERROR。状態はデータソース・変換・SHUFFLE・BATCH-SIZE などを
持たないので、LOADER は状態を取ったローダと同じ引数で作ること（違っても検査できない）。
NUM-WORKERS と PREFETCH は MAKE-DATA-ITERATOR と同じ（状態を取ったイテレータと違ってよい）。"
  (unless (and (iterator-state-p state)
               (%state-leaf-p (iterator-state-key state) '(unsigned-byte 32) '(2))
               (%state-leaf-p (iterator-state-epoch state) '(signed-byte 32) '())
               (%state-leaf-p (iterator-state-position state) '(signed-byte 32) '())
               (%state-leaf-p (iterator-state-version state) '(signed-byte 32) '()))
    (%data-loader-error :bad-state "~S が、:u32 の (2) と rank 0 の :i32 3 つの iterator-state でない"
                        state))
  (unless (= (aref (iterator-state-version state)) +index-sampler-algorithm-version+)
    (%data-loader-error :version-mismatch
                        "状態のシャッフルの算法の版 ~D が、このサンプラーの版 ~D と違う"
                        (aref (iterator-state-version state)) +index-sampler-algorithm-version+))
  (unless (equalp (iterator-state-key state) (data-loader-key loader))
    (%data-loader-error :key-mismatch "状態の key ~S がローダのシード ~S と違う"
                        (iterator-state-key state) (data-loader-key loader)))
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
    (%make-data-iterator loader k (%open-worker-pool num-workers prefetch))))
