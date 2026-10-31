;;;; map / filter の変換（issue #281）。
;;;;
;;;; レコード（PyTree）1 つにかける純粋な変換（Grain の MapTransform / FilterTransform）。
;;;; 変換のリストはローダ（#285）の :transforms に渡し、ローダとワーカー（#286）は同じ
;;;; APPLY-TRANSFORMS でレコードを組み立てる。バッチにまとめるのは変換ではなくローダの
;;;; :batch-size の仕事（batch.lisp の BATCH-RECORDS）で、変換のリストには入れられない
;;;; （「batch はパイプラインの最後に 1 つだけ」を構造で守る）。
;;;; RANDOM-MAP-TRANSFORM（issue #284）はレコードと乱数のキーを受ける変換で、キーはシードと
;;;; サンプラーの通し位置から data 層が作る（並列度・処理の順序・再開に依らない）。

(in-package #:nabla.data)

(defstruct (map-transform (:constructor map-transform (function)) (:copier nil))
  "レコード → レコードの変換。MAP-TRANSFORM で作る。"
  (function nil :type (or function symbol) :read-only t))

(setf (documentation 'map-transform 'function)
      "レコードを受けて新しいレコードを返す関数 FUNCTION の変換を作る（Grain の MapTransform）。
FUNCTION はレコードの木をまるごと受ける（葉ごとではない）。変換はワーカースレッドから同時に
呼ばれるので、FUNCTION は純粋（同じレコードには同じ結果を返し、レコードも大域の状態も
書き換えない）でなければならない。")

(defstruct (filter-transform (:constructor filter-transform (predicate)) (:copier nil))
  "レコードを残すかどうかの述語。FILTER-TRANSFORM で作る。"
  (predicate nil :type (or function symbol) :read-only t))

(setf (documentation 'filter-transform 'function)
      "述語 PREDICATE が真を返すレコードだけを残す変換を作る（Grain の FilterTransform）。
MAP-TRANSFORM の FUNCTION と同じく、PREDICATE は純粋でなければならない。")

(defstruct (random-map-transform (:constructor random-map-transform (function)) (:copier nil))
  "レコードとキー → レコードの変換。RANDOM-MAP-TRANSFORM で作る。"
  (function nil :type (or function symbol) :read-only t))

(setf (documentation 'random-map-transform 'function)
      "レコードとキーを受けて新しいレコードを返す関数 FUNCTION の変換を作る（Grain の
RandomMapTransform）。データ拡張（ランダムな反転・切り抜きなど）を決定的に再開できる形で書くのに使う。
FUNCTION は (FUNCTION レコード キー) と呼ばれ、キーは NABLA:PRNG-KEY と同じ形の :u32 の (2) の
新しい配列。キーはシード・サンプラーの通し位置 k・変換のリストの中でこの変換が何番目の
RANDOM-MAP-TRANSFORM か、だけで決まり（APPLY-TRANSFORMS の :SEED と :POSITION。ローダはそのシード
（無ければ 0）と k を渡す）、並列度や処理の順序に依らない。同じレコードでもエポックが違えば k が
違うので、別の乱数になる。リストに 2 つ以上あれば、それぞれ別のキーを受ける。FUNCTION の中では
NABLA:UNIFORM / NABLA:SPLIT などをキーにそのまま使える（eager。ワーカースレッドから呼んでよいが、
1 回に数十 µs かかるので、必要な乱数は 1 回でまとめて引くとよい）。MAP-TRANSFORM の FUNCTION と
同じく、FUNCTION は (レコード, キー) について純粋でなければならない。ワーカーが動いている間は
コードをロードしない（DEFPRIMITIVE / DEFPYTREE が大域の表を書き換える）。")

(define-condition transform-error (data-error)
  ((reason :initarg :reason :reader transform-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader transform-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "transform: ~A (~S)" (transform-error-detail c) (transform-error-reason c))))
  (:documentation
   "APPLY-TRANSFORMS が変換をかけられないときに signal する。REASON は次のどれか。
  :bad-seed      SEED が nil でも :u32 の (2) の配列（prng-key の形）でもない
  :bad-position  RANDOM-MAP-TRANSFORM をかけるのに、POSITION が 0 以上 2^64 未満の整数でない"))

(defun %transform-error (reason control &rest arguments)
  (error 'transform-error :reason reason :detail (apply #'format nil control arguments)))

(defun %record-key (seed64 ordinal position)
  "シード SEED64・ORDINAL 番目（0 から）の RANDOM-MAP-TRANSFORM・通し位置 POSITION のキー。
状態 mix64(mix64(SEED64) xor mix64(ORDINAL)) の SplitMix64 の列の POSITION 番目（状態 + POSITION × γ を
攪拌した 64 ビット）を [上位 32 ビット, 下位 32 ビット] に割る（PRNG-KEY と同じ並び）。シードを先に
攪拌するので、シードの差と位置の差が打ち消し合わない。導出を変えたら
+INDEX-SAMPLER-ALGORITHM-VERSION+ を上げる（再開した後の拡張が黙って変わらないように）。"
  (let ((bits (%splitmix64 (%mix64 (logxor (%mix64 seed64) (%mix64 ordinal))) position)))
    (make-array 2 :element-type '(unsigned-byte 32)
                  :initial-contents (list (ash bits -32) (ldb (byte 32 0) bits)))))

(defun apply-transforms (transforms record &key seed position)
  "変換のリスト TRANSFORMS を先頭から順に RECORD にかける。どの filter も RECORD を落とさな
ければ (values 変換後のレコード t)、落としたら (values nil nil) を返す（落とした後の変換は
呼ばない）。RANDOM-MAP-TRANSFORM には、SEED（:u32 の (2) の配列。nil なら 0）・RECORD の
サンプラーの通し位置 POSITION・その変換が TRANSFORMS の何番目の RANDOM-MAP-TRANSFORM かから作る
キーを渡す（ローダは自分のシードと位置を渡す）。
TRANSFORMS に MAP-TRANSFORM / FILTER-TRANSFORM / RANDOM-MAP-TRANSFORM でないものがあれば
TYPE-ERROR、SEED の形が違えば :bad-seed、RANDOM-MAP-TRANSFORM をかけるのに POSITION が 0 以上
2^64 未満の整数でなければ :bad-position の TRANSFORM-ERROR。"
  (unless (or (null seed)
              (and (arrayp seed)
                   (equal (array-element-type seed) '(unsigned-byte 32))
                   (equal (array-dimensions seed) '(2))))
    (%transform-error :bad-seed "シード ~S が :u32 の (2) の配列（prng-key の形）でない" seed))
  (let ((seed64 (if seed (%seed64 seed) 0))
        (ordinal 0))
    (dolist (transform transforms (values record t))
      (etypecase transform
        (map-transform
         (setf record (funcall (map-transform-function transform) record)))
        (filter-transform
         (unless (funcall (filter-transform-predicate transform) record)
           (return (values nil nil))))
        (random-map-transform
         (unless (typep position '(unsigned-byte 64))
           (%transform-error :bad-position
                             "random-map-transform には 0 以上 2^64 未満の整数の位置が要るが、~S だった"
                             position))
         (setf record (funcall (random-map-transform-function transform)
                               record (%record-key seed64 ordinal position)))
         (incf ordinal))))))
