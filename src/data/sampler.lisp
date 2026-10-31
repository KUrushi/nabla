;;;; index-sampler（issue #280）。
;;;;
;;;; 通し番号の位置 k（0 から数える）から (エポック, レコードのインデックス) を返す純粋関数
;;;; （Grain の IndexSampler）。位置から直接計算できるので、再開状態は位置だけで済む。
;;;;
;;;; シャッフルは、エポックごとの置換を作って持つ代わりに、添字の全単射で計算する。
;;;; シードとエポックから SplitMix64 でラウンド鍵を作り、均衡 Feistel 置換で 2^(2h) ≥ n の
;;;; 領域を並べ替え、[0, n) の外に出た値はもう一度置換にかける（cycle-walking）。領域は 4n
;;;; 未満なので、期待される反復は 4 回未満。均衡 Feistel は（片側 2 ビット以上では）各ラウンドが
;;;; 偶置換なので、ラウンド数を奇数にしても偶置換のまま。cycle-walking で引き戻した置換の偶奇は
;;;; (-1)^(領域の外の要素数 - 外だけで閉じる巡回の数) で、n が領域の上限に近いほどシードでなく
;;;; n で決まる（n = 2^(2h) - 1 ではほぼ奇、2^(2h) - 2 ではほぼ偶）。そこで最後に鍵のもう 1 ビット
;;;; で決まる互換（インデックス 0 と 1 の入れ替え）をかけて偶奇を五分五分にする。nabla の PRNG（fold-in / uniform）は使わない（eager は呼ぶたびに graph を
;;;; 組むので、位置ごとに引くには遅い）。JAX / Grain とはビット単位で一致しない。

(in-package #:nabla.data)

(define-condition sampler-error (data-error)
  ((reason :initarg :reason :reader sampler-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader sampler-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "index-sampler: ~A (~S)"
                     (sampler-error-detail c) (sampler-error-reason c))))
  (:documentation
   "サンプラーを作れない、または位置を引けないときに signal する。REASON は次のどれか。
  :bad-length            N が 0 以上 2^64 未満の整数でない
  :bad-num-epochs        NUM-EPOCHS が nil か 0 以上の整数でない
  :bad-seed              SEED が :u32 の (2) の配列でない（SHUFFLE が真で SEED が無いときも）
  :position-out-of-range SAMPLER-INDEX の位置が 0 以上 N × NUM-EPOCHS 未満の整数でない"))

(defun %sampler-error (reason control &rest arguments)
  (error 'sampler-error :reason reason :detail (apply #'format nil control arguments)))

;;; --- SplitMix64 ---

(defconstant +splitmix64-gamma+ #x9E3779B97F4A7C15
  "SplitMix64 の状態の増分（黄金比の 2^64 倍の奇数）。")

(defconstant +index-sampler-algorithm-version+ 1
  "シャッフルの算法の版。SplitMix64 の定数・ラウンド数・領域の取り方・偶奇の補正を変えたら上げる。
同じ (シード, エポック, 位置) からは同じインデックスを返すことを版の中で約束し、ローダの再開状態に
この版を入れて、版をまたいで黙って順序が変わるのを防ぐ。テストの固定値はこの版のもので、算法を
変えるときは古い固定値を書き換えず、版を上げて新しい固定値を足す。")

(defconstant +feistel-rounds+ 4
  "Feistel 置換のラウンドの数（Luby–Rackoff の強い擬似ランダム置換になる 4）。")

(defun %mix64 (z)
  "SplitMix64 の出力関数（64 ビットの全単射の攪拌）。"
  (declare (type (unsigned-byte 64) z))
  (let* ((z (ldb (byte 64 0) (* (logxor z (ash z -30)) #xBF58476D1CE4E5B9)))
         (z (ldb (byte 64 0) (* (logxor z (ash z -27)) #x94D049BB133111EB))))
    (declare (type (unsigned-byte 64) z))
    (logxor z (ash z -31))))

(defun %splitmix64 (state i)
  "状態 STATE から始めた SplitMix64 の列の I 番目（1 始まり）の出力。"
  (declare (type (unsigned-byte 64) state i))
  (%mix64 (ldb (byte 64 0) (+ state (* i +splitmix64-gamma+)))))

;;; --- Feistel 置換 ---

(defun %epoch-state (seed64 epoch)
  "シードとエポックを混ぜた、ラウンド鍵の SplitMix64 の状態。エポックは下位 64 ビットを使う。"
  (declare (type (unsigned-byte 64) seed64))
  (%mix64 (logxor seed64 (%mix64 (ldb (byte 64 0) epoch)))))

(defun %feistel (x half-bits state)
  "2 × HALF-BITS ビットの X を、STATE から作るラウンド鍵の均衡 Feistel 置換にかける。"
  (declare (type (unsigned-byte 64) x state) (type (integer 1 32) half-bits))
  (let ((left (ldb (byte half-bits half-bits) x))
        (right (ldb (byte half-bits 0) x)))
    (declare (type (unsigned-byte 32) left right))
    (loop for round from 1 to +feistel-rounds+
          do (psetf left right
                    right (logxor left
                                  (ldb (byte half-bits 0)
                                       (%mix64 (logxor right (%splitmix64 state round)))))))
    (dpb left (byte half-bits half-bits) right)))

(defun %shuffled-index (position n half-bits state)
  "エポック内の位置 POSITION（0..N-1）を、STATE の置換で [0, N) のインデックスに写す。"
  (declare (type (unsigned-byte 64) position n state) (type (integer 1 32) half-bits))
  (let ((index (loop for x = (%feistel position half-bits state)
                       then (%feistel x half-bits state)
                     when (< x n) return x)))
    ;; 偶奇をならす互換。ラウンド鍵の次の SplitMix64 の出力の最下位ビットで決める。
    (if (and (< index 2) (logbitp 0 (%splitmix64 state (1+ +feistel-rounds+))))
        (- 1 index)
        index)))

;;; --- index-sampler ---

(defstruct (index-sampler (:constructor %make-index-sampler (length shuffle seed64 num-epochs))
                          (:copier nil) (:predicate nil))
  "通し番号の位置からレコードのインデックスを決める決定的なサンプラー。MAKE-INDEX-SAMPLER で作り、
SAMPLER-INDEX で引く。"
  (length 0 :type (unsigned-byte 64) :read-only t)
  (shuffle nil :read-only t)
  (seed64 nil :type (or null (unsigned-byte 64)) :read-only t)
  (num-epochs nil :type (or null (integer 0)) :read-only t))

(defun %seed64 (seed)
  "SEED（:u32 の (2) の配列。prng-key の形）の 2 語を [上位, 下位] として 64 ビットに詰める。"
  (unless (and (arrayp seed)
               (equal (array-element-type seed) '(unsigned-byte 32))
               (equal (array-dimensions seed) '(2)))
    (%sampler-error :bad-seed "シード ~S が :u32 の (2) の配列（prng-key の形）でない" seed))
  (logior (ash (aref seed 0) 32) (aref seed 1)))

(defun make-index-sampler (n &key shuffle seed num-epochs)
  "N 個のレコードのインデックス列を決めるサンプラーを作る。位置 k（0 から数える）は
エポック floor(k / N) のエポック内の位置 k mod N に当たる（SAMPLER-INDEX）。
SHUFFLE が nil なら各エポックは 0, 1, ..., N-1 の順。真なら SEED（NABLA:PRNG-KEY が作る :u32 の
(2) の配列。必須）とエポック番号で決まる置換の順で、同じ SEED・エポックからは常に同じ順になる
（SEED は作るときに読み、配列は持たない。SHUFFLE が nil でも渡せば形を検査するが、順序には
効かない）。NUM-EPOCHS が nil（既定）なら列は終わらず、整数なら N × NUM-EPOCHS 個の位置で終わる。N は 0 以上 2^64 未満の整数（0 なら引ける位置は無い）。
不正な引数は SAMPLER-ERROR。複数プロセスでの分割（シャード）は持たない。"
  (unless (typep n '(unsigned-byte 64))
    (%sampler-error :bad-length "レコードの個数 ~S が 0 以上 2^64 未満の整数でない" n))
  (unless (typep num-epochs '(or null (integer 0)))
    (%sampler-error :bad-num-epochs "num-epochs ~S が nil か 0 以上の整数でない" num-epochs))
  (when (and shuffle (null seed))
    (%sampler-error :bad-seed "shuffle にはシード（prng-key の形の配列）が要る"))
  (%make-index-sampler n (and shuffle t) (and seed (%seed64 seed)) num-epochs))

(defun sampler-index (sampler k)
  "SAMPLER の位置 K（0 から数える整数）の (values エポック インデックス) を返す純粋関数。
エポックは floor(K / N)、インデックスはそのエポックの順序の K mod N 番目のレコード（0..N-1）。
K が 0 以上 N × NUM-EPOCHS（NUM-EPOCHS が nil なら無限）未満の整数でなければ
:position-out-of-range の SAMPLER-ERROR。同じ K には常に同じ値を返すので、再開は位置だけで済む。"
  (let* ((n (index-sampler-length sampler))
         (num-epochs (index-sampler-num-epochs sampler))
         ;; 位置の上限（nil なら無限）。レコードが 0 個なら引ける位置は無い。
         (end (cond ((zerop n) 0)
                    (num-epochs (* n num-epochs)))))
    (unless (and (integerp k) (<= 0 k) (or (null end) (< k end)))
      (%sampler-error :position-out-of-range "位置 ~S は 0 以上~:[の~;~:* ~D 未満の~]整数でない"
                      k end))
    (multiple-value-bind (epoch position) (floor k n)
      (values epoch
              (if (and (index-sampler-shuffle sampler) (> n 1))
                  ;; 均衡 Feistel の片側のビット幅 h。2h は n-1 のビット長以上なので n ≤ 2^(2h) < 4n。
                  (%shuffled-index position n (ceiling (integer-length (1- n)) 2)
                                   (%epoch-state (index-sampler-seed64 sampler) epoch))
                  position)))))
