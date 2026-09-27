;;;; bf16 / f16 のビット列 <-> single-float 変換の性質（issue #38）。
;;;;
;;;; nb::decode-float16 / nb::encode-float16 とその配列版は export しない
;;;; 内部関数なので、テストは nb:: で直接呼ぶ。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %f16-max-finite-bits ()
  ;; f16 の最大有限値 65504 のビット列（符号なし）。
  #x7BFF)

(defun %bf16-max-finite-bits ()
  ;; bf16 の最大有限値のビット列（符号なし。指数全1未満、仮数全1）。
  #x7F7F)

(defun %oracle-floor-log2 (r)
  "正の有理数 R について 2^E <= R < 2^(E+1) となる整数 E を求める。
nb::%floor-log2（分子・分母のビット長から求める）とは違う道筋（対数
関数による見積もり）から出発し、最後は有理数のまま正確な比較で補正する
ので、%floor-log2 に同じバグがあっても道連れにならない。"
  (let ((e (floor (log (float r 1.0d0) 2))))
    (loop while (< r (expt 2 e)) do (decf e))
    (loop while (>= r (expt 2 (1+ e))) do (incf e))
    e))

(defun %rational-oracle-encode (x dtype)
  "X (SINGLE-FLOAT, 有限) を、有理数演算だけを使ったオラクルで DTYPE の
ビット列に変換する。nb::encode-float16 とは独立な実装で、最近接偶数丸め
（RNE）を確かめる（.claude/skills/nabla-testing の性質2）。ビット演算
（nb::%single-float-bits / nb::%floor-log2）を経由せず、CL の
FLOAT-SIGN と %ORACLE-FLOOR-LOG2 だけで組み立てる。"
  (let* ((mantissa-bits (ecase dtype (:f16 10) (:bf16 7)))
         (exponent-bits (ecase dtype (:f16 5) (:bf16 8)))
         (exponent-bias (ecase dtype (:f16 15) (:bf16 127)))
         (width (+ mantissa-bits exponent-bits))
         (max-biased-exp (- (ash 1 exponent-bits) 2))
         (sign (if (minusp (float-sign x)) 1 0))
         (r (rational (abs x))))
    (if (zerop r)
        (ash sign width)
        (let* ((e (max (- 1 exponent-bias) (%oracle-floor-log2 r)))
               (shift (- mantissa-bits e))
               (q (if (>= shift 0)
                      (round (* r (expt 2 shift)))
                      (round r (expt 2 (- shift))))))
          (when (>= q (ash 1 (1+ mantissa-bits)))
            (setf q (ash q -1))
            (incf e))
          (let* ((subnormal-p (< q (ash 1 mantissa-bits)))
                 (biased-exp (if subnormal-p 0 (+ e exponent-bias)))
                 (mantissa (if subnormal-p q (- q (ash 1 mantissa-bits)))))
            (if (> biased-exp max-biased-exp)
                (logior (ash sign width) (ash (1+ max-biased-exp) mantissa-bits))
                (logior (ash sign width) (ash biased-exp mantissa-bits) mantissa)))))))

;;; 1. 往復: NaN 以外の全ビット列で decode . encode = id。

(test float16/f16/round-trips-over-all-non-nan-bits
  "f16 の 65536 ビット列すべてで、NaN 以外は encode(decode(bits)) = bits。
不一致は個別に is を出さず、一致しなかったビット列だけを集めて一度に
報告する（失敗時に分かる情報は同じで、FiveAM の記録件数を抑える）。"
  (let ((mismatches
          (loop for bits from 0 below 65536
                for exp16 = (ldb (byte 5 10) bits)
                for mant16 = (ldb (byte 10 0) bits)
                unless (or (and (= exp16 #x1F) (/= mant16 0))
                           (= bits (nb::encode-float16 (nb::decode-float16 bits :f16) :f16)))
                  collect bits)))
    (is (null mismatches) "f16 bits=~{~4,'0X~^ ~}" mismatches)))

(test float16/bf16/round-trips-over-all-non-nan-bits
  "bf16 の 65536 ビット列すべてで、NaN 以外は encode(decode(bits)) = bits。"
  (let ((mismatches
          (loop for bits from 0 below 65536
                for exp16 = (ldb (byte 8 7) bits)
                for mant16 = (ldb (byte 7 0) bits)
                unless (or (and (= exp16 #xFF) (/= mant16 0))
                           (= bits (nb::encode-float16 (nb::decode-float16 bits :bf16) :bf16)))
                  collect bits)))
    (is (null mismatches) "bf16 bits=~{~4,'0X~^ ~}" mismatches)))

(test float16/f16/nan-decodes-to-nan-and-encodes-back-to-canonical-nan
  "f16 の NaN ビット列は decode すると NaN になり、その NaN を encode すると
符号を保った canonical quiet NaN (#x7E00 / #xFE00) に戻る。"
  (loop for bits from 0 below 65536
        for exp16 = (ldb (byte 5 10) bits)
        for mant16 = (ldb (byte 10 0) bits)
        for sign = (ldb (byte 1 15) bits)
        when (and (= exp16 #x1F) (/= mant16 0))
          do (is (sb-ext:float-nan-p (nb::decode-float16 bits :f16)))
             (is (= (logior (ash sign 15) #x7E00)
                    (nb::encode-float16 (nb::decode-float16 bits :f16) :f16)))))

(test float16/bf16/nan-decodes-to-nan-and-encodes-back-to-canonical-nan
  "bf16 の NaN ビット列は decode すると NaN になり、その NaN を encode すると
符号を保った canonical quiet NaN (#x7FC0 / #xFFC0) に戻る。"
  (loop for bits from 0 below 65536
        for exp16 = (ldb (byte 8 7) bits)
        for mant16 = (ldb (byte 7 0) bits)
        for sign = (ldb (byte 1 15) bits)
        when (and (= exp16 #xFF) (/= mant16 0))
          do (is (sb-ext:float-nan-p (nb::decode-float16 bits :bf16)))
             (is (= (logior (ash sign 15) #x7FC0)
                    (nb::encode-float16 (nb::decode-float16 bits :bf16) :bf16)))))

;;; 2. RNE: 有理数オラクルとの一致。

(test float16/f16/matches-rational-oracle-for-unit-interval-values
  "[-1, 1) の SINGLE-FLOAT について、encode-float16 (:f16) が有理数オラクルと一致する。"
  (is (check-it (generator (uniform-real :lo -1.0d0 :hi 1.0d0))
                (lambda (x)
                  (let ((sx (coerce x 'single-float)))
                    (= (nb::encode-float16 sx :f16)
                       (%rational-oracle-encode sx :f16))))
                :regression-id float16/f16/matches-rational-oracle-for-unit-interval-values
                :regression-file (regression-path "float16-f16-oracle-unit-interval"))))

(defun %encode-matches-oracle-or-is-inf-p (x dtype)
  "X が有限なら nb::encode-float16 とオラクルが一致することを、X が
無限大なら（オラクルは有理数を要求するので扱えない）nb::encode-float16
の結果が符号を保った無限大にデコードし直せることを確かめる。X が NaN
なら常に真（NaN は性質1で別に検査する）。"
  (cond
    ((sb-ext:float-nan-p x) t)
    ((sb-ext:float-infinity-p x)
     (let ((decoded (nb::decode-float16 (nb::encode-float16 x dtype) dtype)))
       (and (sb-ext:float-infinity-p decoded)
            (eq (plusp x) (plusp decoded)))))
    (t (= (nb::encode-float16 x dtype) (%rational-oracle-encode x dtype)))))

(test float16/f16/matches-rational-oracle-for-arbitrary-bit-patterns
  "任意の32bitパターンから作った SINGLE-FLOAT（非正規化数・巨大値・inf を
含む。NaN は除く）について、encode-float16 (:f16) が有理数オラクルと一致する。"
  (is (check-it (generator (uniform-integer :lo 0 :hi #xFFFFFFFF))
                (lambda (u32)
                  (%encode-matches-oracle-or-is-inf-p (nb::%make-single-float u32) :f16))
                :regression-id float16/f16/matches-rational-oracle-for-arbitrary-bit-patterns
                :regression-file (regression-path "float16-f16-oracle-arbitrary-bits"))))

(test float16/bf16/matches-rational-oracle-for-unit-interval-values
  "[-1, 1) の SINGLE-FLOAT について、encode-float16 (:bf16) が有理数オラクルと一致する。"
  (is (check-it (generator (uniform-real :lo -1.0d0 :hi 1.0d0))
                (lambda (x)
                  (let ((sx (coerce x 'single-float)))
                    (= (nb::encode-float16 sx :bf16)
                       (%rational-oracle-encode sx :bf16))))
                :regression-id float16/bf16/matches-rational-oracle-for-unit-interval-values
                :regression-file (regression-path "float16-bf16-oracle-unit-interval"))))

(test float16/bf16/matches-rational-oracle-for-arbitrary-bit-patterns
  "任意の32bitパターンから作った SINGLE-FLOAT（非正規化数・巨大値・inf を
含む。NaN は除く）について、encode-float16 (:bf16) が有理数オラクルと一致する。"
  (is (check-it (generator (uniform-integer :lo 0 :hi #xFFFFFFFF))
                (lambda (u32)
                  (%encode-matches-oracle-or-is-inf-p (nb::%make-single-float u32) :bf16))
                :regression-id float16/bf16/matches-rational-oracle-for-arbitrary-bit-patterns
                :regression-file (regression-path "float16-bf16-oracle-arbitrary-bits"))))

;;; 3. 「隣接2値のうち近い方、等距離なら偶数」の直接検査（オラクル2とは
;;;    別の実装で二重に効かせる）。

(defun %adjacent-bits (bits dtype)
  "BITS と同じ符号側で隣り合うビット列（1つ小さい／大きい）のうち、
DTYPE の範囲内にあるものだけを返す。"
  (let* ((sign-mask (ash 1 15))
         (sign (logand bits sign-mask))
         (magnitude (logandc2 bits sign-mask))
         (max-magnitude (ecase dtype (:f16 #x7BFF) (:bf16 #x7F7F))))
    (remove nil
            (list (when (plusp magnitude) (logior sign (1- magnitude)))
                  (when (< magnitude max-magnitude) (logior sign (1+ magnitude)))))))

(defun %nearest-with-ties-to-even-p (x dtype)
  "ENCODE-FLOAT16 (X, DTYPE) の結果が、真値 X に対して隣（同符号側）の
ビット列よりも近く、等距離なら最下位ビットが0であることを確かめる。X が
無限大に丸められる場合（オーバーフロー）は、隣との「距離」が有理数で
測れない（無限大までの距離は無限大）ので、この直接検査の対象外にし、
性質2（有理数オラクル）と性質5（符号付き無限大の直接確認）に委ねる。"
  (or (sb-ext:float-nan-p x)
      (sb-ext:float-infinity-p x)
      (let* ((b (nb::encode-float16 x dtype))
             (decoded (nb::decode-float16 b dtype)))
        (or (sb-ext:float-infinity-p decoded)
            (let ((d (abs (- (rational x) (rational decoded)))))
              (every (lambda (nb-bits)
                       (let ((nb-decoded (nb::decode-float16 nb-bits dtype)))
                         (or (sb-ext:float-infinity-p nb-decoded)
                             (let ((dn (abs (- (rational x) (rational nb-decoded)))))
                               (or (< d dn)
                                   (and (= d dn) (zerop (logand b 1))))))))
                     (%adjacent-bits b dtype)))))))

(test float16/f16/nearest-value-with-ties-to-even
  "エンコード結果の16bit値は、隣（同符号側）よりも真値に近く、等距離な
ら最下位ビットが0になる（最近接偶数丸めの直接検査）。"
  (is (check-it (generator (uniform-integer :lo 0 :hi #xFFFFFFFF))
                (lambda (u32)
                  (%nearest-with-ties-to-even-p (nb::%make-single-float u32) :f16))
                :regression-id float16/f16/nearest-value-with-ties-to-even
                :regression-file (regression-path "float16-f16-nearest-ties-to-even"))))

(test float16/bf16/nearest-value-with-ties-to-even
  "bf16 版の同じ検査。"
  (is (check-it (generator (uniform-integer :lo 0 :hi #xFFFFFFFF))
                (lambda (u32)
                  (%nearest-with-ties-to-even-p (nb::%make-single-float u32) :bf16))
                :regression-id float16/bf16/nearest-value-with-ties-to-even
                :regression-file (regression-path "float16-bf16-nearest-ties-to-even"))))

;;; 3.5. ちょうど等距離な入力を直接構成して、偶数側に丸められることを
;;;      確かめる。性質3（%nearest-with-ties-to-even-p）はランダムな32bit
;;;      パターンから等距離な入力に当たることをほぼ期待できない（f16 は
;;;      仮数下位13bit、bf16 は下位16bitが特定のパターンになる確率が
;;;      2^-13 / 2^-16 しかない）ので、生成器に均等に探させるのではなく、
;;;      隣接するビット列 BITS16 と BITS16+1 のちょうど中間の値を明示的に
;;;      作る（decode-float16 はビット列ごとの往復・既存デコーダとの一致
;;;      で既に確かめてあるので、その値をオラクルとして使ってよい）。

(defun %tie-input (bits16 dtype)
  "BITS16 と (1+ BITS16) が表す DTYPE の隣接する2値のちょうど中間の
SINGLE-FLOAT を返す。両方とも同じ符号側で有限であることを呼び出し側が
保証する。中間値は元の2値より仮数が高々1bit深いだけなので、
SINGLE-FLOAT の24bit精度に対して常に正確に表現できる。"
  (let ((v (rational (nb::decode-float16 bits16 dtype)))
        (v2 (rational (nb::decode-float16 (1+ bits16) dtype))))
    (coerce (/ (+ v v2) 2) 'single-float)))

(defun %expected-tie-bits (bits16)
  "BITS16 と (1+ BITS16) のちょうど中間の値を最近接偶数丸めしたときの
ビット列。どちらか偶数の方（指数部はまたがらないので、BITS16 が偶数なら
そのまま、奇数なら (1+ BITS16) が偶数）。"
  (if (evenp bits16) bits16 (1+ bits16)))

(defun %exact-tie-picks-even-neighbour-p (magnitude sign dtype)
  "符号 SIGN・絶対値ビット MAGNITUDE から作った DTYPE のビット列と、その
1つ大きい絶対値のビット列とのちょうど中間の値を encode-float16 に通すと、
偶数側のビット列に丸められることを確かめる。"
  (let* ((bits16 (logior (ash sign 15) magnitude))
         (x (%tie-input bits16 dtype)))
    (= (nb::encode-float16 x dtype)
       (logior (ash sign 15) (%expected-tie-bits magnitude)))))

(test float16/f16/exact-midpoint-rounds-to-even-neighbour
  "f16: 隣接する2値のちょうど中間の値を明示的に構成し、等距離のときは
偶数側のビット列に丸められることを確かめる（.claude/skills/nabla-testing
の性質2、ties-to-even を直接構成した入力で検査する）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi (1- (%f16-max-finite-bits)))
                                   (uniform-integer :lo 0 :hi 1)))
                (lambda (args)
                  (destructuring-bind (magnitude sign) args
                    (%exact-tie-picks-even-neighbour-p magnitude sign :f16)))
                :regression-id float16/f16/exact-midpoint-rounds-to-even-neighbour
                :regression-file (regression-path "float16-f16-exact-midpoint-ties-to-even"))))

(test float16/bf16/exact-midpoint-rounds-to-even-neighbour
  "bf16 版の同じ検査（正規化数・非正規化数の両方を含む）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi (1- (%bf16-max-finite-bits)))
                                   (uniform-integer :lo 0 :hi 1)))
                (lambda (args)
                  (destructuring-bind (magnitude sign) args
                    (%exact-tie-picks-even-neighbour-p magnitude sign :bf16)))
                :regression-id float16/bf16/exact-midpoint-rounds-to-even-neighbour
                :regression-file (regression-path "float16-bf16-exact-midpoint-ties-to-even"))))

;;; 4. 交差確認: 既存の（切り捨ての）デコーダ tests/support/random-array.lisp
;;;    と一致するか。既存デコーダは exp16 が全1（無限大・NaN 用のビット
;;;    パターン）を正しく扱わない（無限大を有限の巨大な値にデコードして
;;;    しまう）ので、その範囲は比較から除く。bf16 は f32 と指数の幅が
;;;    同じ単純なビット列の切り出しなので、その範囲でも一致する。

(test float16/f16/matches-existing-decoder-outside-inf-nan-range
  "f16 の全ビット列のうち、指数が全1でない範囲（有限）で、nb::decode-float16
が既存の nabla.tests.support::%f16-bits->f32 と一致する。"
  (let ((mismatches
          (loop for bits from 0 below 65536
                for exp16 = (ldb (byte 5 10) bits)
                unless (or (= exp16 #x1F)
                           (= (nb::decode-float16 bits :f16)
                              (nabla.tests.support::%f16-bits->f32 bits)))
                  collect bits)))
    (is (null mismatches) "f16 bits=~{~4,'0X~^ ~}" mismatches)))

(test float16/bf16/matches-existing-decoder-for-all-non-nan-bits
  "bf16 は全ビット列（NaN 以外）で nb::decode-float16 が既存の
nabla.tests.support::%bf16-bits->f32 と一致する（両方とも単純なビット
シフトで、常に正確）。"
  (let ((mismatches
          (loop for bits from 0 below 65536
                for mant8 = (ldb (byte 8 0) bits)
                for exp8 = (ldb (byte 8 7) bits)
                unless (or (and (= exp8 #xFF) (/= mant8 0))
                           (= (nb::decode-float16 bits :bf16)
                              (nabla.tests.support::%bf16-bits->f32 bits)))
                  collect bits)))
    (is (null mismatches) "bf16 bits=~{~4,'0X~^ ~}" mismatches)))

;;; 5. 符号付きゼロ・無限大。

(test float16/signed-zero-and-infinity
  "符号付きゼロと無限大の変換。"
  (is (= #x8000 (nb::encode-float16 -0.0f0 :bf16)))
  (is (= #x8000 (nb::encode-float16 -0.0f0 :f16)))
  (is (minusp (float-sign (nb::decode-float16 #x8000 :bf16))))
  (is (minusp (float-sign (nb::decode-float16 #x8000 :f16))))
  (is (= #x7F80 (nb::encode-float16 sb-ext:single-float-positive-infinity :bf16)))
  (is (= #x7C00 (nb::encode-float16 sb-ext:single-float-positive-infinity :f16)))
  (is (= #xFF80 (nb::encode-float16 sb-ext:single-float-negative-infinity :bf16)))
  (is (= #xFC00 (nb::encode-float16 sb-ext:single-float-negative-infinity :f16)))
  (is (sb-ext:float-infinity-p (nb::decode-float16 #x7F80 :bf16)))
  (is (sb-ext:float-infinity-p (nb::decode-float16 #x7C00 :f16)))
  ;; オーバーフロー: 最大有限値より大きい値は inf に丸められる。
  (is (= #x7C00 (nb::encode-float16 100000.0f0 :f16)))
  (is (= #x7F80 (nb::encode-float16 most-positive-single-float :bf16))))

;;; 6. 配列版。

(test float16/array/decode-and-encode-round-trip
  "ランダムな bf16 / f16 配列で、decode-float16-array の shape と各要素が
decode-float16 と一致し、encode-float16-array で元のビット列に戻る。"
  (is (check-it (generator (array-spec :dtypes '(:bf16 :f16)))
                (lambda (spec)
                  (let* ((array (make-random-array spec))
                         (dtype (array-spec-dtype spec))
                         (decoded (nb::decode-float16-array array dtype))
                         (encoded (nb::encode-float16-array decoded dtype)))
                    (and (equal (array-dimensions decoded) (array-dimensions array))
                         (equal (array-element-type decoded) (upgraded-array-element-type 'single-float))
                         (loop for i below (array-total-size array)
                               always (= (row-major-aref decoded i)
                                         (nb::decode-float16 (row-major-aref array i) dtype)))
                         (equalp encoded array))))
                :regression-id float16/array/decode-and-encode-round-trip
                :regression-file (regression-path "float16-array-decode-encode-round-trip"))))

;;; 7. JAX との数値一致フィクスチャ（例ベース。tests/fixtures/float16/generate.py
;;;    で生成した jax-cross-check.lisp を JAX 側のオラクルとして使う）。

(defun %load-jax-cross-check-fixture ()
  "コミット済みのフィクスチャファイルを読む。READ に既定のリーダ状態を
使うと `#.` などでコードが実行されうるので、フェーズ1 wave 1 契約が
read-graph（issue #29、u1b）に課している約束と同じく
with-standard-io-syntax + *read-eval* nil で読む。"
  (with-standard-io-syntax
    (let ((*read-eval* nil))
      (with-open-file (stream (asdf:system-relative-pathname
                                "nabla" "tests/fixtures/float16/jax-cross-check.lisp"))
        (read stream)))))

(test float16/jax-cross-check/matches-jax-bfloat16-and-float16-conversion
  "tests/fixtures/float16/generate.py が JAX (jnp.asarray(...).view(uint16))
で計算した bf16 / f16 のビット列と、nb::encode-float16 の結果が一致する。"
  (dolist (case (%load-jax-cross-check-fixture))
    (destructuring-bind (bits32 expected-bf16-bits expected-f16-bits) case
      (let ((x (nb::%make-single-float bits32)))
        (is (= expected-bf16-bits (nb::encode-float16 x :bf16))
            "bits32=~D" bits32)
        (is (= expected-f16-bits (nb::encode-float16 x :f16))
            "bits32=~D" bits32)))))
