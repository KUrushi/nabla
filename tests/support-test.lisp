;;;; tests/support/ の共通部品そのものに対する PBT。
;;;;
;;;; ここが落ちるということは、他の全テストが信用できないということなので、
;;;; 小さく・厳密に確かめる。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %expected-element-type (dtype)
  "dtype-tolerance などと同じ dtype 一覧を、独立に書き下したもの。
element-type-for-dtype の実装と1対1で対応させない（実装をなぞるだけの
テストにしないため）。"
  (ecase dtype
    (:f32 'single-float)
    (:f64 'double-float)
    ((:bf16 :f16) '(unsigned-byte 16))))

(defun %coerce-to-double-nested (array shape)
  "ARRAY（SHAPE を持つ）の全要素を double-float に coerce した、
initial-contents に渡せる入れ子リストを作る（reference-* のテスト専用の
小さなヘルパー）。"
  (labels ((build (indices remaining-shape)
             (if (null remaining-shape)
                 (coerce (apply #'aref array (reverse indices)) 'double-float)
                 (loop for i below (first remaining-shape)
                       collect (build (cons i indices) (rest remaining-shape))))))
    (build nil shape)))

(test support/array-spec/within-bounds
  "array-spec の生成器は、rank 0..4、各次元 1..8、指定した dtype だけを出す。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64 :bf16 :f16)
                                        :max-rank 4
                                        :max-dim 8))
                (lambda (spec)
                  (let ((shape (array-spec-shape spec)))
                    (and (<= 0 (length shape) 4)
                         (every (lambda (d) (<= 1 d 8)) shape)
                         (member (array-spec-dtype spec) '(:f32 :f64 :bf16 :f16))
                         (= (array-spec-rank spec) (length shape)))))
                :regression-id support/array-spec/within-bounds
                :regression-file (regression-path "array-spec-within-bounds"))))

(test support/array-spec/respects-dtypes-argument
  "array-spec の :dtypes を絞ると、その中の dtype しか出さない。"
  (is (check-it (generator (array-spec :dtypes '(:f64)))
                (lambda (spec)
                  (eq (array-spec-dtype spec) :f64))
                :regression-id support/array-spec/respects-dtypes-argument
                :regression-file (regression-path "array-spec-respects-dtypes"))))

(test support/array-spec/respects-larger-than-ten-max-rank-and-max-dim
  "array-spec の生成器は :max-rank / :max-dim に check-it::*size*
（既定10）より大きい値を渡しても、実際にその範囲まで rank や次元が届く。

以前は rank・次元・dtype の選択に check-it 組み込みの int-generator を
そのまま使っていたため、:max-rank や :max-dim に10より大きい値を渡して
も、呼び出し側に何のエラーも出さないまま実際の上限が10で頭打ちになって
いた（.claude/skills/nabla-testing/references/properties.md 参照）。
既定値の :max-rank 4 / :max-dim 8 はどちらも10未満なので、この不具合は
既定値だけを使う他のテストには現れない。乱数のシードは固定して、
たまに失敗するテストにしない。"
  (let* ((*random-state* (sb-ext:seed-random-state 42))
         (spec-generator (generator (array-spec :max-rank 20 :max-dim 50)))
         (specs (loop repeat 500 collect (check-it:generate spec-generator))))
    (is (some (lambda (spec) (>= (array-spec-rank spec) 15)) specs)
        "rank が :max-rank 20 の近くまで届いていない")
    (is (some (lambda (spec) (some (lambda (d) (> d 40)) (array-spec-shape spec))) specs)
        "次元が :max-dim 50 の近くまで届いていない")))

(test support/make-random-array/matches-spec
  "make-random-array は spec の shape と、dtype に対応する element-type の配列を返す。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (and (equal (array-dimensions array) (array-spec-shape spec))
                         (subtypep (array-element-type array)
                                   (%expected-element-type (array-spec-dtype spec))))))
                :regression-id support/make-random-array/matches-spec
                :regression-file (regression-path "make-random-array-matches-spec"))))

(test support/make-random-array/deterministic-for-same-seed
  "同じ spec と同じ seed からは、常に同じ配列ができる。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (equalp (make-random-array spec :seed 7)
                          (make-random-array spec :seed 7)))
                :regression-id support/make-random-array/deterministic-for-same-seed
                :regression-file (regression-path "make-random-array-deterministic"))))

(test support/make-random-array/positive-domain-is-positive
  "domain が :positive のとき、f32 / f64 の要素はすべて正になる。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64)))
                (lambda (spec)
                  (let ((array (make-random-array spec :domain :positive)))
                    (loop for i below (array-total-size array)
                          always (plusp (row-major-aref array i)))))
                :regression-id support/make-random-array/positive-domain-is-positive
                :regression-file (regression-path "make-random-array-positive-domain"))))

(test support/make-random-array/positive-domain-stays-in-open-unit-interval
  "domain が :positive のときは、どの dtype でも要素が (0, 1] に入る
（DECODE-ELEMENT で DOUBLE-FLOAT に戻して確かめる）。

bf16 / f16 は仮数部が狭いので、生成した DOUBLE-FLOAT をそのまま
ビット列に切り詰めると、0 に近い値がちょうど 0 に丸まってしまうことが
ある（アンダーフロー）。SEED をいろいろな値にして試すことで、その
アンダーフローが起きる乱数列を広く探す。SEED には check-it 組み込みの
(integer lo hi) ではなく UNIFORM-INTEGER を使う。(integer lo hi) は
check-it::*size*（既定10）で値をクランプしてしまい、実際には 0..10 の
SEED しか試さない（.claude/skills/nabla-testing/references/properties.md
の「check-it の (integer lo hi) / (real lo hi) の落とし穴」参照）。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64 :bf16 :f16))
                                   (uniform-integer :lo 0 :hi 1000000)))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (array (make-random-array spec :seed seed :domain :positive)))
                      (loop for i below (array-total-size array)
                            for value = (decode-element dtype (row-major-aref array i))
                            always (and (> value 0.0d0) (<= value 1.0d0))))))
                :regression-id support/make-random-array/positive-domain-stays-in-open-unit-interval
                :regression-file (regression-path "make-random-array-positive-domain-open-unit-interval"))))

(test support/make-random-array/unit-domain-stays-in-closed-unit-interval
  "domain が :unit のときは、どの dtype でも要素が [0, 1] に入る
（DECODE-ELEMENT で DOUBLE-FLOAT に戻して確かめる）。SEED をいろいろな
値にして試す（UNIFORM-INTEGER を使う理由は
SUPPORT/MAKE-RANDOM-ARRAY/POSITIVE-DOMAIN-STAYS-IN-OPEN-UNIT-INTERVAL と
同じ）。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64 :bf16 :f16))
                                   (uniform-integer :lo 0 :hi 1000000)))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (array (make-random-array spec :seed seed :domain :unit)))
                      (loop for i below (array-total-size array)
                            for value = (decode-element dtype (row-major-aref array i))
                            always (and (>= value 0.0d0) (<= value 1.0d0))))))
                :regression-id support/make-random-array/unit-domain-stays-in-closed-unit-interval
                :regression-file (regression-path "make-random-array-unit-domain-closed-unit-interval"))))

(test support/decode-element/f16-subnormal-round-trips
  "decode-element の f16 は非正規化数（指数部のビットが全部0で仮数部が
非0）を、value = mantissa * 2^-24 として正しく戻す。

以前は非正規化数も正規化数と同じ式（暗黙の先頭1ビットがある前提）で
計算していたため、最小の非正規化数（ビット列 1）が本来の 2^-24
（≈5.96e-8）ではなく、桁違いに大きい 2^-15（≈3.05e-5）相当の値に
デコードされていた。make-random-array の :positive ドメインが 0 を
避けるためにこのビット列 1 へクランプするので、この不具合は
(0, 1] の契約そのものには影響しないが、DECODE-ELEMENT が返す値自体が
不正確だった。"
  (is (= (decode-element :f16 1) (expt 2.0d0 -24)))
  (is (= (decode-element :f16 2) (* 2.0d0 (expt 2.0d0 -24))))
  (is (= (decode-element :f16 #x03FF) (* 1023.0d0 (expt 2.0d0 -24)))))

(test support/decode-element/f16-subnormal-matches-formula
  "decode-element の f16 は、指数部のビットが全部0のとき（非正規化数）、
仮数のビット列を広く振っても value = mantissa * 2^-24 で戻ることを
check-it で確かめる。SUPPORT/DECODE-ELEMENT/F16-SUBNORMAL-ROUND-TRIPS は
2, 3個の固定値だけを見る例ベースのテストなので、ここでは仮数の
10bit 全域 (0..1023) を対象にする。

check-it 組み込みの (integer 0 1023) ではなく UNIFORM-INTEGER を使う。
(integer lo hi) は check-it::*size*（既定10）で値をクランプしてしまい、
実際には 0..10 の仮数しか試さない（過去にこのテストがそれに気づかず
仮数の 99% を見ないまま通っていた。詳しくは
.claude/skills/nabla-testing/references/properties.md 参照）。仮数の
定義域はちょうど 1024 個しかないので、
SUPPORT/DECODE-ELEMENT/F16-SUBNORMAL-EXHAUSTIVE で全数も確かめる。"
  (is (check-it (generator (uniform-integer :lo 0 :hi 1023))
                (lambda (mantissa)
                  (= (decode-element :f16 mantissa)
                     (* mantissa (expt 2.0d0 -24))))
                :regression-id support/decode-element/f16-subnormal-matches-formula
                :regression-file (regression-path "decode-element-f16-subnormal-matches-formula"))))

(test support/decode-element/f16-subnormal-exhaustive
  "非正規化数の仮数は 0..1023 の 1024 通りしかないので、check-it で
サンプリングするのではなく全数を確かめる（PBT がサンプリングで見逃す
範囲がないことを、例ベースのテストで保証する）。"
  (loop for mantissa from 0 to 1023
        do (is (= (decode-element :f16 mantissa)
                  (* mantissa (expt 2.0d0 -24)))
               "decode-element :f16 のビット列 ~A（非正規化数）が想定と違う" mantissa)))

(test support/decode-element/f16-zero-and-max-normal
  "decode-element の f16 は +0 / -0、および最大の正規化数（65504.0）も
正しく戻す（非正規化数の修正が、他の場合を壊していないことを確かめる）。

+0 と -0 は DOUBLE-FLOAT の = では区別できない（(= 0.0d0 -0.0d0) は真）
ので、符号を区別するために FLOAT-SIGN を使う。指数部のビットが全部1
（f16 の Inf / NaN）は decode-element では特別扱いしていない点に注意
（nabla のテストで使う値の絶対値はおおむね 1 以下で、Inf / NaN を
decode-element に渡すことは想定していない）。"
  (is (= (decode-element :f16 #x0000) 0.0d0))
  (is (= (float-sign (decode-element :f16 #x0000)) 1.0d0))
  (is (= (decode-element :f16 #x8000) -0.0d0))
  (is (= (float-sign (decode-element :f16 #x8000)) -1.0d0))
  (is (= (decode-element :f16 #x7BFF) 65504.0d0)))

(test support/uniform-integer/rejects-lo-greater-than-hi
  "MAKE-UNIFORM-INTEGER-GENERATOR / MAKE-UNIFORM-REAL-GENERATOR は、
LO が HI より大きいときに、分かりにくいエラー（RANDOM への負の引数
など）ではなく、その場で意味の分かるエラーを出す。"
  (signals error (make-uniform-integer-generator 10 5))
  (signals error (make-uniform-real-generator 10.0d0 5.0d0)))

(test support/uniform-integer/reaches-both-ends-of-a-wide-range
  "MAKE-UNIFORM-INTEGER-GENERATOR は check-it::*size*（既定10）を無視して
指定した範囲全体から一様に値を選ぶ。2000 回引いて、範囲の下のほう
（20 未満）と上のほう（1000 より大きい）の両方が出ることを確かめる
（check-it 組み込みの (integer 0 1023) はこの範囲を 0..10 にクランプ
してしまうため、その回帰を防ぐ）。乱数のシードは固定して、たまに
失敗するテストにしない。"
  (let* ((*random-state* (sb-ext:seed-random-state 42))
         (generator (make-uniform-integer-generator 0 1023))
         (draws (loop repeat 2000 collect (check-it:generate generator))))
    (is (every (lambda (v) (<= 0 v 1023)) draws))
    (is (some (lambda (v) (< v 20)) draws))
    (is (some (lambda (v) (> v 1000)) draws))))

(test support/uniform-real/reaches-both-ends-of-a-wide-range
  "MAKE-UNIFORM-REAL-GENERATOR も UNIFORM-INTEGER と同じく、
check-it::*size* を無視して [LO, HI) 全体から一様に値を選ぶ。"
  (let* ((*random-state* (sb-ext:seed-random-state 42))
         (generator (make-uniform-real-generator 0.0d0 1000.0d0))
         (draws (loop repeat 2000 collect (check-it:generate generator))))
    (is (every (lambda (v) (<= 0.0d0 v 1000.0d0)) draws))
    (is (some (lambda (v) (< v 20.0d0)) draws))
    (is (some (lambda (v) (> v 900.0d0)) draws))))

(test support/uniform-integer/shrinks-toward-zero-within-range
  "MAKE-UNIFORM-INTEGER-GENERATOR で作った生成器は、check-it 組み込みの
int-generator と同じく、失敗したとき [LO, HI] の範囲内で0に一番近い
反例まで縮小する。

以前は SHRINK が cached-value をそのまま返すだけで、縮小を一切しな
かった（区間全体を一様に選ぶことと、0 に向けて縮小できないことを
混同していた。check-it 組み込みの int-generator も一様に選ぶが、
ちゃんと0に向けて縮小する。generators.lisp の int-generator-function、
shrink.lisp の int-generator への SHRINK メソッド参照）。乱数のシード
は固定して、たまに失敗するテストにしない。"
  (let* ((*random-state* (sb-ext:seed-random-state 1))
         (generator (make-uniform-integer-generator 0 1023)))
    ;; CHECK-IT:SHRINK の TEST 引数は「性質が成り立つなら真」を返す関数。
    ;; ここでは (< x 5) を性質とするので、x < 5 なら真（性質が成り立つ）、
    ;; x >= 5 が反例（性質が破れる）になる。0 が範囲内なので、最小の
    ;; 反例はちょうど 5 になる。
    (loop until (>= (check-it:generate generator) 5))
    (is (= (check-it:shrink generator (lambda (x) (< x 5))) 5))))

(test support/uniform-integer/shrinks-toward-lo-when-zero-is-out-of-range
  "0 が [LO, HI] の範囲外のとき（LO > 0）は、SHRINK は0にではなく LO に
向けて縮小する（範囲外の値を候補にしないという制約を、SHRINK に渡す
TEST 関数でエンコードしている）。常に性質が破れる（TEST が常に偽を
返す）ケースで、縮小結果がちょうど LO になることを確かめる。"
  (let* ((*random-state* (sb-ext:seed-random-state 1))
         (generator (make-uniform-integer-generator 100 1000)))
    (check-it:generate generator)
    (is (= (check-it:shrink generator (constantly nil)) 100))))

(test support/array-spec/shrinks-dimensions-toward-the-minimal-failing-value
  "array-spec の生成器は、次元の選択に UNIFORM-INTEGER を使っているので、
失敗したときに、その次元も0（実際には :max-dim の下限である1）に向けて
縮小できる（UNIFORM-INTEGER の SHRINK 修正が、array-spec がそれを使う
経路 [chained-generator → mapped-generator → tuple-generator のサブジェ
ネレータ] でも効くことを確かめる）。

rank は :max-rank 1 に固定して、rank 自体の縮小（check-it の
chained-generator は事前に選んだ rank を再選択しないので、そもそも
縮小できない）とは切り離し、1次元目の次元だけに注目する。乱数のシード
は固定して、たまに失敗するテストにしない。"
  (let* ((*random-state* (sb-ext:seed-random-state 2))
         (spec-generator (generator (array-spec :max-rank 1 :max-dim 20))))
    (loop for spec = (check-it:generate spec-generator)
          until (and (= (array-spec-rank spec) 1)
                     (>= (first (array-spec-shape spec)) 10)))
    (let ((shrunk (check-it:shrink
                   spec-generator
                   (lambda (spec)
                     (not (and (= (array-spec-rank spec) 1)
                               (>= (first (array-spec-shape spec)) 10)))))))
      (is (= (array-spec-rank shrunk) 1))
      (is (= (first (array-spec-shape shrunk)) 10)))))

(test support/array-spec/prints-readably
  "check-it は失敗例を (format nil \"~S\" value) で保存し、regression
ファイルの LOAD 時に READ-FROM-STRING で読み戻す。array-spec% がこの
往復に耐えないと、最初に失敗した瞬間に regression ファイルが壊れ、
以後 nabla/tests のロードごと失敗するようになる（そのものずばりの
不具合が過去に起きたので、再発を防ぐために固定する）。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64 :bf16 :f16)))
                (lambda (spec)
                  (equalp spec (read-from-string (prin1-to-string spec))))
                :regression-id support/array-spec/prints-readably
                :regression-file (regression-path "array-spec-prints-readably"))))

(test support/reference-matmul/matches-matmul-fixture-documented-values
  "reference-matmul は matmul.mlir フィクスチャのコメントに書かれた期待値
（58 64 139 154）を再現する（フィクスチャの期待値は、コメントとして
明記されているぶんには例ベースのテストで確かめてよい、というスキルの
例外に当たる）。"
  (let ((a (make-array '(2 3) :element-type 'double-float
                        :initial-contents '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))))
        (b (make-array '(3 2) :element-type 'double-float
                        :initial-contents '((7.0d0 8.0d0) (9.0d0 10.0d0) (11.0d0 12.0d0)))))
    (is (equalp (reference-matmul a b)
                (make-array '(2 2) :element-type 'double-float
                            :initial-contents '((58.0d0 64.0d0) (139.0d0 154.0d0)))))))

(test support/reference-add/commutative-and-zero-identity
  "reference-add は可換で、0 を足しても変わらない（x + 0 = x）。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64)))
                (lambda (spec)
                  (let* ((x (make-random-array spec))
                         (y (make-random-array spec :seed 99))
                         (zero (make-array (array-spec-shape spec)
                                           :element-type (%expected-element-type (array-spec-dtype spec))
                                           :initial-element (coerce 0 (%expected-element-type (array-spec-dtype spec))))))
                    (and (equalp (reference-add x y) (reference-add y x))
                         (equalp (reference-add x zero)
                                 (make-array (array-spec-shape spec) :element-type 'double-float
                                             :initial-contents
                                             (%coerce-to-double-nested x (array-spec-shape spec)))))))
                :regression-id support/reference-add/commutative-and-zero-identity
                :regression-file (regression-path "reference-add-commutative-and-zero-identity"))))

(test support/reference-reduce-sum/ones-array-equals-dimension
  "全要素が1の配列を、どの axis に沿って総和しても、その axis の次元数に
等しい（rank 0 の配列には axis が無いので対象外）。"
  (is (check-it (generator (array-spec :dtypes '(:f32 :f64)))
                (lambda (spec)
                  (let ((shape (array-spec-shape spec)))
                    (or (zerop (length shape))
                        (let ((ones (make-array shape :element-type (%expected-element-type (array-spec-dtype spec))
                                                :initial-element (coerce 1 (%expected-element-type (array-spec-dtype spec))))))
                          (loop for axis below (length shape)
                                for result = (reference-reduce-sum ones axis)
                                always (loop for i below (array-total-size result)
                                             always (= (row-major-aref result i)
                                                       (coerce (nth axis shape) 'double-float))))))))
                :regression-id support/reference-reduce-sum/ones-array-equals-dimension
                :regression-file (regression-path "reference-reduce-sum-ones-array-equals-dimension"))))

(test support/reference-matmul/identity-is-identity
  "単位行列を掛けても値が変わらない（A @ I = A）。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 6) (uniform-integer :lo 1 :hi 6)))
                (lambda (dims)
                  (destructuring-bind (m n) dims
                    (let* ((a (make-random-array (make-array-spec (list m n) :f64)))
                           (identity (make-array (list n n) :element-type 'double-float :initial-element 0.0d0)))
                      (dotimes (i n) (setf (aref identity i i) 1.0d0))
                      (allclose (reference-matmul a identity) a :dtype :f64))))
                :regression-id support/reference-matmul/identity-is-identity
                :regression-file (regression-path "reference-matmul-identity-is-identity"))))

(test support/allclose/reflexive
  "allclose は、同じ配列どうしなら常に真になる。"
  (is (check-it (generator (array-spec))
                (lambda (spec)
                  (let ((array (make-random-array spec)))
                    (allclose array array :dtype (array-spec-dtype spec))))
                :regression-id support/allclose/reflexive
                :regression-file (regression-path "allclose-reflexive"))))

(test support/allclose/boundary
  "許容誤差 atol + rtol*|e| のちょうど内側は通り、わずかに外側は落ちる。"
  ;; check-it の (real lo hi) は lo と hi が同符号（0 <= lo < hi）だと
  ;; 常に幅0の範囲になり (random 0.0) で落ちるバグがある
  ;; (check-it b79c9103665be3976915b56b570038f03486e62f の
  ;; real-generator-function 参照)。rtol / atol は (real 0 1) で作り、
  ;; 0 になっても境界の幅が消えないよう定数を足す。
  (is (check-it (generator (tuple (real -100 100) (real 0 1) (real 0 1)))
                (lambda (args)
                  (destructuring-bind (e raw-rtol raw-atol) args
                    (let* ((e (coerce e 'double-float))
                           (rtol (+ 1d-6 (coerce raw-rtol 'double-float)))
                           (atol (+ 1d-6 (coerce raw-atol 'double-float)))
                           (bound (+ atol (* rtol (abs e))))
                           (eps 1d-3)
                           (passing (+ e (* bound (- 1 eps))))
                           (failing (+ e (* bound (+ 1 eps)))))
                      (and (approx= passing e :rtol rtol :atol atol)
                           (not (approx= failing e :rtol rtol :atol atol))))))
                :regression-id support/allclose/boundary
                :regression-file (regression-path "allclose-boundary"))))

(defun %delete-if-exists (path)
  (when (probe-file path)
    (delete-file path)))

(test support/regression-path/defaults-to-nabla-tests-header-regardless-of-caller-package
  "REGRESSION-PATH は、呼び出し時の *PACKAGE* によらず、新規作成する
ファイルの1行目を既定で (in-package #:nabla.tests) にする。

scripts/run-tests.sh は sbcl --eval で複数のシステムを load-system した
あとに fiveam:run! を呼ぶので、テスト実行時の *PACKAGE* は必ずしも
NABLA.TESTS ではない（過去に COMMON-LISP-USER のまま呼ばれ、check-it が
読み込めないヘッダを書いてしまったことがある）。ここでは意図的に
*PACKAGE* を COMMON-LISP-USER にして呼び、それでもヘッダが変わらない
ことを固定する。"
  (let ((path (asdf:system-relative-pathname
               "nabla" "tests/regressions/support-regression-path-default-package-test.lisp")))
    (unwind-protect
        (progn
          (%delete-if-exists path)
          (let ((*package* (find-package "COMMON-LISP-USER")))
            (regression-path "support-regression-path-default-package-test"))
          (is (string= "(in-package #:nabla.tests)"
                       (with-open-file (stream path) (read-line stream)))))
      (%delete-if-exists path))))

(test support/regression-path/package-keyword-overrides-default
  "REGRESSION-PATH の :PACKAGE キーワードで、既定の NABLA.TESTS 以外の
in-package 先を明示的に指定できる。"
  (let ((path (asdf:system-relative-pathname
               "nabla" "tests/regressions/support-regression-path-package-keyword-test.lisp")))
    (unwind-protect
        (progn
          (%delete-if-exists path)
          (regression-path "support-regression-path-package-keyword-test"
                            :package "NABLA.TESTS.SUPPORT")
          (is (string= "(in-package #:nabla.tests.support)"
                       (with-open-file (stream path) (read-line stream)))))
      (%delete-if-exists path))))

(defun %quiet-nan-double ()
  "ビット列から直接 double-float の NaN を作る。
(/ 0d0 0d0) はコンパイル時の定数畳み込みで浮動小数点例外になるため使わない。"
  (sb-kernel:make-double-float #x7FF80000 0))

(test support/approx=/nan-never-equal
  "NaN を含む比較は、境界の値によらず常に不一致になる。"
  (let ((nan (%quiet-nan-double)))
    (is (not (approx= nan 1.0d0 :dtype :f64)))
    (is (not (approx= 1.0d0 nan :dtype :f64)))
    (is (not (approx= nan nan :dtype :f64)))))

(in-suite :nabla.large)

(test support/large-smoke/large-suite-runs
  "large スイートが実行できることだけを確かめるマーカーテスト。
NABLA_TEST_SIZES=large scripts/run-tests.sh が常に失敗する（常に fail する
テストがコミットされていた）不具合の再発を防ぐため、必ず通るようにする。"
  (pass))

(in-suite :nabla.small)

(test support/sizes-from-env/defaults-to-small-and-medium
  "NABLA_TEST_SIZES が未設定（NIL）のときは small と medium だけを既定にする。
large は手動または定期実行のみで、既定のスイートには含めない。"
  (is (equal (sizes-from-env nil) '(:small :medium))))

(test support/sizes-from-env/parses-comma-separated-list
  "NABLA_TEST_SIZES はカンマ区切りの文字列を対応するキーワードのリストにする。
空白を含んでいても、大文字小文字が違っても解釈できる。"
  (is (equal (sizes-from-env "small") '(:small)))
  (is (equal (sizes-from-env "small, MEDIUM ,large") '(:small :medium :large))))

(test support/sizes-from-env/rejects-unknown-size
  "small / medium / large 以外の名前はエラーにする（黙って無視しない）。"
  (signals error (sizes-from-env "huge")))

(test support/sizes-from-env/blank-or-trailing-comma-does-not-signal
  "NABLA_TEST_SIZES が空文字列のときは既定値にフォールバックし、末尾や
連続するカンマが作る空の要素は無視する。どちらも未処理のコンディションで
scripts/run-tests.sh を落としてはいけない。"
  (is (equal (sizes-from-env "") '(:small :medium)))
  (is (equal (sizes-from-env "small,") '(:small)))
  (is (equal (sizes-from-env ",small,,medium,") '(:small :medium))))

;;; run-tests が :nabla.large を既定で触らないことを確かめる統合テスト。
;;; sizes-from-env の既定値が :nabla.large を選ばないことは、内部関数
;;; %size-suite を直接呼ぶのではなく、下の
;;; support/run-tests/default-sizes-do-not-run-large が公開 API
;;; (run-tests) 経由で確かめる。
;;; 検証本体は :nabla.large の下に置く: run-tests (:sizes '(:small :medium))
;;; を、実行中の :nabla.small / :nabla.medium 自身から呼ぶと、実行中のスイート
;;; を fiveam:run! で再度実行することになり無限再帰に陥るため、それらとは
;;; 独立な :nabla.large から検証する。

(defparameter *large-selection-guard-ran-p* nil)

(def-suite %large-selection-guard :in :nabla.large)
(in-suite %large-selection-guard)

(def-test %large-selection-guard/marks-ran ()
  (setf *large-selection-guard-ran-p* t)
  (pass))

(in-suite :nabla.large)

(test support/run-tests/default-sizes-do-not-run-large
  "run-tests に既定値 '(:small :medium) を渡すと、:nabla.large 配下の
テスト（%large-selection-guard）は実行されない。"
  (setf *large-selection-guard-ran-p* nil)
  (run-tests :sizes '(:small :medium))
  (is (null *large-selection-guard-ran-p*)
      "run-tests に :sizes '(:small :medium) を渡したのに :nabla.large 配下のテストが実行された"))

(test support/run-tests/guard-test-itself-can-run
  "対照実験: %large-selection-guard/marks-ran を直接実行すれば
*large-selection-guard-ran-p* は t になる（上のテストが「たまたま
ガードのテストが動かないだけ」で通っているのではないことを確かめる）。
run-tests :sizes '(:large) 経由で呼ぶと :nabla.large（このテスト自身が
属するスイート）を再度実行することになり無限再帰になるため、
fiveam:run で当該テストだけを名指しで呼ぶ。"
  (setf *large-selection-guard-ran-p* nil)
  (fiveam:run '%large-selection-guard/marks-ran)
  (is (eq *large-selection-guard-ran-p* t)))

(in-suite :nabla.small)
