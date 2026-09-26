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

(test support/large-smoke/detects-that-large-ran
  "large スイートが実際に実行されたことを検知するためだけのテスト。
small / medium の既定実行では、このテストは走らないはず
（走ったらここで必ず失敗する）。"
  (fail "large スイートが実行された（NABLA_TEST_SIZES=large で意図的に実行した場合のみ、この失敗は正しい）。"))
