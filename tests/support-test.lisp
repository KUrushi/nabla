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

(test support/make-random-array/positive-domain-stays-in-open-unit-interval
  "domain が :positive のときは、どの dtype でも要素が (0, 1] に入る
（DECODE-ELEMENT で DOUBLE-FLOAT に戻して確かめる）。

bf16 / f16 は仮数部が狭いので、生成した DOUBLE-FLOAT をそのまま
ビット列に切り詰めると、0 に近い値がちょうど 0 に丸まってしまうことが
ある（アンダーフロー）。SEED をいろいろな値にして試すことで、その
アンダーフローが起きる乱数列を広く探す。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64 :bf16 :f16))
                                   (integer 0 1000000)))
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
値にして試す。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64 :bf16 :f16))
                                   (integer 0 1000000)))
                (lambda (args)
                  (destructuring-bind (spec seed) args
                    (let* ((dtype (array-spec-dtype spec))
                           (array (make-random-array spec :seed seed :domain :unit)))
                      (loop for i below (array-total-size array)
                            for value = (decode-element dtype (row-major-aref array i))
                            always (and (>= value 0.0d0) (<= value 1.0d0))))))
                :regression-id support/make-random-array/unit-domain-stays-in-closed-unit-interval
                :regression-file (regression-path "make-random-array-unit-domain-closed-unit-interval"))))

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
