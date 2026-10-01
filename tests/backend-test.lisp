;;;; backend プロトコルの性質（issue #9）。
;;;;
;;;; フェイク backend（tests/support/fake-backend.lisp）を使い、IREE を
;;;; 経由せずにプロトコルの配管（多値、aval、引数の順序、find-backend の
;;;; 同一性、存在しない実行系の扱い）を確かめる。IREE 経由の数値一致は
;;;; tests/iree/backend-test.lisp（medium）にある。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %fake-compile-and-load (backend text)
  "BACKEND-COMPILE してから BACKEND-LOAD した module を返す（このファイルの
テストで何度も出てくる手順をまとめただけ）。"
  (nb:backend-load backend (nb:backend-compile backend text)))

(test backend/fake/round-trip-add-matches-reference
  "フェイク backend で to-device → backend-compile → backend-load →
backend-invoke → to-host した add の結果は reference-add の期待値と
allclose :dtype :f32 で一致し、出力の device-array-aval も一致する。"
  (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                (lambda (seed)
                  (let* ((backend (nb:make-backend :fake))
                         (spec (make-array-spec '(4 8) :f32))
                         (a (make-random-array spec :seed seed))
                         (b (make-random-array spec :seed (1+ seed)))
                         (da (nb:to-device a backend))
                         (db (nb:to-device b backend))
                         (module (%fake-compile-and-load
                                  backend "func.func @main() { stablehlo.add }")))
                    (unwind-protect
                         (multiple-value-bind (result) (nb:backend-invoke backend module "main" da db)
                           (and (equalp (nb:device-array-aval result) (nb:array-aval a :f32))
                                (allclose (nb:to-host result) (reference-add a b) :dtype :f32)))
                      (nb:backend-unload backend module))))
                :regression-id backend/fake/round-trip-add-matches-reference
                :regression-file (regression-path "backend-fake-round-trip-add"))))

(test backend/fake/round-trip-matmul-matches-reference
  "フェイク backend で dot_general（matmul）を実行した結果は reference-matmul
の期待値と allclose :dtype :f32 で一致し、出力の shape も期待どおり。"
  (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                (lambda (seed)
                  (let* ((backend (nb:make-backend :fake))
                         (a (make-random-array (make-array-spec '(2 3) :f32) :seed seed))
                         (b (make-random-array (make-array-spec '(3 2) :f32) :seed (1+ seed)))
                         (da (nb:to-device a backend))
                         (db (nb:to-device b backend))
                         (module (%fake-compile-and-load
                                  backend "\"stablehlo.dot_general\"(%a, %b)")))
                    (unwind-protect
                         (multiple-value-bind (result) (nb:backend-invoke backend module "main" da db)
                           (and (equalp (nb:device-array-aval result) (nb:make-aval '(2 2) :f32))
                                (allclose (nb:to-host result) (reference-matmul a b) :dtype :f32)))
                      (nb:backend-unload backend module))))
                :regression-id backend/fake/round-trip-matmul-matches-reference
                :regression-file (regression-path "backend-fake-round-trip-matmul"))))

(test backend/fake/round-trip-reduce-sum-matches-reference
  "フェイク backend で reduce（総和、dimension 1）を実行した結果は
reference-reduce-sum の期待値と allclose :dtype :f32 で一致する。"
  (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                (lambda (seed)
                  (let* ((backend (nb:make-backend :fake))
                         (a (make-random-array (make-array-spec '(4 8) :f32) :seed seed))
                         (da (nb:to-device a backend))
                         (module (%fake-compile-and-load
                                  backend "\"stablehlo.reduce\"(%a, %init) {dimensions = array<i64: 1>}")))
                    (unwind-protect
                         (multiple-value-bind (result) (nb:backend-invoke backend module "main" da)
                           (and (equalp (nb:device-array-aval result) (nb:make-aval '(4) :f32))
                                (allclose (nb:to-host result) (reference-reduce-sum a 1) :dtype :f32)))
                      (nb:backend-unload backend module))))
                :regression-id backend/fake/round-trip-reduce-sum-matches-reference
                :regression-file (regression-path "backend-fake-round-trip-reduce-sum"))))

(test backend/fake/to-host-round-trips-every-dtype
  "フェイク backend の to-device → to-host は、どの dtype（:f64・:i1 を
含む）の配列も同じ shape・要素型・値の配列に戻す（issue #72。フェイクは
配管の参照実装なので、f64 / i1 を受け付けるようになった本物の IREE
backend と揃える）。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64 :i1))
                                   (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                (lambda (spec-and-seed)
                  (destructuring-bind (spec seed) spec-and-seed
                    (let* ((x (make-random-array spec :seed seed))
                           (roundtripped (nb:to-host (nb:to-device x (nb:make-backend :fake)))))
                      (and (equal (array-element-type roundtripped) (array-element-type x))
                           (equalp roundtripped x)))))
                :regression-id backend/fake/to-host-round-trips-every-dtype
                :regression-file (regression-path "backend-fake-round-trip-every-dtype"))))

(test backend/backend-compile/unsupported-text-signals-backend-error
  "add / dot_general / reduce のどれも含まない TEXT を backend-compile に
渡すと、BACKEND-ERROR の subtype が signal される。"
  (let ((backend (nb:make-backend :fake)))
    (signals nb:backend-error (nb:backend-compile backend "func.func @main() { }"))))

(test backend/make-backend/unknown-kind-signals-backend-not-available
  "登録されていない KIND を make-backend に渡すと BACKEND-NOT-AVAILABLE が
signal され、その KIND が読み出せる。"
  (handler-case
      (progn
        (nb:make-backend :no-such-backend-kind)
        (fiveam:fail "unknown backend kind should have signalled backend-not-available"))
    (nb:backend-not-available (condition)
      (is (eq :no-such-backend-kind (nb:backend-not-available-kind condition))))))

(test backend/find-backend/returns-same-instance-for-same-kind
  "find-backend は同じ KIND に対して、同じ（EQ な）BACKEND インスタンスを
毎回返す。"
  (let ((a (nb:find-backend :fake))
        (b (nb:find-backend :fake)))
    (is (eq a b))))

(test backend/nabla-system/does-not-depend-on-nabla-iree
  "nabla（core）システムは nabla/iree に depends-on していない（core は
実行系の実装を知らない、という設計の約束を ASDF の構成でも守る）。
DEPENDS-ON には (:REQUIRE \"sb-cltl2\") のような文字列でないエントリも
混ざりうる（issue #32、t1）ので、文字列のエントリだけを対象にする。"
  (is (not (member "nabla/iree" (remove-if-not #'stringp (asdf:system-depends-on (asdf:find-system "nabla")))
                    :test #'string-equal))))

(test (backend/core-sources/do-not-mention-iree :suite :nabla.medium)
  "src/*.lisp と src/ffi-support/*.lisp（src/iree/ 以外、非再帰）と nabla.asd の \"nabla\" defsystem
フォームには、大文字小文字を問わず \"iree\" という文字列が一度も現れない
（core は実行系の実装を知らない、という issue #9 の設計の約束）。"
  (let ((offending nil))
    (dolist (path (append (directory (merge-pathnames "*.lisp" (asdf:system-relative-pathname "nabla" "src/")))
                          ;; nabla/ffi-support も IREE の名前を知らない（issue #79。
                          ;; IREE と将来の PJRT の両方から使うため）。
                          (directory (merge-pathnames "*.lisp" (asdf:system-relative-pathname "nabla" "src/ffi-support/")))))
      (with-open-file (stream path :direction :input)
        (let ((text (make-string (file-length stream))))
          (let ((count (read-sequence text stream)))
            (when (search "iree" (string-downcase (subseq text 0 count)))
              (push path offending))))))
    (is (null offending) "iree が現れる core ファイル: ~S" offending)
    ;; core は PJRT の名前も知らない（CLAUDE.md、issue #78）。nabla/ffi-support は
    ;; PJRT との共有が目的なので、ここでは src/*.lisp だけを見る。
    (let ((pjrt-offending nil))
      (dolist (path (directory (merge-pathnames "*.lisp" (asdf:system-relative-pathname "nabla" "src/"))))
        (with-open-file (stream path :direction :input)
          (let* ((text (make-string (file-length stream)))
                 (count (read-sequence text stream)))
            (when (search "pjrt" (string-downcase (subseq text 0 count)))
              (push path pjrt-offending)))))
      (is (null pjrt-offending) "pjrt が現れる core ファイル: ~S" pjrt-offending))
    (let ((asd-path (asdf:system-relative-pathname "nabla" "nabla.asd")))
      (with-open-file (stream asd-path :direction :input)
        (let* ((text (make-string (file-length stream)))
               (count (read-sequence text stream))
               (text (subseq text 0 count))
               (nabla-start (search "(defsystem \"nabla\"" text))
               (nabla-end (search "(defsystem \"nabla/test-support\"" text)))
          (is (and nabla-start nabla-end (< nabla-start nabla-end)))
          (is (not (search "iree" (string-downcase (subseq text nabla-start nabla-end)))))
          (is (not (search "pjrt" (string-downcase (subseq text nabla-start nabla-end))))))))))
