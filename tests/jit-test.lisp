;;;; jit-test: JIT / インメモリのコンパイルキャッシュの性質（issue #34、
;;;; wave 4 j1）。フェイク backend だけを使う small テスト。IREE 経由の
;;;; end-to-end テストは j2（tests/iree/jit-test.lisp）にある。
;;;;
;;;; すべてのテストが NB:*COMPILE-CACHE-DIRECTORY* を NIL に束縛して、
;;;; vmfb のディスクキャッシュ（issue #10）がコンパイル回数を隠さないように
;;;; する。コンパイル回数を数えるテストは、find-backend :fake（プロセスで
;;;; 共有される）ではなく (nb:make-backend :fake) で毎回フレッシュな
;;;; フェイク backend を作る（compile-count が0から始まることを保証する
;;;; ため）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defmacro with-fresh-fake-backend ((var) &body body)
  "VAR に (nb:make-backend :fake) を束縛し、NB:*COMPILE-CACHE-DIRECTORY* を
NIL に束縛した上で BODY を実行する。"
  `(let ((nb:*compile-cache-directory* nil)
         (,var (nb:make-backend :fake)))
     ,@body))

;;; --- 1. jit(f)(x) = f(x)（eager） ---

(test jit/add-matches-eager
  "(+ a b) を jit したものは、フェイク backend 経由でも eager と同じ結果に
なる（フェイクは double で計算してから f32 に丸める。EQUALP ではなく
allclose で比べる）。1回目の呼び出しで *jit-miss-count* が1増え、2回目は
増えない。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (spec (make-array-spec '(2 3) :f32))
           (a (make-random-array spec :seed 1))
           (b (make-random-array spec :seed 2))
           (before nb::*jit-miss-count*))
      (let ((result (funcall jf a b)))
        (is (= 1 (- nb::*jit-miss-count* before)))
        (is (allclose result (funcall f a b) :dtype :f32)))
      (let ((before2 nb::*jit-miss-count*))
        (funcall jf a b)
        (is (= 0 (- nb::*jit-miss-count* before2)))))))

(test jit/dot-matches-eager
  "(nb:dot a w) を jit したものも eager と同じ結果になる。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a w) (nb:dot a w)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 3))
           (w (make-random-array (make-array-spec '(3 4) :f32) :seed 4)))
      (is (allclose (funcall jf a w) (funcall f a w) :dtype :f32)))))

;;; --- 2. 同じキーで2回呼ぶと1回しかコンパイルしない ---

(test jit/same-key-compiles-once
  "同じ shape・dtype の引数で2回呼ぶと、fake-backend-compile-count は1のまま。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 5))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 6)))
      (funcall jf a b)
      (funcall jf a b)
      (is (= 1 (fake-backend-compile-count backend))))))

;;; --- 3. aval・静的引数・ターゲットのどれかが違えば別エントリ ---

(test jit/different-shape-makes-separate-entry
  "(2 3) と (3 2) は別のキャッシュキーになり、それぞれコンパイルする。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend)))
      (funcall jf (make-random-array (make-array-spec '(2 3) :f32) :seed 7)
               (make-random-array (make-array-spec '(2 3) :f32) :seed 8))
      (funcall jf (make-random-array (make-array-spec '(3 2) :f32) :seed 9)
               (make-random-array (make-array-spec '(3 2) :f32) :seed 10))
      (is (= 2 (fake-backend-compile-count backend))))))

(test jit/different-dtype-makes-separate-entry
  "同じ shape でも f64 と f32 は別のキャッシュキーになる。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (f32-a (make-random-array (make-array-spec '(2 3) :f32) :seed 11))
           (f32-b (make-random-array (make-array-spec '(2 3) :f32) :seed 12))
           (f64-a (make-random-array (make-array-spec '(2 3) :f64) :seed 13))
           (f64-b (make-random-array (make-array-spec '(2 3) :f64) :seed 14)))
      (funcall jf f32-a f32-b)
      (funcall jf f64-a f64-b)
      (is (= 2 (fake-backend-compile-count backend))))))

(test jit/different-static-arg-makes-separate-entry
  ":STATIC-ARGS の値が違えば別のキャッシュキーになり、同じ値ならヒットする。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b shape) (nb:reshape (+ a b) shape)))
           (jf (nb:jit f :static-args '(2) :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 15))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 16)))
      (funcall jf a b '(6))
      (is (= 1 (fake-backend-compile-count backend)))
      (funcall jf a b '(3 2))
      (is (= 2 (fake-backend-compile-count backend)))
      (funcall jf a b '(3 2))
      (is (= 2 (fake-backend-compile-count backend))))))

(test jit/different-backend-makes-separate-entry
  "*DEFAULT-BACKEND* を別々の（フィンガープリントも違う）フェイク backend に
束縛して呼ぶと、それぞれ独立に1回だけコンパイルする。"
  (let ((nb:*compile-cache-directory* nil)
        (local (nb:make-backend :fake :fingerprint (list "fake" "target=local")))
        (cuda (nb:make-backend :fake :fingerprint (list "fake" "target=cuda"))))
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 17))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 18)))
      (let ((nb:*default-backend* local)) (funcall jf a b))
      (let ((nb:*default-backend* cuda)) (funcall jf a b))
      (is (= 1 (fake-backend-compile-count local)))
      (is (= 1 (fake-backend-compile-count cuda)))
      (let ((nb:*default-backend* local)) (funcall jf a b))
      (let ((nb:*default-backend* cuda)) (funcall jf a b))
      (is (= 1 (fake-backend-compile-count local)))
      (is (= 1 (fake-backend-compile-count cuda))))))

;;; --- 4. 関数を再定義したら古いキャッシュを使わない ---

(test jit/redefinition-does-not-reuse-cache
  "2つの別々の WITH-TRACING の評価は別々の TRACEABLE-FUNCTION になるので、
それぞれ独立にコンパイルする。%JIT-CACHE-FORGET は捨てたエントリ数を返し、
その後は再びコンパイルする。"
  (with-fresh-fake-backend (backend)
    (let* ((f1 (nb:with-tracing (a b) (+ a b)))
           (f2 (nb:with-tracing (a b) (+ a b)))
           (jf1 (nb:jit f1 :backend backend))
           (jf2 (nb:jit f2 :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 19))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 20)))
      (is (= 0 (nb::%jit-cache-entry-count f1))
          "まだ一度も呼んでいない関数のエントリ数は0")
      (is (= 0 (nb::%jit-cache-forget f1))
          "まだ一度もキャッシュされていない関数を FORGET しても0エントリを捨てる")
      (funcall jf1 a b)
      (funcall jf2 a b)
      (is (= 2 (fake-backend-compile-count backend)))
      (is (= 1 (nb::%jit-cache-forget f1)))
      (is (= 0 (nb::%jit-cache-entry-count f1)))
      (funcall jf1 a b)
      (is (= 3 (fake-backend-compile-count backend))))))

;;; --- 5. %jit-trace / %jit-merge-args を直接 pin する ---

(test jit/%jit-trace-static-position-in-middle
  "静的位置1・静的値3で %JIT-TRACE すると、graph の invar は2個（動的引数の
分だけ）、outvar の shape は (3 2) になる（BROADCAST-IN-DIM の SHAPE に
静的値 N=3 が使われたことの pin）。"
  (let* ((f (nb:with-tracing (a n b) (nb:broadcast-in-dim (+ a b) (list n 2) '(1))))
         (avals (list (nb:make-aval '() :f32) (nb:make-aval '(2) :f32)))
         (graph (nb::%jit-trace f avals '(1) '(3))))
    (is (= 2 (length (nb:graph-invars graph))))
    (is (equal '(3 2) (nb:aval-shape (nb:var-aval (first (nb:graph-outvars graph))))))))

(test jit/%jit-merge-args-golden
  "静的位置 (0 2)・静的値 (:s0 :s2)・動的値 (:d1 :d3) を、arity 4 の元の
位置順に並べ直す。"
  (is (equal (list :s0 :d1 :s2 :d3)
             (nb::%jit-merge-args 4 '(0 2) '(:s0 :s2) '(:d1 :d3)))))

;;; --- 6. エラー ---

(test jit/errors-on-non-traceable-function
  "TRACEABLE-FUNCTION でない関数を JIT に渡すと JIT-ERROR になる。"
  (signals nb:jit-error (nb:jit (lambda (x) x))))

(test jit/errors-on-static-arg-out-of-range
  "arity 2 の関数に :STATIC-ARGS '(2) は範囲外で JIT-ERROR。境界の '(0) と
'(1) はどちらも受け付ける（下限0・上限 arity-1 の両端）。"
  (let ((f (nb:with-tracing (a b) (+ a b))))
    (signals nb:jit-error (nb:jit f :static-args '(2)))
    (is (nb:jit f :static-args '(0)))
    (is (nb:jit f :static-args '(1)))))

(test jit/errors-on-duplicate-static-arg
  ":STATIC-ARGS に重複があると JIT-ERROR。"
  (let ((f (nb:with-tracing (a b) (+ a b))))
    (signals nb:jit-error (nb:jit f :static-args '(0 0)))))

(test jit/errors-on-wrong-arity-call
  "呼び出しの引数の個数が関数の引数の個数と違えば JIT-ERROR。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 21)))
      (signals nb:jit-error (funcall jf a)))))

(test jit/errors-on-bf16-host-array
  "bf16 の生の (unsigned-byte 16) 配列をそのまま渡すと、TO-DEVICE を案内する
JIT-ERROR になる。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (bf16 (make-random-array (make-array-spec '(2 3) :bf16) :seed 22)))
      (signals nb:jit-error (funcall jf bf16 bf16)))))

(test jit/errors-on-unset-default-backend
  "*DEFAULT-BACKEND* が NIL で :BACKEND も渡していなければ JIT-ERROR。"
  (let ((nb:*default-backend* nil))
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 23))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 24)))
      (signals nb:jit-error (funcall jf a b)))))

;;; --- defjit（issue #34、wave 4 j2） ---

(test jit/defjit-basic-and-redefinition
  "DEFJIT で定義した関数は fboundp になり、結果は eager と一致する。2回目の
呼び出しはコンパイルし直さない。同じ DEFJIT フォームを再評価すると、古い
TRACEABLE-FUNCTION のキャッシュエントリは0になり（%JIT-CACHE-FORGET 済み）、
次の呼び出しはまた新しくコンパイルする。"
  (with-fresh-fake-backend (backend)
    (let ((nb:*default-backend* backend))
      (eval '(nb:defjit %jit-test-add (a b) (+ a b)))
      (is (fboundp '%jit-test-add))
      (let ((a (make-random-array (make-array-spec '(2 3) :f32) :seed 30))
            (b (make-random-array (make-array-spec '(2 3) :f32) :seed 31)))
        (let ((old (get '%jit-test-add 'nb::%defjit-traceable)))
          (is (allclose (funcall '%jit-test-add a b) (funcall old a b) :dtype :f32))
          (funcall '%jit-test-add a b)
          (is (= 1 (fake-backend-compile-count backend)))
          (eval '(nb:defjit %jit-test-add (a b) (+ a b)))
          (is (= 0 (nb::%jit-cache-entry-count old))
              "再定義後、古い TRACEABLE-FUNCTION のエントリは捨てられている")
          (funcall '%jit-test-add a b)
          (is (= 2 (fake-backend-compile-count backend))))))))

;;; --- use-eager / recompile リスタート（issue #34、wave 4 j2） ---

(test jit/use-eager-restart-runs-this-call-and-caches-nothing
  "フェイク backend は (- a b) を分類できないので BACKEND-COMPILE が
BACKEND-ERROR を signal し、%JIT-CALL は JIT-COMPILE-ERROR に変換する。
NB:USE-EAGER を invoke するとこの呼び出しだけ EVAL-GRAPH で評価され、
結果は (FUNCALL f a b) と一致する。何もキャッシュされない（エントリ数0）。
ハンドラが無ければ JIT-COMPILE-ERROR がそのまま signal され、
JIT-COMPILE-ERROR-CONDITION は NABLA:BACKEND-ERROR。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (- a b)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 32))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 33)))
      (signals nb:jit-compile-error (funcall jf a b))
      (is (= 0 (nb::%jit-cache-entry-count f)))
      (handler-case (funcall jf a b)
        (nb:jit-compile-error (c)
          (is (typep (nb:jit-compile-error-condition c) 'nb:backend-error))))
      (let ((result nil))
        (handler-bind ((nb:jit-compile-error (lambda (c) (declare (ignore c)) (invoke-restart 'nb:use-eager))))
          (setf result (multiple-value-list (funcall jf a b))))
        (is (allclose (first result) (funcall f a b) :dtype :f32)))
      (is (= 0 (nb::%jit-cache-entry-count f))
          "USE-EAGER は何もキャッシュしない"))))

(defclass %flaky-fake-backend (fake-backend)
  ((attempts :initform 0 :accessor %flaky-fake-backend-attempts))
  (:documentation
   "1回目の BACKEND-COMPILE だけ BACKEND-ERROR を signal し、2回目以降は
FAKE-BACKEND と同じ振る舞いに戻る backend（NB:RECOMPILE リスタートを
確かめるためのテスト専用クラス）。"))

(defmethod nabla:backend-compile ((backend %flaky-fake-backend) text)
  (incf (%flaky-fake-backend-attempts backend))
  (if (= 1 (%flaky-fake-backend-attempts backend))
      (progn
        (incf (fake-backend-compile-count backend))
        (error 'nabla.tests.support::%fake-backend-unsupported-op
               :format-control "flaky: わざと1回目だけ失敗する"
               :format-arguments nil))
      (call-next-method)))

(test jit/recompile-restart-retries-once
  "1回目の BACKEND-COMPILE だけ失敗する %FLAKY-FAKE-BACKEND に対して
NB:RECOMPILE を invoke すると、%JIT-CALL 自体をやり直し、2回目の
BACKEND-COMPILE は成功してキャッシュされる（BACKEND-COMPILE の呼び出し
回数は2）。"
  (let ((nb:*compile-cache-directory* nil)
        (backend (make-instance '%flaky-fake-backend)))
    (let* ((f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 34))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 35))
           (result nil))
      (handler-bind ((nb:jit-compile-error (lambda (c) (declare (ignore c)) (invoke-restart 'nb:recompile))))
        (setf result (funcall jf a b)))
      (is (allclose result (funcall f a b) :dtype :f32))
      (is (= 2 (fake-backend-compile-count backend)))
      (is (= 1 (nb::%jit-cache-entry-count f))))))

;;; --- eqn マッピング: loc が無ければ NIL（issue #34、wave 4 j2） ---

(test jit/compile-error-eqn-is-nil-without-loc
  "フェイク backend のエラーメッセージには loc(\"eqn-N\") が含まれないので、
JIT-COMPILE-ERROR-EQN・JIT-COMPILE-ERROR-EQN-INDEX はどちらも NIL になる
（IREE 経由の正のマッピングは tests/iree/jit-test.lisp で確かめる）。"
  (with-fresh-fake-backend (backend)
    (let* ((f (nb:with-tracing (a b) (- a b)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 36))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 37)))
      (handler-case
          (progn
            (funcall jf a b)
            (fail "フェイク backend の (- a b) がコンパイルに成功してしまった"))
        (nb:jit-compile-error (c)
          (is (null (nb:jit-compile-error-eqn c)))
          (is (null (nb:jit-compile-error-eqn-index c))))))))
