;;;; jit の IREE 経由 end-to-end テスト（issue #34、wave 4 j2）。フェイク
;;;; backend だけを使う small テストは tests/jit-test.lisp にある。
;;;;
;;;; distinct な (テンプレート・shape・dtype) の組は1回の IREE コンパイル
;;;; （約350ms）になるので、PBT の試行回数・演算数は小さく抑える（契約の
;;;; ピットフォール(6)）。bf16/f16 は IREE がエレメントワイズ演算を融合して
;;;; 1回だけ丸めるのに対し eager（EVAL-GRAPH）は演算ごとに丸めるため、
;;;; rtol を (1 + eqn数) 倍に緩める（tests/iree/stablehlo-test.lisp と同じ
;;;; 考え方。ピットフォール(4)）。
;;;;
;;;; このファイルのテストは DEFINE-IREE-TEST/ISOLATED-MEDIUM を使い、
;;;; :NABLA.MEDIUM ではなく独立した :NABLA.ISOLATED-MEDIUM スイートに登録
;;;; する（tests/iree/support.lisp）。理由: tests/iree/ の他の medium テスト
;;;; （IREE/ARITH・DOT-GENERAL・REDUCE-SUM・STABLEHLO 等）をすべて実行した
;;;; *あとに* このファイルのどれか（JIT/PBT-MATCHES-EAGER に限らず、最初に
;;;; 走る jit テストならどれでも）が新しく IREE コンパイラを呼ぶと、
;;;; in-process の libIREECompiler.so（mlir::OpPassManager の構築中）が
;;;; メモリ破壊で落ちる（SB-SYS:MEMORY-FAULT-ERROR、issue #34 PR #67 の
;;;; レビューで発見、issue #68 で追跡）。原因は IREE の *実行* ではなく
;;;; コンパイラ自身の in-process 状態で、medium スイート全体が積み重ねる
;;;; distinct コンパイルの総量に依存するらしい（issue #68 に詳細）。
;;;;
;;;; :NABLA.ISOLATED-MEDIUM は :NABLA.MEDIUM とは別の SBCL プロセスで実行
;;;; する（scripts/run-tests.sh）。まっさらなプロセスから始まるので他の
;;;; medium テストのコンパイルが積み重ならず、このファイル単独ではクラッシュ
;;;; しないことを確認済み。NABLA_TEST_SIZES に "medium" が含まれる限り
;;;; scripts/run-tests.sh が必ずこのスイートも実行するので、CI の既定
;;;; スイートは引き続き #35 の「jit(f)(x) の結果は (f x) の結果と一致する」
;;;; を medium で検査する。

(in-package #:nabla.iree.tests)

(defun %count-substring (needle haystack)
  "HAYSTACK の中に NEEDLE が現れる（重なりを許す）回数を返す。"
  (loop with count = 0
        with start = 0
        for position = (search needle haystack :start2 start)
        while position
        do (incf count) (setf start (1+ position))
        finally (return count)))

;;; --- ランダムな with-tracing 本体で jit(f)(x) = f(x) を確かめる ---
;;;
;;; 各テンプレートは、自分が実際に使う引数だけを WITH-TRACING の仮引数に
;;; 持たせる（%JIT-PBT-ARITY）。未使用引数（例えば A・B だけのテンプレート
;;; に未使用の W）が原因という当初の見立ては誤りだった。実際に
;;; `(with-tracing (a b w) (+ a b))` を（fresh な traceable ごとに
;;; %jit-cache-forget + full GC + finalizer 実行を挟んで）40回 jit しても
;;; 落ちなかった。落ちたときのバックトレースは IREE の *実行* ではなく
;;; libIREECompiler.so の *コンパイラ* 内部（mlir::OpPassManager の構築）に
;;; あり、原因は medium スイート全体で積み重なる distinct コンパイル回数
;;; だった（ファイル冒頭のコメント、issue #34 PR #67 のレビュー参照）。
;;; テンプレートごとに正確な arity でトレースする作りはそのまま残すが
;;; （無害で、実際の使われ方にも近いため）、未使用引数を犯人とする説明は
;;; しない。

(defun %jit-pbt-arity (template-index)
  "TEMPLATE-INDEX が実際に使う引数の組を :AB（A・B、shape (M K) 同士）・
:AW（A・W、shape (M K)・(K N)）・:A（A のみ）のどれかで返す。"
  (case template-index
    ((1 8) :aw)
    ((4 12) :a)
    (t :ab)))

(defun %jit-pbt-lambda-list (template-index)
  (ecase (%jit-pbt-arity template-index)
    (:ab '(a b))
    (:aw '(a w))
    (:a '(a))))

(defun %jit-pbt-body (template-index m k)
  "TEMPLATE-INDEX（0始まり、0..12）に対応する with-tracing の本体（S式）を
返す。RESHAPE・BROADCAST-IN-DIM だけはリテラルの shape に M・K の実際の
値を埋め込む必要があるので、その2つの引数を取る。"
  (ecase template-index
    (0 '(+ a b))
    (1 '(nb:dot a w))
    (2 '(- a b))
    (3 '(tanh (+ a b)))
    (4 '(exp (- a)))
    (5 '(max a b))
    (6 '(min a b))
    (7 '(nb:reduce-sum (+ a b) :axes '(1)))
    (8 '(nb:reduce-max (nb:dot a w) :axes '(1)))
    (9 '(nb:transpose (+ a b)))
    (10 '(nb:where (< a b) a b))
    (11 `(nb:reshape (+ a b) (list ,(* m k))))
    (12 `(nb:broadcast-in-dim (nb:reduce-sum a :axes '(1)) (list ,m ,k) '(0)))))

(defparameter *jit-pbt-template-count* 13)

(defun %jit-pbt-traceable-function (template-index m k)
  "TEMPLATE-INDEX・M・K に対応する (NB:WITH-TRACING (...) <本体>) を EVAL
して TRACEABLE-FUNCTION を返す（仮引数は %JIT-PBT-ARITY が実際に使う分
だけ）。"
  (eval `(nb:with-tracing ,(%jit-pbt-lambda-list template-index) ,(%jit-pbt-body template-index m k))))

(defun %jit-pbt-avals (template-index m k n dtype)
  (ecase (%jit-pbt-arity template-index)
    (:ab (list (nb:make-aval (list m k) dtype) (nb:make-aval (list m k) dtype)))
    (:aw (list (nb:make-aval (list m k) dtype) (nb:make-aval (list k n) dtype)))
    (:a (list (nb:make-aval (list m k) dtype)))))

(defun %jit-pbt-arrays (avals base-seed)
  (loop for aval in avals
        for i from 0
        collect (make-random-array (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval)) :seed (+ base-seed i))))

(defun %jit-pbt-tolerance (graph dtype)
  "GRAPH の中で実際に使われた最も粗い浮動小数点 dtype で許容誤差を決める
（tests/iree/stablehlo-test.lisp と同じ考え方）。bf16/f16 は
(1 + eqn数) 倍に緩める。(values rtol atol)。"
  (let* ((tolerance-dtype (or (graph-worst-float-dtype graph) dtype))
         (n-eqns (length (nb:graph-eqns graph))))
    (multiple-value-bind (rtol atol) (dtype-tolerance tolerance-dtype)
      (values (if (member tolerance-dtype '(:bf16 :f16)) (* rtol (1+ n-eqns)) rtol) atol))))

(defun %jit-pbt-matches-eager-p (backend template-index m k n dtype base-seed)
  "TEMPLATE-INDEX・M・K・N・DTYPE から組み立てた関数を BACKEND 上で JIT した
結果が、f32 なら直接呼んだ eager 実装、bf16 なら EVAL-GRAPH の結果と、
許容誤差つきで一致するかどうかを返す（jit とキャッシュの性質
「jit(f)(x) = f(x)」の medium 版、契約 J2.4）。"
  (let* ((f (%jit-pbt-traceable-function template-index m k))
         (avals (%jit-pbt-avals template-index m k n dtype))
         (arrays (%jit-pbt-arrays avals base-seed))
         (graph (nb:trace-to-graph f avals))
         (jf (nb:jit f :backend backend)))
    (unwind-protect
         (multiple-value-bind (rtol atol) (%jit-pbt-tolerance graph dtype)
           (if (eq dtype :bf16)
               (let* ((device-arrays (mapcar (lambda (array aval) (to-device array backend :dtype (nb:aval-dtype aval)))
                                              arrays avals))
                      (jit-result nil)
                      (eager-result (apply #'nb:eval-graph graph arrays)))
                 (unwind-protect
                      (progn
                        (setf jit-result (apply jf device-arrays))
                        (allclose jit-result eager-result :dtype dtype :rtol rtol :atol atol))
                   (dolist (da device-arrays) (release-device-array da))))
               (allclose (apply jf arrays) (apply f arrays) :dtype dtype :rtol rtol :atol atol)))
      ;; F は毎試行ごとに EVAL で新しく作る TRACEABLE-FUNCTION（1回しか
      ;; 使わない）なので、*JIT-CACHE* のエントリを持ったままにすると
      ;; ロードした IREE の module が二度と解放されない（J1.5 の「GC で
      ;; テーブルごと消えるが BACKEND-UNLOAD は呼ばれない」既知の制約）。
      ;; PBT は多数の distinct な関数を作るので、ここで明示的に
      ;; %JIT-CACHE-FORGET して毎回すぐ解放する。
      (nb::%jit-cache-forget f))))

(define-iree-test/isolated-medium jit/pbt-matches-eager
    "ランダムな with-tracing 本体（+ - max min neg tanh exp dot transpose
where reduce-sum/max reshape broadcast-in-dim）を IREE 上で jit した結果は、
f32 なら直接呼んだ eager 実装、bf16 なら eval-graph の結果と一致する。
:NABLA.ISOLATED-MEDIUM スイート（ファイル冒頭のコメント参照: 他の medium
テストと同じプロセスで実行すると IREE コンパイラの in-process 状態が
壊れることを確認したため、別プロセスで実行する。issue #68）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 15)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (uniform-integer :lo 0 :hi (1- *jit-pbt-template-count*))
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 1 :hi 4)
                                     (or (quote :f32) (quote :bf16))
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (template m k n dtype seed) args
                      (%jit-pbt-matches-eager-p backend template m k n dtype seed)))
                  :regression-id jit/pbt-matches-eager
                  :regression-file (regression-path "iree-jit-matches-eager" :package "NABLA.IREE.TESTS")))
    ;; jit の内部で作った device array は明示的に release せず finalizer に
    ;; 任せる設計（CLAUDE.md）だが、この試行回数（各試行が1回の IREE
    ;; コンパイル + 実行）の直後は未回収の device array が溜まりやすいので、
    ;; ここで明示的に GC + finalizer を走らせて後続テストへの持ち越しを
    ;; 減らす（tests/iree/support.lisp の GC-AND-RUN-FINALIZERS）。
    (gc-and-run-finalizers)))

;;; --- 複数の出力値 ---

(define-iree-test/isolated-medium jit/multiple-values
    "(values (nb:dot x w) (+ x x)) を jit すると、host 配列2つが多値で返る
（E2: 1つは invar そのもの）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (nb:with-tracing (x w) (values (nb:dot x w) (+ x x))))
         (jf (nb:jit f :backend backend))
         (x (make-random-array (make-array-spec '(2 3) :f32) :seed 40))
         (w (make-random-array (make-array-spec '(3 4) :f32) :seed 41)))
    (multiple-value-bind (dot-result doubled-result) (funcall jf x w)
      (multiple-value-bind (expected-dot expected-doubled) (funcall f x w)
        (is (allclose dot-result expected-dot :dtype :f32))
        (is (allclose doubled-result expected-doubled :dtype :f32))))))

;;; --- 再定義しなければ再コンパイルしない ---

(define-iree-test/isolated-medium jit/does-not-recompile-on-repeated-call
    "同じ shape・dtype で2回呼んでも、2回目は NB::*JIT-MISS-COUNT* が増えない
（IREE backend にはコンパイル回数を直接数える手段がないので、代わりに
*JIT-MISS-COUNT* のデルタと %JIT-CACHE-ENTRY-COUNT で確かめる。契約の
ピットフォール(7)）。vmfb ディスクキャッシュは一時ディレクトリに束縛して、
開発機のキャッシュ状態が影響しないようにする。"
  (skip-unless-iree :library :both)
  (with-temporary-directory (dir)
    (let* ((nb:*compile-cache-directory* dir)
           (backend (nabla:find-backend :iree))
           (f (nb:with-tracing (a b) (+ a b)))
           (jf (nb:jit f :backend backend))
           (a (make-random-array (make-array-spec '(2 3) :f32) :seed 42))
           (b (make-random-array (make-array-spec '(2 3) :f32) :seed 43))
           (before nb::*jit-miss-count*))
      (funcall jf a b)
      (is (= 1 (- nb::*jit-miss-count* before)))
      (let ((before2 nb::*jit-miss-count*))
        (funcall jf a b)
        (is (= 0 (- nb::*jit-miss-count* before2))))
      (is (= 1 (nb::%jit-cache-entry-count f))))))

;;; --- #35 end-to-end 完了条件: JAX フィクスチャと一致する MLP ---

(defun %mlp-fixture ()
  "tests/fixtures/jit/mlp.lisp を読み込んで返す（generate.py が生成した
s式そのもの）。"
  (let ((path (asdf:system-relative-pathname "nabla" "tests/fixtures/jit/mlp.lisp")))
    (with-open-file (stream path :direction :input)
      (read stream))))

(defun %fixture-array (dtype entry)
  "ENTRY（(NAME SHAPE BITS) の形。generate.py 参照）から DTYPE の CL 配列を
作る。f32 は各 BITS を NB::%MAKE-SINGLE-FLOAT で SINGLE-FLOAT に、bf16 は
BITS をそのまま (UNSIGNED-BYTE 16) の要素にする。"
  (destructuring-bind (name shape bits) entry
    (declare (ignore name))
    (let ((array (make-array shape :element-type (if (eq dtype :f32) 'single-float '(unsigned-byte 16)))))
      (loop for i from 0 below (array-total-size array)
            for bit in bits
            do (setf (row-major-aref array i)
                     (if (eq dtype :f32) (nb::%make-single-float bit) bit)))
      array)))

(defun %fixture-section (dtype)
  (getf (%mlp-fixture) dtype))

(defun %fixture-inputs (dtype)
  (mapcar (lambda (entry) (%fixture-array dtype entry)) (getf (%fixture-section dtype) :inputs)))

(defun %fixture-outputs (dtype)
  (mapcar (lambda (entry) (%fixture-array dtype entry)) (getf (%fixture-section dtype) :outputs)))

(defparameter *mlp-shapes* '((2 3) (3 4) (4) (4 2) (2))
  "%MLP のシグネチャ (x w1 b1 w2 b2) の shape（B=2 D=3 H=4 C=2、契約 J2.4）。")

(defun %mlp-avals (dtype)
  (mapcar (lambda (shape) (nb:make-aval shape dtype)) *mlp-shapes*))

(nb:defjit %jit-test-mlp (x w1 b1 w2 b2)
  (let* ((h (tanh (+ (nb:dot x w1) (nb:broadcast-in-dim b1 '(2 4) '(1)))))
         (logits (+ (nb:dot h w2) (nb:broadcast-in-dim b2 '(2 2) '(1))))
         (out1 (nb:reduce-max logits :axes '(1)))
         (out2 (nb:reduce-sum (nb:reshape logits (list 4)))))
    (values out1 out2)))

(defun %mlp-check-dtype (backend dtype)
  "DTYPE（:f32・:bf16）で %JIT-TEST-MLP を JAX フィクスチャと EVAL-GRAPH の
両方と突き合わせる。out1（JAX 期待値）・out2（JAX 期待値）・out1（eager）・
out2（eager）の4つを別々の FIVEAM:IS にして、どれが食い違ったかが失敗
メッセージから分かるようにする（1つの AND にまとめない）。"
  (let* ((inputs (%fixture-inputs dtype))
         (expected (%fixture-outputs dtype))
         (traceable (get '%jit-test-mlp 'nb::%defjit-traceable))
         (avals (%mlp-avals dtype))
         (graph (nb:trace-to-graph traceable avals))
         (jit-args (if (eq dtype :bf16)
                       (mapcar (lambda (array) (to-device array backend :dtype dtype)) inputs)
                       inputs)))
    (multiple-value-bind (rtol atol) (%jit-pbt-tolerance graph dtype)
      (unwind-protect
           (multiple-value-bind (out1 out2) (apply #'%jit-test-mlp jit-args)
             (multiple-value-bind (eager-out1 eager-out2) (apply #'nb:eval-graph graph inputs)
               (is (allclose out1 (first expected) :dtype dtype :rtol rtol :atol atol)
                   "~A: out1 (jit) が JAX フィクスチャと一致しない" dtype)
               (is (allclose out2 (second expected) :dtype dtype :rtol rtol :atol atol)
                   "~A: out2 (jit) が JAX フィクスチャと一致しない" dtype)
               (is (allclose out1 eager-out1 :dtype dtype :rtol rtol :atol atol)
                   "~A: out1 (jit) が eval-graph と一致しない" dtype)
               (is (allclose out2 eager-out2 :dtype dtype :rtol rtol :atol atol)
                   "~A: out2 (jit) が eval-graph と一致しない" dtype)))
        (when (eq dtype :bf16) (dolist (da jit-args) (release-device-array da)))))))

(define-iree-test/isolated-medium jit/mlp-matches-jax-fixture-and-eval-graph
    "DEFJIT した小さな MLP 相当の関数（elementwise + dot + reduce-sum/max +
reshape + broadcast-in-dim）は、f32・bf16 のどちらでも、JAX で生成した
フィクスチャ（tests/fixtures/jit/mlp.lisp）および同じ graph を
EVAL-GRAPH で評価した結果と、許容誤差つきで一致する（#35 の完了条件）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*default-backend* (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (%mlp-check-dtype backend :f32)
    (%mlp-check-dtype backend :bf16)))

;;; --- コンパイル診断からどの eqn が原因かを逆引きできる（jit 経由） ---

(define-iree-test/isolated-medium jit/compile-error-maps-back-to-broken-eqn
    "わざと壊した eqn（tests/iree/stablehlo-test.lisp の %TEST-BAD-RESHAPE）
だけを持つ関数を jit すると JIT-COMPILE-ERROR が signal され、
JIT-COMPILE-ERROR-EQN-INDEX が 0（壊れた唯一の eqn）になる。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (nb:with-tracing (x) (nb::%trace-eqn :%test-bad-reshape (list x))))
         (jf (nb:jit f :backend backend))
         (x (make-random-array (make-array-spec '(2) :f32) :seed 44)))
    (handler-case
        (progn
          (funcall jf x)
          (fiveam:fail "型が矛盾した graph の jit が成功してしまった"))
      (nb:jit-compile-error (c)
        (is (= 0 (nb:jit-compile-error-eqn-index c)))
        (is (eq :%test-bad-reshape (nb:primitive-name (nb::eqn-prim (nb:jit-compile-error-eqn c)))))))))

;;; --- README の使用例（examples/jit.lisp）が壊れていないことを確かめる ---

(define-iree-test/isolated-medium example/jit-lisp/prints-expected-sum
    "examples/jit.lisp（README の使用例）を読み込むと、標準出力に4要素の
加算結果 \"11.0 22.0 33.0 44.0\" が2回（1回目・2回目）現れる。"
  (skip-unless-iree :library :both)
  (let ((output (make-string-output-stream))
        (nb:*compile-cache-directory* nil))
    (let ((*standard-output* output))
      (load (asdf:system-relative-pathname "nabla" "examples/jit.lisp")))
    (let ((text (get-output-stream-string output)))
      (is (<= 2 (%count-substring "11.0 22.0 33.0 44.0" text))
          "examples/jit.lisp の出力に期待する和が2回見つからなかった: ~S" text))))
