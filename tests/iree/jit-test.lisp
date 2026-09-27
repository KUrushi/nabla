;;;; jit の IREE 経由 end-to-end テスト（issue #34、wave 4 j2）。フェイク
;;;; backend だけを使う small テストは tests/jit-test.lisp にある。
;;;;
;;;; distinct な (テンプレート・shape・dtype) の組は1回の IREE コンパイル
;;;; （約350ms）になるので、PBT の試行回数・演算数は小さく抑える（契約の
;;;; ピットフォール(6)）。bf16/f16 の期待値は、演算ごとに丸める eager
;;;; （EVAL-GRAPH）ではなく、bf16/f16 を f32 に昇格して丸めをほぼ含まない
;;;; 高精度オラクル（%JIT-PBT-F32-ORACLE）にする。理由は %JIT-PBT-TOLERANCE
;;;; の docstring と、そのすぐ上のコメントを参照（要旨: IREE はエレメント
;;;; ワイズ演算を融合し、丸めずに f32 の累積へ渡すことがあるので、eager の
;;;; 「毎回丸める」という前提がそもそも正しい期待値ではなかった）。
;;;;
;;;; このファイルのテストは DEFINE-IREE-TEST を使い、tests/iree/ の他の
;;;; medium テストと同じ :NABLA.MEDIUM スイート・同じ SBCL プロセスで実行
;;;; する。以前はここが in-process の libIREECompiler.so を壊すという
;;;; issue #68 のため独立プロセスに隔離されていたが、原因（K=0 の
;;;; dot_general が float trap を C++ フレームを越えて非ローカル脱出させ、
;;;; コンパイラの状態を壊す）は PR #69 で修正済み: 該当テストは子プロセスで
;;;; 実行し、非ローカル脱出のあとはコンパイラを poison して以後の呼び出しを
;;;; 即座に IREE-COMPILE-ERROR にする（src/iree/compiler.lisp）。この
;;;; ファイルの jit テストは、他の全 medium テストと同じプロセスで問題なく
;;;; 実行できる。

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
;;; あり、原因は K=0 の dot_general（float traps のテスト）が float trap の
;;; signal 処理中に C++ フレームを越えて非ローカル脱出し、in-process の
;;; コンパイラの状態を壊していたこと（issue #68、PR #69 で修正済み）だった。
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

;;; --- 高精度オラクル: bf16/f16 を f32 に昇格して再トレースする ---
;;;
;;; 以前は eager（EVAL-GRAPH、演算ごとに bf16/f16 に丸める）を期待値にして
;;; rtol を (1 + eqn数) 倍に緩めていたが、これは2つの理由で間違っていた:
;;;
;;; (1) IREE はエレメントワイズ演算を融合し、その結果を直後の
;;;     stablehlo.convert（bf16/f16 → f32、reduce-sum/dot の f32 累積の
;;;     ための変換）にそのまま渡す形に最適化することがある。つまり
;;;     「丸めずに f32 へ渡す」経路が実在し、eager の「毎回丸める」という
;;;     前提はそもそも正しい期待値ではない
;;; (2) この丸め1回分の絶対誤差は出力の大きさではなく丸めが起きた
;;;     *中間値* の大きさに比例するので、rtol を出力の絶対値に掛けるだけ
;;;     では、2項が打ち消し合ってほぼ0になる出力（回帰:
;;;     tests/regressions/iree-jit-matches-eager.lisp の
;;;     (7 4 2 4 :BF16 858539031)）を正しく評価できない
;;;
;;; そこで期待値そのものを「同じ GRAPH を、bf16/f16 の invar/中間値を
;;; すべて f32 に昇格して EVAL-GRAPH で評価した結果」（%JIT-PBT-F32-ORACLE）
;;; に変える。これは各演算ごとに丸めるが f32 の丸め誤差（相対 ~1e-7）は
;;; 無視できるほど小さいので、実質「一切丸めない」高精度な参照値になる。
;;; IREE の実際の出力との差は、原理的には「出力を格納 dtype（bf16/f16）に
;;; 丸めたときの誤差」1つだけになるはずなので、許容誤差は dtype ごとの
;;; 既定値（DTYPE-TOLERANCE、SKILL.md の表）をそのまま使えばよく、
;;; eqn 数や中間値の大きさで人為的に緩める必要が無くなる。

(defun %jit-pbt-promote-dtype (dtype)
  "DTYPE が :bf16 / :f16 なら :f32 に、それ以外はそのまま返す。"
  (if (member dtype '(:bf16 :f16)) :f32 dtype))

(defun %jit-pbt-promote-aval (aval)
  (nb:make-aval (nb:aval-shape aval) (%jit-pbt-promote-dtype (nb:aval-dtype aval))))

(defun %jit-pbt-promote-array (array dtype)
  "ARRAY（DTYPE の格納表現）を、DTYPE が :bf16 / :f16 なら SINGLE-FLOAT の
配列にデコードして返す（f32 / f64 はそのまま）。"
  (if (member dtype '(:bf16 :f16)) (nb::decode-float16-array array dtype) array))

(defun %jit-pbt-f32-oracle (f avals arrays)
  "F（TRACEABLE-FUNCTION）を AVALS の bf16/f16 をすべて f32 に昇格した aval
で再トレースし、ARRAYS も対応する SINGLE-FLOAT 配列に変換したうえで
EVAL-GRAPH した結果を返す（多値）。中間の bf16/f16 丸めが一切無い高精度な
参照値になる（このファイル冒頭のコメント参照）。"
  (let* ((f32-avals (mapcar #'%jit-pbt-promote-aval avals))
         (f32-arrays (mapcar (lambda (array aval) (%jit-pbt-promote-array array (nb:aval-dtype aval)))
                              arrays avals))
         (f32-graph (nb:trace-to-graph f f32-avals)))
    (apply #'nb:eval-graph f32-graph f32-arrays)))

(defun %jit-pbt-tolerance (dtype)
  "DTYPE（jit に渡した実際の dtype。出力の格納 dtype と同じ、このファイルの
どのテンプレートも dtype を変えない）の許容誤差を DTYPE-TOLERANCE から
そのまま返す。期待値が %JIT-PBT-F32-ORACLE（丸めをほぼ含まない）になった
ことで、IREE の実際の出力との差は原理的に「出力を DTYPE に格納するときの
丸め」1つ分に収まるはずなので、SKILL.md の表の値（bf16/f16 は 1 ULP の
数倍程度）をそのまま使う。それでも吸収しきれない差（IREE がまれに
中間値を bf16/f16 のまま保持する場合）が見つかれば、そのケースを
tests/regressions/iree-jit-matches-eager.lisp に固定してから、ここに
理由つきで根拠のある項を足すこと（当てずっぽうに緩めない）。"
  (dtype-tolerance dtype))

(defun %jit-pbt-matches-eager-p (backend template-index m k n dtype base-seed)
  "TEMPLATE-INDEX・M・K・N・DTYPE から組み立てた関数を BACKEND 上で JIT した
結果が、f32 なら直接呼んだ eager 実装、bf16/f16 なら %JIT-PBT-F32-ORACLE
（bf16/f16 を一切丸めない高精度な参照値）と、許容誤差つきで一致するか
どうかを返す（jit とキャッシュの性質「jit(f)(x) = f(x)」の medium 版、
契約 J2.4）。bf16/f16 は生の (unsigned-byte 16) 配列を jit に渡すと
%JIT-ARGUMENT-AVAL の dtype 推論が効かず JIT-ERROR になる（src/jit.lisp）
ので、あらかじめ TO-DEVICE で device array にしてから渡す。"
  (let* ((f (%jit-pbt-traceable-function template-index m k))
         (avals (%jit-pbt-avals template-index m k n dtype))
         (arrays (%jit-pbt-arrays avals base-seed))
         (jf (nb:jit f :backend backend)))
    (unwind-protect
         (multiple-value-bind (rtol atol) (%jit-pbt-tolerance dtype)
           (if (member dtype '(:bf16 :f16))
               (let* ((device-arrays (mapcar (lambda (array aval) (to-device array backend :dtype (nb:aval-dtype aval)))
                                              arrays avals))
                      (jit-result nil)
                      (oracle-result (%jit-pbt-f32-oracle f avals arrays)))
                 (unwind-protect
                      (progn
                        (setf jit-result (apply jf device-arrays))
                        (allclose (decode-array jit-result dtype) (decode-array oracle-result :f32)
                                  :rtol rtol :atol atol))
                   (dolist (da device-arrays) (release-device-array da))))
               (allclose (apply jf arrays) (apply f arrays) :dtype dtype :rtol rtol :atol atol)))
      ;; F は毎試行ごとに EVAL で新しく作る TRACEABLE-FUNCTION（1回しか
      ;; 使わない）なので、*JIT-CACHE* のエントリを持ったままにすると
      ;; ロードした IREE の module が二度と解放されない（J1.5 の「GC で
      ;; テーブルごと消えるが BACKEND-UNLOAD は呼ばれない」既知の制約）。
      ;; PBT は多数の distinct な関数を作るので、ここで明示的に
      ;; %JIT-CACHE-FORGET して毎回すぐ解放する。
      (nb::%jit-cache-forget f))))

(define-iree-test jit/pbt-matches-eager
    "ランダムな with-tracing 本体（+ - max min neg tanh exp dot transpose
where reduce-sum/max reshape broadcast-in-dim）を IREE 上で jit した結果は、
f32 なら直接呼んだ eager 実装、bf16/f16 なら bf16/f16 を f32 に昇格した
高精度オラクル（%JIT-PBT-F32-ORACLE）の結果と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 15)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (uniform-integer :lo 0 :hi (1- *jit-pbt-template-count*))
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 1 :hi 4)
                                     (or (quote :f32) (quote :bf16) (quote :f16))
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

;;; --- jit(f)(x) = f(x) は、reduce-sum / dot の f32 累積が壊れたときにも
;;; 検出できる（Beyoncé rule） ---
;;;
;;; JIT/PBT-MATCHES-EAGER の m・k・n は 1〜4 しか動かさない（コンパイル
;;; コストのため）ので、%REDUCE-ACCUMULATE-IN-F32-P / %DOT-ACCUMULATE-IN-F32-P
;;; を無効化する変異（f32 に一度も昇格せず bf16/f16 のまま累積する）を
;;; 注入しても、その PBT の許容誤差の中に収まってしまうことがレビューで
;;; 分かった。手元で軸長・契約次元 K を増やしながら測ったところ:
;;;
;;;   K/軸長 |  4 |  8 | 16 | 32 | 64  | 512 | 2048  | 65536
;;;   reduce  0% |  0%|  0%|20% |     | 85% | 95%   | 95〜100%
;;;   dot     0% |  0%|  5%|28% |100%*| 100%|100%   |
;;;   (* dot は tests/iree/dot-test.lisp の K64 テストが直接 backend 経由で確認済み)
;;;
;;; （割合は20試行中「オラクルと不一致で検出できた」割合。K が小さいと
;;; bf16/f16 は数項の加算では f32 で計算したのとほぼ同じ値になり
;;; （CPU 上の bf16/f16 演算は内部で f32 に昇格してから丸めるため、単発の
;;; 加算では丸め回数が変わらない）、原理的にほぼ検出不可能）。そこで
;;; K=2048（dot）・軸長 65536（reduce-sum）の固定の1ケースを、jit 経由で
;;; 高精度オラクルと突き合わせるテストをここに置く。PBT を大きな shape に
;;; 広げる（distinct なコンパイルが増える）代わりに、この1点だけを
;;; hermetic な例ベースのテストとして固定する。

(defun %jit-large-dot-matches-oracle-p (backend dtype k)
  "shape (4 K) @ (K 4) の DOT を jit した結果が、bf16/f16 を f32 に昇格した
高精度オラクルと一致するかどうかを返す。K を十分大きくすることで、
%DOT-ACCUMULATE-IN-F32-P が無効化された回帰を高確率で検出できる
（ファイル冒頭のコメント参照）。"
  (let* ((f (nb:with-tracing (a w) (nb:dot a w)))
         (avals (list (nb:make-aval (list 4 k) dtype) (nb:make-aval (list k 4) dtype)))
         (arrays (%jit-pbt-arrays avals 1))
         (jf (nb:jit f :backend backend))
         (device-arrays (mapcar (lambda (array aval) (to-device array backend :dtype (nb:aval-dtype aval)))
                                 arrays avals))
         (oracle-result (%jit-pbt-f32-oracle f avals arrays)))
    (unwind-protect
         (multiple-value-bind (rtol atol) (%jit-pbt-tolerance dtype)
           (allclose (decode-array (apply jf device-arrays) dtype) (decode-array oracle-result :f32)
                     :rtol rtol :atol atol))
      (dolist (da device-arrays) (release-device-array da))
      (nb::%jit-cache-forget f))))

(defun %jit-large-reduce-matches-oracle-p (backend dtype axis-size)
  "shape (2 AXIS-SIZE) を軸1で潰す REDUCE-SUM を jit した結果が、bf16/f16 を
f32 に昇格した高精度オラクルと一致するかどうかを返す。AXIS-SIZE を十分
大きくすることで、%REDUCE-ACCUMULATE-IN-F32-P が無効化された回帰を高確率で
検出できる（ファイル冒頭のコメント参照）。"
  (let* ((f (nb:with-tracing (a) (nb:reduce-sum a :axes '(1))))
         (avals (list (nb:make-aval (list 2 axis-size) dtype)))
         (arrays (%jit-pbt-arrays avals 2))
         (jf (nb:jit f :backend backend))
         (device-arrays (mapcar (lambda (array aval) (to-device array backend :dtype (nb:aval-dtype aval)))
                                 arrays avals))
         (oracle-result (%jit-pbt-f32-oracle f avals arrays)))
    (unwind-protect
         (multiple-value-bind (rtol atol) (%jit-pbt-tolerance dtype)
           (allclose (decode-array (apply jf device-arrays) dtype) (decode-array oracle-result :f32)
                     :rtol rtol :atol atol))
      (dolist (da device-arrays) (release-device-array da))
      (nb::%jit-cache-forget f))))

(define-iree-test jit/large-reduction-catches-accumulation-regression
    "shape (4 2048) @ (2048 4) の DOT と、軸長 65536 の REDUCE-SUM を jit した
結果は、bf16・f16 のどちらでも高精度オラクルと一致する。K・軸長をここまで
大きくする理由と、JIT/PBT-MATCHES-EAGER（K が 1〜4 しかない）では
%REDUCE-ACCUMULATE-IN-F32-P / %DOT-ACCUMULATE-IN-F32-P が無効化された回帰を
ほとんど検出できないことの実測値は、ファイル冒頭のコメントを参照。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (dolist (dtype '(:bf16 :f16))
      (is (%jit-large-dot-matches-oracle-p backend dtype 2048)
          "~A: K=2048 の dot がオラクルと一致しない" dtype)
      (is (%jit-large-reduce-matches-oracle-p backend dtype 65536)
          "~A: 軸長 65536 の reduce-sum がオラクルと一致しない" dtype))
    (gc-and-run-finalizers)))

;;; --- 複数の出力値 ---

(define-iree-test jit/multiple-values
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

(define-iree-test jit/does-not-recompile-on-repeated-call
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
  "DTYPE（:f32・:bf16）で %JIT-TEST-MLP を JAX フィクスチャと高精度オラクル
（%JIT-PBT-F32-ORACLE。bf16 を一切丸めずに同じ graph を評価した参照値）の
両方と突き合わせる。out1（JAX 期待値）・out2（JAX 期待値）・out1
（オラクル）・out2（オラクル）の4つを別々の FIVEAM:IS にして、どれが
食い違ったかが失敗メッセージから分かるようにする（1つの AND にまとめない）。
JAX フィクスチャは JAX/XLA が実際にコンパイル・実行した bf16 の出力
（すでに丸め済み）なので、IREE 側の出力とは「両方とも一度だけ格納 dtype に
丸めた値」同士の比較になり、eval-graph の場合と同じ理由で許容誤差は
DTYPE-TOLERANCE のままでよい（%JIT-PBT-TOLERANCE 参照）。"
  (let* ((inputs (%fixture-inputs dtype))
         (expected (%fixture-outputs dtype))
         (traceable (get '%jit-test-mlp 'nb::%defjit-traceable))
         (avals (%mlp-avals dtype))
         (jit-args (if (eq dtype :bf16)
                       (mapcar (lambda (array) (to-device array backend :dtype dtype)) inputs)
                       inputs)))
    (multiple-value-bind (rtol atol) (%jit-pbt-tolerance dtype)
      (unwind-protect
           (multiple-value-bind (out1 out2) (apply #'%jit-test-mlp jit-args)
             (multiple-value-bind (oracle-out1 oracle-out2)
                 (%jit-pbt-f32-oracle traceable avals inputs)
               (is (allclose out1 (first expected) :dtype dtype :rtol rtol :atol atol)
                   "~A: out1 (jit) が JAX フィクスチャと一致しない" dtype)
               (is (allclose out2 (second expected) :dtype dtype :rtol rtol :atol atol)
                   "~A: out2 (jit) が JAX フィクスチャと一致しない" dtype)
               (is (allclose (decode-array out1 dtype) (decode-array oracle-out1 :f32) :rtol rtol :atol atol)
                   "~A: out1 (jit) が高精度オラクルと一致しない" dtype)
               (is (allclose (decode-array out2 dtype) (decode-array oracle-out2 :f32) :rtol rtol :atol atol)
                   "~A: out2 (jit) が高精度オラクルと一致しない" dtype)))
        (when (eq dtype :bf16) (dolist (da jit-args) (release-device-array da)))))))

(define-iree-test jit/mlp-matches-jax-fixture-and-eval-graph
    "DEFJIT した小さな MLP 相当の関数（elementwise + dot + reduce-sum/max +
reshape + broadcast-in-dim）は、f32・bf16 のどちらでも、JAX で生成した
フィクスチャ（tests/fixtures/jit/mlp.lisp）および同じ graph の高精度
オラクル（%JIT-PBT-F32-ORACLE。f32 は EVAL-GRAPH そのもの）と、許容誤差
つきで一致する（#35 の完了条件）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*default-backend* (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil))
    (%mlp-check-dtype backend :f32)
    (%mlp-check-dtype backend :bf16)))

;;; --- コンパイル診断からどの eqn が原因かを逆引きできる（jit 経由） ---

(define-iree-test jit/compile-error-maps-back-to-broken-eqn
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

(define-iree-test example/jit-lisp/prints-expected-sum
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
