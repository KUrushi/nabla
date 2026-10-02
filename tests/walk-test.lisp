;;;; walk-test: WITH-TRACING の code walker の性質（issue #32、t1）。
;;;;
;;;; UNSUPPORTED-FORM は SB-CLTL2:MACROEXPAND-ALL 呼び出しの中（マクロ展開時）
;;;; に signal されるので、テストは MACROEXPAND-1 で確かめる（COMPILE の中で
;;;; signal すると SBCL がランタイムのスタブに変えてしまうため。契約の
;;;; pitfall）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defmacro %signals-unsupported-form ((form-var path-var) unsupported-form &body body)
  "UNSUPPORTED-FORM を MACROEXPAND-1 (UNSUPPORTED-FORM) から捕まえ、
FORM-VAR・PATH-VAR にその FORM・PATH を束縛して BODY を実行する。捕まらなけ
れば FIVEAM のテスト失敗にする。"
  `(handler-case (progn (macroexpand-1 ,unsupported-form) (fail "UNSUPPORTED-FORM が signal されなかった: ~S" ,unsupported-form))
     (nb:unsupported-form (c)
       (let ((,form-var (nb:unsupported-form-form c)) (,path-var (nb:unsupported-form-path c)))
         ,@body))))

(test walk/setq-signals-unsupported-form
  "本体に SETQ を含む WITH-TRACING は、マクロ展開時に UNSUPPORTED-FORM を
signal する。FORM は SETQ フォームそのもの、PATH は (1)（(PROGN (SETQ X 1))
の中で SETQ フォームが位置1にあるため）。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (setq x 1))
    (is (equal '(setq x 1) form))
    (is (equal '(1) path))))

(test walk/incf-signals-unsupported-form
  "INCF は SETQ に展開されるので、同じく UNSUPPORTED-FORM になる。展開後の
FORM は (SETQ X (+ 1 X))。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (incf x))
    (is (equal '(setq x (+ 1 x)) form))
    (is (equal '(1) path))))

(test walk/dotimes-signals-unsupported-form
  "DOTIMES は BLOCK/TAGBODY に展開されるので UNSUPPORTED-FORM になる（BLOCK
が対応していない特殊形式のため、展開結果の先頭 BLOCK で捕まる）。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (dotimes (i 3) x))
    (is (eq 'block (first form)))
    (is (equal '(1) path))))

(test walk/return-from-signals-unsupported-form
  "RETURN-FROM は対応していない（BLOCK 自体もそう）。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (block nil (return-from nil x)))
    (is (equal '(block nil (return-from nil x)) form))
    (is (equal '(1) path))))

(test walk/multiple-value-bind-is-walked-not-rejected
  "MULTIPLE-VALUE-BIND（SBCL が MULTIPLE-VALUE-CALL + (LAMBDA (&OPTIONAL ...)) に
展開する）は対応している: マクロ展開時に signal せず、本体の中の演算子も
書き換わる（% T-ADD になる）。"
  (let ((expansion (macroexpand-1 '(nb:with-tracing (x y) (multiple-value-bind (a b) (values x y) (+ a b))))))
    (is (search "%T-ADD" (prin1-to-string expansion)))))

(test walk/multiple-value-bind-body-is-still-checked
  "MULTIPLE-VALUE-BIND の値フォームにも本体にも、対応していない形式（SETQ）が
あれば従来どおり UNSUPPORTED-FORM になる。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x y) (multiple-value-bind (a b) (values x y) (setq a 1) b))
    (is (equal '(setq a 1) form)))
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x y) (multiple-value-bind (a b) (values x (setq y 1)) a))
    (is (equal '(setq y 1) form))))

(test walk/multiple-value-call-of-rewritten-cl-function-signals-unsupported-form
  "(MULTIPLE-VALUE-CALL #'+ ...) のように、書き換え表にある CL の関数を直接
渡す形は、実行時にトレーサへ CL:+ を適用して分かりにくく失敗するので、
マクロ展開時に UNSUPPORTED-FORM にする。FORM は MULTIPLE-VALUE-CALL フォーム全体。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x y) (multiple-value-call #'+ (values x y)))
    (is (eq 'multiple-value-call (first form)))
    (is (equal '(1) path))))

(test walk/multiple-value-call-without-function-form-signals-unsupported-form
  "関数フォームの無い (MULTIPLE-VALUE-CALL) は UNSUPPORTED-FORM。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) x (multiple-value-call))
    (is (equal '(multiple-value-call) form))
    (is (equal '(2) path))))

(test walk/multiple-value-prog1-is-walked
  "MULTIPLE-VALUE-PROG1 は対応している（全フォームを歩く）。中の SETQ は従来どおり
UNSUPPORTED-FORM。"
  (is (search "%T-ADD" (prin1-to-string
                        (macroexpand-1 '(nb:with-tracing (x y) (multiple-value-prog1 (+ x y) (+ x 1)))))))
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x y) (multiple-value-prog1 x (setq y 1)))
    (is (equal '(setq y 1) form))
    (is (equal '(1 2) path))))

(test walk/multiple-value-prog1-multiple-values-flow-through
  "MULTIPLE-VALUE-PROG1 は最初のフォームの多値をそのまま返し、残りは副作用だけ。"
  (let ((f (nb:with-tracing (x y) (multiple-value-prog1 (values x y) (+ x y)))))
    (is (equal '(1 2) (multiple-value-list (funcall f 1 2))))))

(test walk/unwind-protect-signals-unsupported-form
  "UNWIND-PROTECT はそのまま対応していない特殊形式。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (unwind-protect x (print 1)))
    (is (equal '(unwind-protect x (print 1)) form))
    (is (equal '(1) path))))

(test walk/optional-lambda-list-signals-unsupported-form
  "仮引数リストが &OPTIONAL などの lambda-list キーワードを含む場合、本体を
マクロ展開する前に UNSUPPORTED-FORM を signal する。FORM はそのまま
lambda-list、PATH は NIL。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x &optional y) x)
    (is (equal '(x &optional y) form))
    (is (null path))))

(test walk/nested-lambda-call-not-operator-signals-unsupported-form
  "呼び出しの演算子が (LAMBDA ...) でも symbol でもない場合（(FOO) を呼び出す
など）は UNSUPPORTED-FORM か、少なくとも with-tracing のマクロ展開が失敗する。
SB-CLTL2:MACROEXPAND-ALL 自身がこの形をエラーにするため、ここでは ERROR が
signal されることだけを確かめる。"
  (signals error (macroexpand-1 '(nb:with-tracing (x) ((foo) x)))))

;;; --- 対応していない形式は他にもある（性質「4種類以上」の確認）が、上の
;;; 6つで setq・incf・dotimes(block/tagbody)・return-from・
;;; multiple-value-bind・unwind-protect の6種類をカバーしている。

(test walk/unrelated-calls-are-left-alone
  "書き換え対象でないシンボルを演算子に持つ呼び出しはそのまま残り、ふつうの
Lisp として実行される。(LIST (IDENTITY X)) を X=1.0 で呼ぶと (1.0) になる。"
  (let ((f (nb:with-tracing (x) (list (identity x)))))
    (is (equal '(1.0) (funcall f 1.0)))))

(test walk/first-of-list-traces-with-no-eqns
  "(FIRST (LIST X)) は演算を1つも足さず、graph の outvar が invar そのもの
になる（0個の eqn）。"
  (let* ((f (nb:with-tracing (x) (first (list x))))
         (graph (nb:trace-to-graph f (list (nb:make-aval '(2) :f32)))))
    (is (null (nb:graph-eqns graph)))
    (is (eq (first (nb:graph-outvars graph)) (first (nb:graph-invars graph))))))

(test walk/function-quote-plus-is-left-alone
  "#'+ はシンボルなので歩かれず、CL の + のまま残る。(REDUCE #'+ (LIST 1 2))
は普通の Lisp として eager に評価され 3 になる。"
  (let ((f (nb:with-tracing () (reduce #'+ (list 1 2)))))
    (is (= 3 (funcall f)))))

(test walk/funcall-lambda-is-walked
  "(FUNCALL (LAMBDA (Y) ...) X) の中の (LAMBDA ...) の本体も歩かれ、書き換え
対象の演算子（+ など）はトレース対象に書き換わる。X=2.0, Y=2.0+1=3.0 の
eager 実行で確かめる。"
  (let ((f (nb:with-tracing (x) (funcall (lambda (y) (+ y 1)) x))))
    (is (= 3.0 (funcall f 2.0)))))

(test walk/setq-inside-nested-lambda-signals-unsupported-form
  "(FUNCALL (LAMBDA (Y) (SETQ Y 1)) X) のように、対応していない特殊形式が
ネストした LAMBDA の中にあっても UNSUPPORTED-FORM になる。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x) (funcall (lambda (y) (setq y 1)) x))
    (is (equal '(setq y 1) form))
    (is (equal '(1 1 2) path))))

;;; --- path の付け方そのものをピン留めする（mutation testing 対策:
;;; %PATH-EXTEND や各 %WALK-* の位置決めがずれると、ここが落ちる） ---

(test walk/path-indexes-progn-body-by-position
  "本体が複数フォームからなるとき、N番目（0始まり、PROGN のオペレータ自身
を位置0として数える）の UNSUPPORTED な本体フォームの path は (N)。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) x (setq x 1))
    (declare (ignore form))
    (is (equal '(2) path))))

(test walk/path-descends-into-let-binding-init-form
  "LET のバインディングの初期値フォームの中に UNSUPPORTED な形式があると、
path はそこまで降りる: LET 自体が位置1、bindings リストが位置1、1番目の
バインディングが位置0、その初期値が位置1 -> (1 1 0 1)。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x) (let ((y (block nil x))) y))
    (is (equal '(block nil x) form))
    (is (equal '(1 1 0 1) path))))

(test walk/path-descends-into-rewritten-call-argument
  "書き換え対象の呼び出し（+ など）の引数の中に UNSUPPORTED な形式が
あっても、その引数を歩く（%WALK-CALL の書き換え分岐が %WALK-ARGS の
BASE-INDEX を正しく1にしていることを確かめる。ずれると引数の位置が
1つずれる）。(+ X (SETQ Y 1)) の2番目の引数（位置2）で捕まる -> (1 2)。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x y) (+ x (setq y 1)))
    (is (equal '(setq y 1) form))
    (is (equal '(1 2) path))))

(test walk/function-lambda-body-is-walked
  "(FUNCTION (LAMBDA ...)) の LAMBDA の本体も歩く。UNSUPPORTED な形式は
LAMBDA の本体（位置2）の中、LAMBDA 自体は FUNCTION の位置1にある ->
(1 1 2)。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x) (function (lambda (y) (setq y 1))))
    (is (equal '(setq y 1) form))
    (is (equal '(1 1 2) path))))

(test walk/lambda-call-operator-body-is-walked
  "呼び出しの演算子自体が (LAMBDA ...) のとき（即時適用）、その本体も歩く。
LAMBDA は呼び出しの位置0にある -> (1 0 2)。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x) ((lambda (y) (setq y 1)) x))
    (is (equal '(setq y 1) form))
    (is (equal '(1 0 2) path))))

(test walk/flet-single-binding-body-is-walked
  "FLET のローカル関数の本体も歩く。FLET 自体が位置1、bindings リストが
位置1、1番目（唯一の）バインディングが位置0、その本体（位置2）に UNSUPPORTED
な形式がある -> (1 1 0 2)。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x) (flet ((f (y) (setq y 1))) (f x)))
    (is (equal '(setq y 1) form))
    (is (equal '(1 1 0 2) path))))

(test walk/flet-second-binding-index-is-pinned
  "FLET に2つのバインディングがあるとき、2番目（位置1）の本体で捕まった
UNSUPPORTED な形式の path はそのバインディングの位置を反映する -> (1 1 1 2)
（バインディングを数える起点がずれると、1番目の (1 1 0 2) と混同される）。"
  (%signals-unsupported-form (form path)
      '(nb:with-tracing (x) (flet ((f (y) y) (g (z) (setq z 1))) (g x)))
    (is (equal '(setq z 1) form))
    (is (equal '(1 1 1 2) path))))

(test walk/flet-supported-form-works
  "FLET 自体は対応する特殊形式で、ローカル関数はふつうに呼べる（トレース
対象の演算子もローカル関数の中で書き換わる）。"
  (let ((f (eval '(nb:with-tracing (x) (flet ((sq (y) (* y y))) (sq x))))))
    (is (= 9.0 (funcall f 3.0)))))

;;; --- + / * の0引数の恒等元、および算術・比較の書き換え関数の arity
;;; チェックをピン留めする（mutation testing: 定数置換・境界の変異対策） ---

(test walk/plus-with-no-args-expands-to-zero-literal
  "(+) は展開結果の本体そのものが 0 になる（%T-ADD を1度も呼ばない）。"
  (is (equal '(nabla::%make-traceable-function 'nil (lambda () (progn 0)))
             (macroexpand-1 '(nb:with-tracing () (+))))))

(test walk/times-with-no-args-expands-to-one-literal
  "(*) は展開結果の本体そのものが 1 になる。"
  (is (equal '(nabla::%make-traceable-function 'nil (lambda () (progn 1)))
             (macroexpand-1 '(nb:with-tracing () (*))))))

(defmacro %def-nullary-rewrite-unsupported-test (test-name form)
  "FORM（0引数の算術・比較呼び出し）が UNSUPPORTED-FORM になり、FORM 自身が
そのまま報告されることを確かめるテストを定義する。"
  `(test ,test-name
     ,(format nil "~S は0引数では書き換えられず UNSUPPORTED-FORM になる。" form)
     (%signals-unsupported-form (f path) (list 'nb:with-tracing '() ',form)
       (is (equal ',form f))
       (is (equal '(1) path)))))

(%def-nullary-rewrite-unsupported-test walk/nullary-minus-signals-unsupported-form (-))
(%def-nullary-rewrite-unsupported-test walk/nullary-divide-signals-unsupported-form (/))
(%def-nullary-rewrite-unsupported-test walk/nullary-max-signals-unsupported-form (max))
(%def-nullary-rewrite-unsupported-test walk/nullary-min-signals-unsupported-form (min))
(%def-nullary-rewrite-unsupported-test walk/nullary-exp-signals-unsupported-form (exp))
(%def-nullary-rewrite-unsupported-test walk/nullary-tanh-signals-unsupported-form (tanh))
(%def-nullary-rewrite-unsupported-test walk/nullary-log-signals-unsupported-form (log))
(%def-nullary-rewrite-unsupported-test walk/nullary-1+-signals-unsupported-form (1+))
(%def-nullary-rewrite-unsupported-test walk/nullary-1--signals-unsupported-form (1-))

(test walk/binary-exp-signals-unsupported-form
  "EXP はちょうど1引数でなければ UNSUPPORTED-FORM になる（2引数の例）。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x y) (exp x y))
    (is (equal '(exp x y) form))
    (is (equal '(1) path))))

(test walk/binary-log-signals-unsupported-form
  "LOG は2引数（底の指定）を対応しない。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x y) (log x y))
    (is (equal '(log x y) form))
    (is (equal '(1) path))))

(test walk/unary-compare-signals-unsupported-form
  "< はちょうど2引数でなければ UNSUPPORTED-FORM になる（1引数の例）。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (< x))
    (is (equal '(< x) form))
    (is (equal '(1) path))))

(test walk/ternary-compare-signals-unsupported-form
  "< に3引数を渡しても UNSUPPORTED-FORM になる。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x y z) (< x y z))
    (is (equal '(< x y z) form))
    (is (equal '(1) path))))

(test walk/nullary-compare-signals-unsupported-form
  "< に0引数を渡しても UNSUPPORTED-FORM になる（1引数・3引数の場合と違い、
0引数はちょうど2引数チェックの境界そのものを踏む。境界の変異対策）。"
  (%signals-unsupported-form (form path) '(nb:with-tracing () (<))
    (is (equal '(<) form))
    (is (equal '(1) path))))

;;; --- IF / LOCALLY / THE のピン留め（mutation testing 対策） ---

(test walk/if-with-else-uses-else-branch-when-test-false
  "(IF TEST THEN ELSE)（長さちょうど4）は TEST が偽なら ELSE を評価する。
HAS-ELSE-P の境界判定 (>= (LENGTH FORM) 4) が (> ...) に変異すると、長さ
ちょうど4の IF で ELSE が使われなくなり、常に NIL を返すようになる。"
  (let ((f (eval '(nb:with-tracing (p) (if p 1.0 2.0)))))
    (is (= 1.0 (funcall f t)))
    (is (= 2.0 (funcall f nil)))))

(test walk/locally-body-position-is-pinned
  "LOCALLY の本体フォームは LOCALLY 自体を基準に位置1から始まる。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (locally (setq x 1)))
    (is (equal '(setq x 1) form))
    (is (equal '(1 1) path))))

(test walk/unary-divide-computes-reciprocal
  "(/ X)（単項）は 1/X を計算する（(%T-DIV 1 X) に書き換わる。定数 1 が
0 に化けていないことを、実際に呼び出して確かめる。トレース対象の
WITH-TRACING は EVAL 経由で作り、マクロ展開が実行時に起きるようにする
（そうしないと、コンパイル時に埋め込まれた展開結果が、%REWRITE-DIV の
変異を検出できない）。"
  (let ((f (eval '(nb:with-tracing (x) (/ x)))))
    (is (= 0.25 (funcall f 4.0)))))

(test walk/the-drops-type-and-walks-value
  "(THE type value) は type を捨てて value を歩く。UNSUPPORTED な value の
path は THE 自体を基準に位置2。"
  (%signals-unsupported-form (form path) '(nb:with-tracing (x) (the single-float (setq x 1.0)))
    (is (equal '(setq x 1.0) form))
    (is (equal '(1 2) path))))
