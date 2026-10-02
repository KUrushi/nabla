;;;; walk: WITH-TRACING が本体を歩くコードウォーカ（issue #32、t1）。
;;;;
;;;; 手順: (1) 仮引数リストが必須引数だけであることを確かめる、(2)
;;;; SB-CLTL2:MACROEXPAND-ALL で本体をマクロ展開する（COND/WHEN/UNLESS/
;;;; INCF/DOTIMES などのマクロは、この時点で IF/SETQ/BLOCK/TAGBODY などの
;;;; 基本形に展開される）、(3) %WALK でその展開結果を歩く。
;;;;
;;;; 対応する形式（マクロ展開後）: LET / LET* / PROGN / IF / FUNCTION /
;;;; FLET / LABELS / THE / SB-EXT:TRULY-THE / LOCALLY / QUOTE / アトム、
;;;; MULTIPLE-VALUE-CALL / MULTIPLE-VALUE-PROG1（MULTIPLE-VALUE-BIND と
;;;; MULTIPLE-VALUE-LIST は SBCL がこれらに展開する。issue #115）、
;;;; および CL の算術・比較関数への呼び出し（下の *REWRITE-TABLE*）と、
;;;; それ以外のシンボルを演算子に持つ通常の関数呼び出し（引数だけ歩き、
;;;; 呼び出しそのものはそのまま残す。documented: ふつうの Lisp として実行
;;;; されるので、トレーサを渡すとその関数の中で失敗する）。それ以外の
;;;; 特殊形式（SETQ・BLOCK・TAGBODY・…）は UNSUPPORTED-FORM を signal する。

(in-package #:nabla)

(define-condition unsupported-form (error)
  ((form :initarg :form :reader unsupported-form-form)
   (path :initarg :path :reader unsupported-form-path))
  (:report
   (lambda (condition stream)
     (format stream "with-tracing: 対応していない形式 ~S（path ~S）"
             (unsupported-form-form condition) (unsupported-form-path condition))))
  (:documentation
   "WITH-TRACING の本体（SB-CLTL2:MACROEXPAND-ALL で展開したあと）に、歩けない
形式が現れたときに、マクロ展開時に signal する。FORM は展開後の問題の
フォーム、PATH は展開した (PROGN ,@BODY) を根とする、そこへ至る0始まりの
要素インデックスのリスト（() は根そのもの）。LAMBDA-LIST が必須引数以外
（&OPTIONAL 等）を含む場合は FORM がその LAMBDA-LIST、PATH が NIL になる。"))

(defun %unsupported (form path)
  (error 'unsupported-form :form form :path path))

(defun %path-extend (path index)
  "PATH の末尾に INDEX を1つ足した新しい path を返す。INDEX は、いま
歩いている（リストである）フォームの中での0始まりの位置（先頭のシンボル
自身も位置0として数える。%WALK-IF 等のドキュメントで具体例を pin する）。"
  (append path (list index)))

(defun %declare-form-p (form)
  "FORM が (DECLARE ...) なら真を返す。LET・LOCALLY・LAMBDA 本体の先頭に
現れる宣言は、歩かずにそのまま残す。"
  (and (consp form) (eq (first form) 'declare)))

(defun %walk-body-with-declares (forms base-index path)
  "FORMS（本体を構成するフォームのリスト）を歩く。DECLARE はそのまま残し、
それ以外は %WALK する。BASE-INDEX は FORMS の先頭が親フォームの中で占める
位置（%PATH-EXTEND に渡す）。"
  (loop for form in forms
        for i from base-index
        collect (if (%declare-form-p form) form (%walk form (%path-extend path i)))))

(defun %walk-args (args base-index path)
  "ARGS（関数呼び出しの引数のリスト）を、それぞれ %WALK して返す。"
  (loop for arg in args
        for i from base-index
        collect (%walk arg (%path-extend path i))))

;;; --- 各特殊形式の WALK ---

(defun %walk-if (form path)
  "(IF TEST THEN [ELSE]) を (%T-IF <test> (LAMBDA () <then>) (LAMBDA () <else>))
に変換する。ELSE を省略した IF は NIL を ELSE の代わりに使う
（ELSE-THUNK を呼ばれても NIL を返す、歩く必要のない値）。"
  (let* ((test (second form))
         (then (third form))
         (has-else-p (>= (length form) 4))
         (else (fourth form)))
    (list '%t-if
          (%walk test (%path-extend path 1))
          (list 'lambda '() (%walk then (%path-extend path 2)))
          (list 'lambda '() (if has-else-p (%walk else (%path-extend path 3)) nil)))))

(defun %walk-progn (form path)
  (list* 'progn (%walk-body-with-declares (rest form) 1 path)))

(defun %walk-let-binding (binding path)
  "1つの LET/LET* バインディング（symbol、または (var init-form)）を歩く。
symbol 単体（init が省略された束縛）はそのまま返す。"
  (if (consp binding)
      (list (first binding) (%walk (second binding) (%path-extend path 1)))
      binding))

(defun %walk-let-bindings (bindings path)
  "BINDINGS 全体（LET/LET* の第1引数、親フォームの中で位置1にある）を歩く。"
  (loop for binding in bindings
        for i from 0
        collect (%walk-let-binding binding (%path-extend path i))))

(defun %walk-let (form path)
  (destructuring-bind (op bindings &rest body) form
    (list* op
           (%walk-let-bindings bindings (%path-extend path 1))
           (%walk-body-with-declares body 2 path))))

(defun %walk-locally (form path)
  (list* 'locally (%walk-body-with-declares (rest form) 1 path)))

(defun %walk-the (form path)
  "(THE type value) / (SB-EXT:TRULY-THE type value) は type を捨て、歩いた
value に置き換える（型注釈はトレース対象の値には意味を持たないため）。"
  (%walk (third form) (%path-extend path 2)))

(defun %walk-lambda-form (lambda-form path)
  "(LAMBDA lambda-list . body) の本体を歩く（lambda-list はそのまま）。"
  (destructuring-bind (op lambda-list &rest body) lambda-form
    (list* op lambda-list (%walk-body-with-declares body 2 path))))

(defun %walk-function (form path)
  "(FUNCTION x)。x がシンボルならそのまま（#'+ は CL の + のまま。
documented limitation: これを N 引数の関数として渡す先で使うと、シンボルの
指す CL の関数がそのまま呼ばれ、トレーサに対しては動かない）。x が
(LAMBDA ...) なら、その本体を歩く。"
  (let ((x (second form)))
    (if (and (consp x) (eq (first x) 'lambda))
        (list 'function (%walk-lambda-form x (%path-extend path 1)))
        form)))

(defun %walk-flet-binding (binding path)
  (destructuring-bind (name lambda-list &rest body) binding
    (list* name lambda-list (%walk-body-with-declares body 2 path))))

(defun %walk-flet-bindings (bindings path)
  (loop for binding in bindings
        for i from 0
        collect (%walk-flet-binding binding (%path-extend path i))))

(defun %walk-flet (form path)
  (destructuring-bind (op bindings &rest body) form
    (list* op
           (%walk-flet-bindings bindings (%path-extend path 1))
           (%walk-body-with-declares body 2 path))))

(defun %walk-lambda-call (form path)
  "呼び出しの演算子自体が (LAMBDA ...) の call（((LAMBDA (X) X) 1) のような
即時適用）。演算子と引数の両方を歩く。"
  (list* (%walk-lambda-form (first form) (%path-extend path 0))
         (%walk-args (rest form) 1 path)))

;;; --- 多値（issue #115） ---
;;;
;;; 歩いたあとのコードは、トレース中もふつうの Lisp として実行される。多値は
;;; CL 自身の多値で、トレーサは単なるオブジェクトとして多値の各要素になる
;;; だけなので、MULTIPLE-VALUE-CALL / MULTIPLE-VALUE-PROG1 は全サブフォームを
;;; 歩いて形をそのまま残せばよい（トレーサ以外の実数・配列の多値も同じ扱い）。
;;; 唯一の落とし穴は関数フォームに書き換え表にある CL の関数（#'+ など）を
;;; 直接渡す形で、CL:+ がトレーサに適用されて実行時に分かりにくく失敗するので、
;;; 展開時に UNSUPPORTED-FORM にする。

(defun %walk-multiple-value-call (form path)
  "(MULTIPLE-VALUE-CALL fn form...) の fn と各 form を歩く。fn が書き換え表に
ある CL の関数への (FUNCTION sym) なら、関数フォームが無い形とともに
UNSUPPORTED-FORM。"
  (let ((fn-form (second form)))
    (when (or (null (rest form))
              (and (consp fn-form) (eq (first fn-form) 'function)
                   (symbolp (second fn-form)) (%find-rewriter (second fn-form))))
      (%unsupported form path))
    (list* 'multiple-value-call (%walk-args (rest form) 1 path))))

(defun %walk-multiple-value-prog1 (form path)
  "(MULTIPLE-VALUE-PROG1 first-form form...) の全フォームを歩く。"
  (when (null (rest form)) (%unsupported form path))
  (list* 'multiple-value-prog1 (%walk-args (rest form) 1 path)))

(defparameter *unsupported-operators*
  '(setq block return-from tagbody go catch throw unwind-protect
    progv eval-when
    load-time-value symbol-macrolet macrolet)
  "MACROEXPAND-ALL のあとも残りうる、WITH-TRACING が対応しない特殊形式の
演算子シンボルのリスト。")

(defun %unsupported-operator-p (op)
  (member op *unsupported-operators*))

;;; --- CL の算術・比較関数をトレース対象の総称関数に書き換える表 ---
;;;
;;; 各エントリの値は (WALKED-ARGS FORM PATH) を受け取り、書き換え後の
;;; フォーム（すでに歩き終えたサブフォームからなる）を返す関数。

(defun %fold-with-identity (op-symbol args identity)
  "ARGS（コード形式のリスト）を OP-SYMBOL で左畳み込みする: 0個なら
IDENTITY、1個ならその要素そのもの、2個以上なら
(OP-SYMBOL (OP-SYMBOL a0 a1) a2 ...) のようにネストしたフォームを作る。"
  (cond
    ((null args) identity)
    ((null (rest args)) (first args))
    (t (reduce (lambda (acc arg) (list op-symbol acc arg)) args))))

(defun %fold-no-identity (op-symbol args)
  "ARGS を OP-SYMBOL で左畳み込みする（0個の恒等値を持たない演算 - / 用。
ARGS は1個以上を前提とする）。"
  (if (null (rest args)) (first args) (reduce (lambda (acc arg) (list op-symbol acc arg)) args)))

(defun %rewrite-add (args form path)
  (declare (ignore form path))
  (%fold-with-identity '%t-add args 0))

(defun %rewrite-mul (args form path)
  (declare (ignore form path))
  (%fold-with-identity '%t-mul args 1))

(defun %rewrite-sub (args form path)
  (cond
    ((null args) (%unsupported form path))
    ((null (rest args)) (list '%t-neg (first args)))
    (t (%fold-no-identity '%t-sub args))))

(defun %rewrite-div (args form path)
  (cond
    ((null args) (%unsupported form path))
    ((null (rest args)) (list '%t-div 1 (first args)))
    (t (%fold-no-identity '%t-div args))))

(defun %rewrite-max (args form path)
  (if (null args) (%unsupported form path) (%fold-no-identity '%t-max args)))

(defun %rewrite-min (args form path)
  (if (null args) (%unsupported form path) (%fold-no-identity '%t-min args)))

(defun %rewrite-unary-1 (op-symbol args form path)
  (if (= 1 (length args)) (list op-symbol (first args)) (%unsupported form path)))

(defun %rewrite-exp (args form path) (%rewrite-unary-1 '%t-exp args form path))
(defun %rewrite-tanh (args form path) (%rewrite-unary-1 '%t-tanh args form path))
(defun %rewrite-log (args form path) (%rewrite-unary-1 '%t-log args form path))

(defun %rewrite-1+ (args form path)
  (if (= 1 (length args)) (list '%t-add (first args) 1) (%unsupported form path)))

(defun %rewrite-1- (args form path)
  (if (= 1 (length args)) (list '%t-sub (first args) 1) (%unsupported form path)))

(defparameter *compare-directions*
  '((< . :lt) (<= . :le) (> . :gt) (>= . :ge) (= . :eq) (/= . :ne))
  "CL の比較関数シンボルと、対応する %T-COMPARE の DIRECTION キーワードの対。")

(defun %compare-direction-for (op)
  (cdr (assoc op *compare-directions*)))

(defun %rewrite-compare (op args form path)
  (if (= 2 (length args))
      (list '%t-compare (first args) (second args) (%compare-direction-for op))
      (%unsupported form path)))

(defparameter *rewrite-table*
  (list (cons '+ '%rewrite-add)
        (cons '* '%rewrite-mul)
        (cons '- '%rewrite-sub)
        (cons '/ '%rewrite-div)
        (cons 'max '%rewrite-max)
        (cons 'min '%rewrite-min)
        (cons 'exp '%rewrite-exp)
        (cons 'tanh '%rewrite-tanh)
        (cons 'log '%rewrite-log)
        (cons '1+ '%rewrite-1+)
        (cons '1- '%rewrite-1-)
        (cons '< (lambda (args form path) (%rewrite-compare '< args form path)))
        (cons '<= (lambda (args form path) (%rewrite-compare '<= args form path)))
        (cons '> (lambda (args form path) (%rewrite-compare '> args form path)))
        (cons '>= (lambda (args form path) (%rewrite-compare '>= args form path)))
        (cons '= (lambda (args form path) (%rewrite-compare '= args form path)))
        (cons '/= (lambda (args form path) (%rewrite-compare '/= args form path))))
  "書き換え対象の CL シンボルから、書き換え関数への対応表。書き換え関数は
(WALKED-ARGS FORM PATH) を取り、書き換え後のフォームを返す。値はシンボル
（+ - * / max min exp tanh log 1+ 1- の各 %REWRITE-* 関数名）か、比較演算子
用のクロージャ。SYMBOL-FUNCTION ではなく SYMBOL そのものを持たせているのは、
FUNCALL に渡す SYMBOL は呼ぶたびに現在のグローバル定義を引く（再定義の
影響を受ける）のに対し、#'NAME はこの DEFPARAMETER が評価された時点の
関数オブジェクトを一度だけ捕まえてしまい、あとから (DEFUN %REWRITE-ADD ...)
を再評価してもこの表からは呼ばれなくなる（mutation testing の runner が
DEFUN を再評価して変異体を試す際にまさにこれを踏む）ため。")

(defun %find-rewriter (op)
  (cdr (assoc op *rewrite-table*)))

(defun %walk-call (form path)
  (let* ((op (first form))
         (raw-args (rest form))
         (rewriter (%find-rewriter op)))
    (if rewriter
        (funcall rewriter (%walk-args raw-args 1 path) form path)
        (list* op (%walk-args raw-args 1 path)))))

;;; --- ディスパッチ本体 ---

(defun %walk (form path)
  (cond
    ((atom form) form)
    ((eq (first form) 'quote) form)
    ((eq (first form) 'if) (%walk-if form path))
    ((eq (first form) 'progn) (%walk-progn form path))
    ((member (first form) '(let let*)) (%walk-let form path))
    ((eq (first form) 'locally) (%walk-locally form path))
    ((member (first form) '(the sb-ext:truly-the)) (%walk-the form path))
    ((eq (first form) 'function) (%walk-function form path))
    ((eq (first form) 'lambda) (%walk-lambda-form form path))
    ((member (first form) '(flet labels)) (%walk-flet form path))
    ((eq (first form) 'multiple-value-call) (%walk-multiple-value-call form path))
    ((eq (first form) 'multiple-value-prog1) (%walk-multiple-value-prog1 form path))
    ((%unsupported-operator-p (first form)) (%unsupported form path))
    ((and (consp (first form)) (eq (first (first form)) 'lambda)) (%walk-lambda-call form path))
    ((symbolp (first form)) (%walk-call form path))
    (t (%unsupported form path))))

;;; --- WITH-TRACING ---

(defun %check-tracing-lambda-list (lambda-list)
  "LAMBDA-LIST が必須引数のシンボルだけからなることを確かめる。
&OPTIONAL 等の lambda-list キーワードや、シンボルでない要素が混ざって
いれば UNSUPPORTED-FORM を signal する（FORM はそのまま LAMBDA-LIST、
PATH は NIL）。"
  (unless (every (lambda (x) (and (symbolp x) (not (member x lambda-list-keywords)))) lambda-list)
    (%unsupported lambda-list nil)))

(defmacro with-tracing ((&rest lambda-list) &body body &environment env)
  "BODY を LAMBDA-LIST を仮引数とする関数として（コードウォークして）
トレース対象にする。返り値は TRACEABLE-FUNCTION（呼び出すと eager に実行
する。TRACE-TO-GRAPH に渡すとトレースして GRAPH にする）。

手順: (1) LAMBDA-LIST が必須引数のシンボルだけであることを確かめる、(2)
(SB-CLTL2:MACROEXPAND-ALL `(PROGN ,@BODY) ENV) で本体をマクロ展開する、(3)
%WALK でその展開結果を歩く（対応していない形式は UNSUPPORTED-FORM を
マクロ展開時に signal する）。CL の + - * / max min exp log tanh 1+ 1-
< <= > >= = /= は、この時点でトレース対象の内部ジェネリック（%T-ADD 等）
への呼び出しに書き換わる。それ以外のシンボルを演算子に持つ呼び出しは
そのまま残す（ふつうの Lisp として実行される。トレーサを渡すと、その
関数の実装が対応していない限り失敗する）。"
  (%check-tracing-lambda-list lambda-list)
  (let* ((expanded (sb-cltl2:macroexpand-all `(progn ,@body) env))
         (walked (%walk expanded '())))
    `(%make-traceable-function ',lambda-list (lambda ,lambda-list ,walked))))
