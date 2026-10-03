;;;; primitive: 演算（プリミティブ）の宣言と登録（issue #29 前半）。
;;;;
;;;; プリミティブは形状推論（abstract-eval）・StableHLO 出力（emit）・
;;;; eager 用の CPU 実装（eager）の3つを束ねた PRIMITIVE 構造体として
;;;; DEFPRIMITIVE で登録する。jvp / transpose のルールは可変スロットで、
;;;; DEFPRIMITIVE の任意キー :JVP / :TRANSPOSE か、後から DEF-JVP-RULE /
;;;; DEF-TRANSPOSE-RULE（src/ad/rules.lisp）で設定する。batch のルール（vmap）は
;;;; :BATCH か DEF-BATCH-RULE（src/vmap.lisp）で設定する。

(in-package #:nabla)

(defstruct (primitive (:constructor %make-primitive) (:copier nil) (:predicate primitive-p))
  "1つの演算（プリミティブ）を表す。NAME は :ADD のようなキーワード、
PARAMS は宣言順に並んだパラメタ名（キーワード）のリスト。ABSTRACT-EVAL /
EMIT / EAGER の呼び出し規約は DEFPRIMITIVE の docstring を見る。JVP と
TRANSPOSE は自動微分のルール、BATCH は vmap のバッチ化ルール（どれも無ければ
NIL）で、他のスロットと違って後から設定できる。呼び出し規約は
src/ad/rules.lisp の DEF-JVP-RULE / DEF-TRANSPOSE-RULE と、src/vmap.lisp の
DEF-BATCH-RULE を見る。MULTIPLE-OUTPUT-P が真のプリミティブは複数の
出力を持てる（契約 C1。呼び出し規約が単一出力と変わる点は DEFPRIMITIVE の
docstring を見る）。"
  (name nil :type keyword :read-only t)
  (params nil :type list :read-only t)
  (multiple-outputs-p nil :read-only t)
  (abstract-eval nil :type function :read-only t)
  (emit nil :type (or null function) :read-only t)
  (eager nil :type (or null function) :read-only t)
  (jvp nil :type (or null function))
  (transpose nil :type (or null function))
  ;; バッチ化ルール（vmap。DEF-BATCH-RULE、src/vmap.lisp）。常に最後のスロット
  (batch nil :type (or null function)))

(defvar *primitives* (make-hash-table :test 'eq)
  "プリミティブ名（キーワード）から PRIMITIVE への表。DEFPRIMITIVE の
再評価は既存のエントリを新しい PRIMITIVE で置き換える。")

(defun register-primitive (primitive)
  "PRIMITIVE を *PRIMITIVES* に（既存の同名エントリを上書きして）登録し、
PRIMITIVE をそのまま返す。"
  (setf (gethash (primitive-name primitive) *primitives*) primitive))

(defun find-primitive (name)
  "NAME（キーワード）に対応する PRIMITIVE を返す。登録が無ければ NIL。"
  (gethash name *primitives*))

(define-condition unknown-primitive (error)
  ((name :initarg :name :reader unknown-primitive-name))
  (:report
   (lambda (condition stream)
     (format stream "未登録のプリミティブ: ~S" (unknown-primitive-name condition))))
  (:documentation
   "MAKE-EQN に、登録されていない名前を渡したときに signal される。NAME は
渡された名前（キーワード）。"))

(define-condition primitive-error (error)
  ((name :initarg :name :reader primitive-error-name)
   (in-avals :initarg :in-avals :initform nil :reader primitive-error-in-avals)
   (format-control :initarg :format-control :reader primitive-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader primitive-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "プリミティブ ~S: ~?"
             (primitive-error-name condition)
             (primitive-error-format-control condition)
             (primitive-error-format-arguments condition))))
  (:documentation
   "プリミティブの abstract-eval が入力の shape / dtype の不一致を検出した
とき、または MAKE-EQN が渡された params の不正（未知キー・欠落・余分）を
検出したときに signal される。NAME はプリミティブ名、IN-AVALS は入力の
AVAL のリスト（分からなければ NIL）。"))

(defun %existing-rule (name accessor)
  "NAME の既に登録されている PRIMITIVE から ACCESSOR（PRIMITIVE-JVP /
PRIMITIVE-TRANSPOSE / PRIMITIVE-BATCH）でルールを取り出す。未登録なら NIL。DEFPRIMITIVE の
再評価がルールを引き継ぐために使う。"
  (let ((old (find-primitive name)))
    (and old (funcall accessor old))))

(defun %check-param-keywords (name param-keywords)
  (dolist (k param-keywords)
    (unless (keywordp k)
      (error "DEFPRIMITIVE ~S: パラメタ ~S はキーワードでなければならない" name k))))

(defmacro defprimitive (name (&rest param-keywords) &key multiple-outputs abstract-eval emit eager jvp transpose batch)
  "NAME（シンボル）を名前に持つプリミティブを宣言し、
*PRIMITIVES* に登録する。登録名は (INTERN (SYMBOL-NAME NAME) :KEYWORD)。

PARAM-KEYWORDS はこのプリミティブが受け取るパラメタ名を宣言順に並べた
キーワードのリスト（キーワード以外を渡すとマクロ展開時にエラーになる）。

:ABSTRACT-EVAL は必須で、
  (lambda (in-avals &key <params>) ...) → aval
という形の関数。入力 AVAL のリストとパラメタから出力 AVAL を計算する
（形状・dtype の不一致は PRIMITIVE-ERROR を signal する）。

:EMIT は省略でき、
  (lambda (in-names in-avals out-name out-aval &key <params>) ...) → string
という形の関数。1つの MLIR 演算を表す文字列（複数行でもよい）を返す。
`loc(...)` は付けない。

:EAGER は省略でき、
  (lambda (arrays in-avals &key <params>) ...) → simple-array
という形の関数。CPU 上で即時に評価する。

:JVP / :TRANSPOSE は省略でき、自動微分のルール関数（呼び出し規約は
DEF-JVP-RULE / DEF-TRANSPOSE-RULE の docstring）。省略すると NIL で、後から
DEF-JVP-RULE / DEF-TRANSPOSE-RULE で設定できる。:BATCH はバッチ化ルール（vmap。
呼び出し規約は DEF-BATCH-RULE の docstring。常に最後のキー）。

:MULTIPLE-OUTPUTS（評価されない真偽値。既定 NIL）が真のプリミティブは、
出力の個数ではなくこのフラグで複数出力の規約に切り替わる（契約 C1）:
  - :ABSTRACT-EVAL は AVAL の「リスト」を返す。
  - :EMIT は (lambda (in-names in-avals out-names out-avals &key <params>) ...)
    で、出力の名前と AVAL を「リスト」で受け取り、\"%8, %9 = ...\" のような
    左辺を自分で書く。
  - :EAGER は配列の「リスト」を返す。
  - :JVP は (primals tangents &key <params>) → (VALUES 主値の出力のリスト 接線のリスト)。
    jvp-graph は主値の eqn を事前に足さず、ルールが自分で足す（足さないと、
    while / scan / cond のような高階プリミティブが2回走る）。全入力の接線が
    ゼロのときだけ、ルールを呼ばず主値を再発行する。
    :TRANSPOSE は (cts invars &key <params>) で、CTS は余接線のリスト
    （SYMBOLIC-ZERO 可）、返り値は invar ごとのリスト。
  - トレースには %TRACE-EQN ではなく %TRACE-EQN*（常にトレーサのリストを返す）
    を使う。
既存の（単一出力の）プリミティブはこのフラグを付けず、何も変わらない。

このマクロは NAME のキーワードを評価値として返す。再評価は登録を
新しい PRIMITIVE 構造体で置き換える（EQ ではなくなる）。ただし jvp /
transpose のルールは引き継ぐ: :JVP / :TRANSPOSE を明示しなければ、古い
PRIMITIVE のルール（DEF-JVP-RULE などで後から設定したものを含む）がそのまま
新しい PRIMITIVE に載る（batch も同じ）。明示すればそちらで上書きする（NIL で消すことは
できない）。"
  (%check-param-keywords name param-keywords)
  (unless abstract-eval
    (error "DEFPRIMITIVE ~S: :ABSTRACT-EVAL は必須" name))
  (let ((keyword (intern (symbol-name name) :keyword)))
    `(progn
       (register-primitive
        (%make-primitive :name ,keyword
                          :params ',param-keywords
                          :multiple-outputs-p ,(and multiple-outputs t)
                          :abstract-eval ,abstract-eval
                          :emit ,emit
                          :eager ,eager
                          :jvp (or ,jvp (%existing-rule ,keyword #'primitive-jvp))
                          :transpose (or ,transpose (%existing-rule ,keyword #'primitive-transpose))
                          :batch (or ,batch (%existing-rule ,keyword #'primitive-batch))))
       ,keyword)))

(defun dtype-mlir-name (dtype)
  "DTYPE に対応する StableHLO/MLIR の要素型の綴りを返す
（\"f32\" \"f64\" \"bf16\" \"f16\" \"i1\"）。"
  (ecase dtype
    (:f32 "f32")
    (:f64 "f64")
    (:bf16 "bf16")
    (:f16 "f16")
    (:i1 "i1")))

(defun tensor-type-string (aval)
  "AVAL を表す MLIR のテンソル型の文字列を返す（\"tensor<2x3xf32>\"）。
rank 0 は \"tensor<f32>\"。"
  (let ((shape (aval-shape aval)))
    (format nil "tensor<~{~D~^x~}~:[~;x~]~A>"
            shape (not (null shape)) (dtype-mlir-name (aval-dtype aval)))))
