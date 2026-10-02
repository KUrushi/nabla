;;;; rules: プリミティブの jvp / transpose ルールを後から設定するマクロと、
;;;; ルールが無いときのコンディション（issue #77、77a）。
;;;;
;;;; ルールは PRIMITIVE 構造体の可変スロット（primitive-jvp /
;;;; primitive-transpose）に入る関数。配列ではなく、現在のトレースに eqn を
;;;; 足すコード（トレーサへの内部演算や %TRACE-EQN）として書く。こうすると
;;;; grad の grad や将来の vmap の grad がそのまま動く。

(in-package #:nabla)

(define-condition autodiff-error (error)
  ((format-control :initarg :format-control :initform "" :reader autodiff-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil
                     :reader autodiff-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "自動微分エラー: ~?"
             (autodiff-error-format-control condition)
             (autodiff-error-format-arguments condition))))
  (:documentation
   "自動微分の変換（jvp / transpose など）が続けられないときに signal する
コンディションの親。子の NO-JVP-RULE / NO-TRANSPOSE-RULE は、プリミティブが
ルールを持たないとき。"))

(define-condition no-jvp-rule (autodiff-error)
  ((name :initarg :name :reader no-jvp-rule-name))
  (:report
   (lambda (condition stream)
     (format stream "プリミティブ ~S に jvp ルールが無い" (no-jvp-rule-name condition))))
  (:documentation
   "jvp 変換が、jvp ルールを持たないプリミティブに出会ったときに signal する。
NAME はそのプリミティブ名（キーワード）。"))

(define-condition no-transpose-rule (autodiff-error)
  ((name :initarg :name :reader no-transpose-rule-name))
  (:report
   (lambda (condition stream)
     (format stream "プリミティブ ~S に transpose ルールが無い" (no-transpose-rule-name condition))))
  (:documentation
   "transpose が、transpose ルールを持たないプリミティブに出会ったときに
signal する。NAME はそのプリミティブ名（キーワード）。"))

(defun require-jvp-rule (primitive)
  "PRIMITIVE の jvp ルール（関数）を返す。無ければ NO-JVP-RULE を signal する。"
  (or (primitive-jvp primitive)
      (error 'no-jvp-rule :name (primitive-name primitive))))

(defun require-transpose-rule (primitive)
  "PRIMITIVE の transpose ルール（関数）を返す。無ければ NO-TRANSPOSE-RULE を
signal する。"
  (or (primitive-transpose primitive)
      (error 'no-transpose-rule :name (primitive-name primitive))))

(defun %primitive-or-error (name)
  "NAME（キーワード）の PRIMITIVE を返す。未登録なら UNKNOWN-PRIMITIVE。"
  (or (find-primitive name)
      (error 'unknown-primitive :name name)))

(defun set-jvp-rule (name function)
  "NAME（キーワード）のプリミティブの jvp スロットを FUNCTION にする。
未登録なら UNKNOWN-PRIMITIVE。FUNCTION を返す。"
  (setf (primitive-jvp (%primitive-or-error name)) function))

(defun set-transpose-rule (name function)
  "NAME（キーワード）のプリミティブの transpose スロットを FUNCTION にする。
未登録なら UNKNOWN-PRIMITIVE。FUNCTION を返す。"
  (setf (primitive-transpose (%primitive-or-error name)) function))

(defmacro def-jvp-rule (name (primals out tangents &rest param-lambda-list) &body body)
  "NAME（シンボル。DEFPRIMITIVE と同じく (INTERN (SYMBOL-NAME NAME) :KEYWORD)）
のプリミティブの jvp ルールを設定する。そのプリミティブが未登録なら、ルールを
定義する時点（ロード時）に UNKNOWN-PRIMITIVE を signal する。

ルールは次の規約の関数になる:
  (lambda (primals out tangents &key <params>) ...) → 出力の接線
PARAM-LAMBDA-LIST はその &key 以降をそのまま書く（例: (primals out tangents
&key shape dims)。パラメタの無いプリミティブは何も書かない。&ALLOW-OTHER-KEYS
は書かない: 知らないパラメタが渡されたらエラーにして気づけるようにする）。呼び出し側は
  (apply rule primals out tangents (eqn-params eqn))
の形で呼ぶ。

- PRIMALS: 入力の主値のトレーサのリスト（現在のトレースのもの）。
- OUT: 出力の主値のトレーサ。変換側が先に %TRACE-EQN で作ってあるので、
  ルールは主値を再計算せず、必要なら OUT を使う。
- TANGENTS: PRIMALS と同じ長さの、各入力の接線（トレーサまたは
  SYMBOLIC-ZERO）のリスト。全部ゼロのときは変換側が短絡するので、ルールには
  来ない。
- 戻り値: OUT と同じ aval を持つ接線（トレーサまたは SYMBOLIC-ZERO）。

制約: 接線はその被演算子について線形な演算にしか流さない。接線どうしの積や、
接線への exp / log / tanh / max / min / compare / reduce-max は禁止（mul / div
/ dot-general は片側だけが接線）。transpose が通せなくなるため。

ルールはプリミティブ構造体に載っているが、DEFPRIMITIVE を再評価しても
（:JVP を明示しない限り）引き継がれる。"
  `(set-jvp-rule ,(intern (symbol-name name) :keyword)
                 (lambda (,primals ,out ,tangents ,@param-lambda-list)
                   ,@body)))

(defmacro def-transpose-rule (name (ct invars &rest param-lambda-list) &body body)
  "NAME のプリミティブの transpose ルールを設定する（DEF-JVP-RULE と同じ名前の
扱い。未登録ならロード時に UNKNOWN-PRIMITIVE）。

ルールは次の規約の関数になる:
  (lambda (ct invars &key <params>) ...) → リスト
PARAM-LAMBDA-LIST は &key 以降。呼び出し側は
  (apply rule ct invars (eqn-params eqn))
の形で呼ぶ。

- CT: 出力の余接線（トレーサ）。常に非ゼロ（ゼロなら変換側が呼ばない）。
- INVARS: 入力ごとに、既知の主値のトレーサか、線形入力を表す UNDEFINED-PRIMAL。
- 戻り値: INVARS と同じ長さのリスト。UNDEFINED-PRIMAL の位置にはその入力の
  余接線（トレーサまたは SYMBOLIC-ZERO）、既知の位置には NIL を返す。mul / div
  で両方が UNDEFINED-PRIMAL のような線形でない使い方は AUTODIFF-ERROR にする。"
  `(set-transpose-rule ,(intern (symbol-name name) :keyword)
                       (lambda (,ct ,invars ,@param-lambda-list)
                         ,@body)))

(defun make-jvp-from-partials (partials)
  "要素ごと（入出力の shape・dtype が同じ）のプリミティブ専用。convert・形状
演算・縮約・dot は DEF-JVP-RULE で書くこと。

偏微分関数のリスト PARTIALS から jvp ルール（DEF-JVP-RULE と同じ規約の
関数）を作る。i 番目の偏微分関数は
  (lambda (primals out &key <params>) ...) → 係数のトレーサ
で、i 番目の入力についての偏微分を返す。戻り値はトレーサ・実数・SYMBOLIC-ZERO
のどれか。rank 0 のトレーサ（や実数）は %T-MUL が接線の shape へ自動で
ブロードキャストする。SYMBOLIC-ZERO ならその項を飛ばす。
ルールは、非ゼロの接線 t_i だけについて項 (%T-MUL 係数 t_i) を作り、
ADD-TANGENTS で足す（ゼロの接線の項は偏微分関数も呼ばない）。全部ゼロなら
OUT の aval の SYMBOLIC-ZERO を返す。係数は接線について線形にしか使わない
（積の相手は接線 1つだけ）。"
  (lambda (primals out tangents &rest params)
    (let ((sum (make-symbolic-zero (tracer-aval out))))
      (loop for partial in partials
            for tangent in tangents
            unless (symbolic-zero-p tangent)
              do (let ((coefficient (apply partial primals out params)))
                   (unless (symbolic-zero-p coefficient)
                     (setf sum (add-tangents sum (%t-mul coefficient tangent))))))
      sum)))

(defmacro def-jvp-partials (name &rest partials)
  "要素ごとのプリミティブ専用（MAKE-JVP-FROM-PARTIALS 参照）。
NAME のプリミティブの jvp ルールを、入力ごとの偏微分関数 PARTIALS（式。
評価すると MAKE-JVP-FROM-PARTIALS が受け取る関数になる）から作って設定する。
たとえば二項演算 f(a, b) なら (def-jvp-partials mul (lambda (primals out)
(second primals)) (lambda (primals out) (first primals)))。"
  `(set-jvp-rule ,(intern (symbol-name name) :keyword)
                 (make-jvp-from-partials (list ,@partials))))
