;;;; while-loop: 反復回数がトレース時に決まらないループ（issue #131）。
;;;; JAX の lax.while_loop に相当する、サブグラフを持つ高階プリミティブ。
;;;;
;;;; 公開 API は (WHILE-LOOP COND-FN BODY-FN INIT)。carry は配列（かトレーサ）の
;;;; リストで、COND-FN / BODY-FN はどちらも carry のリストを1つ受け取る（契約 C4。
;;;; フェーズ4でリストを PyTree に一般化しても API は変わらない）。
;;;;
;;;; 内部表現: プリミティブ :WHILE-LOOP（複数出力）。params は :COND / :BODY（閉じた
;;;; GRAPH）と :N-CARRIES。cond / body が閉包で外側のトレーサを捕まえると（closure
;;;; conversion。契約 C2）、捕まえた値は loop 不変の追加のオペランドとして eqn の入力の
;;;; 末尾に足される。StableHLO の while は、オペランド全部を carry として持つ
;;;; （リージョンのブロック引数も戻り値も全部）ので、eqn のオペランドと出力は次の形に
;;;; 揃える:
;;;;   オペランド = carry（N 個）+ cond が捕まえた値 + body が捕まえた値
;;;;   COND graph = 全オペランドを受け、rank 0 の :i1 を返す（cond が捕まえない値は未使用）
;;;;   BODY graph = 全オペランドを受け、新しい carry（N 個）+ 捕まえた値の素通しを返す
;;;;   出力      = 全オペランドと同じ aval（捕まえた値の分は呼び出し側が捨てる）
;;;; こうすると形状推論・eager・StableHLO の3つが同じ規則で書ける。
;;;; N 番目以降のオペランド（捕まえた値）は loop 不変で、BODY はそれを同じ入力 var のまま
;;;; 返す（abstract-eval が検査する）。jvp / batch のルールは、これらのオペランドの接線や
;;;; バッチ軸を具体化せずに済ませてよい。
;;;;
;;;; 既知の制限（ある実行系のコンパイラのバグ。詳細は docs/stablehlo-ops.md の制御構造の節。issue #131 のレビューで確認）: 本体の中で
;;;; 比較（:i1）から作った値（:i1 のフラグ、またはそれを i32 に変換した値）を carry にして、
;;;; その while の結果を関数の戻り値にすると、その実行系のコンパイラが LLVM の
;;;; "out of memory" / メモリフォルトでプロセスごと落ちる（i1 を i32 として通す、
;;;; optimization_barrier を挟む、select で作る、のどれでも直らない）。戻り値にしない場合、
;;;; および eager は問題ない。nabla 側では防げないので、carry に比較由来のフラグを
;;;; 持つ while の結果を jit の戻り値にしないこと（フラグは戻り値の前に使い切る）。
;;;; tests の medium テスト（while-loop-test）の子プロセスのテストが、この制限（バグ）がその実行系に
;;;; 残っていることを守る。直ったらそのテストが失敗するので、この注意書きごと消す。
;;;;
;;;; 微分: 逆モード（grad）は対応しない（反復回数が分からず、残差を保存できない）。jvp
;;;; ルールを持たないので、接線が流れ込むと NO-JVP-RULE（AUTODIFF-ERROR の子。
;;;; NO-JVP-RULE-NAME が :WHILE-LOOP）になる。jvp のみの対応は「cond / while-loop の
;;;; jvp」の issue（#134）。

(in-package #:nabla)

;;; ---- コンディション ----

(define-condition while-loop-error (error)
  ((format-control :initarg :format-control :initform "" :reader while-loop-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader while-loop-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "while-loop: ~?"
             (while-loop-error-format-control condition)
             (while-loop-error-format-arguments condition))))
  (:documentation
   "WHILE-LOOP が引数を受け付けないときの親のコンディション。具体的な原因は
子の WHILE-LOOP-ARGUMENT-ERROR / WHILE-LOOP-CARRY-MISMATCH /
WHILE-LOOP-CONDITION-ERROR を見る。"))

(define-condition while-loop-argument-error (while-loop-error) ()
  (:documentation
   "WHILE-LOOP の引数の型が不正なときに signal する: INIT が空でない（真の）リスト
でない・要素が配列かトレーサでない、COND-FN / BODY-FN が関数でない、BODY-FN が
リストを返さない。"))

(define-condition while-loop-carry-mismatch (while-loop-error)
  ((expected :initarg :expected :reader while-loop-carry-mismatch-expected)
   (actual :initarg :actual :reader while-loop-carry-mismatch-actual))
  (:documentation
   "BODY-FN が返した carry の AVAL（個数・shape・dtype）が INIT の AVAL と一致
しないときに signal する（トレース時。ループが0回で終わる場合でも検出する）。
EXPECTED は INIT の AVAL のリスト、ACTUAL は BODY-FN の出力の AVAL のリスト。"))

(define-condition while-loop-condition-error (while-loop-error) ()
  (:documentation
   "COND-FN の結果が rank 0 の :i1（スカラーの真偽）でないときに signal する。"))

(defun %while-loop-fail (condition-type control &rest arguments)
  (error condition-type :format-control control :format-arguments arguments))

;;; ---- プリミティブ ----

(defparameter *%while-cond-aval* (make-aval '() :i1)
  "cond の結果の aval（rank 0 の :i1）。")

(defun %while-region-avals (graph)
  (mapcar #'var-aval (graph-invars graph)))

(defun %while-loop-abstract-eval (in-avals &key cond body n-carries)
  (progn
    (unless (<= 0 n-carries (length in-avals))
      (error 'primitive-error :name :while-loop :in-avals in-avals
             :format-control "n-carries ~D が入力の個数 ~D を超えている"
             :format-arguments (list n-carries (length in-avals))))
    (dolist (graph (list cond body))
      (unless (equalp in-avals (%while-region-avals graph))
        (error 'primitive-error :name :while-loop :in-avals in-avals
               :format-control "入力の aval が cond / body の入力と一致しない: ~S / ~S"
               :format-arguments (list in-avals (%while-region-avals graph)))))
    (unless (equalp (mapcar #'var-aval (graph-outvars cond)) (list *%while-cond-aval*))
      (error 'primitive-error :name :while-loop :in-avals in-avals
             :format-control "cond の出力は rank 0 の :i1 が1つでなければならない"))
    ;; carry 以降（n-carries 番目から）のオペランドは loop 不変: body はそのまま返す。
    (loop for k from n-carries below (length in-avals)
          unless (eq (nth k (graph-outvars body)) (nth k (graph-invars body)))
            do (error 'primitive-error :name :while-loop :in-avals in-avals
                      :format-control "body の ~D 番目の出力は、loop 不変のオペランドの素通し（同じ入力 var）でなければならない"
                      :format-arguments (list k)))
    (unless (equalp (mapcar #'var-aval (graph-outvars body)) in-avals)
      (error 'primitive-error :name :while-loop :in-avals in-avals
             :format-control "body の出力の aval が入力（carry + 素通しの値）と一致しない: ~S"
             :format-arguments (list (mapcar #'var-aval (graph-outvars body))))))
  in-avals)

(defun %while-loop-eval (arrays cond body)
  "ARRAYS（全オペランド）から、COND の評価結果が真の間 BODY を回して、最後の
全オペランドの配列のリストを返す。"
  (loop while (= 1 (aref (first (multiple-value-list (apply #'eval-graph cond arrays)))))
        do (setf arrays (multiple-value-list (apply #'eval-graph body arrays))))
  arrays)

(defprimitive while-loop (:cond :body :n-carries)
  :multiple-outputs t
  ;; 名前で呼ぶ（関数オブジェクトを捕まえると、再定義した %WHILE-LOOP-ABSTRACT-EVAL が
  ;; 反映されず、mutation testing の変異が効かない）。
  :abstract-eval (lambda (in-avals &rest params)
                   (apply '%while-loop-abstract-eval in-avals params))
  :emit
  (lambda (in-names in-avals out-names out-avals &key cond body n-carries)
    (declare (ignore in-avals n-carries))
    ;; リージョンの接頭辞が cond → body の順に振られるよう、先に別々に作る。
    (let ((cond-lines (%stablehlo-region-lines cond))
          (body-lines (%stablehlo-region-lines body)))
      (format nil "~{~A~^, ~} = \"stablehlo.while\"(~{~A~^, ~}) ({~%~{  ~A~^~%~}~%}, {~%~{  ~A~^~%~}~%}) : (~{~A~^, ~}) -> (~{~A~^, ~})"
              out-names in-names cond-lines body-lines
              (mapcar #'tensor-type-string (mapcar #'var-aval (graph-invars cond)))
              (mapcar #'tensor-type-string out-avals))))
  :eager
  (lambda (arrays in-avals &key cond body n-carries)
    (declare (ignore in-avals n-carries))
    (%while-loop-eval arrays cond body)))

;;; ---- 公開 API ----

(defun %while-loop-carry-list-p (init)
  (and (consp init) (null (cdr (last init)))))

(defun %while-loop-check-arguments (cond-fn body-fn init)
  (unless (functionp cond-fn)
    (%while-loop-fail 'while-loop-argument-error "COND-FN は関数でなければならない: ~S" cond-fn))
  (unless (functionp body-fn)
    (%while-loop-fail 'while-loop-argument-error "BODY-FN は関数でなければならない: ~S" body-fn))
  (unless (%while-loop-carry-list-p init)
    (%while-loop-fail 'while-loop-argument-error
                      "INIT は空でない配列（かトレーサ）のリストでなければならない: ~S" init))
  (dolist (carry init)
    (unless (or (typep carry 'tracer) (typep carry '(and array (not string))))
      (%while-loop-fail 'while-loop-argument-error
                        "carry は配列かトレーサでなければならない: ~S" carry))
    ;; dtype が一意に決まらない配列（生の (unsigned-byte 16) の bf16 / f16 など）は
    ;; aval を作れない。トレーサ（dtype を持つ）で渡す。
    (when (typep carry '(and array (not string)))
      (handler-case (array-aval carry)
        (error ()
          (%while-loop-fail 'while-loop-argument-error
                            "carry の配列の dtype を決められない（bf16 / f16 はトレーサで渡す）: ~S" carry))))))

(defun %while-loop-trace-regions (cond-fn body-fn avals)
  "COND-FN / BODY-FN を AVALS（carry の aval）でサブグラフにトレースして、検査したうえで、
全オペランド形式（ファイル冒頭）に揃えた (VALUES COND-GRAPH BODY-GRAPH CAPTURED) を返す。
CAPTURED は cond が捕まえた外側のトレーサ、body が捕まえたものの順に並ぶ。"
  (let ((n (length avals)))
    (multiple-value-bind (cond-graph cond-captured)
        (%call-with-trace avals
                          (lambda (&rest carries) (funcall cond-fn carries))
                          *current-trace*)
      (unless (equalp (mapcar #'var-aval (graph-outvars cond-graph)) (list *%while-cond-aval*))
        (%while-loop-fail 'while-loop-condition-error
                          "COND-FN は rank 0 の :i1 を1つ返さなければならない（返した aval: ~S）"
                          (mapcar #'var-aval (graph-outvars cond-graph))))
      (multiple-value-bind (body-graph body-captured)
          (%call-with-trace avals
                            (lambda (&rest carries)
                              (let ((outs (funcall body-fn carries)))
                                (unless (%while-loop-carry-list-p outs)
                                  (%while-loop-fail 'while-loop-argument-error
                                                    "BODY-FN は carry のリストを返さなければならない: ~S" outs))
                                (values-list outs)))
                            *current-trace*)
        (let ((actual (mapcar #'var-aval (graph-outvars body-graph))))
          (unless (equalp actual avals)
            (error 'while-loop-carry-mismatch
                   :expected avals :actual actual
                   :format-control "BODY-FN の出力の aval が INIT と一致しない: 期待 ~S / 実際 ~S"
                   :format-arguments (list avals actual))))
        (let* ((cond-const-avals (mapcar #'tracer-aval cond-captured))
               (body-const-avals (mapcar #'tracer-aval body-captured))
               ;; cond: [carry.. cond-consts..] + 未使用の body-consts
               (cond-unused (mapcar #'make-var body-const-avals))
               (cond-full (make-graph (append (graph-invars cond-graph) cond-unused)
                                      (graph-eqns cond-graph)
                                      (graph-outvars cond-graph)
                                      (graph-constants cond-graph)))
               ;; body: [carry.. 未使用の cond-consts.. body-consts..]。consts は素通し
               (body-unused (mapcar #'make-var cond-const-avals))
               (body-invars (graph-invars body-graph))
               (body-consts (nthcdr n body-invars))
               (body-full (make-graph (append (subseq body-invars 0 n) body-unused body-consts)
                                      (graph-eqns body-graph)
                                      (append (graph-outvars body-graph) body-unused body-consts)
                                      (graph-constants body-graph))))
          (values cond-full body-full (append cond-captured body-captured)))))))

(defun while-loop (cond-fn body-fn init)
  "COND-FN が真の間 BODY-FN を繰り返し、最後の carry のリストを返す（JAX の
lax.while_loop）。INIT は配列（トレース中はトレーサでもよい）の空でないリスト。
COND-FN / BODY-FN は carry のリストを1つ受け取る関数（WITH-TRACING で作る）:
COND-FN は rank 0 の :i1（比較の結果など）を返し、BODY-FN は INIT と同じ AVAL
（個数・shape・dtype）の carry のリストを返す。carry は INIT の個数と順序のまま
返る。

配列だけを渡して WITH-TRACING の外から呼べば eager に、WITH-TRACING / JIT の中では
:WHILE-LOOP の eqn（StableHLO では stablehlo.while）としてトレースされる。COND-FN /
BODY-FN は外側のトレーサを閉包で捕まえてよい（loop 不変の値として扱われる）。

エラー（どれも WHILE-LOOP-ERROR の子）: INIT がリストでない・空・要素が配列でない、
関数でない、BODY-FN がリストを返さない → WHILE-LOOP-ARGUMENT-ERROR。BODY-FN の
出力の AVAL が INIT と違う → WHILE-LOOP-CARRY-MISMATCH（トレース時に検出する。0回で
終わる場合でも）。COND-FN の結果が rank 0 の :i1 でない → WHILE-LOOP-CONDITION-ERROR。

一部の実行系の制限: 本体の中で比較から作ったフラグ（:i1）を carry にした while の結果を jit の
戻り値にすると、その実行系のコンパイラが落ちる（docs/stablehlo-ops.md の制御構造の節）。

vmap: 条件がバッチされなければ本体をバッチ化した :while-loop のままになる（バッチされる
carry は不動点まで広げる）。条件がバッチされると、どれかの要素の条件が真の間回し、
条件が偽になった要素の carry は SELECT で据え置く（要素ごとに反復回数が違ってよい）。

微分: 逆モード（GRAD）は対応しない（反復回数が分からず、残差を保存できない）。
GRAD を通すと、原因のプリミティブ名 :WHILE-LOOP を持つ NO-JVP-RULE
（AUTODIFF-ERROR の子）になる。jvp のみの対応は別の issue（#134）。"
  (%while-loop-check-arguments cond-fn body-fn init)
  (let ((traced (or *current-trace* (some (lambda (x) (typep x 'tracer)) init))))
    (when (and traced (null *current-trace*))
      (error 'tracing-error
             :format-control "while-loop の init に、終わったトレースのトレーサが混ざっている: ~S"
             :format-arguments (list init)))
    (let* ((operands (if traced
                         (mapcar (lambda (x)
                                   (if (typep x 'tracer)
                                       x
                                       (%lift-constant x (array-aval x) *current-trace*)))
                                 init)
                         init))
           (avals (mapcar (lambda (x) (if (typep x 'tracer) (tracer-aval x) (array-aval x))) init))
           (n (length init)))
      (multiple-value-bind (cond-graph body-graph captured)
          (%while-loop-trace-regions cond-fn body-fn avals)
        (if traced
            (subseq (apply #'%trace-eqn* :while-loop (append operands captured)
                           (list :cond cond-graph :body body-graph :n-carries n))
                    0 n)
            ;; トレースの外（captured は常に空）: Lisp のループで評価する。
            (subseq (%while-loop-eval init cond-graph body-graph) 0 n))))))
