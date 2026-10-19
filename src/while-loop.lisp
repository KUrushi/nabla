;;;; while-loop: 反復回数がトレース時に決まらないループ（issue #131）。
;;;; JAX の lax.while_loop に相当する、サブグラフを持つ高階プリミティブ。
;;;;
;;;; 公開 API は (WHILE-LOOP COND-FN BODY-FN INIT)。carry は配列・トレーサ・実数（rank 0 の配列に持ち上げる。issue #227）を葉とする
;;;; PyTree（issue #201。リストは既定の節なのでこれまでの呼び出しはそのまま動く）で、
;;;; COND-FN / BODY-FN はどちらも carry の木を1つ受け取る（契約 C4）。内部ではこれまでどおり
;;;; フラットな葉のリストで扱い、BODY-FN の戻り値の treedef が INIT と違えば
;;;; WHILE-LOOP-CARRY-MISMATCH にする。
;;;;
;;;; 内部表現: プリミティブ :WHILE-LOOP（複数出力）。params は :COND / :BODY（閉じた
;;;; GRAPH）と :N-CARRIES。cond / body が閉包で外側のトレーサを捕まえると（closure
;;;; conversion。契約 C2）、捕まえた値は loop 不変の追加のオペランドとして eqn の入力の
;;;; 末尾に足される。StableHLO の while は、オペランド全部を carry として持つ
;;;; （リージョンのブロック引数も戻り値も全部）ので、eqn のオペランドと出力は次の形に
;;;; 揃える:
;;;;   オペランド = carry（N 個）+ cond と body が捕まえた値の和集合（EQ で重複なし。
;;;;                cond が捕まえた順、続いて body だけが捕まえた順。issue #166）
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
;;;; もう1つの既知のバグ（ある実行系のコンパイラ。:i1 の carry とは別）: 定数で初期化したカウンタの carry と
;;;; ほかの carry 2 つ以上（rank 1 以上を含む）を持つ while はコンパイラが非決定的に落ちるので、
;;;; emit は定数のオペランドを optimization_barrier に通す（%WHILE-BARRIER-LINES。詳細は
;;;; docs/stablehlo-ops.md の制御構造の節）。
;;;;
;;;; 微分: 前進モード（jvp）だけ対応する（src/ad/rules-control.lisp、issue #134）。
;;;; 逆モード（grad）は対応しない（反復回数が分からず、残差を保存できない）ので、
;;;; linearize が AUTODIFF-ERROR（メッセージにプリミティブ名 :WHILE-LOOP を含む）にする。

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
   "WHILE-LOOP の引数の型が不正なときに signal する: INIT が PyTree として不正か葉を持たない・
葉が配列・トレーサ・実数でない、COND-FN / BODY-FN が関数でない、BODY-FN の戻り値が PyTree として不正。"))

(define-condition while-loop-carry-mismatch (while-loop-error)
  ((expected :initarg :expected :reader while-loop-carry-mismatch-expected)
   (actual :initarg :actual :reader while-loop-carry-mismatch-actual))
  (:documentation
   "BODY-FN が返した carry の AVAL（個数・shape・dtype）が INIT の AVAL と一致
しないときに signal する（トレース時。ループが0回で終わる場合でも検出する）。
EXPECTED は INIT の葉の AVAL のリスト、ACTUAL は BODY-FN の戻り値の葉の AVAL のリスト。
木の形（treedef）が違うときも、これらは葉の AVAL のリストで、木の形はメッセージにだけ出る。"))

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

(defun %while-barrier-lines (in-names in-avals out-name)
  "while のオペランド名 IN-NAMES（型は IN-AVALS）のうち、いま出している graph の定数（*STABLEHLO-CONSTANT-NAMES*）
の名前のものを、まとめて1つの stablehlo.optimization_barrier に通す。
(VALUES barrier の行のリスト（無ければ NIL） 置き換えたオペランド名のリスト) を返す。
同じ定数が複数のオペランドに現れてもよい（それぞれ別の barrier の結果になる）。
定数のオペランドがあるときは、モジュールの中で一意な整数（%STABLEHLO-UNIQUE-ID）の constant
\"%wbar_<out>_u_c\" も同じ barrier に通す（結果は使わない。MLIR の SSA 名は数字で始まると
数字だけでなければならないので、接頭辞を付ける）。同じ値の定数は CSE で1つになるので、
通さないと同じ定数で初期化した2つの while の barrier がまったく同じ形になって CSE で
まとめられ、2つのループが同じ SSA 値から始まる（issue #179。%SCAN-YS-INIT-LINES と同じやり方）。"
  (let ((positions (loop for name in in-names for i from 0
                         when (member name *stablehlo-constant-names* :test #'string=)
                           collect i)))
    (if (null positions)
        (values '() in-names)
        (let* ((results (loop for i in positions
                              collect (format nil "%wbar~D_~A" i (subseq out-name 1))))
               (salt (format nil "%wbar_~A_u" (subseq out-name 1)))
               (names (copy-list in-names)))
          (loop for i in positions for r in results do (setf (nth i names) r))
          (values (list (format nil "~A_c = stablehlo.constant dense<~D> : tensor<i32>" salt (%stablehlo-unique-id))
                        (format nil "~{~A~^, ~} = stablehlo.optimization_barrier ~{~A~^, ~} : ~{~A~^, ~}"
                                (append results (list salt))
                                (append (mapcar (lambda (i) (nth i in-names)) positions)
                                        (list (format nil "~A_c" salt)))
                                (append (mapcar (lambda (i) (tensor-type-string (nth i in-avals))) positions)
                                        (list "tensor<i32>"))))
                  names)))))

(defprimitive while-loop (:cond :body :n-carries)
  :multiple-outputs t
  ;; 名前で呼ぶ（関数オブジェクトを捕まえると、再定義した %WHILE-LOOP-ABSTRACT-EVAL が
  ;; 反映されず、mutation testing の変異が効かない）。
  :abstract-eval (lambda (in-avals &rest params)
                   (apply '%while-loop-abstract-eval in-avals params))
  :emit
  (lambda (in-names in-avals out-names out-avals &key cond body n-carries)
    (declare (ignore n-carries))
    ;; graph の定数（stablehlo.constant）をそのまま while のオペランドにすると、ある実行系の
    ;; コンパイラが落ちることがある。定数のオペランドは optimization_barrier を通す
    ;; （docs/stablehlo-ops.md の制御構造の節。%WHILE-BARRIER-LINES）。
    (multiple-value-bind (barrier-lines operand-names)
        (%while-barrier-lines in-names in-avals (first out-names))
      ;; リージョンの接頭辞が cond → body の順に振られるよう、先に別々に作る。
      (let ((cond-lines (%stablehlo-region-lines cond))
            (body-lines (%stablehlo-region-lines body)))
        (format nil "~{~A~%~}~{~A~^, ~} = \"stablehlo.while\"(~{~A~^, ~}) ({~%~{  ~A~^~%~}~%}, {~%~{  ~A~^~%~}~%}) : (~{~A~^, ~}) -> (~{~A~^, ~})"
                barrier-lines out-names operand-names cond-lines body-lines
                (mapcar #'tensor-type-string (mapcar #'var-aval (graph-invars cond)))
                (mapcar #'tensor-type-string out-avals)))))
  :eager
  (lambda (arrays in-avals &key cond body n-carries)
    (declare (ignore in-avals n-carries))
    (%while-loop-eval arrays cond body)))

;;; ---- 公開 API ----

(defun %while-loop-flatten (tree what)
  "TREE（carry の木）を (VALUES 葉のリスト treedef) にする。PyTree として不正（真リストでない
コンスなど）なら WHILE-LOOP-ARGUMENT-ERROR。"
  (handler-case (tree-flatten tree)
    (pytree-error (condition)
      (%while-loop-fail 'while-loop-argument-error "~A が PyTree として不正: ~A" what condition))))

(defun %while-loop-check-arguments (cond-fn body-fn init)
  "引数を検査して、INIT の (VALUES 葉のリスト treedef) を返す。"
  (unless (functionp cond-fn)
    (%while-loop-fail 'while-loop-argument-error "COND-FN は関数でなければならない: ~S" cond-fn))
  (unless (functionp body-fn)
    (%while-loop-fail 'while-loop-argument-error "BODY-FN は関数でなければならない: ~S" body-fn))
  (multiple-value-bind (leaves treedef) (%while-loop-flatten init "INIT")
    (when (null leaves)
      (%while-loop-fail 'while-loop-argument-error
                        "INIT は葉（配列・トレーサ・実数）を1つ以上持つ PyTree でなければならない: ~S" init))
    (values (%while-loop-check-leaves leaves) treedef)))

(defun %while-loop-check-leaves (leaves)
  "葉のリストを検査して、実数の葉を rank 0 の配列に持ち上げたリストを返す（持ち上げ規則は
SCAN と共有する %REAL-SCALAR-ARRAY。double-float は :F64、整数などそれ以外は :F32）。"
  (mapcar
   (lambda (carry)
     (typecase carry
       (tracer carry)
       (real (%real-scalar-array carry))
       ((and array (not string))
        ;; dtype が一意に決まらない配列（生の (unsigned-byte 16) の bf16 / f16 など）は
        ;; aval を作れない。トレーサ（dtype を持つ）で渡す。
        (handler-case (array-aval carry)
          (error ()
            (%while-loop-fail 'while-loop-argument-error
                              "carry の配列の dtype を決められない（bf16 / f16 はトレーサで渡す）: ~S" carry)))
        carry)
       (t (%while-loop-fail 'while-loop-argument-error
                            "carry は配列・トレーサ・実数でなければならない: ~S" carry))))
   leaves))

(defun %while-loop-trace-regions (cond-fn body-fn avals treedef)
  "COND-FN / BODY-FN を AVALS（carry の葉の aval、木の形は TREEDEF）でサブグラフにトレースして、検査したうえで、
全オペランド形式（ファイル冒頭）に揃えた (VALUES COND-GRAPH BODY-GRAPH CAPTURED) を返す。
CAPTURED は cond と body が捕まえた外側のトレーサの和集合（EQ で重複なし）で、cond が
捕まえた順、続いて body だけが捕まえた順に並ぶ。"
  (let ((n (length avals))
        (body-treedef nil))
    (multiple-value-bind (cond-graph cond-captured)
        (%call-with-trace avals
                          (lambda (&rest carries) (funcall cond-fn (tree-unflatten treedef carries)))
                          *current-trace*)
      (unless (equalp (mapcar #'var-aval (graph-outvars cond-graph)) (list *%while-cond-aval*))
        (%while-loop-fail 'while-loop-condition-error
                          "COND-FN は rank 0 の :i1 を1つ返さなければならない（返した aval: ~S）"
                          (mapcar #'var-aval (graph-outvars cond-graph))))
      (multiple-value-bind (body-graph body-captured)
          (%call-with-trace avals
                            (lambda (&rest carries)
                              (multiple-value-bind (outs out-treedef)
                                  (%while-loop-flatten
                                   (funcall body-fn (tree-unflatten treedef carries)) "BODY-FN の戻り値")
                                (setf body-treedef out-treedef)
                                (values-list outs)))
                            *current-trace*)
        (let ((actual (mapcar #'var-aval (graph-outvars body-graph))))
          (unless (treedef= treedef body-treedef)
            (error 'while-loop-carry-mismatch
                   :expected avals :actual actual
                   :format-control "BODY-FN の戻り値の木の形が INIT と一致しない: 期待 ~S / 実際 ~S"
                   :format-arguments (list treedef body-treedef)))
          (unless (equalp actual avals)
            (error 'while-loop-carry-mismatch
                   :expected avals :actual actual
                   :format-control "BODY-FN の出力の aval が INIT と一致しない: 期待 ~S / 実際 ~S"
                   :format-arguments (list avals actual))))
        ;; 両方が捕まえた値は1回だけ（cond が先に捕まえた順、続いて body だけが捕まえた順）
        ;; オペランドにする。cond / body の invars はどちらも「carry、CAPTURED」に揃え、
        ;; 自分が使わない捕捉値の位置には使われない invar を置く（%COND-UNIFY-BRANCHES）。
        (let* ((captured (remove-duplicates (append cond-captured body-captured)
                                            :test #'eq :from-end t))
               (cond-full (%cond-unify-branches cond-graph cond-captured captured n))
               (body-unified (%cond-unify-branches body-graph body-captured captured n))
               (body-consts (nthcdr n (graph-invars body-unified)))
               ;; body は捕捉値を同じ invar のまま素通しで返す。
               (body-full (make-graph (graph-invars body-unified)
                                      (graph-eqns body-unified)
                                      (append (graph-outvars body-unified) body-consts)
                                      (graph-constants body-unified))))
          (values cond-full body-full captured))))))

(defun while-loop (cond-fn body-fn init)
  "COND-FN が真の間 BODY-FN を繰り返し、最後の carry を返す（JAX の lax.while_loop）。
INIT は PyTree（リスト・ベクタ・DEFPYTREE / DEFMODULE の型を入れ子にした木。配列そのもの
でもよい）で、葉は配列（トレース中はトレーサでもよい）か Lisp の実数、葉は1つ以上。
実数の葉は rank 0 の配列に持ち上げてから cond-fn / body-fn に渡す（SCAN と同じ規則）:
double-float は :f64、それ以外（整数・有理数・single-float）は :f32。**整数も :f32 になる**
ので、整数のカウンタが要るなら :i32 の配列で渡す。BODY-FN が返す実数リテラルも同じ規則で
持ち上がる（INIT の葉と同じ実数なら aval が一致する）。
COND-FN / BODY-FN は carry の木を1つ受け取る関数（WITH-TRACING で作る）:
COND-FN は rank 0 の :i1（比較の結果など）を返し、BODY-FN は INIT と同じ木の形（TREEDEF=）で、
葉の AVAL（shape・dtype）も同じ carry の木を返す。結果は INIT と同じ木の形で返る。eager の結果は INIT・閉包で捕まえた配列と EQ でありうる（README「配列の不変性」）。

配列だけを渡して WITH-TRACING の外から呼べば eager に、WITH-TRACING / JIT の中では
:WHILE-LOOP の eqn（StableHLO では stablehlo.while）としてトレースされる。COND-FN /
BODY-FN は外側のトレーサを閉包で捕まえてよい（loop 不変の値として扱われる）。

エラー（どれも WHILE-LOOP-ERROR の子）: INIT が PyTree として不正・葉が無い・葉が配列・トレーサ・実数でない（文字列など）、
関数でない、BODY-FN の戻り値が PyTree として不正 → WHILE-LOOP-ARGUMENT-ERROR。BODY-FN の
戻り値の木の形か葉の AVAL が INIT と違う → WHILE-LOOP-CARRY-MISMATCH（トレース時に検出する。
0回で終わる場合でも）。COND-FN の結果が rank 0 の :i1 でない → WHILE-LOOP-CONDITION-ERROR。

一部の実行系の制限: 本体の中で比較から作ったフラグ（:i1）を carry にした while の結果を jit の
戻り値にすると、その実行系のコンパイラが落ちる（docs/stablehlo-ops.md の制御構造の節）。

vmap: 条件がバッチされなければ本体をバッチ化した :while-loop のままになる（バッチされる
carry は不動点まで広げる）。条件がバッチされると、どれかの要素の条件が真の間回し、
条件が偽になった要素の carry は SELECT で据え置く（要素ごとに反復回数が違ってよい）。

微分: 前進モード（jvp）だけ対応する。逆モード（GRAD）は対応しない（反復回数が
分からず、残差を保存できない）。GRAD を通すと、プリミティブ名 :WHILE-LOOP を含む
AUTODIFF-ERROR になる。"
  (multiple-value-bind (init treedef) (%while-loop-check-arguments cond-fn body-fn init)
    ;; 以下の INIT は葉のリスト。内部は今までどおりフラットなリストで扱い、最後に木へ戻す。
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
            (%while-loop-trace-regions cond-fn body-fn avals treedef)
          (tree-unflatten
           treedef
           (if traced
               (subseq (apply #'%trace-eqn* :while-loop (append operands captured)
                              (list :cond cond-graph :body body-graph :n-carries n))
                       0 n)
               ;; トレースの外（captured は常に空）: Lisp のループで評価する。
               (subseq (%while-loop-eval init cond-graph body-graph) 0 n))))))))
