;;;; vmap: バッチ化ルールのスロットと、公開 API の VMAP（issue #125）。
;;;;
;;;; VMAP は GRAD（src/ad/grad.lisp）と同じ形の変換: f を「バッチ軸を取り除いた」
;;;; aval で新しいトレースに1回トレースして graph にし、その eqn を順に歩いて、
;;;; 値ごとに「バッチ軸の位置（無ければ NIL）」を伝播させながら、バッチ化ルール
;;;; （DEF-BATCH-RULE）で現在のトレースに発行し直す。呼び出しが別のトレースの中
;;;; なら、そのトレースへそのまま展開する（jit・grad・入れ子の vmap が動く）。
;;;; そうでなければ、バッチされた引数の aval で新しいトレースを作り、できた graph を
;;;; EVAL-GRAPH で評価する（eager の vmap）。
;;;;
;;;; バッチ軸を持たない値だけを入力とする eqn は、ルールを呼ばずにそのまま
;;;; 発行し直す（不要な broadcast をしない）。バッチ軸は、ルールが決めた位置のまま
;;;; 伝播し、最後に出力ごとに OUT-AXES へ移す（動かす必要があるときだけ transpose 1つ）。
;;;;
;;;; 既知の制限: GRAD と同じく、f が外側のトレースのトレーサを閉包で捕まえると
;;;; TRACING-ERROR になる。外側の値は f の引数として渡すこと。

(in-package #:nabla)

(define-condition vmap-error (error)
  ((format-control :initarg :format-control :initform "" :reader vmap-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil
                     :reader vmap-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "vmap エラー: ~?"
             (vmap-error-format-control condition)
             (vmap-error-format-arguments condition))))
  (:documentation
   "VMAP が続けられないときに signal するコンディションの親。IN-AXES / OUT-AXES の
不正（範囲外・個数の不一致・型）、バッチ軸の長さの不一致、バッチされた引数が1つも
ないとき、f が vmap の対象にできない関数のとき、などで signal する。子の
NO-BATCH-RULE は、プリミティブがバッチ化ルールを持たないとき。"))

(define-condition no-batch-rule (vmap-error)
  ((name :initarg :name :reader no-batch-rule-name))
  (:report
   (lambda (condition stream)
     (format stream "vmap エラー: プリミティブ ~S にバッチ化ルールが無い" (no-batch-rule-name condition))))
  (:documentation
   "VMAP が、バッチ軸を持つ入力を受けるプリミティブのうち、バッチ化ルールを持たない
ものに出会ったときに signal する。NAME はそのプリミティブ名（キーワード）。"))

(defun %vmap-error (control &rest arguments)
  (error 'vmap-error :format-control control :format-arguments arguments))

(defun require-batch-rule (primitive)
  "PRIMITIVE のバッチ化ルール（関数）を返す。無ければ NO-BATCH-RULE を signal する。"
  (or (primitive-batch primitive)
      (error 'no-batch-rule :name (primitive-name primitive))))

(defun set-batch-rule (name function)
  "NAME（キーワード）のプリミティブの batch スロットを FUNCTION にする。
未登録なら UNKNOWN-PRIMITIVE。FUNCTION を返す。"
  (setf (primitive-batch (%primitive-or-error name)) function))

(defmacro def-batch-rule (name (args batch-dims &rest param-lambda-list) &body body)
  "NAME（シンボル。DEFPRIMITIVE と同じく (INTERN (SYMBOL-NAME NAME) :KEYWORD)）
のプリミティブのバッチ化ルールを設定する。未登録ならロード時に UNKNOWN-PRIMITIVE。

ルールは次の規約の関数になる:
  (lambda (args batch-dims &key <params>) ...) → (values outs out-dims)
PARAM-LAMBDA-LIST は &key 以降をそのまま書く（&ALLOW-OTHER-KEYS は書かない）。
呼び出し側は (apply rule args batch-dims (eqn-params eqn)) の形で呼ぶ。

- ARGS: 入力のトレーサのリスト（新しい外側のトレースのもの。バッチ軸を持つ）。
- BATCH-DIMS: ARGS と同じ長さのリスト。各要素は、その引数のバッチ軸の位置（整数）か、
  バッチされていないことを表す NIL。全部 NIL のときは変換側が短絡するので、ルールには
  来ない。バッチされていない引数のトレーサは形にバッチ軸を持たない（元の形のまま）。
- 戻り値: 多値 (OUTS OUT-DIMS)。**単一出力のプリミティブでも常にリスト**で、
  どちらも eqn の出力の個数と同じ長さ。OUT-DIMS の各要素は出力のバッチ軸の位置か、
  出力がバッチされないときは NIL。
- ルールは、現在のトレースに %TRACE-EQN などで演算を足して書く（配列は扱わない）。
  バッチされていない入力は、必要なときだけ（要素ごとの演算のように形を揃える必要が
  あるとき）ルールの中で broadcast する。"
  `(set-batch-rule ,(intern (symbol-name name) :keyword)
                   (lambda (,args ,batch-dims ,@param-lambda-list)
                     ,@body)))

;;; --- バッチ軸の移動（ルールと vmap 本体が共有する） ---

(defun %vmap-move-axis (tracer from to)
  "TRACER の軸 FROM を軸 TO の位置へ動かす（他の軸の相対順は保つ）。FROM = TO なら
TRACER をそのまま返す。"
  (if (= from to)
      tracer
      (let* ((rank (aval-rank (tracer-aval tracer)))
             (rest (loop for i below rank unless (= i from) collect i))
             (perm (append (subseq rest 0 to) (list from) (nthcdr to rest))))
        (%trace-eqn :transpose (list tracer) :perm perm))))

(defun %vmap-broadcast-batch (tracer axis size)
  "バッチされていない TRACER に、長さ SIZE の新しい軸を AXIS の位置へ足し、
中身を複製する。"
  (let* ((shape (aval-shape (tracer-aval tracer)))
         (out-shape (append (subseq shape 0 axis) (list size) (nthcdr axis shape)))
         (dims (loop for i below (length out-shape) unless (= i axis) collect i)))
    (%trace-eqn :broadcast-in-dim (list tracer) :shape out-shape :dims dims)))

;;; --- 引数の検査 ---

(defun %vmap-traceable (f)
  "F（TRACEABLE-FUNCTION か、静的引数の無い JITTED-FUNCTION）の TRACEABLE-FUNCTION
を返す。それ以外は VMAP-ERROR。"
  (typecase f
    (traceable-function f)
    (jitted-function
     (when (%jitted-function-static-positions f)
       (%vmap-error "静的引数のある jit した関数は vmap の対象にできない: ~S" f))
     (%jitted-function-fn f))
    (t (%vmap-error "vmap の対象は WITH-TRACING / DEFJIT / JIT が作った関数でなければならない: ~S" f))))

(defun %vmap-axis-spec-p (spec)
  (or (null spec) (integerp spec)))

(defun %vmap-per-item (spec count what)
  "SPEC（整数か NIL なら全体に共通、それ以外はリスト）を COUNT 個の軸指定のリストにする。"
  (cond
    ((%vmap-axis-spec-p spec) (make-list count :initial-element spec))
    ((and (listp spec) (every #'%vmap-axis-spec-p spec))
     (unless (= (length spec) count)
       (%vmap-error "~A の個数 ~D が ~D 個と一致しない: ~S" what (length spec) count spec))
     (copy-list spec))
    (t (%vmap-error "~A は整数・NIL、またはそのリストでなければならない: ~S" what spec))))

(defun %vmap-normalize-axis (axis rank what)
  "AXIS（整数。負なら RANK からの相対）を [0, RANK) の整数にする。範囲外は VMAP-ERROR。"
  (let ((n (if (minusp axis) (+ axis rank) axis)))
    (unless (< -1 n rank)
      (%vmap-error "~A の軸 ~D は範囲外（rank ~D）" what axis rank))
    n))

(defun %vmap-argument (arg)
  "ARG を TRACER か配列にして返す（実数は rank 0 の配列。DOUBLE-FLOAT は :F64、それ以外は :F32）。"
  (typecase arg
    (tracer arg)
    (real (%scalar-array arg (if (typep arg 'double-float) :f64 :f32)))
    (string (%vmap-error "vmap した関数の引数に文字列は渡せない: ~S" arg))
    (array arg)
    (t (%vmap-error "vmap した関数の引数は配列・実数・トレーサでなければならない: ~S" arg))))

(defun %vmap-argument-aval (arg)
  (if (typep arg 'tracer) (tracer-aval arg) (array-aval arg)))

;;; --- graph の歩き ---

(defun %vmap-check-rule-result (eqn outs out-dims size)
  "ルールの戻り値 OUTS / OUT-DIMS が、EQN の出力と矛盾しないことを確かめる。"
  (let ((outvars (eqn-outvars eqn)))
    (unless (and (listp outs) (listp out-dims)
                 (= (length outs) (length outvars) (length out-dims)))
      (%vmap-error "~S のバッチ化ルールは、出力 ~D 個に対して長さの揃ったリスト (outs out-dims) を返さなければならない: ~S ~S"
                   (primitive-name (eqn-prim eqn)) (length outvars) outs out-dims))
    (loop for out in outs
          for dim in out-dims
          for outvar in outvars
          do (unless (typep out 'tracer)
               (%vmap-error "~S のバッチ化ルールの出力はトレーサでなければならない: ~S"
                            (primitive-name (eqn-prim eqn)) out))
             (let ((shape (aval-shape (tracer-aval out)))
                   (expected (aval-shape (var-aval outvar))))
               (unless (eq (aval-dtype (tracer-aval out)) (aval-dtype (var-aval outvar)))
                 (%vmap-error "~S のバッチ化ルールの出力の dtype ~S が、元の出力の dtype ~S と一致しない"
                              (primitive-name (eqn-prim eqn)) (aval-dtype (tracer-aval out))
                              (aval-dtype (var-aval outvar))))
               (unless (if dim
                           (and (integerp dim) (< -1 dim (length shape)) (= (nth dim shape) size)
                                (equal (append (subseq shape 0 dim) (nthcdr (1+ dim) shape)) expected))
                           (equal shape expected))
                 (%vmap-error "~S のバッチ化ルールの出力の形 ~S・軸 ~S が、元の出力の形 ~S と整合しない（バッチ軸の長さ ~D）"
                              (primitive-name (eqn-prim eqn)) shape dim expected size))))))

(defun %vmap-finish-output (tracer dim var spec size)
  "出力 TRACER（バッチ軸 DIM。無ければ NIL。元の出力の var は VAR）を OUT-AXES の
指定 SPEC の位置へ動かす（バッチされていなければ複製する）。"
  (let ((rank (1+ (aval-rank (var-aval var)))))
    (cond
      ((and dim (null spec))
       (%vmap-error "出力がバッチされているのに OUT-AXES が NIL（出力の軸 ~D）" dim))
      ((null spec) tracer)
      (dim (%vmap-move-axis tracer dim (%vmap-normalize-axis spec rank "OUT-AXES")))
      (t (%vmap-broadcast-batch tracer (%vmap-normalize-axis spec rank "OUT-AXES") size)))))

(defun %vmap-walk-values (graph tracers dims size)
  "GRAPH（f をバッチ軸なしでトレースしたもの）の eqn を現在のトレースに、バッチ化して
発行し直し、(VALUES OUT-TRACERS OUT-DIMS)（GRAPH-OUTVARS ごとのトレーサと、そのバッチ軸
（無ければ NIL））を返す。TRACERS は現在のトレースの GRAPH-INVARS に対応するトレーサ
（バッチ軸を持つ）、DIMS はその軸（無ければ NIL）、SIZE はバッチ軸の長さ。
複数出力の eqn（制御構造。契約 C1）も扱う: バッチされていない入力だけならそのまま
%TRACE-EQN* で発行し直し、そうでなければルールが出力ごとのリストを返す。
cond / while-loop / scan のルールも、本体のサブグラフをこの関数で再帰的に
バッチ化する（rules-batch-control.lisp の %VMAP-SUBGRAPH）。"
  (let ((env (make-hash-table :test 'eq)))
    (loop for tracer in tracers for dim in dims for invar in (graph-invars graph)
          do (setf (gethash invar env) (cons tracer dim)))
    (loop for (var . array) in (graph-constants graph)
          do (setf (gethash var env)
                   (cons (%lift-constant array (var-aval var) *current-trace*) nil)))
    (dolist (eqn (graph-eqns graph))
      (let* ((entries (mapcar (lambda (v) (gethash v env)) (eqn-invars eqn)))
             (args (mapcar #'car entries))
             (arg-dims (mapcar #'cdr entries))
             (prim (eqn-prim eqn)))
        (if (notany #'identity arg-dims)
            ;; バッチされていない入力だけ: ルールを呼ばずにそのまま発行し直す。
            (let ((results (apply #'%trace-eqn* (primitive-name prim) args (eqn-params eqn))))
              (loop for var in (eqn-outvars eqn) for result in results
                    do (setf (gethash var env) (cons result nil))))
            (multiple-value-bind (outs out-dims)
                (apply (require-batch-rule prim) args arg-dims (eqn-params eqn))
              (%vmap-check-rule-result eqn outs out-dims size)
              (loop for var in (eqn-outvars eqn) for out in outs for dim in out-dims
                    do (setf (gethash var env) (cons out dim)))))))
    (let ((entries (mapcar (lambda (v) (gethash v env)) (graph-outvars graph))))
      (values (mapcar #'car entries) (mapcar #'cdr entries)))))

(defun %vmap-walk (graph tracers dims size out-axes)
  "%VMAP-WALK-VALUES で GRAPH をバッチ化して発行し直し、出力ごとに OUT-AXES（整数か NIL。
正規化前。個数は呼び出し側 %VMAP-CALL が %VMAP-PER-ITEM で出力の個数に揃えてある）の
位置へ動かした、出力のトレーサのリストを返す。"
  (multiple-value-bind (outs out-dims) (%vmap-walk-values graph tracers dims size)
    (loop for out in outs
          for dim in out-dims
          for var in (graph-outvars graph)
          for spec in out-axes
          collect (%vmap-finish-output out dim var spec size))))

(defun %vmap-batch-dims (specs avals)
  "引数ごとの IN-AXES 指定 SPECS を、正規化したバッチ軸（無ければ NIL）のリストにする。"
  (loop for spec in specs for aval in avals for i from 0
        collect (and spec (%vmap-normalize-axis spec (aval-rank aval)
                                                (format nil "IN-AXES（引数 ~D）" i)))))

(defun %vmap-inner-aval (aval dim)
  "AVAL からバッチ軸 DIM（NIL ならそのまま）を取り除いた aval。"
  (if dim
      (let ((shape (aval-shape aval)))
        (make-aval (append (subseq shape 0 dim) (nthcdr (1+ dim) shape)) (aval-dtype aval)))
      aval))

(defun %vmap-call (fn in-axes out-axes args)
  "FN（TRACEABLE-FUNCTION）を ARGS でバッチ化して呼び、出力を多値で返す。"
  (let ((arity (length (traceable-function-lambda-list fn))))
    (unless (= (length args) arity)
      (%vmap-error "引数の個数 ~D が関数の引数の個数 ~D と一致しない" (length args) arity)))
  (let* ((arguments (mapcar #'%vmap-argument args))
         (avals (mapcar #'%vmap-argument-aval arguments))
         (dims (%vmap-batch-dims (%vmap-per-item in-axes (length args) "IN-AXES") avals))
         (sizes (loop for dim in dims for aval in avals when dim collect (nth dim (aval-shape aval))))
         (size (first sizes)))
    (unless sizes
      (%vmap-error "バッチされた引数が1つもない（IN-AXES ~S）" in-axes))
    (unless (every (lambda (s) (= s size)) sizes)
      (%vmap-error "バッチ軸の長さが揃わない: ~S（IN-AXES ~S）" sizes in-axes))
    (let* ((graph (%call-with-fresh-trace (mapcar #'%vmap-inner-aval avals dims)
                                          (%traceable-function-function fn)))
           (out-specs (%vmap-per-item out-axes (length (graph-outvars graph)) "OUT-AXES")))
      (if *current-trace*
          (values-list
           (%vmap-walk graph
                       (mapcar (lambda (a aval)
                                 (if (typep a 'tracer) a (%lift-constant a aval *current-trace*)))
                               arguments avals)
                       dims size out-specs))
          (let ((batched (%call-with-fresh-trace
                          avals
                          (lambda (&rest tracers)
                            (values-list (%vmap-walk graph tracers dims size out-specs))))))
            (apply #'eval-graph batched arguments))))))

(defun vmap (f &key (in-axes 0) (out-axes 0))
  "F（WITH-TRACING / DEFJIT / JIT が作った関数。JIT は静的引数が無いものだけ）を、
引数のバッチ軸について一度にまとめて適用する関数を返す。結果は、バッチ軸で切り出した
各要素に F を適用して、出力の OUT-AXES の位置に積み直したものと一致する。

IN-AXES は引数ごとのバッチ軸（0始まりの整数。負なら末尾から数える）か NIL（その引数は
バッチしない。F にそのまま渡る）。整数か NIL を1つ渡すと全引数に共通、リストなら引数ごと
（個数は引数の個数と同じ）。OUT-AXES は出力ごとの、出力のバッチ軸の位置（整数）か NIL。
NIL にできるのは、出力が引数のバッチに依存しないとき（そうでなければ VMAP-ERROR）。
バッチに依存しない出力が整数の軸を指定されると、その軸に沿って複製する。

次のときは VMAP-ERROR（またはその子）を signal する: IN-AXES / OUT-AXES の個数・型・範囲が
不正、バッチ軸の長さが引数の間で揃わない、バッチされた引数が1つもない、バッチ軸を持つ値が
バッチ化ルールの無いプリミティブに渡る（NO-BATCH-RULE）。IN-AXES の個数と型はここで、
範囲と軸長は呼び出し時に検査する。

戻り値はトレースできる関数（TRACEABLE-FUNCTION）なので、配列・実数を渡して直接呼ぶ（eager）
ほか、(JIT (VMAP F))、WITH-TRACING の本体の中、GRAD の対象の中、別の VMAP の対象の中
（(VMAP (VMAP F)) の形で、入れ子のバッチ軸）で使える。トレース中の呼び出しは、バッチ化した
graph を呼び出し元のトレースへ展開する。バッチされていない入力だけの演算は、そのまま
（バッチ化ルールを呼ばずに）残る。

既知の制限: F が外側のトレースのトレーサを閉包で捕まえていると TRACING-ERROR になる。
外側の値は F の引数として渡すこと。F はトレーサ・配列・実数を返す（多値は可）が、リストは
返せない（必要なら (WITH-TRACING (...) (VALUES-LIST ...)) で包む）。複数出力の eqn を
持つ F（cond* / while-loop / scan）も使える（本体のサブグラフを再帰的にバッチ化する）。jit した関数を渡すと、中の
TRACEABLE-FUNCTION だけを使い、その JIT の :BACKEND は無視される（GRAD と同じ）。

注意: (VMAP F) は呼ぶたびに新しい関数オブジェクトを作る。JIT キャッシュは関数の同一性が
キーなので、ループの中で (JIT (VMAP F)) を作ると毎回コンパイルされる。ループの外で
1回だけ作ること（GRAD と同じ）。"
  (let* ((fn (%vmap-traceable f))
         (lambda-list (traceable-function-lambda-list fn)))
    (%vmap-per-item in-axes (length lambda-list) "IN-AXES")
    (unless (or (%vmap-axis-spec-p out-axes)
                (and (listp out-axes) (every #'%vmap-axis-spec-p out-axes)))
      (%vmap-error "OUT-AXES は整数・NIL、またはそのリストでなければならない: ~S" out-axes))
    (%make-traceable-function
     (copy-list lambda-list)
     (lambda (&rest args) (%vmap-call fn in-axes out-axes args)))))
