;;;; ad/grad: 公開 API の GRAD / VALUE-AND-GRAD（issue #86）。
;;;;
;;;; f を引数の aval で新しいトレース（%CALL-WITH-FRESH-TRACE）に1回トレースして
;;;; graph にし、VJP-GRAPH（#82）で reverse モードの微分 graph にする。その結果を、
;;;; 呼び出しが別のトレースの中なら INLINE-GRAPH で現在のトレースへ流し込み
;;;; （jit・with-tracing の本体・入れ子の grad が動く）、そうでなければ EVAL-GRAPH で
;;;; その場で評価する（eager の grad）。jit キャッシュは「関数の同一性（EQ）」が
;;;; キーなので、GRAPH はキャッシュしない: (GRAD F) を呼ぶたびに新しい関数オブジェクト
;;;; ができ、トレースは呼び出しごとに行う。
;;;;
;;;; 既知の制限（設計の正本 plan.md §8）: f が外側のトレースのトレーサを閉包で
;;;; 捕まえていると、新しいトレースの中でその値を使った時点で TRACING-ERROR になる。
;;;; 外側の値は f の引数として渡すこと。

(in-package #:nabla)

(define-condition grad-requires-scalar-output (autodiff-error)
  ((aval :initarg :aval :initform nil :reader grad-requires-scalar-output-aval))
  (:documentation
   "GRAD / VALUE-AND-GRAD の対象の関数が、スカラー（rank 0）の浮動小数点を
ちょうど1つ返さなかったときに signal する。AVAL は問題の出力の aval
（出力が1つでないときは NIL）。"))

(defun %grad-signal-autodiff (control &rest arguments)
  (error 'autodiff-error :format-control control :format-arguments arguments))

(defun %grad-traceable (f)
  "F（TRACEABLE-FUNCTION か、静的引数の無い JITTED-FUNCTION）の TRACEABLE-FUNCTION
を返す。それ以外、静的引数のある JITTED-FUNCTION は AUTODIFF-ERROR。"
  (typecase f
    (traceable-function f)
    (jitted-function
     (when (%jitted-function-static-positions f)
       (%grad-signal-autodiff "静的引数のある jit した関数は grad の対象にできない: ~S" f))
     (%jitted-function-fn f))
    (t (%grad-signal-autodiff "grad の対象は WITH-TRACING / DEFJIT / JIT が作った関数でなければならない: ~S" f))))

(defun %grad-normalize-argnums (argnums arity)
  "ARGNUMS（整数、または整数のリスト）を、ARITY に対して検査した整数のリストにして返す。"
  (let ((positions (if (listp argnums) argnums (list argnums))))
    (unless (and positions (every #'integerp positions))
      (%grad-signal-autodiff "ARGNUMS は整数か、空でない整数のリストでなければならない: ~S" argnums))
    (dolist (p positions)
      (unless (< -1 p arity)
        (%grad-signal-autodiff "ARGNUMS の ~D が範囲外（関数の引数は ~D 個）" p arity)))
    (unless (= (length positions) (length (remove-duplicates positions)))
      (%grad-signal-autodiff "ARGNUMS に重複がある: ~S" argnums))
    positions))

(defun %grad-argument (arg)
  "ARG を TRACER か配列にして返す（実数は rank 0 の配列にする。DOUBLE-FLOAT は :F64、
それ以外は :F32）。"
  (typecase arg
    (tracer arg)
    (real (%scalar-array arg (if (typep arg 'double-float) :f64 :f32)))
    (string (%grad-signal-autodiff "grad した関数の引数に文字列は渡せない: ~S" arg))
    (array arg)
    (t (%grad-signal-autodiff "grad した関数の引数は配列・実数・トレーサでなければならない: ~S" arg))))

(defun %grad-check-output (graph)
  "GRAPH の出力が、rank 0 の浮動小数点ちょうど1つであることを確かめる。そうでなければ
GRAD-REQUIRES-SCALAR-OUTPUT。"
  (let ((outvars (graph-outvars graph)))
    (unless (= 1 (length outvars))
      (error 'grad-requires-scalar-output
             :format-control "微分する関数は出力をちょうど1つ返さなければならない（~D 個返した）"
             :format-arguments (list (length outvars))))
    (let ((aval (var-aval (first outvars))))
      (unless (and (zerop (aval-rank aval)) (%float-dtype-p (aval-dtype aval)))
        (error 'grad-requires-scalar-output
               :aval aval
               :format-control "微分する関数の出力は rank 0 の浮動小数点でなければならない: shape ~S、dtype ~S"
               :format-arguments (list (aval-shape aval) (aval-dtype aval)))))
    (var-aval (first outvars))))

(defun %grad-call (fn positions argnums value-p args)
  "FN（TRACEABLE-FUNCTION）を ARGS でトレースして微分し、GRAD / VALUE-AND-GRAD の
結果を返す（多値）。POSITIONS は検査済みの argnums の整数のリスト、ARGNUMS は
利用者が渡したそのままの値（整数かリストか。戻り値の形を決める）。"
  (let ((arity (length (traceable-function-lambda-list fn))))
    (unless (= (length args) arity)
      (%grad-signal-autodiff "引数の個数 ~D が関数の引数の個数 ~D と一致しない" (length args) arity)))
  (let* ((arguments (mapcar #'%grad-argument args))
         (avals (mapcar (lambda (a) (if (typep a 'tracer) (tracer-aval a) (array-aval a))) arguments))
         (graph (%call-with-fresh-trace avals (%traceable-function-function fn)))
         (out-aval (%grad-check-output graph)))
    (loop for p in positions
          unless (%float-dtype-p (aval-dtype (nth p avals)))
            do (%grad-signal-autodiff "引数 ~D の dtype ~S は浮動小数点でないので勾配を取れない"
                                      p (aval-dtype (nth p avals))))
    (let* ((sorted (sort (copy-list positions) #'<))
           (nonzero (loop for i below (length avals) collect (and (member i sorted) t)))
           (vjp (vjp-graph graph :nonzero nonzero))
           (seed (%scalar-array 1 (aval-dtype out-aval)))
           (results
             (if *current-trace*
                 (inline-graph vjp
                               (append (mapcar (lambda (a aval)
                                                 (if (typep a 'tracer) a (%lift-constant a aval *current-trace*)))
                                               arguments avals)
                                       (list (%lift-constant seed out-aval *current-trace*))))
                 (multiple-value-list (apply #'eval-graph vjp (append arguments (list seed))))))
           (value (first results))
           (grads (mapcar (lambda (p) (nth (position p sorted) (rest results))) positions)))
      (let ((grad (if (integerp argnums) (first grads) grads)))
        (if value-p (values value grad) grad)))))

(defun %make-grad-function (f argnums value-p)
  (let* ((fn (%grad-traceable f))
         (lambda-list (traceable-function-lambda-list fn))
         (positions (%grad-normalize-argnums argnums (length lambda-list))))
    (%make-traceable-function
     (copy-list lambda-list)
     (lambda (&rest args) (%grad-call fn positions argnums value-p args)))))

(defun grad (f &key (argnums 0))
  "F（WITH-TRACING / DEFJIT / JIT が作った関数。JIT は静的引数が無いものだけ）の、
スカラー出力についての勾配を計算する関数を返す。F は rank 0 の浮動小数点を1つ
返さなければならない（そうでなければ GRAD-REQUIRES-SCALAR-OUTPUT）。

ARGNUMS は微分する引数の位置（0始まりの整数、または整数のリスト。既定は0）。
整数なら勾配の配列を1つ、リストならリストの順に勾配のリストを返す（JAX と同じ）。
勾配の shape・dtype はその引数と同じ。微分する引数は浮動小数点でなければならない。
ARGNUMS が範囲外・重複などで不正なら、ここで AUTODIFF-ERROR を signal する。

戻り値はトレースできる関数（TRACEABLE-FUNCTION）なので、次のどれでも使える:
- 配列・実数を渡して直接呼ぶ（eager。引数ごとにトレースし、graph を評価する）。
  実数（DOUBLE-FLOAT は :F64、それ以外は :F32）は rank 0 の配列として扱い、勾配は配列で返る
- (JIT (GRAD F))、WITH-TRACING の本体の中、別の GRAD の対象の中（高階微分。
  (GRAD (GRAD F)) の形で使える）。トレース中の呼び出しは、微分した graph を
  呼び出し元のトレースへ展開する
  - ARGNUMS がリストのとき、勾配のリストは JIT の出力にできない（JIT は配列か
    トレーサだけを返せる）。(WITH-TRACING (...) (VALUES-LIST (FUNCALL g ...))) で包む

既知の制限: F が外側のトレースのトレーサを閉包で捕まえていると TRACING-ERROR になる。
外側の値は F の引数として渡すこと。

jit した関数（JITTED-FUNCTION）を F に渡すと、中の TRACEABLE-FUNCTION だけを使う。
その JIT の :BACKEND は無視される。backend 上で動かすには、外側を
(JIT (GRAD ...) :BACKEND b) にする。

注意: (GRAD F) は呼ぶたびに新しい関数オブジェクトを作る。JIT キャッシュは関数の同一性が
キーなので、ループの中で (JIT (GRAD F)) を作ると毎回コンパイルされる（JAX と同じ）。
ループの外で (JIT (GRAD F)) を1回だけ作るか、DEFJIT の本体の中で GRAD を使う。"
  (%make-grad-function f argnums nil))

(defun value-and-grad (f &key (argnums 0))
  "GRAD と同じ引数・同じ制限で、F の値と勾配を一度に計算する関数を返す。戻り値は
多値の (値 勾配)。値は F の出力（rank 0）、勾配の形は GRAD の ARGNUMS の説明を参照。
値と勾配を別々に求めるより、トレースが1回で済む。

(VALUE-AND-GRAD F) も呼ぶたびに新しい関数オブジェクトを作る。ループの中で
(JIT (VALUE-AND-GRAD F)) を作ると毎回コンパイルされる（GRAD の注意を参照）。"
  (%make-grad-function f argnums t))
