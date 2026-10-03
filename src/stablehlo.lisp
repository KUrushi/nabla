;;;; stablehlo: graph を StableHLO のテキストに変換する emitter（issue #33）。
;;;;
;;;; StableHLO は出力先であって内部表現ではない（CLAUDE.md）。core（このファイル）
;;;; は実行系を知らない。実行系の診断テキストからどの eqn が失敗したかを
;;;; 逆引きする GRAPH-EQN-FOR-DIAGNOSTIC も、渡された文字列を解析するだけで
;;;; 実行系固有の型は一切扱わない。
;;;;
;;;; SSA 名は src/ir-print.lisp の %ASSIGN-VAR-NUMBERS / %VAR-NAME をそのまま
;;;; 再利用する（invars → constants → 各 eqn の outvars の順）。テンソル型と
;;;; dtype の綴りは src/primitive.lisp の TENSOR-TYPE-STRING / DTYPE-MLIR-NAME
;;;; を再利用する。

(in-package #:nabla)

(define-condition primitive-not-emittable (error)
  ((name :initarg :name :reader primitive-not-emittable-name))
  (:report
   (lambda (condition stream)
     (format stream "プリミティブ ~S は :EMIT を持たないため EMIT-STABLEHLO できない"
             (primitive-not-emittable-name condition))))
  (:documentation
   "EMIT-STABLEHLO が、:EMIT を持たないプリミティブを使う EQN に当たったときに
signal される。NAME はそのプリミティブ名（キーワード）。PRIMITIVE-NOT-EVALUABLE
（src/eval.lisp）の EMIT 版。"))

;;; ---- 定数リテラル ----

(defun %stablehlo-f32-bits (value)
  "SINGLE-FLOAT VALUE のビット列を符号なし32bit整数として返す。"
  (logand (sb-kernel:single-float-bits value) #xFFFFFFFF))

(defun %stablehlo-f64-bits (value)
  "DOUBLE-FLOAT VALUE のビット列を符号なし64bit整数として返す。"
  (logior (ash (logand (sb-kernel:double-float-high-bits value) #xFFFFFFFF) 32)
          (logand (sb-kernel:double-float-low-bits value) #xFFFFFFFF)))

(defun %stablehlo-non-finite-literal (value dtype)
  "非有限（NaN / ±inf）の :F32 / :F64 の値を、16進ビット列のリテラルにする
（\"0x7FC00000\" のような形。MLIR は非有限の10進リテラルを受け付けないため）。"
  (ecase dtype
    (:f32 (format nil "0x~8,'0X" (%stablehlo-f32-bits value)))
    (:f64 (format nil "0x~16,'0X" (%stablehlo-f64-bits value)))))

(defun %stablehlo-finite-float-literal (value dtype)
  "有限の :F32 / :F64 の値を、10進の丸め込み無し（round-trip）リテラルにする。
*READ-DEFAULT-FLOAT-FORMAT* を要素型に束縛して PRIN1 するので、SBCL が
DOUBLE-FLOAT に付ける \"d0\" のような MLIR が読めないマーカーは付かない
（f32 は \"1.0e10\" のような指数表記になることがあるが、MLIR はこれを読める）。"
  (let ((element-type (ecase dtype (:f32 'single-float) (:f64 'double-float))))
    (with-standard-io-syntax
      (let ((*read-default-float-format* element-type)
            (*print-readably* nil))
        (prin1-to-string value)))))

(defun %stablehlo-float-literal (value dtype)
  ":F32 / :F64 の1要素のリテラルを返す（有限なら10進、NaN / ±inf なら16進ビット列）。"
  (if (or (sb-ext:float-nan-p value) (sb-ext:float-infinity-p value))
      (%stablehlo-non-finite-literal value dtype)
      (%stablehlo-finite-float-literal value dtype)))

(defun %stablehlo-f16-literal (value)
  "bf16 / f16 の1要素のリテラルを返す。格納されているビット列
（(UNSIGNED-BYTE 16)）をそのまま16進で書く（有限・非有限を区別しない。
CLAUDE.md の約束どおり bf16/f16 はビット列のまま持つため、デコードせずに
そのまま書くのが最も忠実で、丸め誤差も入らない）。"
  (format nil "0x~4,'0X" value))

(defun %stablehlo-element-literal (value dtype)
  "AVAL の要素1つのリテラルを返す（\"dense<...>\" の中身に埋め込む文字列）。"
  (ecase dtype
    ((:f32 :f64) (%stablehlo-float-literal value dtype))
    ((:bf16 :f16) (%stablehlo-f16-literal value))
    (:i1 (if (= value 1) "true" "false"))))

(defun %nest-elements (array dtype dims next-index)
  "DIMS（残りの次元のリスト）ぶんだけネストした \"[...]\" を組み立てる。
NEXT-INDEX は呼ぶたびに次の row-major index を返す関数（副作用で進む）。"
  (if (null dims)
      (%stablehlo-element-literal (row-major-aref array (funcall next-index)) dtype)
      (format nil "[~{~A~^, ~}]"
              (loop repeat (first dims) collect (%nest-elements array dtype (rest dims) next-index)))))

(defun %constant-literal (array dtype)
  "ARRAY（DTYPE の値を持つ多次元配列）の \"dense<...>\" 全体を返す。
rank 0 は角括弧無し、要素数0は \"dense<>\"、rank ≥ 1 は行優先の次元順に
ネストした角括弧。"
  (let ((shape (array-dimensions array)))
    (cond
      ((zerop (array-total-size array)) "dense<>")
      ((null shape) (format nil "dense<~A>" (%stablehlo-element-literal (row-major-aref array 0) dtype)))
      (t (let ((index 0))
           (flet ((next-index () (prog1 index (incf index))))
             (format nil "dense<~A>" (%nest-elements array dtype shape #'next-index))))))))

;;; ---- ヘッダ・戻り値 ----

(defun %stablehlo-arg-string (numbers var)
  (format nil "~A: ~A" (%var-name numbers var) (tensor-type-string (var-aval var))))

(defun %stablehlo-result-clause (outvars)
  "func.func の \" -> (T1, T2)\" 節を返す。OUTVARS が空なら空文字列
（\"-> ()\" は MLIR が受け付けないので、出力の無い graph は丸ごと省く）。"
  (if outvars
      (format nil " -> (~{~A~^, ~})" (mapcar (lambda (v) (tensor-type-string (var-aval v))) outvars))
      ""))

(defun %stablehlo-header-line (numbers graph function-name)
  (format nil "  func.func @~A(~{~A~^, ~})~A {"
          function-name
          (mapcar (lambda (v) (%stablehlo-arg-string numbers v)) (graph-invars graph))
          (%stablehlo-result-clause (graph-outvars graph))))

(defun %stablehlo-return-line (numbers outvars)
  (if outvars
      (format nil "func.return ~{~A~^, ~} : ~{~A~^, ~}"
              (mapcar (lambda (v) (%var-name numbers v)) outvars)
              (mapcar (lambda (v) (tensor-type-string (var-aval v))) outvars))
      "func.return"))

;;; ---- 定数・eqn の本体行 ----

(defun %stablehlo-constant-line (numbers var array)
  (format nil "~A = stablehlo.constant ~A : ~A"
          (%var-name numbers var)
          (%constant-literal array (aval-dtype (var-aval var)))
          (tensor-type-string (var-aval var))))

(defun %split-lines (text)
  "TEXT を #\\Newline で分割した文字列のリストを返す（primitive の :EMIT が
複数行を返すことがある。reduce-sum/reduce-max が既にそう）。"
  (loop for start = 0 then (1+ pos)
        for pos = (position #\Newline text :start start)
        collect (subseq text start pos)
        while pos))

(defun %stablehlo-eqn-emit-text (numbers eqn)
  "EQN の1つの MLIR 演算（複数行のこともある）のテキストを、PRIMITIVE の
:EMIT を呼んで得る。:EMIT が無ければ PRIMITIVE-NOT-EMITTABLE を signal する。
複数出力のプリミティブ（契約 C1）の :EMIT には、出力の名前と AVAL をリストで渡す
（\"%8, %9 = ...\" の左辺は :EMIT が書く）。"
  (let* ((prim (eqn-prim eqn))
         (emit (primitive-emit prim)))
    (unless emit
      (error 'primitive-not-emittable :name (primitive-name prim)))
    (let* ((outvars (eqn-outvars eqn))
           (invars (eqn-invars eqn))
           (in-names (mapcar (lambda (v) (%var-name numbers v)) invars))
           (in-avals (mapcar #'var-aval invars))
           (out-names (mapcar (lambda (v) (%var-name numbers v)) outvars))
           (out-avals (mapcar #'var-aval outvars)))
      (cond
        ((primitive-multiple-outputs-p prim)
         (apply emit in-names in-avals out-names out-avals (eqn-params eqn)))
        ((= 1 (length outvars))
         (apply emit in-names in-avals (first out-names) (first out-avals) (eqn-params eqn)))
        (t
         (error "emit-stablehlo: 単一出力のプリミティブの eqn の outvars が1つでない: ~S" eqn))))))

(defun %brace-balance (line)
  "LINE の中の \"{\" の個数から \"}\" の個数を引いた値（リージョンの深さの増減）。"
  (- (count #\{ line) (count #\} line)))

(defun %stablehlo-eqn-lines (numbers eqn index)
  "EQN の出力行に \" loc(\\\"eqn-INDEX\\\")\" を付けたリストを返す。リージョンの
外（深さ 0 で終わる行）の全ての行に付ける（複数行を返す :EMIT でも、補助 SSA 名の
行（\"%idx = stablehlo.constant ...\" など）まで付けて、実行系の診断から eqn を逆引き
できるようにする。契約 §3 のピットフォール(3)）。リージョンの途中の行（\"{\" で開いた
まま終わる行や、リージョンの中身）には付けない。MLIR の loc は演算の後にしか置けず、
リージョンを閉じる最後の行（\"}) : ... -> ...\"）に付ければ演算全体に効くため。
INDEX が NIL なら loc を付けない（リージョンの中の eqn。外側の GRAPH-EQNS の位置と
対応しないため）。"
  (let ((lines (%split-lines (%stablehlo-eqn-emit-text numbers eqn)))
        (depth 0))
    (if (null index)
        lines
        (loop for line in lines
              do (incf depth (%brace-balance line))
              collect (if (zerop depth)
                          (format nil "~A loc(\"eqn-~D\")" line index)
                          line)))))

;;; ---- 本体 ----

(defun %stablehlo-body-lines (numbers graph)
  "GRAPH の func.func 本体（定数・eqn・return）の行のリストを、インデント
無しで返す。NUMBERS は %ASSIGN-VAR-NUMBERS が返す var → 番号の表
（EMIT-STABLEHLO がヘッダ行と共有して1回だけ計算する）。"
  (append
   (mapcar (lambda (entry) (%stablehlo-constant-line numbers (car entry) (cdr entry)))
           (graph-constants graph))
   (loop for eqn in (graph-eqns graph)
         for index from 0
         append (%stablehlo-eqn-lines numbers eqn index))
   (list (%stablehlo-return-line numbers (graph-outvars graph)))))

(defvar *stablehlo-region-counter* nil
  "EMIT-STABLEHLO が 0 に束縛する整数。%STABLEHLO-REGION-LINES がリージョンを
出すたびに 1 ずつ増やし、その値 k からリージョンの SSA 名の接頭辞 \"%s<k>_\" を作る
（入れ子のリージョンや、同じ graph から出す複数のリージョンの名前が衝突しない）。")

(defun %stablehlo-region-return-line (numbers outvars)
  "リージョンの終わりの stablehlo.return。OUTVARS が無ければ裸の \"stablehlo.return\"。"
  (if outvars
      (format nil "stablehlo.return ~{~A~^, ~} : ~{~A~^, ~}"
              (mapcar (lambda (v) (%var-name numbers v)) outvars)
              (mapcar (lambda (v) (tensor-type-string (var-aval v))) outvars))
      "stablehlo.return"))

(defun %stablehlo-region-lines (graph &key (arg-names nil arg-names-p))
  "サブグラフ GRAPH を StableHLO のリージョンの中身として出した行のリストを返す
（契約 C1。リージョンを持つ演算を出す :EMIT が、\"{\" と \"}\" の間に置く）。
先頭と末尾の波括弧は含まない。リージョンは stablehlo.return で終わる
（出力が無ければ裸の stablehlo.return）。

ARG-NAMES は GRAPH の invars と同じ長さのリストで、要素ごとに invar の扱いを決める。
  - 文字列: その invar を外側の SSA 名に結びつける（ブロック引数にしない。
    stablehlo.if / case の枝用。リージョンは外側の値を直接参照できる）。
  - NIL: その invar をブロック引数にする。
ブロック引数になる invar が1つでもあれば、先頭に \"^bb0(%s<k>_0: tensor<...>, ...):\"
を出し、ブロック引数を invars の順に並べる（stablehlo.while の cond / body 用。
carry をブロック引数、閉包で捕まえた値を外側の名前にするなら
(nil nil \"%7\" \"%3\") のように渡す）。全て文字列なら ^bb0 の行は出さない。
ARG-NAMES を渡さないと、全ての invar がブロック引数になる。

リージョンの中の SSA 名には、EMIT-STABLEHLO ごとの *STABLEHLO-REGION-COUNTER*
から作る接頭辞 \"%s<k>_\" を付ける。定数はリージョンの中で出す。リージョンの中の
eqn の行には loc を付けない。サブグラフは外側の var を参照しない閉じた graph
なので、外側の名前に結びつくのは ARG-NAMES の文字列だけ。"
  (unless *stablehlo-region-counter*
    (error "%stablehlo-region-lines は EMIT-STABLEHLO の中（*STABLEHLO-REGION-COUNTER* が束縛されている間）でしか呼べない"))
  (when (and arg-names-p (/= (length arg-names) (length (graph-invars graph))))
    (error "%stablehlo-region-lines: arg-names の個数 ~D が graph の入力の個数 ~D と一致しない"
           (length arg-names) (length (graph-invars graph))))
  (let* ((k (incf *stablehlo-region-counter*))
         (*var-name-prefix* (format nil "%s~D_" k))
         (*var-name-overrides* (make-hash-table :test 'eq))
         (numbers (%assign-var-numbers graph))
         (names (if arg-names-p arg-names (make-list (length (graph-invars graph)))))
         (block-args '()))
    (loop for var in (graph-invars graph)
          for name in names
          do (if name
                 (setf (gethash var *var-name-overrides*) name)
                 (push var block-args)))
    (append
     (when block-args
       (list (format nil "^bb0(~{~A~^, ~}):"
                     (mapcar (lambda (v) (%stablehlo-arg-string numbers v)) (reverse block-args)))))
     (mapcar (lambda (entry) (%stablehlo-constant-line numbers (car entry) (cdr entry)))
             (graph-constants graph))
     (loop for eqn in (graph-eqns graph)
           append (%stablehlo-eqn-lines numbers eqn nil))
     (list (%stablehlo-region-return-line numbers (graph-outvars graph))))))

(defun emit-stablehlo (graph &key (function-name "main"))
  "GRAPH を StableHLO のテキストに変換して返す。無名の module の中に
FUNCTION-NAME（既定 \"main\"）という1つの func.func を出す（BACKEND-INVOKE が
\"module.<name>\" を仮定しているため、名前付き module にはしない）。

各 var の SSA 名は %ASSIGN-VAR-NUMBERS / %VAR-NAME（src/ir-print.lisp）が
決める出現順（invars → constants → 各 eqn の outvars）の \"%N\"。関数の
引数・戻り値の型は TENSOR-TYPE-STRING（src/primitive.lisp）が決める。
戻り値の型は常に丸括弧で括る（\"-> (T1, T2)\"）。出力が無い graph は
\"->\" 節ごと省く（MLIR が \"-> ()\" を受け付けないため）。同じ var が
複数回 GRAPH-OUTVARS に現れれば、その分だけ繰り返して返す。

各 eqn の唯一の出力行には \" loc(\\\"eqn-N\\\")\"（N は GRAPH-EQNS 中の
0始まりの位置）を付ける。実行系のコンパイルエラーがこの loc を含む
診断を返したとき、GRAPH-EQN-FOR-DIAGNOSTIC でどの eqn が原因かを逆引き
できる。:EMIT が複数行の文字列を返すプリミティブ（reduce-sum など）は、
その全ての行に同じ loc を付ける。

複数出力のプリミティブ（契約 C1）の :EMIT には出力の名前と AVAL をリストで渡す。
リージョンの外の行にはすべて loc が付き、リージョンの途中の行には付かない。

次の graph はエラーになる: 単一出力のプリミティブで eqn の outvars が1つでない
もの（ERROR）、:EMIT を持たないプリミティブを使うもの
（PRIMITIVE-NOT-EMITTABLE）、invars/constants/他の eqn の outvars の
どれにも定義されていない var を参照しているもの（MALFORMED-GRAPH、
%VAR-NAME 経由）。"
  (let ((numbers (%assign-var-numbers graph))
        (*stablehlo-region-counter* 0))
    (format nil "module {~%~A~%~{    ~A~%~}  }~%}"
            (%stablehlo-header-line numbers graph function-name)
            (%stablehlo-body-lines numbers graph))))

(defun graph-eqn-for-diagnostic (graph text)
  "TEXT（何らかの実行系がコンパイルエラーとして返す診断テキスト、
または任意の文字列）の中に現れる最初の \"loc(\\\"eqn-N\\\"\" を探し、
(VALUES eqn n) を返す。TEXT に loc が無い、N が GRAPH-EQNS の範囲外
（負・大きすぎる）のときは (VALUES NIL NIL)。core は実行系を知らないので、
呼び出し側（実行系との連携層など）が診断オブジェクトを (PRINC-TO-STRING condition)
のような文字列にしてから渡す。内部の関数（このワークではまだ export
しない。jit のエラー報告と合わせてフェーズ1の後半で export を検討する）。"
  (let ((marker "loc(\"eqn-"))
    (let ((pos (search marker text)))
      (if (null pos)
          (values nil nil)
          (let* ((start (+ pos (length marker)))
                 (end (or (position-if-not #'digit-char-p text :start start) (length text))))
            (if (= start end)
                (values nil nil)
                (let* ((n (parse-integer text :start start :end end))
                       (eqn (and (<= 0 n) (nth n (graph-eqns graph)))))
                  (if eqn (values eqn n) (values nil nil)))))))))
