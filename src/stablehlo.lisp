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

(defun %stablehlo-eqn-out (eqn)
  "EQN の唯一の outvar を返す。outvars がちょうど1つでなければ ERROR を
signal する（フェーズ1は常に1つのはずで、これは #29/#39 と同じ制約）。"
  (let ((outvars (eqn-outvars eqn)))
    (unless (= 1 (length outvars))
      (error "emit-stablehlo: eqn の outvars が1つでない graph は出力できない（フェーズ1では対象外）: ~S" eqn))
    (first outvars)))

(defun %stablehlo-eqn-emit-text (numbers eqn)
  "EQN の1つの MLIR 演算（複数行のこともある）のテキストを、PRIMITIVE の
:EMIT を呼んで得る。:EMIT が無ければ PRIMITIVE-NOT-EMITTABLE を signal する。"
  (let* ((prim (eqn-prim eqn))
         (emit (primitive-emit prim)))
    (unless emit
      (error 'primitive-not-emittable :name (primitive-name prim)))
    (let* ((out-var (%stablehlo-eqn-out eqn))
           (invars (eqn-invars eqn))
           (in-names (mapcar (lambda (v) (%var-name numbers v)) invars))
           (in-avals (mapcar #'var-aval invars))
           (out-name (%var-name numbers out-var))
           (out-aval (var-aval out-var)))
      (apply emit in-names in-avals out-name out-aval (eqn-params eqn)))))

(defun %stablehlo-eqn-lines (numbers eqn index)
  "EQN の全ての出力行に \" loc(\\\"eqn-INDEX\\\")\" を付けたリストを返す
（複数行を返す :EMIT でも、その全ての行に付ける。契約 §3 のピットフォール(3):
補助 SSA 名の行に loc が付いても無害）。"
  (let ((text (%stablehlo-eqn-emit-text numbers eqn)))
    (mapcar (lambda (line) (format nil "~A loc(\"eqn-~D\")" line index))
            (%split-lines text))))

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

次の graph はエラーになる: eqn の outvars が1つでないもの（ERROR、
フェーズ1では対象外）、:EMIT を持たないプリミティブを使うもの
（PRIMITIVE-NOT-EMITTABLE）、invars/constants/他の eqn の outvars の
どれにも定義されていない var を参照しているもの（MALFORMED-GRAPH、
%VAR-NAME 経由）。"
  (let ((numbers (%assign-var-numbers graph)))
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
