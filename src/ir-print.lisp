;;;; ir-print: graph を jaxpr 風のテキストに印字する / 読み戻す（issue #29 後半、u1b）。
;;;;
;;;; レイアウトは手書きの FORMAT で固定し（*print-pretty* には依存しない）、
;;;; 文字列一致の往復（PRINT-GRAPH → READ-GRAPH → PRINT-GRAPH）が
;;;; string= で成り立つようにする。テキスト形式そのものはフェーズ1では
;;;; 公開契約にしないので、READ-GRAPH は内部シンボルのままにする（PRINT-GRAPH
;;;; だけを export する）。

(in-package #:cl-user)

(defpackage #:nabla.graph-syntax
  (:use)
  (:documentation
   "READ-GRAPH がテキストを読むときの *PACKAGE*。何も USE しない空の
パッケージにすることで、graph のテキストに現れるシンボル（var 名・
プリミティブ名・dtype 名など）が nabla や CL のシンボルと衝突しない
ようにする（シンボルの照合は SYMBOL-NAME の文字列比較で行うので、
このパッケージにインターンされること自体しか使わない）。"))

(in-package #:nabla)

(define-condition graph-syntax-error (error)
  ((form :initarg :form :initform nil :reader graph-syntax-error-form)
   (format-control :initarg :format-control :reader graph-syntax-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader graph-syntax-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "graph のテキストが不正: ~?"
             (graph-syntax-error-format-control condition)
             (graph-syntax-error-format-arguments condition))))
  (:documentation
   "READ-GRAPH が、graph のテキスト表現の構造上の誤り（節が欠けている、
`:=` が無い、印字された aval と再計算した aval が食い違う、など）を
検出したときに signal する。FORM は問題があったフォーム（分からなければ
NIL）。未登録のプリミティブは UNKNOWN-PRIMITIVE、未定義の var 参照は
MALFORMED-GRAPH のまま伝わる（この条件にはならない）。"))

;;; ---- 印字（PRINT-GRAPH） ----

(defun %assign-var-numbers (graph)
  "GRAPH に現れる各 var に、出現順（invars → constants → 各 eqn の
outvars）の番号を振り、var から番号への EQ ハッシュ表を返す。"
  (let ((numbers (make-hash-table :test 'eq))
        (n 0))
    (flet ((assign (var)
             (setf (gethash var numbers) n)
             (incf n)))
      (dolist (var (graph-invars graph)) (assign var))
      (dolist (entry (graph-constants graph)) (assign (car entry)))
      (dolist (eqn (graph-eqns graph))
        (dolist (var (eqn-outvars eqn)) (assign var))))
    numbers))

(defvar *var-name-prefix* "%"
  "%VAR-NAME が番号の前に付ける接頭辞。StableHLO のリージョン（契約 C1）の
中だけ \"%s<k>_\" に束縛し、外側の SSA 名と衝突させない。")

(defvar *var-name-overrides* nil
  "NIL か、var → 名前（文字列）の EQ ハッシュ表。あれば %VAR-NAME は番号より
先にこちらを引く。StableHLO のリージョンの invars を外側の SSA 名に結びつける
ために使う（%STABLEHLO-REGION-LINES の :ARG-NAMES）。")

(defun %var-name (numbers var)
  "VAR の印字名 \"%N\" を返す。*VAR-NAME-OVERRIDES* に VAR があればその名前。
VAR が NUMBERS に無ければ（GRAPH が未定義の
var を参照している、壊れた graph）、~D に NIL を渡して \"%nil\" のような
壊れた文字列を返す代わりに MALFORMED-GRAPH を signal する。"
  (multiple-value-bind (n found) (gethash var numbers)
    (when (and *var-name-overrides* (gethash var *var-name-overrides*))
      (return-from %var-name (gethash var *var-name-overrides*)))
    (unless found
      (error 'malformed-graph :graph nil
             :format-control "var ~S が graph のどこにも定義されていない（未定義参照）"
             :format-arguments (list var)))
    (format nil "~A~D" *var-name-prefix* n)))

(defun %print-shape (stream shape)
  "SHAPE（非負整数のリスト）を \"(2 3)\" のように印字する。rank 0（NIL）は
\"()\"（~S で NIL を印字すると \"nil\" になってしまうので、ここだけ手で
書く）。"
  (write-char #\( stream)
  (loop for (dim . rest) on shape
        do (format stream "~D" dim)
           (when rest (write-char #\Space stream)))
  (write-char #\) stream))

(defun %check-finite-element (value)
  "VALUE（single-float / double-float）が NaN でも ±inf でもないことを
確かめ、そのまま返す。非有限なら PRINT-GRAPH の呼び出しをエラーにする
（フェーズ1では対象外。CLAUDE.md/契約のとおり）。"
  (when (or (sb-ext:float-nan-p value) (sb-ext:float-infinity-p value))
    (error "print-graph: 非有限の定数（NaN / ±inf）は印字できない: ~S" value))
  value)

(defun %print-const-elements (stream array dtype)
  (dotimes (i (array-total-size array))
    (write-char #\Space stream)
    (let ((value (row-major-aref array i)))
      (format stream "~S" (if (member dtype '(:f32 :f64))
                               (%check-finite-element value)
                               value)))))

(defun %print-entry-header (stream numbers var)
  "\"(%N dtype shape\" を書く（末尾の \")\" や、その後に続く内容は呼び出し側
が書く）。:in / :const / :eqns の3つの節エントリで共通の先頭部分。"
  (let ((aval (var-aval var)))
    (format stream "(~A ~A " (%var-name numbers var) (dtype-mlir-name (aval-dtype aval)))
    (%print-shape stream (aval-shape aval))))

(defun %print-in-entry (stream numbers var)
  (%print-entry-header stream numbers var)
  (write-char #\) stream))

(defun %print-const-entry (stream numbers var array)
  (%print-entry-header stream numbers var)
  (%print-const-elements stream array (aval-dtype (var-aval var)))
  (write-char #\) stream))

(defun %print-list-tail (stream tail)
  (cond
    ((null tail) nil)
    ((consp tail)
     (write-char #\Space stream)
     (%print-form stream (car tail))
     (%print-list-tail stream (cdr tail)))
    (t
     (write-string " . " stream)
     (%print-form stream tail))))

(defvar *print-indent* 0
  "印字中の graph の入れ子の深さに応じた、各行の先頭に足す空白の数
（サブグラフの中の行を、深さごとに 4 桁ずつ字下げする）。")

(defun %print-newline (stream)
  "改行し、*PRINT-INDENT* 桁の空白を書く。"
  (write-char #\Newline stream)
  (dotimes (i *print-indent*) (write-char #\Space stream)))

(defun %print-form (stream form)
  "FORM を ~S に近い形で印字するが、NIL（空リスト）だけは \"()\" にする
（~S の NIL は *print-pretty* が NIL のとき \"nil\" になり、READ-GRAPH が
*package* に NABLA.GRAPH-SYNTAX を束縛して読むと、CL:NIL とは EQ でない
別のシンボルになってしまう。eqn の params のように空リストがありうる値に
~S をそのまま使わない理由）。"
  (cond
    ((null form) (write-string "()" stream))
    ((graph-p form)
     (let ((*print-indent* (+ *print-indent* 4)))
       (%print-graph-body form stream)))
    ((consp form)
     (write-char #\( stream)
     (%print-form stream (car form))
     (%print-list-tail stream (cdr form))
     (write-char #\) stream))
    (t (format stream "~S" form))))

(defun %print-eqn-entry (stream numbers eqn)
  "eqn を1つ印字する。出力が1つなら (%N dtype shape := prim params invars...)、
複数（契約 C1）なら出力のヘッダを並べた
((%N dtype shape) (%M dtype shape) := prim params invars...)。params に
サブグラフがあれば、その場で入れ子に印字する（%PRINT-FORM）。"
  (let ((outvars (eqn-outvars eqn))
        (prim-name (primitive-name (eqn-prim eqn))))
    (if (primitive-multiple-outputs-p (eqn-prim eqn))
        (progn
          (write-char #\( stream)
          (loop for (out . rest) on outvars
                do (write-char #\( stream)
                   (let ((aval (var-aval out)))
                     (format stream "~A ~A " (%var-name numbers out) (dtype-mlir-name (aval-dtype aval)))
                     (%print-shape stream (aval-shape aval)))
                   (write-char #\) stream)
                   (when rest (write-char #\Space stream))))
        (progn
          (unless (= 1 (length outvars))
            (error "print-graph: 単一出力のプリミティブの eqn の outvars が1つでない: ~S" eqn))
          (%print-entry-header stream numbers (first outvars))))
    (format stream " := ~A " (string-downcase (symbol-name prim-name)))
    (%print-form stream (eqn-params eqn))
    (dolist (v (eqn-invars eqn))
      (format stream " ~A" (%var-name numbers v)))
    (write-char #\) stream)))

(defun %print-graph-body (graph stream)
  (let ((numbers (%assign-var-numbers graph)))
    (write-string "(graph" stream)
    (%print-newline stream)
    (write-string " (:in" stream)
    (dolist (var (graph-invars graph))
      (write-char #\Space stream)
      (%print-in-entry stream numbers var))
    (write-char #\) stream)
    (%print-newline stream)
    (write-string " (:const" stream)
    (dolist (entry (graph-constants graph))
      (write-char #\Space stream)
      (%print-const-entry stream numbers (car entry) (cdr entry)))
    (write-char #\) stream)
    (%print-newline stream)
    (write-string " (:eqns" stream)
    (dolist (eqn (graph-eqns graph))
      (%print-newline stream)
      (write-string "  " stream)
      (%print-eqn-entry stream numbers eqn))
    (write-char #\) stream)
    (%print-newline stream)
    (write-string " (:out" stream)
    (dolist (var (graph-outvars graph))
      (write-char #\Space stream)
      (write-string (%var-name numbers var) stream))
    (write-string "))" stream)))

(defun print-graph (graph &optional stream)
  "GRAPH を jaxpr 風のテキストに変換する。呼び方は FORMAT と同じ規約:
STREAM が NIL（既定）なら文字列を返し、T なら *STANDARD-OUTPUT* に、
それ以外はそのストリームに書いて NIL を返す。

var の名前は出現順（invars → constants → 各 eqn の outvars）の \"%N\"、
dtype とプリミティブ名は小文字、shape はリスト（rank 0 は \"()\"）で
印字する。READ-GRAPH で読み戻すと同じ graph になる（内部の関数。
テキスト形式はフェーズ1では公開契約にしない）。

eqn の params にサブグラフ（GRAPH 構造体。契約 C2）があれば、その場で入れ子に
印字する（サブグラフの var 名は、サブグラフの中で 0 から振り直し、サブグラフの
中の行は入れ子の深さごとに 4 桁字下げする）。複数出力の eqn（契約 C1）は
((%N dtype shape) (%M dtype shape) := ...) の形で印字する。

次の graph は印字できずエラーになる（いずれもフェーズ1では対象外）:
非有限（NaN / ±inf）の f32 / f64 定数を持つもの。また、
どこかの eqn の invars や GRAPH-OUTVARS が invars / constants / 他の eqn の
outvars のどれでも定義されていない var を参照している（未定義参照の
壊れた graph）場合も MALFORMED-GRAPH を signal する（\"%nil\" のような壊れた
テキストを黙って出力しない）。"
  (let ((text (with-standard-io-syntax
                (let ((*print-case* :downcase)
                      (*print-readably* nil)
                      (*print-indent* 0))
                  (with-output-to-string (out) (%print-graph-body graph out))))))
    (cond
      ((null stream) text)
      ((eq stream t) (write-string text *standard-output*) nil)
      (t (write-string text stream) nil))))

;;; ---- 読み込み（READ-GRAPH） ----

(defun %symbol-named-p (x name)
  (and (symbolp x) (string= (symbol-name x) name)))

(defun %graph-syntax-error (form format-control &rest format-arguments)
  (error 'graph-syntax-error :form form
         :format-control format-control :format-arguments format-arguments))

(defun %parse-dtype (sym)
  (unless (symbolp sym)
    (%graph-syntax-error sym "dtype が symbol でない: ~S" sym))
  (let ((keyword (intern (symbol-name sym) :keyword)))
    (unless (typep keyword 'dtype)
      (%graph-syntax-error sym "未知の dtype: ~S" sym))
    keyword))

(defun %normalize-graph-syntax-symbol (sym)
  "SYM が NABLA.GRAPH-SYNTAX パッケージにインターンされた \"T\" という
シンボルなら CL:T に、そうでなければ SYM をそのまま返す。READ-GRAPH は
*PACKAGE* を NABLA.GRAPH-SYNTAX（何も USE しない）に束縛して読むため、
params に現れうる T のような CL の定数シンボルは、そのままでは
CL:T と EQ でない別のシンボルとして読まれてしまう（キーワードは
パッケージに関係なく :KEYWORD にインターンされるので影響を受けない。
NIL は空リストとして特別に読まれるので同様に影響を受けない）。"
  (if (and (symbolp sym) (not (keywordp sym)) (%symbol-named-p sym "T"))
      t
      sym))

(defun %normalize-graph-syntax-form (form)
  "FORM（READ の結果）を再帰的に walk し、%NORMALIZE-GRAPH-SYNTAX-SYMBOL を
すべてのアトムに適用する。"
  (cond
    ((consp form)
     (cons (%normalize-graph-syntax-form (car form))
           (%normalize-graph-syntax-form (cdr form))))
    ((symbolp form) (%normalize-graph-syntax-symbol form))
    (t form)))

(defun %whitespace-char-p (char)
  (member char '(#\Space #\Tab #\Newline #\Return #\Linefeed #\Page)))

(defun %read-graph-form (source)
  "SOURCE を1つの form として読む。form の後に空白以外の文字が残っていれば
（末尾に余分なテキストがある壊れたソース）GRAPH-SYNTAX-ERROR を signal する。"
  (let (form)
    (handler-case
        (with-standard-io-syntax
          (let ((*package* (find-package '#:nabla.graph-syntax))
                (*read-eval* nil))
            (etypecase source
              (string
               (multiple-value-bind (f pos) (read-from-string source)
                 (setf form f)
                 (when (position-if-not #'%whitespace-char-p source :start pos)
                   (%graph-syntax-error form "form の後に余分なテキストがある: ~S" (subseq source pos)))))
              (stream
               (setf form (read source))
               (unless (eq (peek-char t source nil :eof) :eof)
                 (%graph-syntax-error form "form の後に余分なテキストがある"))))))
      (graph-syntax-error (c) (error c))
      (error (c)
        (%graph-syntax-error nil "graph のテキストを読めない: ~A" c)))
    (%normalize-graph-syntax-form form)))

(defun %require-section (form name)
  (unless (and (consp form) (eq (first form) name))
    (%graph-syntax-error form "~S 節が無い、または壊れている: ~S" name form))
  form)

(defun %split-graph-form (form)
  "FORM（READ の結果）が (graph :in-section :const-section :eqns-section
:out-section) の形をしていることを確かめ、4つの節を多値で返す。"
  (unless (%symbol-named-p (first form) "GRAPH")
    (%graph-syntax-error form "先頭が graph でない: ~S" form))
  (let ((sections (rest form)))
    (unless (= 4 (length sections))
      (%graph-syntax-error form "graph は :in :const :eqns :out の4節が必要: ~S" form))
    (destructuring-bind (in-section const-section eqns-section out-section) sections
      (values (%require-section in-section :in)
              (%require-section const-section :const)
              (%require-section eqns-section :eqns)
              (%require-section out-section :out)))))

(defun %build-subgraphs-in-params (form)
  "params の FORM を再帰的にたどり、(graph ...) の形をしたリスト（PRINT-GRAPH が
サブグラフを印字した形）を %BUILD-GRAPH-FROM-FORM で GRAPH に戻す。"
  (cond
    ((and (consp form) (%symbol-named-p (first form) "GRAPH"))
     (%build-graph-from-form form))
    ((consp form)
     ;; リストは要素ごとにたどる（cdr に再帰すると、途中の尾が graph という
     ;; シンボルで始まるだけの params をサブグラフと取り違える）。
     (loop for tail = form then (cdr tail)
           while (consp tail)
           collect (%build-subgraphs-in-params (car tail)) into items
           finally (return (if tail (nconc items tail) items))))
    (t form)))

(defun %build-graph-from-form (form)
  (multiple-value-bind (in-section const-section eqns-section out-section)
      (%split-graph-form form)
    (let ((vars (make-hash-table :test 'equal))
          (invars '()) (constants '()) (eqns '()) (outvars '()))
      (flet ((define (name-symbol var)
               (let ((name (symbol-name name-symbol)))
                 ;; VARS には常に VAR 構造体（NIL でない）だけを格納するので、
                 ;; プライマリ値の真偽で「既に定義済みか」を判定できる
                 ;; （MULTIPLE-VALUE-BIND で第2値の FOUND を見る必要が無い）。
                 (when (gethash name vars)
                   (error 'malformed-graph :graph nil
                          :format-control "var 名 ~A が :in/:const/:eqns の中で複数回定義されている"
                          :format-arguments (list name)))
                 (setf (gethash name vars) var)))
             (resolve (name-symbol)
               (multiple-value-bind (var found) (gethash (symbol-name name-symbol) vars)
                 (if found
                     var
                     (error 'malformed-graph :graph nil
                            :format-control "var ~A が未定義のまま参照されている"
                            :format-arguments (list (symbol-name name-symbol)))))))
        (dolist (entry (rest in-section))
          (destructuring-bind (name dtype-sym shape) entry
            (let ((var (make-var (make-aval shape (%parse-dtype dtype-sym)))))
              (define name var)
              (push var invars))))
        (dolist (entry (rest const-section))
          (destructuring-bind (name dtype-sym shape &rest elements) entry
            (let* ((dtype (%parse-dtype dtype-sym))
                   (array (make-array shape :element-type (dtype-element-type dtype)))
                   (var (make-var (make-aval shape dtype))))
              (unless (= (length elements) (array-total-size array))
                (%graph-syntax-error entry "const ~S の要素数が shape と一致しない" entry))
              (loop for i from 0 for element in elements
                    do (setf (row-major-aref array i) element))
              (define name var)
              (push (cons var array) constants))))
        (dolist (entry (rest eqns-section))
          ;; 出力が1つなら (name dtype shape := ...)、複数なら
          ;; ((name dtype shape) (name dtype shape) ... := ...)（契約 C1）。
          (let* ((multiple (consp (first entry)))
                 (assign-position (if multiple (position := entry) 3))
                 (headers (if multiple (subseq entry 0 assign-position) (list (subseq entry 0 3))))
                 (rest-form (nthcdr (or assign-position (length entry)) entry)))
            (destructuring-bind (assign-sym prim-sym params &rest invar-names) rest-form
              (unless (eq assign-sym :=)
                (%graph-syntax-error entry ":= が無い eqn: ~S" entry))
              (let* ((prim-key (intern (symbol-name prim-sym) :keyword))
                     (resolved-invars (mapcar #'resolve invar-names))
                     (eqn (apply #'make-eqn prim-key resolved-invars (%build-subgraphs-in-params params)))
                     (outs (eqn-outvars eqn)))
                (unless (= (length outs) (length headers))
                  (%graph-syntax-error entry "eqn ~S: 出力の個数 ~D が再計算した個数 ~D と一致しない"
                                       entry (length headers) (length outs)))
                (loop for (name dtype-sym shape) in headers
                      for out in outs
                      for printed-aval = (make-aval shape (%parse-dtype dtype-sym))
                      do (unless (equalp printed-aval (var-aval out))
                           (%graph-syntax-error entry
                                                 "eqn ~S: 印字された aval ~S と再計算した aval ~S が一致しない"
                                                 entry printed-aval (var-aval out)))
                         (define name out))
                (push eqn eqns)))))
        (dolist (name (rest out-section))
          (push (resolve name) outvars)))
      (check-graph
       (make-graph (nreverse invars) (nreverse eqns) (nreverse outvars) (nreverse constants))))))

(defun read-graph (source)
  "SOURCE（文字列またはストリーム）を PRINT-GRAPH と同じテキスト表現として
読み、GRAPH を返す。内部の関数（テキスト形式はフェーズ1では公開契約に
しない）。

構造が不正なテキスト（graph で始まらない、節が欠ける、`:=` が無い、
印字された aval と再計算した aval が食い違う、form の後に余分なテキストが
ある、など）は GRAPH-SYNTAX-ERROR を signal する。未登録のプリミティブ名は
UNKNOWN-PRIMITIVE、未定義の var 参照は MALFORMED-GRAPH のまま伝わる。
:in / :const / :eqns のいずれかで同じ var 名が2回以上定義されている
（後の定義が前を黙って上書きし、前の var が到達不能になる壊れたテキスト）
場合も MALFORMED-GRAPH を signal する。最後に CHECK-GRAPH を呼ぶ。"
  (let ((form (%read-graph-form source)))
    (handler-case (%build-graph-from-form form)
      (unknown-primitive (c) (error c))
      (primitive-error (c) (error c))
      (malformed-graph (c) (error c))
      (graph-syntax-error (c) (error c))
      (error (c)
        (%graph-syntax-error form "graph の構造が不正: ~A" c)))))
