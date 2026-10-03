;;;; scan: 配列の先頭の軸に沿って状態（carry）を持ち回す高階プリミティブ
;;;; （issue #132。JAX の lax.scan に相当。順方向の計算だけで、jvp / transpose は
;;;; #135 以降、バッチ化は #140）。
;;;;
;;;; eqn の params は JAX の scan に揃えてある（後続の jvp / partial eval / transpose が
;;;; JAX の _scan_jvp などをそのまま写せるように）:
;;;;   :NUM-CONSTS  ループ不変な入力（本体が閉包で捕まえた外側の値）の個数
;;;;   :NUM-CARRY   carry の個数
;;;;   :LENGTH      繰り返し回数
;;;;   :REVERSE     真なら添字 length-1 から 0 へ辿る
;;;;   :BODY        サブグラフ。入力は consts ++ carry ++ x_t、出力は carry ++ y_t
;;;; eqn の invars は consts ++ init ++ xs、outvars は 最終 carry ++ ys（ys は
;;;; 各 y_t を先頭の軸に積んだもの）。reverse でも ys[t] には添字 t のステップの y が入る。
;;;;
;;;; StableHLO は :i32 のカウンタを carry の先頭に足した stablehlo.while。x_t は
;;;; dynamic_slice（先頭の軸が 1 の切り出し）+ reshape で読み、ys は 0 で初期化した
;;;; バッファを carry に加えて dynamic_update_slice で書く。consts は外側の SSA 名を
;;;; そのまま参照する（%STABLEHLO-REGION-LINES の :ARG-NAMES に文字列で渡す）。
;;;; これらの補助の op は IR のプリミティブではなく、:scan の :emit の中だけに現れる。
;;;; 長さ 0 のときは dynamic_slice の切り出し幅 1 が軸の長さ 0 を超えて不正になるので、
;;;; ループを出さず、carry は入力の素通し、ys は空の定数にする。

(in-package #:nabla)

;;; ---- コンディション ----

(define-condition scan-error (error)
  ((format-control :initarg :format-control :initform "" :reader scan-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader scan-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "scan エラー: ~?"
             (scan-error-format-control condition)
             (scan-error-format-arguments condition))))
  (:documentation
   "SCAN の引数や本体 f が不正なときに signal するコンディションの親。init / xs が
リストでない、f が WITH-TRACING で作った2引数の関数でない、f の戻り値が
(values carry-list y-list) でない、carry も ys も無い、など。子の
SCAN-CARRY-MISMATCH は carry の aval の不一致、SCAN-LENGTH-ERROR は繰り返し回数の
不整合。"))

(define-condition scan-carry-mismatch (scan-error)
  ()
  (:documentation
   "f が返した carry の個数・shape・dtype が、init（carry の入力）と一致しないときに
signal する（トレース時に検出する）。carry はステップをまたいで同じ aval でなければ
ならない。"))

(define-condition scan-length-error (scan-error)
  ()
  (:documentation
   "繰り返し回数が決まらない、または食い違うときに signal する。xs が空なのに
LENGTH が無い、xs の要素が rank 0、xs の先頭の軸の長さが揃わない、LENGTH が xs の
先頭の軸と合わない、LENGTH が負または整数でない、など。"))

(defun %scan-error (class control &rest arguments)
  (error class :format-control control :format-arguments arguments))

;;; ---- プリミティブ ----

(defun %scan-split (list num-consts num-carry)
  "LIST を (values consts carry xs) の3つに分ける。"
  (values (subseq list 0 num-consts)
          (subseq list num-consts (+ num-consts num-carry))
          (nthcdr (+ num-consts num-carry) list)))

(defun %scan-stacked-aval (length aval)
  (make-aval (cons length (aval-shape aval)) (aval-dtype aval)))

(defun %scan-abstract-eval (in-avals &key num-consts num-carry length reverse body)
  (declare (ignore reverse))
  (flet ((fail (control &rest arguments)
           (error 'primitive-error :name :scan :in-avals in-avals
                                   :format-control control :format-arguments arguments)))
    (let ((body-in (mapcar #'var-aval (graph-invars body)))
          (body-out (mapcar #'var-aval (graph-outvars body))))
      (unless (and (integerp length) (>= length 0))
        (fail "length は 0 以上の整数でなければならない: ~S" length)
        )
      (unless (and (integerp num-consts) (integerp num-carry) (<= 0 num-consts) (<= 0 num-carry)
                   (<= (+ num-consts num-carry) (length in-avals)))
        (fail "num-consts ~S / num-carry ~S が入力の個数 ~D に合わない" num-consts num-carry (length in-avals)))
      (unless (= (length body-in) (length in-avals))
        (fail "本体の入力の個数 ~D が eqn の入力の個数 ~D と一致しない" (length body-in) (length in-avals)))
      (multiple-value-bind (consts carry xs) (%scan-split in-avals num-consts num-carry)
        (multiple-value-bind (body-consts body-carry body-xs) (%scan-split body-in num-consts num-carry)
          (unless (equalp (append consts carry) (append body-consts body-carry))
            (fail "consts / carry の aval が本体の入力と一致しない: ~S / ~S" (append consts carry)
                  (append body-consts body-carry)))
          (unless (equalp (mapcar (lambda (a) (%scan-stacked-aval length a)) body-xs) xs)
            (fail "xs の aval が、本体の x_t を先頭に長さ ~D の軸を足したものと一致しない: ~S" length xs))
          (unless (>= (length body-out) num-carry)
            (fail "本体の出力が carry の個数 ~D より少ない" num-carry))
          (unless (equalp carry (subseq body-out 0 num-carry))
            (fail "本体が返す carry の aval が入力の carry と一致しない: ~S / ~S"
                  carry (subseq body-out 0 num-carry)))
          (append carry
                  (mapcar (lambda (a) (%scan-stacked-aval length a)) (nthcdr num-carry body-out))))))))

(defun %scan-row-copy (source index row)
  "SOURCE の先頭の軸の INDEX 番目を ROW（先頭の軸を落とした形の配列）に写す。"
  (let ((size (array-total-size row)))
    (dotimes (j size row)
      (setf (row-major-aref row j) (row-major-aref source (+ (* index size) j))))))

(defun %scan-fresh-row (source index)
  "SOURCE の先頭の軸の INDEX 番目を、新しく確保した配列に写して返す。本体が x_t を
そのまま carry や y として返すと、その配列が次のステップまで生き残るので、
行のバッファは使い回さない。"
  (%scan-row-copy source index
                  (make-array (rest (array-dimensions source)) :element-type (array-element-type source))))

(defun %scan-store-row (target index row)
  "ROW を TARGET の先頭の軸の INDEX 番目に書き込む。"
  (let ((size (array-total-size row)))
    (dotimes (j size)
      (setf (row-major-aref target (+ (* index size) j)) (row-major-aref row j)))))

(defun %scan-eager (arrays in-avals &key num-consts num-carry length reverse body)
  (let ((out-avals (%scan-abstract-eval in-avals :num-consts num-consts :num-carry num-carry
                                                 :length length :reverse reverse :body body)))
    (multiple-value-bind (consts carry xs) (%scan-split arrays num-consts num-carry)
      (let* ((ys (mapcar (lambda (aval)
                           (make-array (aval-shape aval) :element-type (dtype-element-type (aval-dtype aval))))
                         (nthcdr num-carry out-avals))))
        (dotimes (step length)
          (let* ((i (if reverse (- length 1 step) step))
                 (results (multiple-value-list
                           (apply #'eval-graph body
                                  (append consts carry
                                          (mapcar (lambda (x) (%scan-fresh-row x i)) xs))))))
            (setf carry (subseq results 0 num-carry))
            (loop for target in ys
                  for y in (nthcdr num-carry results)
                  do (%scan-store-row target i y))))
        (append carry ys)))))

;;; ---- StableHLO ----

(defun %scan-zero-literal (dtype)
  "DTYPE の 0 の要素リテラル。"
  (ecase dtype
    ((:f32 :f64) "0.0")
    ((:bf16 :f16) "0x0000")
    (:i1 "false")
    ((:i32 :u32 :u64) "0")))

(defun %scan-zero-constant-line (name aval)
  (format nil "~A = stablehlo.constant ~A : ~A" name
          (if (zerop (aval-size aval))
              "dense<>"
              (format nil "dense<~A>" (%scan-zero-literal (aval-dtype aval))))
          (tensor-type-string aval)))

(defun %scan-return-names (line)
  "リージョンの最後の行 \"stablehlo.return %a, %b : ...\"（または裸の
\"stablehlo.return\"）が返す値の名前のリスト。"
  (let ((start (length "stablehlo.return"))
        (end (search " : " line)))
    (if (null end)
        '()
        (mapcar (lambda (s) (string-trim " " s))
                (uiop:split-string (subseq line start end) :separator ",")))))

(defun %scan-index-types (rank)
  "i32 のスカラー RANK 個の型リスト（\"tensor<i32>, ...\"）。"
  (format nil "~{~A~^, ~}" (make-list rank :initial-element "tensor<i32>")))

(defun %scan-emit-empty (in-names in-avals out-names out-avals num-consts num-carry)
  "長さ 0: carry は入力の素通し（同じ型の reshape）、ys は空の定数。"
  (multiple-value-bind (consts carry-names) (%scan-split in-names num-consts num-carry)
    (declare (ignore consts))
    (format nil "~{~A~^~%~}"
            (append
             (loop for name in carry-names
                   for out in out-names
                   for aval in (nthcdr num-consts in-avals)
                   collect (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                                   out name (tensor-type-string aval) (tensor-type-string aval)))
             (loop for out in (nthcdr num-carry out-names)
                   for aval in (nthcdr num-carry out-avals)
                   collect (%scan-zero-constant-line out aval))))))

(defun %scan-emit (in-names in-avals out-names out-avals &key num-consts num-carry length reverse body)
  (when (zerop length)
    (return-from %scan-emit (%scan-emit-empty in-names in-avals out-names out-avals num-consts num-carry)))
  (multiple-value-bind (const-names init-names xs-names) (%scan-split in-names num-consts num-carry)
    (let* ((p (format nil "%scan_~A" (subseq (first out-names) 1)))
           (carry-avals (subseq out-avals 0 num-carry))
           (ys-avals (nthcdr num-carry out-avals))
           (xs-avals (nthcdr (+ num-consts num-carry) in-avals))
           (i32 "tensor<i32>")
           (counter (format nil "~A_i0" p))
           (ys-init (loop for j below (length ys-avals) collect (format nil "~A_y~D_0" p j)))
           (carry-types (mapcar #'tensor-type-string carry-avals))
           (ys-types (mapcar #'tensor-type-string ys-avals))
           (all-types (append (list i32) carry-types ys-types)))
      (flet ((names (prefix count) (loop for k below count collect (format nil "~A_~A~D" p prefix k)))
             (block-args (names types)
               (format nil "^bb0(~{~A~^, ~}):" (mapcar (lambda (n ty) (format nil "~A: ~A" n ty)) names types))))
        (let* ((cond-carry (names "cc" num-carry))
               (cond-ys (names "cy" (length ys-avals)))
               (body-carry (names "bc" num-carry))
               (body-ys (names "by" (length ys-avals)))
               (x-names (names "x" (length xs-avals)))
               (idx (if reverse (format nil "~A_idx" p) (format nil "~A_bi" p)))
               (zero (format nil "~A_z" p))
               (inner (%stablehlo-region-lines
                       body :arg-names (append const-names body-carry x-names)))
               (return-names (%scan-return-names (car (last inner))))
               (new-carry (subseq return-names 0 num-carry))
               (y-values (nthcdr num-carry return-names)))
          (format nil "~{~A~^~%~}"
                  (append
                   ;; ループ前: カウンタと ys バッファの初期値
                   ;; while の定数オペランド（カウンタと ys の0初期値）は optimization_barrier を
                   ;; 通す。特定の版のバックエンドのコンパイラ（記録は docs/stablehlo-ops.md）は、cond を決める carry が constant 初期化の
                   ;; while（他に carry が2つ以上、うち1つは rank 1 以上）で、Stream の
                   ;; AffinityAnalysis が非決定的にクラッシュする（docs/stablehlo-ops.md）。
                   ;; 長さ 1 は そのコンパイラが while を scf.for にしてしまい、barrier があると
                   ;; 型の不一致（stream.resource<transient> / <external>）でコンパイルに
                   ;; 失敗するので、barrier を付けない（constant のまま。この長さでは
                   ;; クラッシュしない）。
                   (if (= length 1)
                       (list (format nil "~A = stablehlo.constant dense<0> : ~A" counter i32))
                       (list (format nil "~A_c = stablehlo.constant dense<0> : ~A" counter i32)
                             (format nil "~A = stablehlo.optimization_barrier ~A_c : ~A" counter counter i32)))
                   (loop for name in ys-init for aval in ys-avals
                         append (if (= length 1)
                                    (list (%scan-zero-constant-line name aval))
                                    (list (%scan-zero-constant-line (format nil "~A_c" name) aval)
                                          (format nil "~A = stablehlo.optimization_barrier ~A_c : ~A"
                                                  name name (tensor-type-string aval)))))
                   (list
                    (format nil "~{~A~^, ~} = \"stablehlo.while\"(~{~A~^, ~}) ({"
                            (cons (format nil "~A_n" p) out-names)
                            (append (list counter) init-names ys-init))
                    ;; cond: カウンタ < length
                    (block-args (append (list (format nil "~A_ci" p)) cond-carry cond-ys) all-types)
                    (format nil "~A_len = stablehlo.constant dense<~D> : ~A" p length i32)
                    (format nil "~A_lt = stablehlo.compare LT, ~A_ci, ~A_len : (~A, ~A) -> tensor<i1>" p p p i32 i32)
                    (format nil "stablehlo.return ~A_lt : tensor<i1>" p)
                    "}, {"
                    ;; body
                    (block-args (append (list (format nil "~A_bi" p)) body-carry body-ys) all-types))
                   (when reverse
                     (list (format nil "~A_last = stablehlo.constant dense<~D> : ~A" p (1- length) i32)
                           (format nil "~A = stablehlo.subtract ~A_last, ~A_bi : ~A" idx p p i32)))
                   (list (format nil "~A = stablehlo.constant dense<0> : ~A" zero i32))
                   ;; x_t を読む
                   (loop for x in x-names for outer in xs-names for aval in xs-avals for j from 0
                         for rank = (aval-rank aval)
                         for slice = (format nil "~A_s~D" p j)
                         for row = (make-aval (rest (aval-shape aval)) (aval-dtype aval))
                         for one-row = (make-aval (cons 1 (rest (aval-shape aval))) (aval-dtype aval))
                         append (list
                                 (format nil "~A = stablehlo.dynamic_slice ~A, ~A~{, ~A~}, sizes = [~{~D~^, ~}] : (~A, ~A) -> ~A"
                                         slice outer idx (make-list (1- rank) :initial-element zero)
                                         (aval-shape one-row)
                                         (tensor-type-string aval) (%scan-index-types rank)
                                         (tensor-type-string one-row))
                                 (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                                         x slice (tensor-type-string one-row) (tensor-type-string row))))
                   ;; 本体（最後の stablehlo.return は外す）
                   (butlast inner)
                   ;; y_t を書き込む
                   (loop for y in y-values for buffer in body-ys for aval in ys-avals for j from 0
                         for rank = (aval-rank aval)
                         for row = (make-aval (rest (aval-shape aval)) (aval-dtype aval))
                         for one-row = (make-aval (cons 1 (rest (aval-shape aval))) (aval-dtype aval))
                         for reshaped = (format nil "~A_u~D" p j)
                         append (list
                                 (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                                         reshaped y (tensor-type-string row) (tensor-type-string one-row))
                                 (format nil "~A_w~D = stablehlo.dynamic_update_slice ~A, ~A, ~A~{, ~A~} : (~A, ~A, ~A) -> ~A"
                                         p j buffer reshaped idx (make-list (1- rank) :initial-element zero)
                                         (tensor-type-string aval) (tensor-type-string one-row)
                                         (%scan-index-types rank) (tensor-type-string aval))))
                   (list (format nil "~A_one = stablehlo.constant dense<1> : ~A" p i32)
                         (format nil "~A_next = stablehlo.add ~A_bi, ~A_one : ~A" p p p i32)
                         (format nil "stablehlo.return ~{~A~^, ~} : ~{~A~^, ~}"
                                 (append (list (format nil "~A_next" p)) new-carry
                                         (loop for j below (length ys-avals) collect (format nil "~A_w~D" p j)))
                                 all-types)
                         (format nil "}) : (~{~A~^, ~}) -> (~{~A~^, ~})" all-types all-types)))))))))

(defprimitive scan (:num-consts :num-carry :length :reverse :body)
  :multiple-outputs t
  ;; 関数オブジェクトではなくシンボル経由で呼ぶ（#' で捕まえると、関数の再定義が
  ;; プリミティブに反映されず、mutation testing が変異を差し込めない）。
  :abstract-eval (lambda (in-avals &rest params) (apply #'%scan-abstract-eval in-avals params))
  :emit (lambda (in-names in-avals out-names out-avals &rest params)
          (apply #'%scan-emit in-names in-avals out-names out-avals params))
  :eager (lambda (arrays in-avals &rest params) (apply #'%scan-eager arrays in-avals params)))

;;; ---- 公開 API ----

(defun %scan-argument (value what)
  "VALUE（トレーサ・配列・実数）を、トレーサか配列にして返す。"
  (typecase value
    (tracer value)
    (real (%scalar-array value (if (typep value 'double-float) :f64 :f32)))
    (string (%scan-error 'scan-error "~A に文字列は渡せない: ~S" what value))
    (array value)
    (t (%scan-error 'scan-error "~A はトレーサ・配列・実数でなければならない: ~S" what value))))

(defun %scan-value-aval (value)
  (if (typep value 'tracer) (tracer-aval value) (array-aval value)))

(defun %scan-resolve-length (xs-avals length)
  "XS-AVALS の先頭の軸と LENGTH から繰り返し回数を決める（不整合は SCAN-LENGTH-ERROR）。"
  (when (and length (not (and (integerp length) (>= length 0))))
    (%scan-error 'scan-length-error "length は 0 以上の整数でなければならない: ~S" length))
  (dolist (aval xs-avals)
    (when (zerop (aval-rank aval))
      (%scan-error 'scan-length-error "xs の要素は rank 1 以上でなければならない（rank 0）: ~S" aval)))
  (let ((leading (remove-duplicates (mapcar (lambda (a) (first (aval-shape a))) xs-avals))))
    (cond
      ((> (length leading) 1)
       (%scan-error 'scan-length-error "xs の先頭の軸の長さが揃っていない: ~S" leading))
      ((and leading length (/= length (first leading)))
       (%scan-error 'scan-length-error "length ~D が xs の先頭の軸の長さ ~D と一致しない" length (first leading)))
      (leading (first leading))
      (length length)
      (t (%scan-error 'scan-length-error "xs が空のときは length が必要")))))

(defun %scan-check-list (value what)
  (unless (listp value)
    (%scan-error 'scan-error "~A はリストでなければならない: ~S" what value))
  value)

(defun %scan-trace-body (f carry-avals x-avals)
  "F を (carry-list x-list) で呼ぶ本体をサブグラフにトレースして、
(values GRAPH CAPTURED) を返す。GRAPH の invars は carry ++ x ++ captured、
outvars は carry ++ ys。"
  ;; %TRACE-SUBGRAPH ではなく %CALL-WITH-TRACE を直接使う。公開の f は
  ;; (carry-list x-list) の2引数（リストを受ける）で、%TRACE-SUBGRAPH が要求する
  ;; 「avals と同じ個数の引数を取る TRACEABLE-FUNCTION」と形が合わないため、
  ;; トレース用の平らな lambda でリストに詰め直してから f を呼ぶ。親トレースは
  ;; %TRACE-SUBGRAPH と同じ *CURRENT-TRACE*（閉包の closure conversion も同じ）。
  (let ((n-carry (length carry-avals)))
    (%call-with-trace
     (append carry-avals x-avals)
     (lambda (&rest tracers)
       (multiple-value-bind (new-carry ys)
           (funcall (%traceable-function-function f) (subseq tracers 0 n-carry) (nthcdr n-carry tracers))
         (%scan-check-list new-carry "f が返す carry")
         (%scan-check-list ys "f が返す ys")
         (unless (= (length new-carry) n-carry)
           (%scan-error 'scan-carry-mismatch "f が返した carry の個数 ~D が init の個数 ~D と一致しない"
                        (length new-carry) n-carry))
         (values-list (append new-carry ys))))
     *current-trace*)))

(defun scan (f init xs &key length reverse)
  "先頭の軸に沿って F を回し、(VALUES 最終の carry のリスト ys のリスト) を返す
（JAX の lax.scan に相当）。

F は WITH-TRACING で作った2引数の関数 (carry-list x-list) で、
(VALUES 新しい carry のリスト y のリスト) を返す。INIT は carry の初期値のリスト、
XS は走査する配列（トレーサ・配列・実数）のリストで、どれも先頭の軸（長さ N）で
走査する。ys は各ステップの y を先頭の軸に積んだ配列のリスト。F が返す carry は
INIT と個数・shape・dtype が同じでなければならない（違えば SCAN-CARRY-MISMATCH）。

LENGTH は繰り返し回数。XS が空のときは必須で、そうでなければ XS の先頭の軸の長さと
一致しなければならない（SCAN-LENGTH-ERROR）。長さ 0 の scan は INIT をそのまま返し、
ys は先頭の軸が 0 の空の配列になる。REVERSE が真なら添字 N-1 から 0 へ辿る
（ys[t] には、そのときも添字 t のステップの y が入る）。

F が閉包で捕まえた外側の値は、ループ不変な入力（consts）になる。トレース中
（with-tracing・jit の中）でもその場（eager）でも使える。grad は scan を通る（jvp は #135、partial eval と transpose は #139。
src/ad/rules-scan-reverse.lisp）。vmap は未対応。引数や F の戻り値の形が不正なときは
SCAN-ERROR。"
  (unless (and (typep f 'traceable-function)
               (= 2 (length (traceable-function-lambda-list f))))
    (%scan-error 'scan-error "f は WITH-TRACING で作った2引数 (carry x) の関数でなければならない: ~S" f))
  (%scan-check-list init "init")
  (%scan-check-list xs "xs")
  (let* ((init (mapcar (lambda (v) (%scan-argument v "init の要素")) init))
         (xs (mapcar (lambda (v) (%scan-argument v "xs の要素")) xs))
         (carry-avals (mapcar #'%scan-value-aval init))
         (xs-avals (mapcar #'%scan-value-aval xs))
         (length (%scan-resolve-length xs-avals length))
         (x-avals (mapcar (lambda (a) (make-aval (rest (aval-shape a)) (aval-dtype a))) xs-avals))
         (n-carry (length init)))
    (multiple-value-bind (graph captured) (%scan-trace-body f carry-avals x-avals)
      (let* ((n-xs (length xs))
             (n-consts (length captured))
             (invars (graph-invars graph))
             ;; JAX の並び consts ++ carry ++ xs に直す（トレースの並びは carry ++ xs ++ consts）。
             (body (check-graph
                    (make-graph (append (nthcdr (+ n-carry n-xs) invars)
                                        (subseq invars 0 n-carry)
                                        (subseq invars n-carry (+ n-carry n-xs)))
                                (graph-eqns graph) (graph-outvars graph) (graph-constants graph)))))
        (when (null (graph-outvars body))
          (%scan-error 'scan-error "carry も ys も無い scan は作れない"))
        (unless (equalp carry-avals (mapcar #'var-aval (subseq (graph-outvars body) 0 n-carry)))
          (%scan-error 'scan-carry-mismatch "f が返した carry の aval ~S が init の aval ~S と一致しない"
                       (mapcar #'var-aval (subseq (graph-outvars body) 0 n-carry)) carry-avals))
        (let* ((operands (append captured init xs))
               (params (list :num-consts n-consts :num-carry n-carry :length length
                             :reverse (and reverse t) :body body))
               (results
                 (if *current-trace*
                     (apply #'%trace-eqn* :scan
                            (mapcar (lambda (v)
                                      (if (typep v 'tracer)
                                          v
                                          (%lift-constant v (array-aval v) *current-trace*)))
                                    operands)
                            params)
                     (apply (primitive-eager (find-primitive :scan))
                            operands (mapcar #'%scan-value-aval operands) params))))
          (values (subseq results 0 n-carry) (nthcdr n-carry results)))))))
