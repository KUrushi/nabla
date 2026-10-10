;;;; nb:scan の性質（issue #132）。
;;;;
;;;; 期待値は scan に依存しない参照実装 %SCAN-REF（Lisp のループで f を eager に
;;;; 回し、各ステップの y を先頭の軸に積み直す）。f は WITH-TRACING で作った関数で、
;;;; funcall すると eager に実行される。StableHLO / IREE での実行は
;;;; tests/iree/scan-test.lisp。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; ---- 参照実装 ----

(defun %scan-row (array i)
  "ARRAY の先頭の軸の i 番目を、先頭の軸を落とした新しい配列として返す。"
  (let* ((dims (rest (array-dimensions array)))
         (row (make-array dims :element-type (array-element-type array)))
         (size (array-total-size row)))
    (dotimes (j size row)
      (setf (row-major-aref row j) (row-major-aref array (+ (* i size) j))))))

(defun %scan-stack (rows shape element-type)
  "ROWS（同じ形の配列のリスト。空でもよい）を先頭の軸に積んだ配列。空のときの
1ステップぶんの形は SHAPE、要素型は ELEMENT-TYPE で与える。"
  (let* ((out (make-array (cons (length rows) shape) :element-type element-type))
         (size (reduce #'* shape)))
    (loop for row in rows for i from 0
          do (dotimes (j size)
               (setf (row-major-aref out (+ (* i size) j)) (row-major-aref row j))))
    out))

(defun %scan-ref (f init xs length reverse y-specs)
  "f を Lisp のループで回す参照実装。Y-SPECS は ys ごとの (1ステップの形 . 要素型)。
(values carry-list ys-list) を返す。"
  (let ((carry init)
        (rows (make-list (length y-specs) :initial-element nil)))
    (dolist (i (if reverse
                   (loop for i from (1- length) downto 0 collect i)
                   (loop for i below length collect i)))
      (multiple-value-bind (new-carry ys) (funcall f carry (mapcar (lambda (x) (%scan-row x i)) xs))
        (setf carry new-carry)
        ;; reverse のときも ys は添字 i の位置に入る（走査の順ではなく）ので、
        ;; 添字つきで溜めて最後に添字順に並べる。
        (setf rows (mapcar (lambda (acc y) (acons i y acc)) rows ys))))
    (values carry
            (loop for acc in rows for (shape . type) in y-specs
                  collect (%scan-stack (mapcar #'cdr (sort (copy-list acc) #'< :key #'car)) shape type)))))

;;; ---- テスト対象の本体 ----

(defparameter *scan-mixed*
  (nb:with-tracing (carry x)
    (let ((h (first carry)) (c (second carry)) (u (first x)) (v (second x)))
      (values (list (tanh (+ (* h 0.5) u)) (+ c 1))
              (list (* h u) v c))))
  "carry = (h:浮動小数点 [n], c:i32 スカラー)、x = (u:[n], v:[k])、y = (h*u, v, c)。
ys の形・dtype が混ざる（[n]、[k]、i32 スカラー）。")

(defun %scan-mixed-case (seed length n k dtype)
  "*SCAN-MIXED* 用の (values init xs) を返す。"
  (flet ((random-of (shape s)
           (make-random-array (make-array-spec shape dtype) :seed (+ seed s))))
    (values (list (random-of (list n) 1)
                  (make-array '() :element-type '(signed-byte 32) :initial-element (mod seed 7)))
            (list (random-of (list length n) 2)
                  (random-of (list length k) 3)))))

(defun %scan-mixed-y-specs (n k dtype)
  (list (cons (list n) (nb::dtype-element-type dtype))
        (cons (list k) (nb::dtype-element-type dtype))
        (cons '() '(signed-byte 32))))

(defun %scan-lists-allclose (actual expected dtypes)
  (and (= (length actual) (length expected))
       (every (lambda (a e dt)
                (and (equal (array-dimensions a) (array-dimensions e))
                     (allclose a e :dtype dt)))
              actual expected dtypes)))

;;; ---- 性質: scan = Lisp のループ ----

(test scan/equals-lisp-loop
  "scan の結果は、Lisp のループで f を回して ys を積んだ結果と一致する
（reverse、長さ 0 と 1、複数の carry / xs / ys、f32 と f64、i32 の carry と ys を含む）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 4) (integer 1 4) (integer 1 3) (integer 0 1) (integer 0 1)))
       (lambda (case)
         (destructuring-bind (seed length n k reverse-code dtype-code) case
           (let ((dtype (if (zerop dtype-code) :f32 :f64))
                 (reverse (= 1 reverse-code)))
             (multiple-value-bind (init xs) (%scan-mixed-case seed length n k dtype)
               (multiple-value-bind (carry ys) (nb:scan *scan-mixed* init xs :reverse reverse)
                 (multiple-value-bind (ref-carry ref-ys)
                     (%scan-ref *scan-mixed* init xs length reverse (%scan-mixed-y-specs n k dtype))
                   (and (%scan-lists-allclose carry ref-carry (list dtype :i32))
                        (%scan-lists-allclose ys ref-ys (list dtype dtype :i32)))))))))
       :regression-id scan/equals-lisp-loop
       :regression-file (regression-path "scan-matches-lisp-loop"))))

(test scan/length-zero-returns-init-and-empty-ys
  "長さ 0 の scan は init と同じ値（新しい配列）を返し、ys は先頭の軸が 0 の空の配列になる。"
  (multiple-value-bind (init xs) (%scan-mixed-case 3 0 2 3 :f32)
    (multiple-value-bind (carry ys) (nb:scan *scan-mixed* init xs)
      (is (every #'equalp carry init))
      (is (equal '((0 2) (0 3) (0)) (mapcar #'array-dimensions ys)))
      (is (equal '(single-float single-float (signed-byte 32)) (mapcar #'array-element-type ys))))))

(test scan/reverse-visits-last-step-first-but-stores-ys-by-index
  "reverse の scan は添字 length-1 から 0 へ辿る。carry の最終値は添字 0 のステップの後の値、
ys は添字 t のステップの y が ys[t] に入る。"
  (let* ((f (nb:with-tracing (carry x) (values (list (+ (* (first carry) 2.0) (first x))) (list (first carry)))))
         (xs (list (make-array 3 :element-type 'single-float :initial-contents '(1.0 10.0 100.0))))
         (init (list (make-array '() :element-type 'single-float :initial-element 0.0))))
    ;; 辿る順は x = 100, 10, 1。carry: 0 -> 100 -> 210 -> 421。y は各ステップの前の carry
    ;; （添字 2 のステップが 0、添字 1 が 100、添字 0 が 210）。
    (multiple-value-bind (carry ys) (nb:scan f init xs :reverse t)
      (is (= 421.0 (row-major-aref (first carry) 0)))
      (is (equalp #(210.0 100.0 0.0) (first ys))))))

;;; ---- xs が空 ----

(test scan/without-xs-uses-length
  "xs が空のリストなら length だけで回数が決まる（PBT。reverse を含む）。"
  (let ((f (nb:with-tracing (carry x)
             x
             (values (list (* (first carry) 2.0)) (list (first carry))))))
    (is (check-it
         (generator (tuple (integer 0 100000) (integer 0 5) (integer 0 1)))
         (lambda (case)
           (destructuring-bind (seed length reverse-code) case
             (let ((init (list (make-random-array (make-array-spec '(2) :f32) :seed seed))))
               (multiple-value-bind (carry ys)
                   (nb:scan f init '() :length length :reverse (= 1 reverse-code))
                 (multiple-value-bind (ref-carry ref-ys)
                     (%scan-ref f init '() length (= 1 reverse-code) (list (cons '(2) 'single-float)))
                   (and (%scan-lists-allclose carry ref-carry '(:f32))
                        (%scan-lists-allclose ys ref-ys '(:f32))
                        (equal (list length 2) (array-dimensions (first ys)))))))))
         :regression-id scan/without-xs
         :regression-file (regression-path "scan-without-xs")))))

;;; ---- トレース: 閉包は consts、graph の形 ----

(defun %scan-traced-graph (length reverse)
  "外側のトレーサ w を閉包で捕まえる本体の scan を含む graph。入力は (h0 xs w)。"
  (nb::trace-to-graph
   (nb:with-tracing (h0 xs w)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x) (let ((h (tanh (+ (* (first carry) w) (first x)))))
                                                (values (list h) (list (* h w)))))
                  (list h0) (list xs) :length length :reverse reverse)
       (values (first carry) (first ys))))
   (list (nb:make-aval '(3) :f32) (nb:make-aval (list length 3) :f32) (nb:make-aval '(3) :f32))))

(defun %scan-eqn (graph)
  (find :scan (nb:graph-eqns graph) :key (lambda (e) (nb::primitive-name (nb:eqn-prim e)))))

(test scan/closure-over-outer-tracer-becomes-const
  "本体が閉包で捕まえた外側のトレーサは、JAX と同じ並び（consts ++ carry ++ xs）の
サブグラフの先頭の入力になり、eqn の params に num-consts / num-carry / length / reverse が入る。"
  (let* ((graph (%scan-traced-graph 4 t))
         (eqn (%scan-eqn graph))
         (params (nb:eqn-params eqn))
         (body (getf params :body)))
    (is (not (null eqn)))
    (is (= 1 (getf params :num-consts)))
    (is (= 1 (getf params :num-carry)))
    (is (= 4 (getf params :length)))
    (is (eq t (getf params :reverse)))
    ;; eqn の入力は (w h0 xs)、本体の入力は (consts carry x_t)。
    (is (equal (list '(3) '(3) '(4 3)) (mapcar (lambda (v) (nb:aval-shape (nb:var-aval v))) (nb:eqn-invars eqn))))
    (is (equal (list '(3) '(3) '(3)) (mapcar (lambda (v) (nb:aval-shape (nb:var-aval v))) (nb:graph-invars body))))
    (is (equal (list '(3) '(4 3)) (mapcar (lambda (v) (nb:aval-shape (nb:var-aval v))) (nb:eqn-outvars eqn))))))

(test scan/traced-graph-evaluates-like-eager-loop
  "scan を含む graph の eval-graph は、閉包の値を使う Lisp のループと一致する（PBT）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 4) (integer 0 1)))
       (lambda (case)
         (destructuring-bind (seed length reverse-code) case
           (let* ((reverse (= 1 reverse-code))
                  (graph (%scan-traced-graph length reverse))
                  (h0 (make-random-array (make-array-spec '(3) :f32) :seed seed))
                  (xs (make-random-array (make-array-spec (list length 3) :f32) :seed (+ seed 1)))
                  (w (make-random-array (make-array-spec '(3) :f32) :seed (+ seed 2)))
                  (f (nb:with-tracing (carry x) (let ((h (tanh (+ (* (first carry) w) (first x)))))
                                                  (values (list h) (list (* h w))))))
                  (results (multiple-value-list (nb:eval-graph graph h0 xs w))))
             (multiple-value-bind (ref-carry ref-ys)
                 (%scan-ref f (list h0) (list xs) length reverse (list (cons '(3) 'single-float)))
               (and (%scan-lists-allclose (list (first results)) ref-carry '(:f32))
                    (%scan-lists-allclose (list (second results)) ref-ys '(:f32)))))))
       :regression-id scan/with-consts
       :regression-file (regression-path "scan-with-consts"))))

(test scan/emits-while-with-dynamic-slices
  "StableHLO は stablehlo.while で、xs は dynamic_slice、ys は dynamic_update_slice。
reverse のときだけ添字を length-1 から引く。"
  (let ((forward (nb:emit-stablehlo (%scan-traced-graph 4 nil)))
        (backward (nb:emit-stablehlo (%scan-traced-graph 4 t))))
    (dolist (text (list forward backward))
      (is (search "stablehlo.while" text))
      (is (search "stablehlo.dynamic_slice" text))
      (is (search "stablehlo.dynamic_update_slice" text)))
    (is (null (search "stablehlo.subtract" forward)))
    (is (search "stablehlo.subtract" backward))))

(defun %scan-two-ys-graph (length)
  "carry h:f32 [3] と、同じ形と dtype の ys を2つ (h, -h) 持つ長さ LENGTH の scan の graph。"
  (nb::trace-to-graph
   (nb:with-tracing (h xs)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    (let ((h (+ (first carry) (first x))))
                      (values (list h) (list h (- h)))))
                  (list h) (list xs))
       (values (first carry) (first ys) (second ys))))
   (list (nb:make-aval '(3) :f32) (nb:make-aval (list length 3) :f32))))

(defun %scan-text-lines (text)
  (mapcar (lambda (line) (string-trim " " line)) (uiop:split-string text :separator '(#\Newline))))

(defun %scan-line-defining (lines name)
  "LINES のうち NAME を（多出力の左辺の1つとしても）定義する行。"
  (find-if (lambda (line)
             (let ((eq (search " = " line)))
               (and eq (member name (uiop:split-string (subseq line 0 eq) :separator '(#\, #\Space))
                               :test #'string=))))
           lines))

(test scan/emits-ys-buffers-through-barriers
  "長さ 2 以上の scan は、本体で ys のバッファ（%<p>by<j>）をそれぞれ optimization_barrier に通し、
その結果（%<p>bk<j>）を dynamic_update_slice に渡す。初期値は ys ごとに broadcast_in_dim と
optimization_barrier の組を1つずつ持ち、同じ型の2つの ys も別々のバッファになる。長さ 1 の scan は
どちらも持たない（issue #159。実行時間ではなく出力の形で、in-place の書き込みを守る）。"
  (let* ((lines (%scan-text-lines (nb:emit-stablehlo (%scan-two-ys-graph 4))))
         (bb0 (find-if (lambda (line) (and (search "^bb0(" line) (search "bi: " line))) lines))
         (prefix (subseq bb0 (length "^bb0(") (search "bi: " bb0)))
         (while-line (find-if (lambda (line) (search "\"stablehlo.while\"(" line)) lines))
         (operands (uiop:split-string
                    (subseq while-line (+ (search "(" while-line :start2 (search "while" while-line)) 1)
                            (search ")" while-line))
                    :separator '(#\, #\Space)))
         (ys-init (last (remove "" operands :test #'string=) 2)))
    (dotimes (j 2)
      (let ((by (format nil "~Aby~D" prefix j))
            (bk (format nil "~Abk~D" prefix j)))
        (is (member (format nil "~A = stablehlo.optimization_barrier ~A : tensor<4x3xf32>" bk by) lines
                    :test #'string=)
            "~A が optimization_barrier に通されていない" by)
        (is (find-if (lambda (line) (search (format nil "stablehlo.dynamic_update_slice ~A," bk) line)) lines)
            "dynamic_update_slice が ~A を受け取っていない" bk)))
    (is (= 2 (length (remove-duplicates ys-init :test #'string=)))
        "同じ型の2つの ys の初期値が同じバッファ: ~S" ys-init)
    (let ((broadcast-sources '()))
      (dolist (name ys-init)
        (let* ((def (%scan-line-defining lines name))
               (source (and def (search "stablehlo.optimization_barrier " def)
                            (string-right-trim
                             " " (subseq def (+ (search "optimization_barrier " def) (length "optimization_barrier "))
                                         (search " :" def)))))
               (source-def (and source (%scan-line-defining lines source))))
          (is (and source-def (search "stablehlo.broadcast_in_dim" source-def))
              "ys の初期値 ~A が broadcast_in_dim → optimization_barrier の組で作られていない: ~S" name def)
          (when source-def
            (push (subseq source-def (+ (search "broadcast_in_dim " source-def) (length "broadcast_in_dim "))
                          (search "," source-def))
                  broadcast-sources))))
      (is (= 2 (length (remove-duplicates broadcast-sources :test #'string=)))
          "2つの ys の broadcast_in_dim が同じスカラーを広げている: ~S" broadcast-sources)))
  (let ((text (nb:emit-stablehlo (%scan-two-ys-graph 1))))
    (is (null (search "optimization_barrier" text)) "長さ 1 の scan に optimization_barrier がある")
    (is (null (search "broadcast_in_dim" text)) "長さ 1 の scan の ys の初期値が broadcast_in_dim で作られている")
    (is (null (search "bk0" text)) "長さ 1 の scan の本体に ys の barrier がある")))

(test scan/emits-a-distinct-salt-for-each-scan-ys-init
  "同じ入力の順方向と逆方向の scan を並べても、カウンタと ys の初期値のスカラーを通す
optimization_barrier のオペランドは scan ごとに違う（モジュールの中で一意な整数の constant も通す）。
同じなら barrier どうしが CSE でまとめられ、2つの scan が同じカウンタや ys のバッファを書き換える
（先の scan が進めたカウンタから逆方向の scan が始まり、1回も回らない）。"
  (let* ((text (nb:emit-stablehlo
                (nb::trace-to-graph
                 (nb:with-tracing (h xs)
                   (multiple-value-bind (fwd-carry fwd-ys)
                       (nb:scan (nb:with-tracing (carry x)
                                  (let ((h (+ (first carry) (first x)))) (values (list h) (list h))))
                                (list h) (list xs))
                     (multiple-value-bind (rev-carry rev-ys)
                         (nb:scan (nb:with-tracing (carry x)
                                    (let ((h (+ (first carry) (first x)))) (values (list h) (list h))))
                                  (list h) (list xs) :reverse t)
                       (values (first fwd-carry) (first fwd-ys) (first rev-carry) (first rev-ys)))))
                 (list (nb:make-aval '(3) :f32) (nb:make-aval '(4 3) :f32)))))
         (salts (loop for line in (uiop:split-string text :separator '(#\Newline))
                      for pos = (search "_u_c = stablehlo.constant dense<" line)
                      when pos
                        collect (let ((start (+ (position #\< line :start pos) 1)))
                                  (subseq line start (position #\> line :start start))))))
    (is (= 2 (length salts)) "ys の初期値の一意な整数が scan ごとに1つずつ無い: ~S" salts)
    (is (= 2 (length (remove-duplicates salts :test #'string=))) "2つの scan の一意な整数が同じ: ~S" salts)
    (let ((counter-lines (remove-if-not (lambda (line) (and (search "_i0, " line) (search "optimization_barrier" line)))
                                        (uiop:split-string text :separator '(#\Newline)))))
      (is (= 2 (length counter-lines)) "カウンタが scan ごとに1つの多出力の barrier から出ていない")
      (is (every (lambda (line) (search "_u_c :" line)) counter-lines)
          "カウンタの barrier に一意な整数が通っていない: ~S" counter-lines))))

;;; ---- コンディション ----

(defun %scan-one (shape &optional (dtype :f32))
  (make-random-array (make-array-spec shape dtype) :seed 1))

(test scan/rejects-carry-aval-mismatch
  "f が返す carry の aval（shape / dtype / 個数）が init と違うと SCAN-CARRY-MISMATCH。"
  (let ((init (list (%scan-one '(2))))
        (scalar-init (list (%scan-one '()))))
    ;; shape が違う（rank 0 のリテラル）
    (signals nb:scan-carry-mismatch
      (nb:scan (nb:with-tracing (c x) c x (values (list 1.0) '())) init '() :length 2))
    ;; dtype が違う（f64 の配列）
    (signals nb:scan-carry-mismatch
      (nb:scan (nb:with-tracing (c x) c x
                 (values (list (make-array '() :element-type 'double-float :initial-element 1d0)) '()))
               scalar-init '() :length 2))
    ;; 個数が違う
    (signals nb:scan-carry-mismatch
      (nb:scan (nb:with-tracing (c x) x (values (list (first c) (first c)) '()))
               init '() :length 2))))

(test scan/rejects-length-problems
  "xs の先頭の軸が揃わない、length と合わない、xs が空で length が無い、
xs の要素が rank 0、length が負、で SCAN-LENGTH-ERROR。"
  (let ((body (nb:with-tracing (c x) x (values c '())))
        (init (list (%scan-one '(2)))))
    (signals nb:scan-length-error
      (nb:scan body init (list (%scan-one '(3 2)) (%scan-one '(4 2)))))
    (signals nb:scan-length-error
      (nb:scan body init (list (%scan-one '(3 2))) :length 4))
    (signals nb:scan-length-error (nb:scan body init '()))
    (signals nb:scan-length-error (nb:scan body init (list (%scan-one '()))))
    (signals nb:scan-length-error (nb:scan body init '() :length -1))))

(test scan/accepts-matching-explicit-length
  "length が xs の先頭の軸と一致していれば受け付ける。"
  (finishes (nb:scan (nb:with-tracing (c x) x (values c '()))
                     (list (%scan-one '(2))) (list (%scan-one '(3 2))) :length 3)))

(test scan/rejects-malformed-arguments
  "init / xs がリストでない、f が WITH-TRACING の関数でない、f の戻り値がリストでない、
carry も ys も無い、で SCAN-ERROR。"
  (let ((body (nb:with-tracing (c x) x (values c '())))
        (init (list (%scan-one '(2)))))
    (signals nb:scan-error (nb:scan body (%scan-one '(2)) '() :length 1))
    (signals nb:scan-error (nb:scan body init (%scan-one '(3 2))))
    (signals nb:scan-error (nb:scan #'identity init '() :length 1))
    (signals nb:scan-error
      (nb:scan (nb:with-tracing (c x) x (values (first c) '())) init '() :length 1))
    (signals nb:scan-error
      (nb:scan (nb:with-tracing (c x) c x (values '() '())) '() '() :length 1))))

(test scan/grad-through-scan-works
  "scan を通る grad は動く（#139）。h' = 2h を3回回すので、d(最後の h の総和)/d h0 は全要素で 8。"
  (let* ((f (nb:with-tracing (h0)
              (multiple-value-bind (carry ys)
                  (nb:scan (nb:with-tracing (c x) x (values (list (* 2.0d0 (first c))) '()))
                           (list h0) '() :length 3)
                (declare (ignore ys))
                (nb:reduce-sum (first carry)))))
         (g (funcall (nb:grad f) (make-array '(2) :element-type 'double-float :initial-element 0.5d0))))
    ;; h' = 2h を3回: d/dh0 = 8
    (is (equalp #(8.0d0 8.0d0) g))))

;;; ---- 回帰: carry が入力の x_t をそのまま返す（eager が行バッファを使い回さない） ----

(test scan/carry-aliasing-x-row-is-not-overwritten
  "f が x_t をそのまま次の carry にしても、次のステップが x_t の領域を書き換えて
carry を壊さない（ys は1ステップ前の x、最終の carry は最後の x）。"
  (let* ((f (nb:with-tracing (carry x) (values (list (first x)) (list (first carry)))))
         (init (list (make-array 2 :element-type 'single-float :initial-contents '(0.0 0.0))))
         (xs (list (make-array '(3 2) :element-type 'single-float
                                      :initial-contents '((1.0 2.0) (3.0 4.0) (5.0 6.0))))))
    (multiple-value-bind (carry ys) (nb:scan f init xs)
      (is (equalp #(5.0 6.0) (first carry)))
      (is (equalp #2A((0.0 0.0) (1.0 2.0) (3.0 4.0)) (first ys))))))

;;; ---- eager の scan は入力を書き換えない（issue #166） ----

(defparameter *scan-host-constant*
  (make-array 2 :element-type 'single-float :initial-contents '(7.0 8.0))
  "*SCAN-ALIAS-BODIES* の本体が閉包で捕まえるホストの配列（本体のサブグラフの定数になる）。")

(defparameter *scan-alias-bodies*
  (list
   ;; carry をそのまま返す（長さ 0 でも 1 以上でも、結果の carry は init と同じ値）
   (nb:with-tracing (carry x) x (values carry (list (first carry))))
   ;; 捕まえたホストの配列を carry にする
   (nb:with-tracing (carry x) carry (values (list *scan-host-constant*) x))
   ;; x_t をそのまま carry にする
   (nb:with-tracing (carry x) (values (list (first x)) (list (first carry)))))
  "本体が入力（init・捕捉した配列・x_t）をそのまま出力に回す scan の本体。
どれも carry = ([2] の f32)、x = ([2] の f32)。")

(defun %scan-snapshot (arrays)
  "ARRAYS の各配列の中身を写した新しい配列のリスト。"
  (mapcar (lambda (a)
            (let ((copy (make-array (array-dimensions a) :element-type (array-element-type a))))
              (dotimes (j (array-total-size a) copy)
                (setf (row-major-aref copy j) (row-major-aref a j)))))
          arrays))

(test scan/eager-does-not-modify-inputs
  "eager の scan は init・xs・本体が捕まえたホストの配列を書き換えず、結果は参照実装
%SCAN-REF と一致する（長さ 0 を含む）。結果が入力と EQ かどうかは問わない
（README「配列の不変性」）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 3) (integer 0 2)))
       (lambda (case)
         (destructuring-bind (seed length body-index) case
           (let* ((f (nth body-index *scan-alias-bodies*))
                  (init (list (make-random-array (make-array-spec '(2) :f32) :seed seed)))
                  (xs (list (make-random-array (make-array-spec (list length 2) :f32) :seed (1+ seed))))
                  (inputs (append init xs (list *scan-host-constant*)))
                  (before (%scan-snapshot inputs)))
             (multiple-value-bind (carry ys) (nb:scan f init xs)
               (multiple-value-bind (ref-carry ref-ys)
                   (%scan-ref f (%scan-snapshot init) (%scan-snapshot xs) length nil
                              (list (cons '(2) 'single-float)))
                 (and (every #'equalp before inputs)
                      (equalp ref-carry carry)
                      (equalp ref-ys ys)))))))
       :regression-id scan/eager-does-not-modify-inputs
       :regression-file (regression-path "scan-eager-does-not-modify-inputs"))))

(test scan/rejects-raw-16-bit-arrays
  "dtype が一意に決まらない生の (unsigned-byte 16) の配列（bf16 / f16）を init や xs に
渡すと SCAN-ERROR（トレーサで渡す）。"
  (let ((body (nb:with-tracing (c x) x (values c '())))
        (raw-carry (make-array 2 :element-type '(unsigned-byte 16) :initial-element 0))
        (raw-xs (make-array '(3 2) :element-type '(unsigned-byte 16) :initial-element 0)))
    (signals nb:scan-error (nb:scan body (list raw-carry) '() :length 1))
    (signals nb:scan-error (nb:scan body (list (%scan-one '(2))) (list raw-xs)))))

;;; ---- 実数リテラル、ys の形、eqn の整合性検査（primitive-error） ----

(test scan/real-literal-init-takes-float-dtype-of-the-literal
  "init に実数を渡すと、single-float は f32、double-float は f64 の rank 0 の carry になる。"
  (let ((f (nb:with-tracing (c x) x (values (list (+ (first c) 1)) '()))))
    (is (eq 'single-float (array-element-type (first (nb:scan f (list 0.5) '() :length 2)))))
    (is (eq 'double-float (array-element-type (first (nb:scan f (list 0.5d0) '() :length 2)))))
    (is (= 2.5 (row-major-aref (first (nb:scan f (list 0.5) '() :length 2)) 0)))))

(test scan/rejects-non-list-ys
  "f の2つ目の戻り値（ys）がリストでないと SCAN-ERROR。"
  (signals nb:scan-error
    (nb:scan (nb:with-tracing (c x) x (values c (first c))) (list (%scan-one '(2))) '() :length 1)))

(defun %scan-eqn-case ()
  "(values eqn-invars-avals params)。長さ 4、consts 1、carry 1、xs 1、ys 1 の正しい scan。"
  (let ((eqn (%scan-eqn (%scan-traced-graph 4 nil))))
    (values (mapcar #'nb:var-aval (nb:eqn-invars eqn)) (nb:eqn-params eqn))))

(defun %scan-make-eqn (avals params)
  (apply #'nb::make-eqn :scan (mapcar #'nb::make-var avals) params))

(defun %scan-with-param (params key value)
  (let ((copy (copy-list params))) (setf (getf copy key) value) copy))

(test scan/make-eqn-accepts-consistent-params-and-infers-outputs
  "整合した params の eqn は、carry の aval と、先頭に長さ length の軸を足した ys の aval を出力にする。"
  (multiple-value-bind (avals params) (%scan-eqn-case)
    (is (equal (list '(3) '(4 3)) (mapcar (lambda (v) (nb:aval-shape (nb:var-aval v)))
                                           (nb:eqn-outvars (%scan-make-eqn avals params)))))
    ;; num-consts / num-carry が 0 でもよい（本体の入力が合っている限り）。
    (let* ((body (nb::trace-to-graph (nb:with-tracing (x) (* x 2.0)) (list (nb:make-aval '(3) :f32))))
           (eqn (%scan-make-eqn (list (nb:make-aval '(0 3) :f32))
                                (list :num-consts 0 :num-carry 0 :length 0 :reverse nil :body body))))
      (is (equal '((0 3)) (mapcar (lambda (v) (nb:aval-shape (nb:var-aval v))) (nb:eqn-outvars eqn)))))))

(test scan/make-eqn-rejects-inconsistent-params
  "abstract-eval は、params と入力・本体の aval の不整合を PRIMITIVE-ERROR にする。"
  (multiple-value-bind (avals params) (%scan-eqn-case)
    (flet ((bad (avals params) (signals nb:primitive-error (%scan-make-eqn avals params))))
      (bad avals (%scan-with-param params :length -1))
      (bad avals (%scan-with-param params :length 2.0))
      (bad avals (%scan-with-param params :num-consts -1))
      (bad avals (%scan-with-param params :num-carry -1))
      (bad avals (%scan-with-param params :num-consts 2.5))
      (bad avals (%scan-with-param params :num-carry 3))
      ;; 入力の個数が本体と合わない
      (bad (append avals (list (nb:make-aval '(3) :f32))) params)
      ;; consts / carry の aval が本体と合わない
      (bad (list (nb:make-aval '(2) :f32) (second avals) (third avals)) params)
      (bad (list (first avals) (nb:make-aval '(3) :f64) (third avals)) params)
      ;; xs の先頭の軸が length と合わない
      (bad (list (first avals) (second avals) (nb:make-aval '(5 3) :f32)) params)
      (bad (list (first avals) (second avals) (nb:make-aval '(4 2) :f32)) params)
      ;; carry の aval が違う（reduce-sum で rank 0 になる本体）
      (bad avals (%scan-with-param params :num-carry 3))
      (bad avals (%scan-with-param
                  params :body
                  (nb::trace-to-graph (nb:with-tracing (w h x) (nb:reduce-sum (* w (* h x))))
                                      (list (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))))
    ;; 出力が無い本体（carry の個数 1 に足りない）
    (signals nb:primitive-error
      (%scan-make-eqn
       avals
       (%scan-with-param params :body
                         (nb::make-graph (nb:graph-invars (getf params :body)) '() '() '()))))))

(test scan/primitive-checks-carry-aval-against-body-output
  "本体の carry の出力が入力の carry と違う aval なら PRIMITIVE-ERROR（i32 の carry に f32 を返す本体）。"
  (let ((body (nb::trace-to-graph (nb:with-tracing (c) (+ c 1.0)) (list (nb:make-aval '() :f32)))))
    (signals nb:primitive-error
      (%scan-make-eqn (list (nb:make-aval '() :i32))
                      (list :num-consts 0 :num-carry 1 :length 2 :reverse nil :body body)))))
