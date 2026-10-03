;;;; with-tracing の中の do ループを scan に展開する（issue #137）。
;;;;
;;;; 期待値は、同じループを Lisp の do で、単精度浮動小数点のリストに対して回した
;;;; 結果（参照実装は scan にも with-tracing にも依存しない）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; ---- 参照実装と、展開されるループ ----

(defun %do-ref (h0 w n)
  "(values h k): h <- tanh(h*s + 0.1)、k <- k+1 を n 回。s は w のまま不変。"
  (let ((h (map 'list #'identity h0))
        (w (map 'list #'identity w)))
    (do ((i 0 (1+ i))
         (k 0.0 (+ k 1.0)))
        ((>= i n) (values h k))
      (setf h (mapcar (lambda (hh ww) (tanh (+ (* hh ww) 0.1))) h w)))))

;;; with-tracing は展開が（テストのコンパイル時ではなく）実行時に起きるよう、フォームを
;;; EVAL する。mutation testing は src の関数を実行時に差し替えるので、コンパイル済みの
;;; 展開結果では変異体が見えない。

(defun %eval-tracing (form)
  (eval form))

(defun %do-traced (n)
  "h・k が carry、s は不変（step を省略した変数）。n はリテラルとして埋め込む。"
  (%eval-tracing
   `(nb:with-tracing (h0 w)
      (do ((i 0 (1+ i))
           (h h0 (tanh (+ (* h s) 0.1)))
           (s w)
           (k 0.0 (+ k 1.0)))
          ((>= i ,n) (values h k))))))

(defun %do-traced-from (start n)
  "カウンタを start から始め、= で止める形。"
  (%eval-tracing
   `(nb:with-tracing (h0)
      (do ((i ,start (+ i 1))
           (h h0 (* h 2.0)))
          ((= i ,n) h)))))

(defun %loop-vec (seed size)
  (make-random-array (make-array-spec (list size) :f32) :seed seed :domain :unit))

(test loop-scan/do-equals-lisp-loop
  "展開した scan の結果は、同じループを Lisp で回した結果と一致する
（反復回数 0 と 1 を含む。複数の carry、不変な変数、スカラーの carry）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 6) (integer 1 4)))
       (lambda (case)
         (destructuring-bind (seed n size) case
           (let* ((h0 (%loop-vec seed size)) (w (%loop-vec (1+ seed) size)))
             (multiple-value-bind (h k) (funcall (%do-traced n) h0 w)
               (multiple-value-bind (ref-h ref-k) (%do-ref h0 w n)
                 (and (allclose h (make-array size :element-type 'single-float :initial-contents ref-h)
                                :dtype :f32)
                      (allclose k (make-array '() :element-type 'single-float :initial-element ref-k)
                                :dtype :f32)))))))
       :regression-id loop-scan/do-equals-lisp-loop
       :regression-file (regression-path "loop-scan-do-equals-lisp-loop"))))

(test loop-scan/counter-start-and-equal-test
  "カウンタの初期値が0でなくても、= で止める形でも、(n - start) 回だけ回る。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 3) (integer 0 5)))
       (lambda (case)
         (destructuring-bind (seed start extra) case
           (let* ((n (+ start extra)) (h0 (%loop-vec seed 2))
                  (h (funcall (%do-traced-from start n) h0)))
             (allclose h (make-array 2 :element-type 'single-float
                                       :initial-contents (map 'list (lambda (x) (* x (expt 2.0 extra))) h0))
                       :dtype :f32))))
       :regression-id loop-scan/counter-start
       :regression-file (regression-path "loop-scan-counter-start"))))

(test loop-scan/result-sees-final-counter
  "結果形式の中のカウンタ変数は最終値（反復回数 + 初期値）の Lisp の整数になる。"
  (let ((f (%eval-tracing '(nb:with-tracing (h0) (do ((i 1 (1+ i)) (h h0 (+ h 1.0))) ((>= i 4) i))))))
    (is (= 4 (funcall f (%loop-vec 1 1))))
    (is (= 1 (funcall (%eval-tracing '(nb:with-tracing (h0) (do ((i 1 (1+ i)) (h h0 (+ h 1.0))) ((>= i 0) i))))
                      (%loop-vec 1 1))))))

(test loop-scan/counter-only-loop-returns-final-counter
  "carry が無い（カウンタだけの）ループも動き、最終のカウンタだけ返す。"
  (is (= 5 (funcall (%eval-tracing '(nb:with-tracing (x) (do ((i 0 (1+ i))) ((>= i 5) i)))) 1.0))))

;;; ---- eqn の数 ----

(defun %loop-funcall-scalar (f)
  "H0 に 0 を渡して F を呼び、スカラーの結果を Lisp の数にして返す。"
  (row-major-aref (funcall f (make-array '() :element-type 'single-float :initial-element 0.0)) 0))

(defun %do-graph (n)
  (nb::trace-to-graph (%do-traced n) (list (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32))))

(test loop-scan/graph-has-one-scan-eqn-independent-of-bound
  "反復回数を大きくしても eqn の数は増えず、:scan の eqn が1つ、その :length が反復回数になる。"
  (flet ((scan-eqns (g) (remove :scan (nb:graph-eqns g) :key (lambda (e) (nb::primitive-name (nb:eqn-prim e)))
                                                        :test-not #'eq)))
    (let ((g1 (%do-graph 1)) (g200 (%do-graph 200)))
      (is (= (length (nb:graph-eqns g1)) (length (nb:graph-eqns g200))))
      (is (= 1 (length (scan-eqns g200))))
      (is (= 200 (getf (nb:eqn-params (first (scan-eqns g200))) :length))))))

(test loop-scan/graph-evaluates-like-lisp-loop
  "トレースした graph の eval-graph も Lisp のループと一致する（PBT）。"
  (is (check-it
       (generator (tuple (integer 0 100000) (integer 0 6)))
       (lambda (case)
         (destructuring-bind (seed n) case
           (let* ((h0 (%loop-vec seed 3)) (w (%loop-vec (1+ seed) 3))
                  (results (multiple-value-list (nb:eval-graph (%do-graph n) h0 w))))
             (multiple-value-bind (ref-h ref-k) (%do-ref h0 w n)
               (and (allclose (first results)
                              (make-array 3 :element-type 'single-float :initial-contents ref-h) :dtype :f32)
                    (allclose (second results)
                              (make-array '() :element-type 'single-float :initial-element ref-k)
                              :dtype :f32))))))
       :regression-id loop-scan/graph-evaluates
       :regression-file (regression-path "loop-scan-graph-evaluates"))))

;;; ---- 対応しない形 ----

(defmacro %signals-unsupported-do ((do-form) with-tracing-form)
  "WITH-TRACING-FORM を展開すると、フォームとして DO-FORM そのものを持つ
UNSUPPORTED-FORM が signal されること。"
  `(handler-case (progn (macroexpand-1 ,with-tracing-form) (fail "UNSUPPORTED-FORM が signal されなかった"))
     (nb:unsupported-form (c)
       (is (equal ,do-form (nb:unsupported-form-form c))))))

(test loop-scan/unsupported-shapes-name-the-do-form
  "定型でない do は、原因の do フォームそのものを示す UNSUPPORTED-FORM になる
（本体にフォームがある、終了条件が無い、カウンタの step が (1+ i) でない、
終了条件が >= か = でない、終了条件の上限が do 変数を参照する）。"
  (let ((body '(do ((i 0 (1+ i)) (h h0 h)) ((>= i 3) h) (print h)))
        (no-end '(do ((i 0 (1+ i)) (h h0 h)) () h))
        (step2 '(do ((i 0 (+ i 2)) (h h0 h)) ((>= i 3) h)))
        (other-test '(do ((i 0 (1+ i)) (h h0 h)) ((< 5 i) h)))
        (bound-var '(do ((i 0 (1+ i)) (h h0 h)) ((>= i h) h)))
        (no-counter '(do ((h h0 h)) ((>= h 3) h))))
    (dolist (form (list body no-end step2 other-test bound-var no-counter))
      (%signals-unsupported-do (form) `(nb:with-tracing (h0) ,form)))))

(test loop-scan/nested-do-inside-do-body-is-lowered
  "do の step の中にある do もそれぞれ scan になる（外側の本体が内側のトレースで閉包になる）。"
  (let* ((f (%eval-tracing
             '(nb:with-tracing (h0)
               (do ((i 0 (1+ i))
                    (h h0 (do ((j 0 (1+ j)) (g h (* g 2.0))) ((>= j 2) g))))
                   ((>= i 3) h)))))
         (h0 (%loop-vec 7 2))
         (h (funcall f h0)))
    (is (allclose h (make-array 2 :element-type 'single-float
                                  :initial-contents (map 'list (lambda (x) (* x 64.0)) h0))
                  :dtype :f32))))

(test loop-scan/large-bound-does-not-allocate-per-iteration
  "反復回数に比例するメモリを確保しない（トレースだけなら上限が 10 億でも速い）。"
  (let ((g (nb::trace-to-graph (%do-traced 1000000000) (list (nb:make-aval '(3) :f32) (nb:make-aval '(3) :f32)))))
    (is (= 1000000000 (getf (nb:eqn-params (find :scan (nb:graph-eqns g)
                                                 :key (lambda (e) (nb::primitive-name (nb:eqn-prim e)))))
                            :length)))))

(defvar *loop-scan-trail* '())
(defun %note (tag value)
  "TAG を *LOOP-SCAN-TRAIL* に記録して VALUE を返す（with-tracing の中では setq が使えないので、外の関数で副作用を起こす）。"
  (push tag *loop-scan-trail*)
  value)

(test loop-scan/init-order-follows-do
  "init 式は do と同じ束縛の順に評価される（副作用の順序が変わらない）。"
  (let* ((*loop-scan-trail* '())
         (f (%eval-tracing
             '(nb:with-tracing (h0)
               (do ((a (%note :a h0) a)
                    (i (%note :i 0) (1+ i))
                    (b (%note :b h0) b))
                   ((>= i (%note :bound 1)) a))))))
    (funcall f (%loop-vec 1 1))
    (is (equal '(:a :i :b :bound) (reverse *loop-scan-trail*)))))

(test loop-scan/non-integer-bound-signals-scan-error
  "上限や初期値がトレース時に決まる整数でない（トレーサや実数）ときは SCAN-ERROR。"
  (let ((f (%eval-tracing '(nb:with-tracing (h0 n) (do ((i 0 (1+ i)) (h h0 (+ h 1.0))) ((>= i n) h))))))
    (signals nb:scan-error (funcall f (%loop-vec 1 1) 2.5))
    (signals nb:scan-error (funcall f (%loop-vec 1 1) (%loop-vec 2 1)))
    ;; カウンタは :i32 のスカラーなので、収まらない上限も SCAN-ERROR。
    (signals nb:scan-error (funcall f (%loop-vec 1 1) (expt 2 31)))))

(test loop-scan/counter-is-an-i32-tracer-inside-steps
  "step の中のカウンタは 0 始まり（初期値始まり）で1ずつ増える :i32 のトレーサ
（step の書き方 (1+ i) / (+ i 1) / (+ 1 i) のどれでも同じ）。i の総和を carry に溜めて確かめる。"
  (is (check-it
       (generator (tuple (integer 0 3) (integer 0 5) (integer 0 2)))
       (lambda (case)
         (destructuring-bind (start extra variant) case
           (let* ((n (+ start extra))
                  (step (ecase variant (0 '(1+ i)) (1 '(+ i 1)) (2 '(+ 1 i))))
                  (f (%eval-tracing
                      `(nb:with-tracing (h0)
                         (do ((i ,start ,step)
                              (s 0.0 (+ s (nb:convert i :f32))))
                             ((>= i ,n) s))))))
             (= (%loop-funcall-scalar f) (/ (* extra (+ start n -1)) 2.0)))))
       :regression-id loop-scan/counter-values
       :regression-file (regression-path "loop-scan-counter-values"))))

(test loop-scan/counter-may-reach-the-i32-maximum
  "カウンタの最終値が :i32 の最大値 2^31-1 ちょうどでも通る（収まらないのはその1つ上から）。"
  (let ((f (%eval-tracing '(nb:with-tracing (h0)
                            (do ((i 2147483646 (1+ i)) (h h0 (+ h 1.0))) ((>= i 2147483647) i))))))
    (is (= 2147483647 (funcall f (%loop-vec 1 1))))))

(test loop-scan/test-form-semantics
  "(>= i n) は n が初期値以下なら0回、(= i n) は n が初期値より小さいと SCAN-LENGTH-ERROR。"
  (let ((ge (%eval-tracing '(nb:with-tracing (h0) (do ((i 5 (1+ i)) (h h0 (+ h 1.0))) ((>= i 3) h)))))
        (eq* (%eval-tracing '(nb:with-tracing (h0) (do ((i 5 (1+ i)) (h h0 (+ h 1.0))) ((= i 3) h))))))
    (is (allclose (funcall ge (%loop-vec 1 2)) (%loop-vec 1 2) :dtype :f32))
    (signals nb:scan-length-error (funcall eq* (%loop-vec 1 2)))))

(test loop-scan/more-unsupported-shapes
  "終了条件が > や引数の個数の違う形、カウンタの step が (- i 1) / (+ i 0) / (+ 2 i) / (+ i 1 1)
のものも、do フォームを示す UNSUPPORTED-FORM。"
  (dolist (form '((do ((i 0 (1+ i)) (h h0 h)) ((> i 3) h))
                  (do ((i 0 (1+ i)) (h h0 h)) ((>= i) h))
                  (do ((i 0 (1+ i)) (h h0 h)) ((>= i 1 2) h))
                  (do ((i 0 (- i 1)) (h h0 h)) ((>= i 3) h))
                  (do ((i 0 (+ i 0)) (h h0 h)) ((>= i 3) h))
                  (do ((i 0 (+ 2 i)) (h h0 h)) ((>= i 3) h))
                  (do ((i 0 (+ i 1 1)) (h h0 h)) ((>= i 3) h))
                  (do ((i 0) (h h0 h)) ((>= i 3) h))))
    (%signals-unsupported-do (form) `(nb:with-tracing (h0) ,form))))

(test loop-scan/non-do-forms-keep-their-behaviour
  "dotimes と loop は従来どおり（展開後の BLOCK が）UNSUPPORTED-FORM になる。"
  (signals nb:unsupported-form (macroexpand-1 '(nb:with-tracing (x) (dotimes (i 3) x))))
  (signals nb:unsupported-form
    (macroexpand-1 '(nb:with-tracing (x) (loop for i below 3 for h = x then (+ h 1.0) finally (return h))))))

(test loop-scan/quoted-and-backquoted-do-are-left-untouched
  "quote とバッククォートの中の (do ...) は書き換えない。"
  (let ((q '(do ((i 0 (1+ i))) ((>= i 3) x))))
    (is (equal `(progn ',q) (nb::%expand-do-loops `(progn ',q))))
    (is (equal '(progn `(do ((i 0 (1+ i))) ((>= i 3) x)) x)
               (nb::%expand-do-loops '(progn `(do ((i 0 (1+ i))) ((>= i 3) x)) x))))))

(test loop-scan/local-function-named-do-is-not-a-loop
  "flet / labels / macrolet の局所関数の定義は、名前が do でもループとして読まれず、書き換えられない
（SBCL は CL の do の局所束縛をパッケージロックで拒否するので、前処理だけを見る。
ロックを外した別パッケージの do 相当を使う利用者のための挙動）。"
  (dolist (op '(flet labels macrolet))
    (let ((form `(,op ((do (a b) (+ a b))) (do 1 2))))
      (is (equal form (nb::%expand-do-loops form))))))
