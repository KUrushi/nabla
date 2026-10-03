;;;; with-tracing の中の定型の DO ループを SCAN に展開する（issue #137）。
;;;;
;;;; 展開は WITH-TRACING の最初の処理（マクロ展開 SB-CLTL2:MACROEXPAND-ALL より前）で、
;;;; 展開前のフォームに対して行う。DO は展開されると BLOCK / TAGBODY / SETQ になって
;;;; 元の形が分からなくなるため。
;;;;
;;;; 対応する形（これ以外の DO は、原因の DO フォームを示す UNSUPPORTED-FORM）:
;;;;
;;;;   (do ((i INIT (1+ i))            ; カウンタ。step は (1+ i) か (+ i 1) か (+ 1 i)
;;;;        (h INIT-H STEP-H)          ; carry。STEP-H は純粋な式（SETQ は書けない）
;;;;        (s INIT-S))                ; step の無い変数は不変（毎回そのまま carry する）
;;;;       ((>= i N) RESULT...)        ; 終了条件は (>= i N) か (= i N)
;;;;     )                            ; 本体のフォームは無し（宣言だけ可）
;;;;
;;;; カウンタの INIT と N はトレース時に決まる整数（実行時に整数でなければ SCAN-ERROR。
;;;; N は do 変数を参照できない）。反復回数は (>= i N) なら max(0, N - INIT)、
;;;; (= i N) なら N - INIT（N < INIT なら Lisp では止まらないので SCAN-LENGTH-ERROR）。
;;;; 展開すると carry を持つ1つの SCAN になる（反復回数に依らず eqn は1つ）。
;;;; step は並列に評価される（DO と同じ。全 step が古い値を見る）。カウンタは step の
;;;; 中では :i32 のスカラーのトレーサ（scan の carry の先頭に足す）、結果形式の中では Lisp の整数（最終値）になる。
;;;; 各 step の dtype・shape は init と同じでなければならない（SCAN-CARRY-MISMATCH）。
;;;;
;;;; 展開しないもの: DO*、DOTIMES、LOOP（従来どおり展開後の BLOCK が UNSUPPORTED-FORM
;;;; になる。DOTIMES は carry を SETQ でしか渡せず、LOOP の for ... = ... then ... は
;;;; 更新と終了判定の順序が DO と違うので、同じ意味にできない）。QUOTE / バッククォートの中と、
;;;; FLET / LABELS / MACROLET の局所関数の定義（名前が do でも）は見ない。ユーザーのマクロが DO に展開されるものは、この段階では
;;;; 見えないので展開しない。また、DO の形をした束縛（(let ((do ...)))）は DO の構文として
;;;; 読めれば DO として扱う。

(in-package #:nabla)

(defun %do-binding-p (binding)
  (or (and (symbolp binding) binding (not (keywordp binding)))
      (and (consp binding) (symbolp (first binding)) (first binding)
           (listp (cdr binding)) (null (cdddr binding))
           (null (cdr (last binding))))))

(defun %do-syntax-p (form)
  "FORM が DO の構文（(do (binding...) (end-test result...) body...)）として読めるか。"
  (and (consp (cdr form)) (consp (cddr form))
       (null (cdr (last form)))
       (listp (second form)) (null (cdr (last (second form))))
       (every #'%do-binding-p (second form))
       (listp (third form)) (null (cdr (last (third form))))))

(defun %do-binding-var (binding) (if (consp binding) (first binding) binding))
(defun %do-binding-init (binding) (if (consp binding) (second binding) nil))
(defun %do-binding-step-p (binding) (and (consp binding) (cddr binding) t))
(defun %do-binding-step (binding) (third binding))

(defun %tree-mentions-p (symbols tree)
  (cond ((symbolp tree) (and (member tree symbols) t))
        ((consp tree) (or (%tree-mentions-p symbols (car tree))
                          (%tree-mentions-p symbols (cdr tree))))))

(defun %do-counter-step-p (var step)
  (or (equal step `(1+ ,var)) (equal step `(+ ,var 1)) (equal step `(+ 1 ,var))))

(defun %do-parse-test (test vars)
  "終了条件 TEST が (>= v N) か (= v N)（v は VARS の1つ、N は VARS を参照しない）なら
(values v N キー) を返す。違えば NIL。"
  (when (and (consp test) (member (first test) '(>= =)) (= 3 (length test))
             (symbolp (second test)) (member (second test) vars)
             (not (%tree-mentions-p vars (third test))))
    (values (second test) (third test) (if (eq (first test) '>=) :ge :eq))))

(defun %lower-do (form)
  "定型の DO フォームを SCAN を呼ぶフォームに書き換える。定型でなければ、FORM を
示す UNSUPPORTED-FORM を signal する。"
  (destructuring-bind (bindings end-clause &rest body) (rest form)
    (let* ((vars (mapcar #'%do-binding-var bindings))
           ;; 本体のフォームは宣言（DECLARE）だけ許す。
           (others (remove-if (lambda (f) (and (consp f) (eq (first f) 'declare))) body)))
      (multiple-value-bind (counter bound test-key) (%do-parse-test (first end-clause) vars)
        (let ((counter-binding (and counter (find counter bindings :key #'%do-binding-var))))
          (when (or (null counter) others
                    (not (%do-binding-step-p counter-binding))
                    (not (%do-counter-step-p counter (%do-binding-step counter-binding))))
            (error 'unsupported-form :form form :path nil))
          (let* ((carries (remove counter bindings :key #'%do-binding-var))
                 (carry-vars (mapcar #'%do-binding-var carries))
                 (c (gensym "CARRY")) (x (gensym "X")) (count (gensym "COUNT")) (finals (gensym "FINALS"))
                 ;; DO と同じく、init は束縛の順に評価する（そのあとで上限）。
                 (counter-tmp (gensym "INIT"))
                 (carry-tmps (mapcar (lambda (b) (declare (ignore b)) (gensym "INIT")) carries))
                 (init-bindings
                   (loop for b in bindings
                         collect (list (if (eq b counter-binding)
                                           counter-tmp
                                           (nth (position b carries) carry-tmps))
                                       (%do-binding-init b))))
                 ;; scan の carry は (カウンタ 他の carry...)。カウンタは i32 のスカラー。
                 (step-body
                   `(let (,@(list `(,counter (nth 0 ,c)))
                          ,@(loop for v in carry-vars for k from 1 collect `(,v (nth ,k ,c))))
                      (declare (ignorable ,counter ,@carry-vars))
                      (values (list (+ ,counter 1)
                                    ,@(loop for b in carries
                                            collect (if (%do-binding-step-p b) (%do-binding-step b) (%do-binding-var b))))
                              nil))))
            `(let* ,init-bindings
               (multiple-value-bind (,count ,finals)
                   (%do-scan ,counter-tmp ,bound ,test-key (list ,@carry-tmps)
                             (with-tracing (,c ,x) ,x ,step-body))
                 (declare (ignorable ,count ,finals))
                 (let (,@(list `(,counter ,count))
                       ,@(loop for v in carry-vars for k from 0 collect `(,v (nth ,k ,finals))))
                   (declare (ignorable ,counter ,@carry-vars))
                   ,@(or (rest end-clause) '(nil)))))))))))

(defun %expand-do-loops (form)
  "FORM（展開前の WITH-TRACING の本体）の中の DO を、構文として読めるものだけ
%LOWER-DO で書き換えて返す。QUOTE の中は見ない。"
  (cond ((atom form) form)
        ;; QUOTE とバッククォート（sb-int:quasiquote）の中はデータなので見ない。
        ((member (first form) '(quote sb-int:quasiquote)) form)
        ((and (member (first form) '(flet labels macrolet)) (consp (cdr form)) (listp (second form)))
         (%expand-do-loops-local-functions form))
        ((and (eq (first form) 'do) (%do-syntax-p form))
         (%expand-do-loops (%lower-do form)))
        (t (%expand-do-loops-list form))))

(defun %expand-do-loops-local-functions (form)
  "(FLET|LABELS|MACROLET ((name lambda-list . body)...) . body) の中を書き換える。
局所関数の定義 (name lambda-list . body) 自体は DO の構文に見えうる（(flet ((do (a b) ...)))）
ので、定義の先頭 2 要素（名前と仮引数リスト）は見ず、本体だけを書き換える。
局所関数の呼び出し側の (do ...) が DO の構文に読めるときは、DO として扱ってしまう（制限）。"
  (list* (first form)
         (mapcar (lambda (definition)
                   (if (and (consp definition) (consp (cdr definition)))
                       (list* (first definition) (second definition)
                              (%expand-do-loops-list (cddr definition)))
                       definition))
                 (second form))
         (%expand-do-loops-list (cddr form))))

(defun %expand-do-loops-list (list)
  (cond ((atom list) list)
        (t (cons (%expand-do-loops (car list)) (%expand-do-loops-list (cdr list))))))

(defun %do-scan (init bound test-key carry-inits f)
  "%LOWER-DO が出すフォームの実行時の本体。(values 最終のカウンタ 最終の carry のリスト)
を返す。INIT・BOUND はカウンタの初期値と上限（整数）、TEST-KEY は :GE か :EQ、
F は (carry-list x-list) の2引数の traceable-function。carry-list の先頭はカウンタ
（:i32 のスカラー）で、そのあとに CARRY-INITS が続く。x-list は空。反復回数に比例する
配列は作らない（カウンタは carry で持つ）。"
  (unless (and (integerp init) (integerp bound))
    (error 'scan-error
           :format-control "do ループのカウンタの初期値と上限はトレース時に決まる整数でなければならない: ~S, ~S"
           :format-arguments (list init bound)))
  (when (and (eq test-key :eq) (< bound init))
    (error 'scan-length-error
           :format-control "(= i ~D) は i が ~D から増える do では成り立たない（Lisp では止まらない）"
           :format-arguments (list bound init)))
  (let* ((length (max 0 (- bound init)))
         (final (+ init length)))
    (unless (typep final '(signed-byte 32))
      (error 'scan-error
             :format-control "do ループのカウンタが :i32 に収まらない: 初期値 ~D、最終値 ~D"
             :format-arguments (list init final)))
    (let ((counter (make-array '() :element-type '(signed-byte 32) :initial-element init)))
      (values final
              ;; scan は公開 API。ロード順のため、シンボル経由で呼ぶ。
              (rest (funcall 'scan f (cons counter carry-inits) '() :length length))))))
