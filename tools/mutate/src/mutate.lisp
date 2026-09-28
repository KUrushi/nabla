;;;; mutate.lisp -- 変異演算子と arid node の判定
;;;;
;;;; フォームを深さ優先・前順（pre-order）で走査し、演算子を適用できる
;;;; 箇所ごとに1つずつ変異体を作る（MUTATION-SITES）。同じ構造には常に
;;;; 同じ順で辿り着くので、変異体の並びは決定的になる。
;;;; :arith-swap / :boundary / :branch-swap / :negate-condition は、
;;;; 最初の箇所に2回適用すると元に戻る（対合）。

(in-package #:nabla.mutate)

(defparameter *mutation-operators*
  '(:arith-swap :boundary :constant :off-by-one :negate-condition
    :delete-form :equality-swap :member-drop :string-constant)
  "runner が定義ごとに試す演算子と、その順序。各演算子を適用できる
すべての箇所に1つずつ変異体を作る。:branch-swap は実装しているが
既定では使わない（4引数の `if` では :negate-condition と意味が同じ
変異体になり、テストの実行時間を倍にするだけのため）。")

(defparameter *arid-heads*
  '("FORMAT" "ERROR" "WARN" "CERROR" "ASSERT" "DECLARE" "DECLAIM" "CHECK-TYPE" "GENSYM")
  "変異させないフォームの先頭シンボル名。ログ出力・エラーメッセージ・
型宣言・gensym の名前（印字にしか効かない）はここに変異を入れても
テストの抜けを教えてくれないので、
サブフォームごと走査から除外する。")

(defun arid-node-p (form)
  "FORM が arid node（変異させても意味のないノード）なら T。
`the` は最初の引数が型指定なので、それ自体は arid とはみなさない
（値を返す式は変異の対象になり得る）が、宣言・ログ出力・エラー
メッセージの類はここで止める。"
  (and (consp form)
       (symbolp (car form))
       (member (symbol-name (car form)) *arid-heads* :test #'string=)
       t))

(defun %defmethod-form-p (form)
  "FORM がトップレベルの defmethod 定義なら T。"
  (and (consp form)
       (symbolp (car form))
       (string= (symbol-name (car form)) "DEFMETHOD")))

(defun %defmethod-lambda-list-index (form)
  "FORM が defmethod のとき、specialized lambda list が並ぶ位置
（0始まり、defmethod 自身が0）を返す。method qualifier（:before など）は
リストでない atom として名前の直後に並ぶ約束なので、名前（インデックス1）
より後で最初に listp（NIL を含む）になった要素が lambda list になる。
そのような要素が見つからなければ NIL を返す。"
  (loop for index from 2
        for tail on (cddr form)
        for elt = (car tail)
        when (listp elt)
          return index))

(defparameter *arith-swap-table*
  '((+ . -) (- . +) (* . /) (/ . *)))

(defparameter *boundary-swap-table*
  '((< . <=) (<= . <) (> . >=) (>= . >)))

(defun %symbol-swap (symbol table)
  (loop for (from . to) in table
        when (and (symbolp symbol) (string= (symbol-name symbol) (symbol-name from)))
          return to))

(defun %if-form-p (form)
  (and (consp form)
       (symbolp (car form))
       (string= (symbol-name (car form)) "IF")
       (= (length form) 4)))

(defun constant-candidate (n)
  "N を置き換える定数を1つ選ぶ。0, 1, (- n) の順で、N と等しくない
最初のものを使う。N が実数なら 0 と 1 は必ず異なるので、この2つの
うちどちらかが必ず選ばれ、(- n) まで進むことはない
（0 と 1 が両方 N と等しいことはあり得ないため）。それでも
(- n) を候補に残しているのは、mutation.md の仕様の順序どおりに
実装し、将来 N が実数以外（比較の意味が変わる型）を取り得るように
なったときの安全側の振る舞いを保つため。"
  (dolist (candidate (list 0 1 (list '- n)))
    (unless (equal candidate n)
      (return-from constant-candidate candidate)))
  nil)


(defparameter *equality-swap-table*
  '((equal . eq) (equalp . equal))
  "等価述語を「より厳しい」ものへ置き換える表。`equal` → `eq` は新しく
作ったリスト（形状など）の比較を壊す。逆向き（`eq` → `equal`）は
keyword やシンボルの比較では常に等価変異体になるだけなので入れない。")

(defparameter *body-start-table*
  '(("PROGN" 1) ("WHEN" 2) ("UNLESS" 2) ("LET" 2) ("LET*" 2) ("FLET" 2)
    ("LABELS" 2) ("DOLIST" 2) ("DOTIMES" 2) ("LAMBDA" 2) ("HANDLER-BIND" 2)
    ("MULTIPLE-VALUE-BIND" 3) ("DESTRUCTURING-BIND" 3) ("DEFUN" 3) ("DEFMACRO" 3)
    ("UNWIND-PROTECT" 2 t))
  ":delete-form が本体とみなす位置。(先頭シンボル名 本体の開始位置
[最後のフォームも消してよいか])。最後のフォームは返り値なので、
既定では消さない（`unwind-protect` の後始末フォームは返り値にならない
ので、最後のものも消す）。")

(defun %head-p (form name)
  (and (consp form) (symbolp (car form)) (string= (symbol-name (car form)) name)))

(defun %replace-nth (list index new)
  (let ((copy (copy-list list)))
    (setf (nth index copy) new)
    copy))

(defun %negate (test)
  "TEST を反転させた式。すでに (not x) なら x に戻す（対合にするため）。"
  (if (and (%head-p test "NOT") (consp (cdr test)) (null (cddr test)))
      (second test)
      (list 'not test)))

(defun %negate-condition-sites (form)
  (cond
    ((and (%head-p form "IF") (<= 3 (length form) 4))
     (list (%replace-nth form 1 (%negate (second form)))))
    ((and (%head-p form "WHEN") (consp (cdr form)))
     (list (cons 'unless (cdr form))))
    ((and (%head-p form "UNLESS") (consp (cdr form)))
     (list (cons 'when (cdr form))))
    ((%head-p form "COND")
     (loop for clause in (cdr form)
           for index from 1
           when (and (consp clause)
                     (not (and (symbolp (car clause))
                               (member (symbol-name (car clause)) '("T" "OTHERWISE")
                                       :test #'string=))))
             collect (%replace-nth form index (cons (%negate (car clause)) (cdr clause)))))))

(defun %body-start (form)
  "FORM の本体が始まる位置と、最後のフォームも消してよいかを返す。
本体を持つ形式でなければ NIL。"
  (cond
    ((%defmethod-form-p form)
     (let ((index (%defmethod-lambda-list-index form)))
       (and index (values (1+ index) nil))))
    ((and (consp form) (symbolp (car form)))
     (let ((entry (assoc (symbol-name (car form)) *body-start-table* :test #'string=)))
       (and entry (values (second entry) (third entry)))))))

(defun %delete-form-sites (form)
  (multiple-value-bind (start allow-last) (%body-start form)
    (when (and start (ignore-errors (list-length form)))
      (let ((last-index (1- (length form))))
        (loop for elt in (nthcdr start form)
              for index from start
              when (and (consp elt)
                        (not (arid-node-p elt))
                        (or allow-last (< index last-index)))
                collect (append (subseq form 0 index) (nthcdr (1+ index) form)))))))

(defun %member-drop-sites (form)
  (let ((items (and (%head-p form "MEMBER")
                    (<= 3 (length form))
                    (%head-p (third form) "QUOTE")
                    (second (third form)))))
    (when (and (consp items) (ignore-errors (list-length items)) (<= 2 (length items)))
      (loop for index below (length items)
            collect (%replace-nth form 2 (list 'quote (append (subseq items 0 index)
                                                               (nthcdr (1+ index) items))))))))

(defun %swap-head-sites (form table)
  (let ((to (and (consp form) (%symbol-swap (car form) table))))
    (and to (list (cons to (cdr form))))))

(defun %node-sites (form operator)
  "FORM 自体（子には降りない）に OPERATOR をかけた変異体のリスト。"
  (ecase operator
    (:arith-swap (%swap-head-sites form *arith-swap-table*))
    (:boundary (%swap-head-sites form *boundary-swap-table*))
    (:equality-swap (%swap-head-sites form *equality-swap-table*))
    (:constant (and (numberp form) (constant-candidate form)
                    (list (constant-candidate form))))
    (:off-by-one (and (integerp form) (list (1+ form) (1- form))))
    (:branch-swap (and (%if-form-p form)
                       (list (list (first form) (second form) (fourth form) (third form)))))
    (:negate-condition (%negate-condition-sites form))
    (:delete-form (%delete-form-sites form))
    (:member-drop (%member-drop-sites form))
    (:string-constant (and (%token-string-p form) (list "")))))

(defun %token-string-p (object)
  "OBJECT が、出力するトークン（StableHLO の演算名 \"add\" など）らしい
文字列なら T。空白・非 ASCII・~ を含む文字列は、自前のエラー関数に
渡すメッセージや format の制御文字列であることがほとんどで、変異させても
テストの抜けを教えないので対象にしない。"
  (and (stringp object)
       (plusp (length object))
       (every (lambda (c) (and (graphic-char-p c) (char/= c #\Space) (char/= c #\~)
                               (< (char-code c) 128)))
              object)))

(defun %docstring-index (form)
  "FORM が本体を持つ形式（%BODY-START）で、本体の先頭が docstring なら
その位置。本体の先頭の文字列は、後ろにまだフォームがあるときだけ
docstring になる（最後なら返り値）。"
  (let ((start (%body-start form)))
    (and start
         (ignore-errors (list-length form))
         (stringp (nth start form))
         (nthcdr (1+ start) form)
         start)))

(defun %report-string-p (previous element)
  "ELEMENT が :report / :documentation の直後の文字列（restart-case の
説明など、利用者向けの文言）なら T。docstring と同じく変異させない。"
  (and (stringp element)
       (keywordp previous)
       (member (symbol-name previous) '("REPORT" "DOCUMENTATION") :test #'string=)
       t))

(defun mutation-sites (form operator)
  "FORM の中で OPERATOR をかけられるすべての箇所について、1箇所だけを
変えた FORM のリストを返す。順序は前順・深さ優先で決まっている
（同じ FORM には常に同じ順）。arid node の内側と、defmethod の
specialized lambda list の中には降りない（MUTATE-FORM を見よ）。"
  (unless (arid-node-p form)
    (append (%node-sites form operator)
            (when (consp form)
              (let ((skip-index (and (%defmethod-form-p form)
                                     (%defmethod-lambda-list-index form)))
                    (docstring-index (%docstring-index form)))
                (loop for tail on form
                      for previous = nil then element
                      for element = (car tail)
                      for index from 0
                      unless (or (eql index skip-index) (eql index docstring-index)
                                 (%report-string-p previous element))
                        nconc (loop for mutated in (mutation-sites (car tail) operator)
                                    collect (%replace-nth form index mutated))))))))

(defun mutate-form (form operator &optional (site 0))
  "FORM の中で OPERATOR を適用できる SITE 番目（0始まり、前順・深さ優先、
arid node の内側は探さない）の箇所だけを書き換える。
戻り値は (values mutated-form applied-p)。SITE 番目の箇所が
なければ (values form nil) を返す。

defmethod の specialized lambda list（`((x (eql 0)) ...)` のような
specializer を含むもの）は、この中の値を変異させても再評価時に
元のメソッドと違うメソッド（別の specializer の組）を新しく作って
しまうだけで、後始末（%evaluate-mutant が元の defmethod を評価し
直す）でも消えずに残ってしまう。そのため specialized lambda list は
まるごと arid として扱い、中には決して降りない
（qualifier はリストでない atom なのでもともと変異の対象にならない）。"
  (let ((sites (mutation-sites form operator)))
    (if (< site (length sites))
        (values (nth site sites) t)
        (values form nil))))
