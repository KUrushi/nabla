;;;; mutate.lisp -- 変異演算子と arid node の判定
;;;;
;;;; フォームを深さ優先・前順（pre-order）で走査し、演算子ごとに
;;;; 最初に見つかった適用可能なノードを1つだけ書き換える。
;;;; MUTATE-FORM は同じ構造には常に同じ順で辿り着くので、
;;;; :arith-swap / :boundary / :branch-swap は対合（2回適用すると元に戻る）。

(in-package #:nabla.mutate)

(defparameter *mutation-operators*
  '(:arith-swap :boundary :constant :branch-swap)
  "runner が定義ごとに試す順序。1つの定義につき、最初に適用できた
演算子だけを使う。")

(defparameter *arid-heads*
  '("FORMAT" "ERROR" "WARN" "CERROR" "ASSERT" "DECLARE" "DECLAIM" "CHECK-TYPE")
  "変異させないフォームの先頭シンボル名。ログ出力・エラーメッセージ・
型宣言はここに変異を入れてもテストの抜けを教えてくれないので、
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

(defun %try-node (form operator)
  "FORM 自体（子には降りない）に OPERATOR を適用できれば
(values mutated-form t)、できなければ (values form nil) を返す。"
  (ecase operator
    (:arith-swap
     (if (and (consp form) (symbolp (car form)) (%symbol-swap (car form) *arith-swap-table*))
         (values (cons (%symbol-swap (car form) *arith-swap-table*) (cdr form)) t)
         (values form nil)))
    (:boundary
     (if (and (consp form) (symbolp (car form)) (%symbol-swap (car form) *boundary-swap-table*))
         (values (cons (%symbol-swap (car form) *boundary-swap-table*) (cdr form)) t)
         (values form nil)))
    (:constant
     (if (and (numberp form) (constant-candidate form))
         (values (constant-candidate form) t)
         (values form nil)))
    (:branch-swap
     (if (%if-form-p form)
         (values (list (first form) (second form) (fourth form) (third form)) t)
         (values form nil)))))

(defun mutate-form (form operator)
  "FORM の中で OPERATOR を適用できる最初のノード（前順・深さ優先、
arid node の内側は探さない）を1つだけ書き換える。
戻り値は (values mutated-form applied-p)。適用できるノードが
なければ (values form nil) を返す。"
  (if (arid-node-p form)
      (values form nil)
      (multiple-value-bind (mutated applied) (%try-node form operator)
        (if applied
            (values mutated t)
            (if (consp form)
                (let ((applied-anywhere nil))
                  (labels ((walk (tail)
                             (if (or applied-anywhere (not (consp tail)))
                                 tail
                                 (multiple-value-bind (new-elt elt-applied)
                                     (mutate-form (car tail) operator)
                                   (if elt-applied
                                       (progn (setf applied-anywhere t)
                                              (cons new-elt (cdr tail)))
                                       (cons (car tail) (walk (cdr tail))))))))
                    (let ((new-form (walk form)))
                      (values new-form applied-anywhere))))
                (values form nil))))))
