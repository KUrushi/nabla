;;;; reader.lisp -- ソースをトップレベルの定義単位で読み込む
;;;;
;;;; Lisp の reader でファイルをそのまま読み、各トップレベルフォームが
;;;; ファイル中の何行目から何行目までかを記録する。`in-package` を
;;;; 追跡し、以降のフォームをそのパッケージで読む。

(in-package #:nabla.mutate)

(defstruct source-form
  "1つのトップレベルフォームと、その位置情報。"
  (form nil)
  (start-line 1 :type fixnum)
  (end-line 1 :type fixnum)
  (package (find-package "COMMON-LISP-USER")))

(setf (documentation 'source-form-form 'function) "読み込んだトップレベルフォームそのもの。")
(setf (documentation 'source-form-start-line 'function) "そのフォームが始まる行番号（1始まり）。")
(setf (documentation 'source-form-end-line 'function) "そのフォームが終わる行番号（1始まり、その行を含む）。")
(setf (documentation 'source-form-package 'function) "そのフォームを読んだときの *PACKAGE*。")

(defparameter *mutable-definition-heads*
  '("DEFUN" "DEFMETHOD" "DEFMACRO" "DEFPRIMITIVE")
  "変異の対象になるトップレベル定義の先頭シンボル名（パッケージ非依存）。")

(defun mutable-definition-p (form)
  "FORM が変異対象の定義（defun / defmethod / defmacro / defprimitive）なら T。"
  (and (consp form)
       (symbolp (car form))
       (member (symbol-name (car form)) *mutable-definition-heads*
               :test #'string=)
       t))

(defun %count-newlines (string end)
  (loop for i of-type fixnum from 0 below end
        count (char= (char string i) #\Newline)))

(defun %in-package-form-p (form)
  (and (consp form)
       (symbolp (car form))
       (string= (symbol-name (car form)) "IN-PACKAGE")))

(defun %resolve-package (designator)
  "IN-PACKAGE フォームの第2引数から、パッケージオブジェクトを探す。
見つからなければ現在の *PACKAGE* のままにする。"
  (or (find-package (string designator)) *package*))

(defun read-source-forms (pathname)
  "PATHNAME をトップレベルフォームの列として読み、SOURCE-FORM のリストを返す。
`in-package` を見つけたら、以降のフォームをそのパッケージで読む。
reader エラーに出会ったら、そこまでに読めたフォームを返す（既知の制限:
`#.` などファイルの残りを読み進められないリーダーマクロを含むファイルは
末尾が欠ける）。"
  (let* ((text (alexandria:read-file-into-string pathname))
         (len (length text)))
    (with-input-from-string (stream text)
      (let ((*package* (find-package "COMMON-LISP-USER"))
            (results nil))
        (loop
          (let ((start-pos (file-position stream)))
            (multiple-value-bind (form errorp)
                (handler-case (values (read stream nil '%%eof%%) nil)
                  (error () (values nil t)))
              (when errorp (return))
              (when (eq form '%%eof%%) (return))
              (let* ((end-pos (min len (file-position stream)))
                     (start-line (1+ (%count-newlines text start-pos)))
                     (end-line (1+ (%count-newlines text end-pos))))
                (push (make-source-form :form form
                                         :start-line start-line
                                         :end-line end-line
                                         :package *package*)
                      results)
                (when (%in-package-form-p form)
                  (setf *package* (%resolve-package (second form))))))))
        (nreverse results)))))

(defun %ranges-intersect-p (a-start a-end b-start b-end)
  (and (<= a-start b-end) (>= a-end b-start)))

(defun definitions-in-range (forms start end)
  "FORMS（SOURCE-FORM のリスト）のうち、変異対象の定義であり、かつ
その行範囲 [start-line,end-line] が [START,END] と重なるものだけを返す。"
  (remove-if-not
   (lambda (sf)
     (and (mutable-definition-p (source-form-form sf))
          (%ranges-intersect-p (source-form-start-line sf)
                                (source-form-end-line sf)
                                start end)))
   forms))
