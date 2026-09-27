;;;; reader.lisp -- ソースをトップレベルの定義単位で読み込む
;;;;
;;;; Lisp の reader でファイルをそのまま読み、各トップレベルフォームが
;;;; ファイル中の何行目から何行目までかを記録する。`in-package` を
;;;; 追跡し、以降のフォームをそのパッケージで読む。

(in-package #:nabla.mutate)

(define-condition missing-source-file (file-error)
  ()
  (:report (lambda (condition stream)
             (format stream "read-source-forms: ファイルが見つからない: ~A"
                     (file-error-pathname condition))))
  (:documentation "READ-SOURCE-FORMS に存在しないファイルを渡したときに
signal する。FILE-ERROR のサブタイプなので、run.sh のように
`(handler-case ... (file-error (e) ...))` で受けている呼び出し元は、
生の FILE-ERROR のバックトレースの代わりにこれを掴める。"))

(defstruct source-form
  "1つのトップレベルフォームと、その位置情報。"
  (form nil)
  (start-line 1 :type fixnum)
  (end-line 1 :type fixnum)
  (package (find-package "COMMON-LISP-USER")))

(setf (documentation 'make-source-form 'function)
      "SOURCE-FORM を作るコンストラクタ。:FORM :START-LINE :END-LINE :PACKAGE
の各キーワード引数は SOURCE-FORM の同名のスロットに対応する。")
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
見つからなければエラーを signal する。ここで CL-USER に黙って
フォールバックすると、そのパッケージのはずの定義がすべて CL-USER で
読まれてしまい、mutation の生存判定が意味をなさなくなる（元の
パッケージのシンボルと衝突せず、テストが変異を検出できない）。
対象システムを事前に `asdf:load-system` してからこの reader を
呼ぶこと。"
  (or (find-package (string designator))
      (error "read-source-forms: in-package の対象パッケージ ~S が見つからない。~
対象のシステムを先に asdf:load-system してから読み込むこと。"
             designator)))

(defun %skip-block-comment (stream)
  "STREAM から `#|` の直後（`|` の次）を読み進め、対応する `|#` の
直後まで読み捨てる。`#| ... #| ... |# ... |#` のようなネストにも対応する。"
  (let ((depth 1))
    (loop while (> depth 0)
          do (let ((c (read-char stream nil nil)))
               (cond
                 ((null c) (return))
                 ((and (char= c #\#) (eql (peek-char nil stream nil nil) #\|))
                  (read-char stream) (incf depth))
                 ((and (char= c #\|) (eql (peek-char nil stream nil nil) #\#))
                  (read-char stream) (decf depth)))))))

(defun %skip-whitespace-and-comments (stream)
  "STREAM の現在位置から、空白・`;` の行コメント・`#| ... |#` の
ブロックコメントを読み飛ばす。フォームの直前に付いたコメントが
そのフォームの一部として start-line に取り込まれるのを防ぐため
（read-source-forms のトップレベルループから使う）。"
  (loop
    (peek-char t stream nil nil)
    (let ((c (peek-char nil stream nil nil)))
      (cond
        ((null c) (return))
        ((char= c #\;)
         (read-line stream nil ""))
        ((char= c #\#)
         (let ((saved (file-position stream)))
           (read-char stream)
           (if (eql (peek-char nil stream nil nil) #\|)
               (progn (read-char stream) (%skip-block-comment stream))
               (progn (file-position stream saved) (return)))))
        (t (return))))))

(defun read-source-forms (pathname)
  "PATHNAME をトップレベルフォームの列として読み、SOURCE-FORM のリストを返す。
`in-package` を見つけたら、以降のフォームをそのパッケージで読む。
reader エラーに出会ったら、そこまでに読めたフォームを返す（既知の制限:
`#.` などファイルの残りを読み進められないリーダーマクロを含むファイルは
末尾が欠ける）。PATHNAME が存在しなければ、生の FILE-ERROR ではなく
分かりやすいメッセージのエラーを signal する。"
  (unless (probe-file pathname)
    (error 'missing-source-file :pathname pathname))
  (let* ((text (alexandria:read-file-into-string pathname))
         (len (length text)))
    (with-input-from-string (stream text)
      (let ((*package* (find-package "COMMON-LISP-USER"))
            ;; 変異対象のソースは信用できるコードとはいえ、read のここでの
            ;; 目的はトップレベルフォームの位置を数えることだけなので、
            ;; #. のような read time evaluation は動かさない。
            (*read-eval* nil)
            (results nil))
        (loop
          ;; フォームの前の空白（改行を含む）・行コメント・ブロックコメント
          ;; を読み飛ばしてから位置を記録する。空白だけ飛ばして止まると、
          ;; 定義の直前に付いたコメント行がその定義の一部として
          ;; start-line に取り込まれてしまう。
          (%skip-whitespace-and-comments stream)
          (let ((start-pos (file-position stream)))
            (multiple-value-bind (form errorp)
                ;; 素の READ は、フォームを読み終えたあとその直後の空白
                ;; 文字を1つ読み捨てる（CLHS: READ-PRESERVING-WHITESPACE
                ;; との違い）。これを使うと、フォームの直後に改行がある
                ;; ときに end-line が実際より1行あとにずれる。
                ;; READ-PRESERVING-WHITESPACE を使い、その読み捨てを
                ;; させないことで end-line を正しく保つ。
                (handler-case (values (read-preserving-whitespace stream nil '%%eof%%) nil)
                  (error (e)
                    (warn "read-source-forms: ~A の位置 ~D でリーダーエラー、~
それ以降は読まずに打ち切る: ~A" pathname start-pos e)
                    (values nil t)))
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
