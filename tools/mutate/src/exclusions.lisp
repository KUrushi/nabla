;;;; exclusions.lisp -- 除外リストの読み込みと照合
;;;;
;;;; `tools/mutate/exclusions.lisp`（データファイル）は次の形の
;;;; plist を要素とするリスト1つをトップレベルフォームとして持つ。
;;;;
;;;;   ((:file "src/core/primitives/add.lisp"
;;;;     :form "(defprimitive add ...)"
;;;;     :mutation "(* 1 x) -> (/ x 1)"
;;;;     :reason "x に 1 をかけても割っても値が変わらない等価変異体"))
;;;;
;;;; 照合ルール（README にも記載）:
;;;; - :file は変異体のファイル名の末尾と一致すること（namestring の suffix）
;;;; - :mutation は "<変異前> -> <変異後>" を PRIN1 で印字した文字列と完全一致
;;;; - :form があれば、変異前フォームを PRIN1 した文字列に部分文字列として含まれること
;;;; - :reason は記録のためだけで、照合には使わない

(in-package #:nabla.mutate)

(defun default-exclusions-path ()
  "tools/mutate/exclusions.lisp の絶対パスを返す。"
  (asdf:system-relative-pathname "nabla-mutate" "exclusions.lisp"))

(defun load-exclusions (&optional (path (default-exclusions-path)))
  "PATH から除外リスト（plist のリスト）を読む。ファイルが存在しない
か中身が空なら空リストを返す。"
  (if (probe-file path)
      (with-open-file (stream path)
        (or (read stream nil nil) nil))
      nil))

(defun %prin1-fixed (object)
  "OBJECT を PRIN1 で印字した文字列を返す。呼び出し元の *PRINT-PRETTY* や
*PACKAGE* などに関わらず結果が同じになるよう、印字変数を固定する
（*PACKAGE* を固定しないと、呼び出し元のパッケージによってシンボルの
印字にパッケージ修飾子が付いたり付かなかったりし、同じ変異体でも
除外リストの文字列と一致したりしなかったりする）。"
  (let ((*print-pretty* nil)
        (*print-case* :upcase)
        (*print-circle* nil)
        (*print-length* nil)
        (*print-level* nil)
        (*package* (find-package "COMMON-LISP-USER")))
    (prin1-to-string object)))

(defun %mutation-string (original mutated)
  "ORIGINAL と MUTATED を印字して \"<変異前> -> <変異後>\" にする。"
  (format nil "~A -> ~A" (%prin1-fixed original) (%prin1-fixed mutated)))

(defun %file-matches-p (pathname pattern)
  "PATTERN が PATHNAME の namestring の末尾と、パスの区切りとして一致
するか。単なる文字列の suffix 一致だと、\"add.lisp\" が
\"broadcast_add.lisp\" のような無関係のファイルにもマッチしてしまう
ので、一致箇所の直前が `/` か文字列の先頭であることも確かめる。"
  (let* ((namestring (namestring pathname))
         (plen (length pattern))
         (nlen (length namestring))
         (start (- nlen plen)))
    (and (<= plen nlen)
         (string= pattern namestring :start2 start)
         (or (zerop start) (char= (char namestring (1- start)) #\/)))))

(defun excluded-p (exclusions pathname operator original mutated)
  "EXCLUSIONS（LOAD-EXCLUSIONS の戻り値）の中に、この変異体に一致する
エントリがあれば、その plist を返す。なければ NIL。OPERATOR はここでは
照合に使わない（:mutation 文字列に演算の変化が現れるため）。"
  (declare (ignore operator))
  (let ((mutation-string (%mutation-string original mutated)))
    (find-if (lambda (entry)
               (and (%file-matches-p pathname (getf entry :file))
                    (string= mutation-string (getf entry :mutation))
                    (or (null (getf entry :form))
                        (search (getf entry :form) (%prin1-fixed original)))))
             exclusions)))
