;;;; tests/regressions/ の下にある回帰テストのファイルを、すべて load する。
;;;;
;;;; check-it が失敗例を保存するときは、そのファイルに
;;;; (check-it:regression-case ...) フォームを追記する。ここで load して
;;;; おかないと、次回以降そのケースが再実行されない。

(in-package #:nabla.tests)

;;; ASDF の system-relative-pathname は "*.lisp" をワイルドカードとしてでは
;;; なく、リテラルなファイル名の一部として解釈する（parse-unix-namestring の
;;; 挙動）。そのため (directory ...) にそのまま渡しても常に NIL になり、
;;; regressions/ 以下のファイルが1つも load されない。uiop:directory-files
;;; はディレクトリと globパターンを別引数で受け取るので、これを使う。
;;;
;;; tests/regressions/ は nabla/tests 以外のテストシステム（nabla/iree/tests
;;; など）も regression-path 経由で共有しており、そこに書かれるファイルの
;;; 1行目は (in-package #:nabla.iree.tests) のような、そのシステム自身の
;;; パッケージを指す。nabla/tests は nabla/iree/tests に依存しない
;;; （システム構成は契約どおり独立）ので、nabla/tests を単体でロードする
;;; ときはそのパッケージがまだ存在せず、load がエラーになる。そのシステムの
;;; 側で（load-time に）自分の regression ファイルを読み直すので、ここでは
;;; パッケージが無いことによる失敗は警告に落として読み飛ばす。
(dolist (path (sort (uiop:directory-files
                      (asdf:system-relative-pathname "nabla" "tests/regressions/")
                      "*.lisp")
                     #'string<
                     :key #'namestring))
  (handler-case (load path)
    (sb-ext:package-does-not-exist ()
      (warn "tests/regressions.lisp: ~A のパッケージが今はロードされていないので読み飛ばした（そのテストシステム自身がロード時に読み直す）" path))))

(in-suite :nabla.small)

(test regressions/loader-registers-committed-cases
  "tests/regressions/*.lisp が実際に load され、check-it の
regression-case が登録されていることを確かめる。ここが NIL に戻ったら、
regressions.lisp のグロブが再び効かなくなった（今回の直った不具合の再発）
ということ。"
  (is (>= (length (get 'support/allclose/boundary 'check-it::regression-cases))
          1)))

;;; ---- 回帰 id の一意性（issue #220） ----
;;;
;;; check-it は失敗例を (get regression-id 'regression-cases) に積み、次の
;;; check-it が同じ id で再生する。id を2つのテストが共有すると、先に落ちた
;;; テストが保存した失敗例を後のテストが再生してしまう。しかも保存直後の
;;; エントリは regression-case オブジェクトではなく文字列なので、同じプロセスで
;;; 後のテストが再生すると NO-APPLICABLE-METHOD CHECK-IT::DATUM ("94169") で
;;; 落ちて、本当の失敗が隠れる。マクロが name 引数から id を作るテストも
;;; あるので、ソースの文面で「id そのもの」と「%with-...-check マクロ + name 引数」の重複を調べる。
;;; それ以外のマクロ（:regression-id ,name など）が作る id は走査しない。

(defun %test-source-files ()
  (directory (merge-pathnames #p"**/*.lisp"
                              (truename (asdf:system-relative-pathname "nabla" "tests/")))))

(defun %symbol-like-token-p (token)
  "TOKEN が（ASCII の）シンボルの綴りか。docstring やコメントの文章を id と取り違えない。"
  (and (plusp (length token))
       (every (lambda (c) (and (< (char-code c) 128)
                               (or (alphanumericp c) (find c "-/*+.:<>=%?!_"))))
              token)))

(defun %source-regression-keys (text)
  "TEXT にある回帰 id の持ち主を、重複を含むキーのリストで返す。
キーは、:regression-id に直接書かれたシンボル（\"id:SYMBOL\"）か、%with-...-check マクロ呼び出しの
最初の文字列リテラル（\"MACRO:NAME\"）。:regression-id にコンマで始まる式（,name や ,(intern ...)）を
書いてマクロの中で id を作るテストは、その式が何になるかをソースから読めないので走査しない
（マクロ呼び出し側の name の一意性は、各マクロの docstring に書く）。"
  (let ((keys '()))
    (let ((start 0))
      (loop for pos = (search ":regression-id " text :start2 start)
            while pos
            do (let* ((from (+ pos (length ":regression-id ")))
                      (to (position-if (lambda (c) (member c '(#\Space #\Newline #\))))
                                       text :start from))
                      (token (subseq text from to)))
                 (when (%symbol-like-token-p token)
                   (push (format nil "id:~A" (string-upcase token)) keys))
                 (setf start from))))
    (let ((start 0))
      (loop for pos = (search "(%with-" text :start2 start)
            while pos
            do (let* ((to (position-if (lambda (c) (member c '(#\Space #\Newline))) text :start pos))
                      (macro (subseq text (1+ pos) to))
                      (open (position #\" text :start to))
                      (close (and open (position #\" text :start (1+ open)))))
                 (when (and open close (%symbol-like-token-p macro) (search "-check" macro))
                   (push (format nil "~A:~A" macro (subseq text (1+ open) close)) keys))
                 (setf start (1+ pos)))))
    keys))

(test regressions/ids-are-unique
  "2つのテストが同じ回帰 id（またはマクロ経由で同じ id になる name）を持たない。"
  (let ((table (make-hash-table :test #'equal))
        (duplicates '()))
    (dolist (file (%test-source-files))
      (dolist (key (%source-regression-keys (uiop:read-file-string file)))
        (when (gethash key table)
          (push (format nil "~A (~A)" key (file-namestring file)) duplicates))
        (setf (gethash key table) t)))
    (is (plusp (hash-table-count table)) "ソースから回帰 id を1つも拾えなかった")
    (is (null duplicates) "回帰 id が重複している: ~{~A~^, ~}" duplicates)))
