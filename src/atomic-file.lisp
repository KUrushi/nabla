;;;; atomic-file: 一時ファイルに書いてから rename で置き換える、ファイルの
;;;; アトミックで永続的な書き込み（issue #263）。vmfb のディスクキャッシュ
;;;; （src/compile-cache.lisp）と safetensors（src/nn/safetensors.lisp）が使う。
;;;;
;;;; 手順は POSIX の定石: 一時ファイルに書く → finish-output → 一時ファイルを
;;;; fsync → close → rename → 親ディレクトリを fsync。rename の前の fsync が
;;;; 無いと、OS のクラッシュ・電源断の後に「rename は済んだが中身が届いて
;;;; いない」空か壊れたファイルが残りうる。後の fsync はディレクトリの
;;;; エントリ（rename そのもの）を永続化する。
;;;;
;;;; ロジックは関数 %CALL-WITH-ATOMIC-OUTPUT-FILE に置き、マクロは薄い包みに
;;;; する（tools/mutate はマクロの展開形を変異させられない）。

(in-package #:nabla)

(defvar *full-fsync* nil
  "真なら、WITH-ATOMIC-OUTPUT-FILE（WRITE-SAFETENSORS / SAVE-CHECKPOINT と
vmfb のディスクキャッシュが使う）の fsync を、macOS では
fcntl(F_FULLFSYNC) にする。macOS の fsync(2) はデータをドライブに渡すだけで、
ドライブの書き込みキャッシュまでは書き出さないため、電源断に対して
書き込みを保証するにはこれが要る。遅い（1回あたり数〜数十 ms）ので既定は
NIL。F_FULLFSYNC に対応しないファイルシステム（ENOTSUP / EINVAL / ENOTTY）
では fsync に戻る。macOS 以外
では何もしない（Linux の fsync(2) はドライブのキャッシュの書き出しも
要求する）。")

(defvar *fcntl-function* 'sb-posix:fcntl
  "F_FULLFSYNC の fcntl に使う関数（引数は fd とコマンド）。テストが呼び出しを
観測するために束縛し直す内部のフック。")

(defun %fsync-fd (fd)
  "FD を fsync する。macOS で *FULL-FSYNC* が真なら fcntl(F_FULLFSYNC) にし、
ファイルシステムが対応しない（ENOTSUP / EINVAL / ENOTTY）ときだけ fsync に戻る。"
  #+darwin
  (when *full-fsync*
    (handler-case
        ;; 51 = F_FULLFSYNC（<sys/fcntl.h>。sb-posix に定数が無い）
        (progn (funcall *fcntl-function* fd 51) (return-from %fsync-fd))
      (sb-posix:syscall-error (e)
        ;; 45 = macOS の ENOTSUP（sb-posix に定数が無い）
        (unless (member (sb-posix:syscall-errno e) (list 45 sb-posix:einval sb-posix:enotty))
          (error e)))))
  (sb-posix:fsync fd))

(defvar *fsync-function* '%fsync-fd
  "fsync に使う関数（引数はファイル記述子）。テストが呼び出しを観測するために
束縛し直す内部のフック。")

(defun %fsync-directory (directory)
  "DIRECTORY を O_RDONLY で開いて fsync する。ディレクトリの fsync に対応しない
ファイルシステム（EINVAL を返す）では何もしない。"
  (let ((fd (sb-posix:open (sb-ext:native-namestring directory) sb-posix:o-rdonly)))
    (unwind-protect
         (handler-case (funcall *fsync-function* fd)
           (sb-posix:syscall-error (e)
             (unless (eql (sb-posix:syscall-errno e) sb-posix:einval)
               (error e))))
      (sb-posix:close fd))))

(deftype %octet ()
  "一時ファイルの要素型。tools/mutate は deftype を変異させない。(UNSIGNED-BYTE 7) /
(UNSIGNED-BYTE 9) への変異は、SBCL がオクテットのストリームとして同じに書くので
テストで区別できない（等価変異体）。"
  '(unsigned-byte 8))

(defun %missing-directories (directory)
  "DIRECTORY とその祖先のうち、まだ存在しないディレクトリのリスト（深い順）。"
  (loop for dir = directory then (uiop:pathname-parent-directory-pathname dir)
        until (probe-file dir)
        collect dir))

(defun %call-with-atomic-output-file (path function)
  "WITH-ATOMIC-OUTPUT-FILE の本体。"
  (let* ((path (merge-pathnames path (uiop:getcwd)))
         ;; 型の無い PATH は :unspecific にしないと、rename が移動元の型（tmp）を移動先に引き継ぐ。
         (target (make-pathname :type (or (pathname-type path) :unspecific) :defaults path))
         (tmp (merge-pathnames (format nil "~A.~D.tmp" (file-namestring path)
                                       (random most-positive-fixnum (make-random-state t)))
                               path))
         (created (%missing-directories (uiop:pathname-directory-pathname path))))
    (ensure-directories-exist path)
    (unwind-protect
         (progn
           (with-open-file (stream tmp :direction :output :element-type '%octet
                                       :if-exists :supersede :if-does-not-exist :create)
             (funcall function stream)
             (finish-output stream)
             (funcall *fsync-function* (sb-sys:fd-stream-fd stream)))
           (uiop:rename-file-overwriting-target tmp target)
           (%fsync-directory (uiop:pathname-directory-pathname path))
           ;; 新しく作ったディレクトリのエントリも永続化する（親を fsync する）
           (dolist (dir created)
             (%fsync-directory (uiop:pathname-parent-directory-pathname dir))))
      ;; rename の後なら tmp はもう無い。後始末の失敗で元のエラーを隠さない。
      (ignore-errors (uiop:delete-file-if-exists tmp)))
    path))

(defmacro with-atomic-output-file ((stream path) &body body)
  "PATH と同じディレクトリの一時ファイルを (UNSIGNED-BYTE 8) の出力用に開いて STREAM に束縛し、BODY を
評価してから、一時ファイルを fsync して PATH へ rename し、親ディレクトリを
fsync する。PATH（現在のディレクトリを基準に絶対化したもの）を返す。
足りないディレクトリは作り、作ったディレクトリのエントリもその親の fsync で
永続化する。

保証: 読む側は書きかけのファイルを見ない。BODY が非局所脱出（エラーを含む）
したり、一時ファイルの fsync・rename が失敗したりすれば、一時ファイルを
消して PATH の既存のファイルを変えない（rename の後に親ディレクトリを
開く・fsync するのに失敗したときは、PATH は新しい中身に置き換わった後で
エラーになる。ディレクトリの fsync に対応しない（EINVAL を返す）ファイル
システムでは、ディレクトリの fsync を飛ばす）。WITH-ATOMIC-OUTPUT-FILE から正常に戻った後は、
プロセスの失敗に加えて OS のクラッシュ・電源断の後も PATH は完全な新しい
中身になる。ただし macOS では *FULL-FSYNC* を真にしない限り、ドライブの
書き込みキャッシュに残ったデータは電源断で失われうる。同じ PATH へ同時に
書くと、最後の rename が勝つ（ロックは取らない）。"
  `(%call-with-atomic-output-file ,path (lambda (,stream) ,@body)))
