;;;; WITH-ATOMIC-OUTPUT-FILE（src/atomic-file.lisp、issue #263）の性質。
;;;;
;;;; ファイル I/O を使うので :nabla.medium。fsync が実際に呼ばれたことは、
;;;; 内部のフック NABLA::*FSYNC-FUNCTION* を差し替えて、呼ばれた瞬間の
;;;; ファイルシステムの状態（fd がファイルかディレクトリか、PATH が新しい
;;;; 中身になっているか）で確かめる（OS の fsync は外から観測できないため）。

(in-package #:nabla.tests)

(in-suite :nabla.medium)

(defun %read-octets (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((octets (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (read-sequence octets in)
      octets)))

(defun %write-octets-atomically (path octets)
  (nb:with-atomic-output-file (out path)
    (write-sequence octets out)))

(defun %directory-names (dir)
  (sort (mapcar #'file-namestring (directory (merge-pathnames "*.*" dir))) #'string<))

(defun %random-octets ()
  (let ((octets (make-array (random 64) :element-type '(unsigned-byte 8))))
    (dotimes (i (length octets) octets)
      (setf (aref octets i) (random 256)))))

(test atomic-file/roundtrip-leaves-only-the-target
  "書いたバイト列がそのまま読め、上書きもでき、ディレクトリには PATH だけが残る。
戻り値は絶対化した PATH。拡張子の無い名前も、そのままの名前で書かれる。"
  (with-temporary-directory (dir)
    (dolist (name '("model" "model.ckpt"))
      (let ((path (merge-pathnames name dir))
            (first (%random-octets))
            (second (%random-octets)))
        (is (equal path (%write-octets-atomically path first)))
        (is (equalp first (%read-octets path)))
        (%write-octets-atomically path second)
        (is (equalp second (%read-octets path)))))
    (is (equal '("model" "model.ckpt") (%directory-names dir)))))

(test atomic-file/relative-path-and-missing-directories
  "相対パスは現在のディレクトリを基準にし（#180）、足りないディレクトリは作る。"
  (with-temporary-directory (dir)
    (uiop:with-current-directory (dir)
      ;; macOS の /var は /private/var へのシンボリックリンクなので truename で比べる。
      (let ((written (%write-octets-atomically "a/b/rel.bin" (%random-octets))))
        (is (equal (truename (merge-pathnames "a/b/rel.bin" dir)) (truename written))))
      (is (uiop:absolute-pathname-p (%write-octets-atomically "rel.bin" (%random-octets))))
      (is (probe-file (merge-pathnames "a/b/rel.bin" dir))))))

(test atomic-file/error-in-body-keeps-the-existing-file
  "本体が error で抜けると、エラーはそのまま伝わり、一時ファイルは残らず、
既存のファイルは変わらない。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "x.bin" dir))
          (old (%random-octets)))
      (%write-octets-atomically path old)
      (signals simple-error
        (nb:with-atomic-output-file (out path)
          (write-sequence (%random-octets) out)
          (error "boom")))
      (is (equalp old (%read-octets path)))
      (is (equal '("x.bin") (%directory-names dir))))))

(test atomic-file/fsyncs-the-file-before-and-the-directory-after-the-rename
  "fsync はちょうど2回: 1回目は rename の前に一時ファイル（通常ファイル、PATH は
まだ古い中身）、2回目は rename の後に親ディレクトリ（PATH は新しい中身）。"
  (with-temporary-directory (dir)
    (let* ((path (merge-pathnames "x.bin" dir))
           (old (%random-octets))
           ;; old と必ず違う中身（どちらも空になりうる乱数だけだと偶然一致する）
           (new (concatenate '(vector (unsigned-byte 8)) old #(1)))
           (calls '()))
      (%write-octets-atomically path old)
      (let ((nabla::*fsync-function*
              (lambda (fd)
                (push (list (if (sb-posix:s-isdir (sb-posix:stat-mode (sb-posix:fstat fd)))
                                :directory
                                :file)
                            (equalp new (%read-octets path)))
                      calls)
                (nabla::%fsync-fd fd))))
        (%write-octets-atomically path new))
      (is (equal '((:file nil) (:directory t)) (reverse calls))))))

(test atomic-file/failed-fsync-keeps-the-existing-file
  "一時ファイルの fsync が失敗すると、エラーが伝わり、rename せず、一時ファイルも残さない。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "x.bin" dir))
          (old (%random-octets)))
      (%write-octets-atomically path old)
      (let ((nabla::*fsync-function* (lambda (fd) (declare (ignore fd)) (error "fsync failed"))))
        (signals simple-error (%write-octets-atomically path (%random-octets))))
      (is (equalp old (%read-octets path)))
      (is (equal '("x.bin") (%directory-names dir))))))

(test atomic-file/full-fsync-writes-the-same-file
  "*FULL-FSYNC* を真にしても（macOS では F_FULLFSYNC、ほかでは fsync）同じように書ける。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "x.bin" dir))
          (octets (%random-octets))
          (nb:*full-fsync* t))
      (%write-octets-atomically path octets)
      (is (equalp octets (%read-octets path))))))

(defun %fd-kind (fd)
  (if (sb-posix:s-isdir (sb-posix:stat-mode (sb-posix:fstat fd))) :directory :file))

(test atomic-file/failed-rename-leaves-no-temporary-file
  "rename が失敗すると（PATH が既存のディレクトリ）、エラーが伝わり、一時ファイルは残らない。"
  (with-temporary-directory (dir)
    (ensure-directories-exist (merge-pathnames "x.bin/" dir))
    (signals error (%write-octets-atomically (merge-pathnames "x.bin" dir) (%random-octets)))
    (is (equal '("x.bin") (mapcar (lambda (p) (car (last (pathname-directory p))))
                                  (directory (merge-pathnames "*.*" dir)))))))

(defun %directory-fsync-signals (errno)
  "ディレクトリの fsync だけが ERRNO の syscall-error になるフック。"
  (lambda (fd)
    (if (eq :directory (%fd-kind fd))
        (error 'sb-posix:syscall-error :errno errno :name 'fsync)
        (nabla::%fsync-fd fd))))

(test atomic-file/directory-fsync-einval-is-skipped
  "ディレクトリの fsync に対応しない（EINVAL）ファイルシステムでは、それを飛ばして正常に戻る。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "x.bin" dir))
          (octets (%random-octets)))
      (let ((nabla::*fsync-function* (%directory-fsync-signals sb-posix:einval)))
        (%write-octets-atomically path octets))
      (is (equalp octets (%read-octets path))))))

(test atomic-file/directory-fsync-error-signals-after-the-rename
  "ディレクトリの fsync が EINVAL 以外で失敗すると、PATH は新しい中身に置き換わった後でエラーになる。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "x.bin" dir))
          (octets (%random-octets)))
      (let ((nabla::*fsync-function* (%directory-fsync-signals sb-posix:eio)))
        (signals sb-posix:syscall-error (%write-octets-atomically path octets)))
      (is (equalp octets (%read-octets path)))
      (is (equal '("x.bin") (%directory-names dir))))))

(defun %fcntl-commands-of-a-write (full-fsync)
  "*FULL-FSYNC* を FULL-FSYNC に束縛して1回書き、F_FULLFSYNC のフックに渡った (fd の種類 コマンド) の列を返す。"
  (with-temporary-directory (dir)
    (let ((calls '())
          (nb:*full-fsync* full-fsync))
      (let ((nabla::*fcntl-function*
              (lambda (fd command)
                (push (list (%fd-kind fd) command) calls)
                (sb-posix:fcntl fd command))))
        (%write-octets-atomically (merge-pathnames "x.bin" dir) (%random-octets)))
      (reverse calls))))

(test atomic-file/full-fsync-uses-f-fullfsync-only-on-macos
  "*FULL-FSYNC* が真なら、macOS ではファイルとディレクトリに fcntl(F_FULLFSYNC = 51) を使う。
偽のとき、macOS 以外のときは使わない。"
  (is (null (%fcntl-commands-of-a-write nil)))
  (is (equal #+darwin '((:file 51) (:directory 51)) #-darwin '()
             (%fcntl-commands-of-a-write t))))

#+darwin
(test atomic-file/full-fsync-falls-back-only-when-unsupported
  "F_FULLFSYNC が ENOTSUP（45）なら fsync に戻って書け、EIO（5）ならエラーが伝わる。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "x.bin" dir))
          (octets (%random-octets))
          (nb:*full-fsync* t))
      (flet ((failing-with (errno)
               (lambda (fd command)
                 (declare (ignore fd command))
                 (error 'sb-posix:syscall-error :errno errno :name 'fcntl))))
        (let ((nabla::*fcntl-function* (failing-with 45)))
          (%write-octets-atomically path octets))
        (is (equalp octets (%read-octets path)))
        (let ((nabla::*fcntl-function* (failing-with 5)))
          (signals sb-posix:syscall-error (%write-octets-atomically path (%random-octets))))
        (is (equalp octets (%read-octets path)))))))

(test atomic-file/created-directories-are-made-durable
  "足りないディレクトリを作って書くと、PATH の親に加えて、作ったディレクトリそれぞれの親も
fsync する（a/b/ を作れば b・a・元のディレクトリ）。既にあるディレクトリの祖先は fsync しない。"
  (with-temporary-directory (dir)
    (let ((inodes '()))
      (let ((nabla::*fsync-function*
              (lambda (fd)
                (when (eq :directory (%fd-kind fd))
                  (push (sb-posix:stat-ino (sb-posix:fstat fd)) inodes))
                (nabla::%fsync-fd fd))))
        (%write-octets-atomically (merge-pathnames "a/b/x.bin" dir) (%random-octets)))
      (flet ((ino (name) (sb-posix:stat-ino (sb-posix:stat (merge-pathnames name dir)))))
        (is (equal (list (ino "a/b/") (ino "a/") (ino "")) (reverse inodes)))))))

(test atomic-file/does-not-leak-file-descriptors
  "書いた後にファイル記述子が残らない（次に開く fd の番号が変わらない）。"
  (with-temporary-directory (dir)
    (flet ((next-fd ()
             (let ((fd (sb-posix:open "/dev/null" sb-posix:o-rdonly)))
               (sb-posix:close fd)
               fd)))
      (let ((before (next-fd)))
        (dotimes (i 3)
          (%write-octets-atomically (merge-pathnames "x.bin" dir) (%random-octets)))
        (is (= before (next-fd)))))))
