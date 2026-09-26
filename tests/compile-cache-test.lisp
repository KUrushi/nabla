;;;; vmfb ディスクキャッシュ（issue #10）の性質。
;;;;
;;;; フェイク backend（tests/support/fake-backend.lisp）の
;;;; FAKE-BACKEND-COMPILE-COUNT で「コンパイラを実際に呼んだ回数」を数え、
;;;; BACKEND-COMPILE の :AROUND キャッシュ（src/compile-cache.lisp）が
;;;; ヒット・ミスを正しく判定していることを確かめる。
;;;;
;;;; ファイル I/O とスレッド（sb-thread）を使うため :nabla.small ではなく
;;;; :nabla.medium に置く（skill の small の定義は「FFI・ファイル・スレッド
;;;; を使わない」なので、issue #10 の完了条件が書く「small」より1段階
;;;; 重いサイズにする。PR 本文にも理由を書く）。
;;;;
;;;; ほとんどのテストは NB:*COMPILE-CACHE-DIRECTORY* を一時ディレクトリに
;;;; 束縛し、~/.cache には一切触らない。例外は :DEFAULT 解決（環境変数
;;;; NABLA_CACHE_DIR の有無での分岐）を確かめる2つのテストで、そのうち
;;;; NABLA_CACHE_DIR 未設定側は実際に ${XDG_CACHE_HOME:-~/.cache}/nabla/vmfb/
;;;; に1ファイルだけ書き、UNWIND-PROTECT でそのファイルだけ削除する。

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :sb-posix))

(in-package #:nabla.tests)

(in-suite :nabla.medium)

(defparameter +add-text+ "func.func @main() { stablehlo.add }"
  "フェイク backend が :add に分類する、有効な StableHLO テキスト。")

(defparameter +matmul-text+ "\"stablehlo.dot_general\"(%a, %b)"
  "フェイク backend が :matmul に分類する、有効な StableHLO テキスト。")

(defun %module-file-count (directory)
  "DIRECTORY 直下にある *.module ファイルの個数を返す。"
  (length (directory (make-pathname :name :wild :type "module" :defaults directory))))

(test compile-cache/backend-compile/same-key-hits-cache-on-second-call
  "同じ backend・同じ text で2回 backend-compile すると、フェイクの
compile-count は 1 のまま（2回目はディスクキャッシュがヒットし、
コンパイラを呼ばない）。2回とも返るバイト列は等しい。"
  (with-temporary-directory (dir)
    (let ((nb:*compile-cache-directory* dir)
          (backend (nb:make-backend :fake)))
      (let ((first (nb:backend-compile backend +add-text+)))
        (is (= 1 (fake-backend-compile-count backend)))
        (let ((second (nb:backend-compile backend +add-text+)))
          (is (= 1 (fake-backend-compile-count backend)))
          (is (equalp first second))
          (is (= 1 (%module-file-count dir))))))))

(test compile-cache/backend-compile/different-text-makes-separate-entry
  "同じ backend でも text が違えば、別のキャッシュエントリになり、
compile-count はそれぞれ1ずつ増える（合計2）。"
  (with-temporary-directory (dir)
    (let ((nb:*compile-cache-directory* dir)
          (backend (nb:make-backend :fake)))
      (nb:backend-compile backend +add-text+)
      (nb:backend-compile backend +matmul-text+)
      (is (= 2 (fake-backend-compile-count backend)))
      (is (= 2 (%module-file-count dir))))))

(test compile-cache/backend-compile/different-fingerprint-makes-separate-entry
  "text が同じでも backend-fingerprint が違えば（ターゲットが違う想定）、
別のキャッシュエントリになる。"
  (with-temporary-directory (dir)
    (let ((nb:*compile-cache-directory* dir)
          (backend-a (nb:make-backend :fake :fingerprint '("fake" "target=a")))
          (backend-b (nb:make-backend :fake :fingerprint '("fake" "target=b"))))
      (nb:backend-compile backend-a +add-text+)
      (nb:backend-compile backend-b +add-text+)
      (is (= 1 (fake-backend-compile-count backend-a)))
      (is (= 1 (fake-backend-compile-count backend-b)))
      (is (= 2 (%module-file-count dir))))))

(test compile-cache/backend-compile/corrupted-file-is-recompiled-then-cached-again
  "キャッシュファイルを切り詰めて壊すと、次の backend-compile は
（マジック・digest の検査に失敗して）ミス扱いになり再コンパイルして
正しいファイルを書き直す。そのさらに次の呼び出しはまたヒットする。"
  (with-temporary-directory (dir)
    (let ((nb:*compile-cache-directory* dir)
          (backend (nb:make-backend :fake)))
      (nb:backend-compile backend +add-text+)
      (is (= 1 (fake-backend-compile-count backend)))
      (let ((path (first (directory (make-pathname :name :wild :type "module" :defaults dir)))))
        ;; 8バイト（マジックの途中）だけ残して切り詰める。
        (with-open-file (in path :direction :input :element-type '(unsigned-byte 8))
          (let ((truncated (make-array 8 :element-type '(unsigned-byte 8))))
            (read-sequence truncated in)
            (with-open-file (out path :direction :output :element-type '(unsigned-byte 8)
                                      :if-exists :supersede)
              (write-sequence truncated out)))))
      (let ((result (nb:backend-compile backend +add-text+)))
        (is (= 2 (fake-backend-compile-count backend)))
        (is (equalp result (sb-ext:string-to-octets +add-text+ :external-format :utf-8))))
      (nb:backend-compile backend +add-text+)
      (is (= 2 (fake-backend-compile-count backend))
          "壊れたファイルを直した後の呼び出しは、またヒットするはず"))))

(test compile-cache/backend-compile/exactly-header-length-file-with-empty-payload-is-a-hit
  "キャッシュファイルがちょうど（マジック8バイト + digest 32バイトの）
40バイトで、payload が空（0バイト）でも、マジックと digest（空バイト列の
SHA-256）さえ正しければ壊れたファイルとして扱わず、ヒットとして0バイトの
payload を返す（『40バイト未満なら壊れている』という境界の、未満ではない
側の値。< と <= を取り違えていないことを確かめる）。"
  (with-temporary-directory (dir)
    (let ((nb:*compile-cache-directory* dir)
          (backend (nb:make-backend :fake)))
      (nb:backend-compile backend +add-text+)
      (is (= 1 (fake-backend-compile-count backend)))
      (let ((path (first (directory (make-pathname :name :wild :type "module" :defaults dir)))))
        (with-open-file (out path :direction :output :element-type '(unsigned-byte 8)
                                  :if-exists :supersede)
          (write-sequence (sb-ext:string-to-octets "NBLMOD01" :external-format :ascii) out)
          (write-sequence (ironclad:digest-sequence :sha256 (make-array 0 :element-type '(unsigned-byte 8)))
                          out)))
      (let ((result (nb:backend-compile backend +add-text+)))
        (is (= 1 (fake-backend-compile-count backend))
            "40バイトちょうどの、正しい空 payload のファイルはヒットのはず")
        (is (= 0 (length result)))))))

(test compile-cache/backend-compile/concurrent-same-key-writes-one-valid-file
  "2〜4スレッドが同時に同じキーで backend-compile しても、compile-count は
スレッド数を超えず、最終的にキャッシュディレクトリには壊れていない
ファイルが1つだけ残り、その後の呼び出しは増分なしでヒットする。

SBCL の SB-THREAD:MAKE-THREAD は、生成元スレッドの LET による特殊変数の
束縛を引き継がない（子スレッドから見えるのは大域値だけ）。そのため
NB:*COMPILE-CACHE-DIRECTORY* は各スレッドのラムダの中で束縛し直す
（外側の LET で束縛するだけでは、子スレッドは既定の ~/.cache を見てしまう）。

各スレッドの本体は IGNORE-ERRORS でくるむ。SBCL は非対話モードで
スレッド内の未処理コンディションを拾うと、その場でプロセス全体を
落とす（デバッガに入れないため）。これはこのテストが仮に
（バグやミューテーションで）失敗するときに、失敗そのものを
FIVEAM の失敗として報告する代わりにプロセスごと落として mutation
testing の runner を壊してしまう、という別の問題を生む。BACKEND-COMPILE
が実際に signal したかどうかは、この性質（ファイルが1つだけ残る・
compile-count がスレッド数を超えない）の判定には関係しない。"
  (with-temporary-directory (dir)
    (let* ((backend (nb:make-backend :fake))
           (thread-count 4)
           (threads (loop repeat thread-count
                          collect (sb-thread:make-thread
                                   (lambda ()
                                     (let ((nb:*compile-cache-directory* dir))
                                       (ignore-errors (nb:backend-compile backend +add-text+))))))))
      (mapc #'sb-thread:join-thread threads)
      (is (<= (fake-backend-compile-count backend) thread-count))
      (is (>= (fake-backend-compile-count backend) 1))
      (is (= 1 (%module-file-count dir)))
      (let ((count-before (fake-backend-compile-count backend))
            (nb:*compile-cache-directory* dir))
        (nb:backend-compile backend +add-text+)
        (is (= count-before (fake-backend-compile-count backend)))))))

(test compile-cache/backend-compile/default-directory-honors-nabla-cache-dir-env
  "NB:*COMPILE-CACHE-DIRECTORY* が :DEFAULT のとき、環境変数
NABLA_CACHE_DIR が指すディレクトリの vmfb/ 以下にキャッシュファイルが
できる（issue #10 が求める『環境変数などで変えられるようにする』の
:DEFAULT 分岐そのもの。src/compile-cache.lisp の %COMPILE-CACHE-ROOT が
NABLA_CACHE_DIR の有無で分岐する2つの枝を、実際に環境変数を設定・解除
して両方確かめる）。

このテストだけは公開 API（NB:BACKEND-COMPILE と NB:*COMPILE-CACHE-DIRECTORY*）
を通して振る舞いを確かめるため、%COMPILE-CACHE-ROOT のような内部関数は
一切呼ばない。SB-POSIX:SETENV / SB-POSIX:UNSETENV でプロセスの環境変数を
書き換えるので、UNWIND-PROTECT で元の値に戻す。"
  (with-temporary-directory (dir)
    (let ((original (sb-ext:posix-getenv "NABLA_CACHE_DIR")))
      (unwind-protect
           (progn
             (sb-posix:setenv "NABLA_CACHE_DIR" (namestring dir) 1)
             (let ((nb:*compile-cache-directory* :default)
                   (backend (nb:make-backend :fake)))
               (nb:backend-compile backend +add-text+)
               (is (= 1 (%module-file-count (merge-pathnames "vmfb/" dir)))
                   "NABLA_CACHE_DIR/vmfb/ の下に .module ファイルができるはず")))
        (if original
            (sb-posix:setenv "NABLA_CACHE_DIR" original 1)
            (sb-posix:unsetenv "NABLA_CACHE_DIR"))))))

(test compile-cache/backend-compile/default-directory-falls-back-to-xdg-cache-home
  "環境変数 NABLA_CACHE_DIR が設定されていなければ、
NB:*COMPILE-CACHE-DIRECTORY* が :DEFAULT のとき
(UIOP:XDG-CACHE-HOME \"nabla/vmfb/\") の下にキャッシュファイルができる
（%COMPILE-CACHE-ROOT の :DEFAULT 分岐の、もう一方の枝）。

実際にホームディレクトリ配下の共有キャッシュに書き込むので、この
テストが作ったファイルを UNWIND-PROTECT で確実に削除する（書き込み前後の
*.module 一覧の差分から新しくできたファイルを特定し、1つに限らず残らず
消す）。text には毎回変わる乱数を混ぜて専用のキャッシュキーにし、途中で
プロセスが落ちて前回の削除が行われなかった場合でも、キャッシュに
ヒットして誤ってスキップされることなく毎回新しいファイルを作る（自己
修復する）。"
  (let* ((original (sb-ext:posix-getenv "NABLA_CACHE_DIR"))
         (unique-text (format nil "func.func @main() { stablehlo.add } ; compile-cache-test-default-xdg-probe-~D"
                               (random most-positive-fixnum (make-random-state t))))
         (directory (uiop:xdg-cache-home "nabla/vmfb/"))
         (created-files nil))
    (unwind-protect
         (progn
           (sb-posix:unsetenv "NABLA_CACHE_DIR")
           (let ((before (directory (make-pathname :name :wild :type "module" :defaults directory))))
             (let ((nb:*compile-cache-directory* :default)
                   (backend (nb:make-backend :fake)))
               (nb:backend-compile backend unique-text))
             (let* ((after (directory (make-pathname :name :wild :type "module" :defaults directory)))
                    (new-files (set-difference after before :test #'equal)))
               (is (= 1 (length new-files))
                   "NABLA_CACHE_DIR 未設定なら XDG_CACHE_HOME 側に .module ファイルが1つできるはず")
               (setf created-files new-files))))
      (dolist (file created-files)
        (ignore-errors (delete-file file)))
      (if original
          (sb-posix:setenv "NABLA_CACHE_DIR" original 1)
          (sb-posix:unsetenv "NABLA_CACHE_DIR")))))

(test compile-cache/backend-compile/nil-directory-disables-cache
  "NB:*COMPILE-CACHE-DIRECTORY* が NIL なら、毎回コンパイラを呼び、
（束縛前から存在する）一時ディレクトリにも何もファイルを作らない。"
  (with-temporary-directory (dir)
    (let ((nb:*compile-cache-directory* nil)
          (backend (nb:make-backend :fake)))
      (nb:backend-compile backend +add-text+)
      (nb:backend-compile backend +add-text+)
      (is (= 2 (fake-backend-compile-count backend)))
      (is (= 0 (%module-file-count dir))))))
