;;;; nabla/iree/tests が共有するテスト部品。
;;;;
;;;; skip-unless-iree: IREE の共有ライブラリが無い環境ではテストをスキップし
;;;; (CI では NABLA_REQUIRE_IREE=1 で失敗にする)、あるときは何もしない。
;;;; stablehlo-fixture: tests/fixtures/stablehlo/ 配下の手書き StableHLO を
;;;; 文字列として読む。*vmfb-magic*: vmfb が実際に先頭に持つ ZIP の
;;;; local-file-header シグネチャ（フラットバッファの識別子ではない。
;;;; IREE は既定で「polyglot zip」形式の vmfb を出す）。

(in-package #:nabla.iree.tests)

(defparameter *vmfb-magic* #(#x50 #x4B #x03 #x04)
  "IREE が出す vmfb（polyglot zip）の先頭4バイト（ZIP local-file-header
シグネチャ \"PK\\3\\4\"）。フラットバッファ自体の識別子ではない。")

(defmacro define-iree-test (name docstring &body body)
  "fiveam:test 相当だが、本体を (block iree-test ...) でくるみ、常に
:nabla.medium スイートに登録する。skip-unless-iree はこの block から
return-from して、IREE が無い環境ではテストの残りを実行しない。

このファイルの中の IREE を使うテストは、fiveam:test の代わりに必ずこちらを
使う。スイートを :nabla.medium 固定にしているのは、ASDF がこのシステムを
ロードする過程でファイルをまたいで fiveam::*suite* の値が（このファイルの
in-suite を経由せずに）変わることがあり、周囲の *suite* に頼ると登録先の
スイートが不安定になるため。"
  `(fiveam:test (,name :suite :nabla.medium) ,docstring
     (block iree-test
       ,@body)))

(defmacro define-iree-test/large (name docstring &body body)
  "DEFINE-IREE-TEST と同じ形（本体を (block iree-test ...) でくるむ）だが、
:nabla.medium ではなく :nabla.large スイートに登録する。GPU (cuda) を
実際に使うテスト（issue #12 の local/cuda 数値一致など）はこちらを使う。"
  `(fiveam:test (,name :suite :nabla.large) ,docstring
     (block iree-test
       ,@body)))

(defmacro define-iree-test/isolated-medium (name docstring &body body)
  "DEFINE-IREE-TEST と同じ形・同じ意味論（IREE の local backend での
実行を確かめる medium テスト。skip-unless-iree も同じ）だが、
:nabla.medium ではなく独立したスイート :nabla.isolated-medium に登録する。

対象は、tests/iree/ の他の medium テストと同じ SBCL プロセス内で実行すると
in-process の libIREECompiler.so が壊れて SB-SYS:MEMORY-FAULT-ERROR で
プロセスごと落ちることが分かっているテスト（issue #34 の jit end-to-end
テスト、issue #68 で追跡）。scripts/run-tests.sh は NABLA_TEST_SIZES に
\"medium\" が含まれる限り、:nabla.medium を実行するのとは別の SBCL
プロセスでこのスイートを実行する。CI の既定スイートには含まれる
（つまり #35 の「jit(f)(x) の結果は (f x) の結果と一致する」性質は
medium で検査され続ける）が、他の medium テストが積み重ねる distinct な
IREE コンパイルの総量とは無関係な、まっさらなプロセスから始まる。"
  `(fiveam:test (,name :suite :nabla.isolated-medium) ,docstring
     (block iree-test
       ,@body)))

(defmacro skip-unless-iree (&key (library :both))
  "(iree-available-p :library LIBRARY) が偽なら、NABLA_REQUIRE_IREE 環境変数が
設定されていれば fiveam:fail で失敗させ、無ければ fiveam:skip でこのテストを
スキップし、どちらの場合も呼び出し元の test 本体（define-iree-test /
define-iree-test/large の block）から return-from する。真ならなにもしない。
fiveam:fail は非局所脱出しない（process-failure を呼ぶだけ）ので、
return-from を省くとテスト本体がそのまま実行を続けてしまう（skip-unless-cuda
の同種のバグを参照）。"
  `(unless (iree-available-p :library ,library)
     (if (let ((value (sb-ext:posix-getenv "NABLA_REQUIRE_IREE")))
           (and value (plusp (length value))))
         (progn
           (fiveam:fail "IREE libraries required but not found under ~A" (nabla.iree::iree-home))
           (return-from iree-test))
         (progn
           (fiveam:skip "IREE libraries not found under ~A (set NABLA_IREE_HOME or run scripts/build-iree.sh)"
                        (nabla.iree::iree-home))
           (return-from iree-test)))))

(defmacro skip-unless-cuda ()
  "\"cuda\" ドライバが (driver-names) にあり、かつ (make-device :cuda) が
実際に成功する（成功すればその場で release-device する）環境でだけ何もしない。
どちらか一方でも満たさなければ、NABLA_REQUIRE_CUDA 環境変数が設定されて
いれば fiveam:fail、無ければ fiveam:skip し、どちらの場合も続けて
このテストの残り（DEFINE-IREE-TEST / DEFINE-IREE-TEST/LARGE の
BLOCK IREE-TEST）から return-from する。fiveam:fail はテストを失敗
扱いにするだけで非局所脱出しない（process-failure を呼ぶだけ）ため、
return-from を省くと呼び出し元のテスト本体がそのまま実行を続けてしまう
（cuda backend の作成や check-it の実行に進み、iree-status-error が
不可解な二重の失敗として記録される上、check-it が失敗例を
tests/regressions/ に書き出してしまう）。

呼び出し側は、この前に (SKIP-UNLESS-IREE :LIBRARY :BOTH) を呼んでおくこと
（IREE の共有ライブラリ自体が無いと DRIVER-NAMES の呼び出し自体が失敗する）。"
  `(unless (and (member "cuda" (driver-names) :test #'string=)
                (handler-case
                    (let ((device (make-device :cuda)))
                      (release-device device)
                      t)
                  (iree-status-error () nil)))
     (if (let ((value (sb-ext:posix-getenv "NABLA_REQUIRE_CUDA")))
           (and value (plusp (length value))))
         (progn
           (fiveam:fail "CUDA device required (NABLA_REQUIRE_CUDA is set) but not available")
           (return-from iree-test))
         (progn
           (fiveam:skip "\"cuda\" driver or device not available (set NABLA_REQUIRE_CUDA to fail instead of skip)")
           (return-from iree-test)))))

(defun gc-and-run-finalizers ()
  "(sb-ext:gc :full t) してから (sb-kernel:run-pending-finalizers) する
（#11）。SBCL 2.2.9 は finalizer を別スレッド（finalizer thread）で
非同期に実行するため、gc :full t の直後に確認しても、ほとんどの
finalizer はまだ実行されていない（この環境での計測では約3%）。
sb-kernel:run-pending-finalizers はキューに溜まった finalizer を
呼び出しスレッドで同期的に実行するので、これを続けて呼ぶことで
テストから確実に観測できる。それでも保守的なスタックルート（GC が
「もしかしたらポインタかもしれない」ビット列をルートとして残すこと）
のせいで、run-pending-finalizers まで呼んでも少数のオブジェクトが
実行されずに生き残ることがある（この環境での計測では 10000 個中 1個
程度、つまり 9999 個は実行された）。finalizer 系のテストは、この
わずかな残留を許容する形で許容量を設定すること（0 で比較しない）。"
  (sb-ext:gc :full t)
  (sb-kernel:run-pending-finalizers))

(defmacro with-device-arrays ((&rest bindings) &body body)
  "BINDINGS の各 (VAR FORM) を、書いた順に評価して VAR に束縛し、BODY を
評価してから、束縛した順とは逆順に release-device-array する（テスト専用の
ヘルパー。finalizer は #11 まで無いので、ここで明示的に解放しないと
テストごとに IREE のバッファがリークする）。BODY やどれかの FORM が
非局所脱出しても、それまでに束縛が済んだ VAR はすべて解放する。"
  (let ((vars (mapcar #'first bindings)))
    `(let ,(mapcar (lambda (var) (list var nil)) vars)
       (unwind-protect
            (progn
              ,@(mapcar (lambda (binding) `(setf ,(first binding) ,(second binding)))
                        bindings)
              ,@body)
         ,@(mapcar (lambda (var) `(when ,var (release-device-array ,var)))
                   (reverse vars))))))

(defun stablehlo-fixture (name)
  "tests/fixtures/stablehlo/NAME.mlir の内容を文字列として返す。"
  (let ((path (asdf:system-relative-pathname
               "nabla" (format nil "tests/fixtures/stablehlo/~A.mlir" name))))
    (with-open-file (stream path :direction :input)
      (let ((text (make-string (file-length stream))))
        (let ((count (read-sequence text stream)))
          (subseq text 0 count))))))

(defun %child-source-registry ()
  "子プロセスの ASDF に、このリポジトリと依存の置き場所を見せる
CL_SOURCE_REGISTRY の値。scripts/run-tests.sh と同じ組み立て方
（リポジトリは非再帰、依存は再帰）にする。"
  (let* ((repo (namestring (asdf:system-source-directory "nabla")))
         (deps (or (sb-ext:posix-getenv "NABLA_LISP_DEPS")
                   (namestring (merge-pathnames ".local/share/nabla/lisp-deps/"
                                                 (user-homedir-pathname))))))
    (format nil "~A:~A//:" repo deps)))

(defparameter *child-sbcl-forwarded-env-vars*
  '("PATH" "HOME" "LD_LIBRARY_PATH" "LANG" "LC_ALL"
    "TMPDIR" "XDG_CACHE_HOME" "NABLA_LISP_DEPS" "SBCL_HOME")
  "子 SBCL プロセスへ、その値があれば転送する環境変数名の共通リスト
（sbcl・ASDF・CFFI の動作に関わるもの）。子プロセスを立てるテストヘルパー
（このファイルの %CHILD-SBCL-ENVIRONMENT、compiler-test.lisp の
%RUN-WITH-MISSING-IREE-HOME・%RUN-SIGNAL-REGISTRATION-CHECK-CHILD）は
このリストを共有し、転送する変数の組をここ1箇所だけで管理する。
NABLA_IREE_HOME・CL_SOURCE_REGISTRY は呼び出し側ごとに要件が違う
（転送するだけでよいか、明示的に上書き・組み立てるか）ため、ここには
含めない。")

(defun %forward-env-vars (names)
  "NAMES の各環境変数について、このプロセスに値が設定されていれば
\"NAME=VALUE\" の文字列を、無ければ何も作らずに、SB-EXT:RUN-PROGRAM の
:environment にそのまま渡せるリストにして返す。"
  (remove nil
          (mapcar (lambda (name)
                    (let ((v (sb-ext:posix-getenv name)))
                      (and v (format nil "~A=~A" name v))))
                  names)))

(defparameter *child-sbcl-timeout-seconds* 120
  "%RUN-IN-CHILD-SBCL が子プロセスに与えるタイムアウト（秒）。coreutils の
timeout(1) を使って強制終了させる。無応答（ハング）な子プロセスがテスト
スイート全体を止めてしまわないための上限で、通常の子プロセスの所要時間
（数十回の distinct なコンパイルでも数秒程度）よりかなり長く取っている。")

(defun %child-sbcl-environment ()
  "%RUN-IN-CHILD-SBCL が子プロセスに渡す環境変数のリスト。SB-EXT:RUN-PROGRAM
は :environment を渡さないと空の環境で子プロセスを起動する（マニュアル
参照）ため、*CHILD-SBCL-FORWARDED-ENV-VARS* に加えて IREE 関連の変数を
明示的に転送する。CL_SOURCE_REGISTRY はこのプロセスのものをそのまま
転送するのではなく、%CHILD-SOURCE-REGISTRY で明示的に組み立て直す。"
  (append (%forward-env-vars (list* "NABLA_IREE_HOME" "NABLA_REQUIRE_IREE"
                                     *child-sbcl-forwarded-env-vars*))
          (list (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))

(defun %run-in-child-sbcl (source &key (timeout-seconds *child-sbcl-timeout-seconds*))
  "SOURCE（Lisp のトップレベルフォームを並べた文字列）を一時ファイルへ書き、
真っさらな子 SBCL プロセスで --load して実行する。(終了コード . 標準出力 .
標準エラー出力) を多値で返す。一時ファイルは UIOP:WITH-TEMPORARY-FILE が
（mkstemp 相当のアトミックな一意名で）作り、呼び出し後に削除する
——SBCL の既定の *RANDOM-STATE* は毎回同じシードから始まるので、
(RANDOM N) で名前を作ると複数の子プロセスが同じファイル名を選んでしまい
うる（実際に踏んだ）。

環境は %CHILD-SBCL-ENVIRONMENT で明示的に組み立てる。子プロセスは
TIMEOUT-SECONDS 秒（既定 *CHILD-SBCL-TIMEOUT-SECONDS*）で coreutils の
timeout(1) から SIGTERM される。ハングした場合、終了コードは 124
（timeout(1) の慣習）になり、呼び出し側の IS 節がその値を含めて報告する
ので、原因不明のまま無限に待つことはない。

issue #68: このプロセス自身の中で libIREECompiler.so を壊しうるコンパイル
（ゼロサイズの contracting 次元を持つ dot_general の #DE 等）や、
プロセスを poisoned にする操作を試すテストは、:nabla.medium を実行している
共有プロセス自身を汚染しないよう、常にこのヘルパー経由で子プロセスの中で
行うこと。"
  (uiop:with-temporary-file (:pathname script :prefix "nabla-iree-child-" :type "lisp" :keep t)
    (unwind-protect
         (progn
           (with-open-file (stream script :direction :output :if-exists :supersede)
             (write-string source stream))
           (let ((output (make-string-output-stream))
                 (error-output (make-string-output-stream)))
             (let ((process (sb-ext:run-program
                              "timeout"
                              (list (princ-to-string timeout-seconds)
                                    "sbcl" "--non-interactive" "--load" (namestring script))
                              :search t
                              :environment (%child-sbcl-environment)
                              :output output :error error-output)))
               (values (sb-ext:process-exit-code process)
                       (get-output-stream-string output)
                       (get-output-stream-string error-output)))))
      (ignore-errors (delete-file script)))))

;;; tests/regressions/ は nabla/tests の tests/regressions.lisp が全ファイルを
;;; load しているが、そちらは nabla/iree/tests に依存しない（システム構成が
;;; 独立している）ため、nabla/tests を単体でロードする時点では
;;; nabla.iree.tests パッケージがまだ無く、この iree-*.lisp のファイルは
;;; 読み飛ばされる（tests/regressions.lisp 参照）。そのため、この
;;; パッケージ自身の regression ファイル（"iree-" で始まる名前。
;;; regression-path の呼び出し側の慣習）は、ここで自分でロードし直す。
(dolist (path (sort (uiop:directory-files
                      (asdf:system-relative-pathname "nabla" "tests/regressions/")
                      "iree-*.lisp")
                     #'string<
                     :key #'namestring))
  (load path))
