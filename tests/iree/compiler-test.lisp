;;;; nabla.iree のコンパイラ埋め込み C API バインディングのテスト。
;;;;
;;;; CFFI バインディング自体は mutation testing の対象外（CLAUDE.md /
;;;; issue #5 の補足）。ここでの主眼は「公開 API を通した疎通」と
;;;; 「壊れた入力でプロセスが落ちないこと」。

(in-package #:nabla.iree.tests)

(defun %locked-iree-commit ()
  "third_party/iree.lock の commit=... の値を返す。"
  (let ((path (asdf:system-relative-pathname "nabla" "third_party/iree.lock")))
    (with-open-file (stream path :direction :input)
      (loop for line = (read-line stream nil nil)
            while line
            when (and (>= (length line) 7) (string= "commit=" line :end2 7))
              return (string-trim '(#\Space #\Tab #\Return) (subseq line 7))))))

(defun %vmrss-kb ()
  "/proc/self/status の VmRSS（キロバイト）。Linux 専用。"
  (with-open-file (stream "/proc/self/status" :direction :input)
    (loop for line = (read-line stream nil nil)
          while line
          when (and (>= (length line) 6) (string= "VmRSS:" line :end2 6))
            return (parse-integer line :start 6 :junk-allowed t))))

(define-iree-test compiler/compile-stablehlo/matmul-fixture-produces-zip-wrapped-vmfb
    "手書きの matmul フィクスチャをコンパイルすると、非空で先頭4バイトが
ZIP local-file-header シグネチャ（IREE の polyglot zip 形式の vmfb）の
バイト列が得られる。"
  (skip-unless-iree :library :compiler)
  (let ((bytes (compile-stablehlo (stablehlo-fixture "matmul"))))
    (is (typep bytes '(simple-array (unsigned-byte 8) (*))))
    (is (plusp (length bytes)))
    (is (equalp *vmfb-magic* (subseq bytes 0 4)))))

(defun %garbage-after-open-brace (garbage)
  "関数本体の直後に、意味のない ASCII の断片を続けるだけの壊れ方。"
  (concatenate 'string "func.func @main() { " garbage))

(defun %truncated-valid-module (cut)
  "有効な matmul フィクスチャを途中で切り詰めた、構文的に不完全な StableHLO。
CUT はそのまま使わず、全体の長さの [1, len-2] に必ず収まるよう mod で
丸める（末尾2文字だけ落とすケースは、閉じ括弧が全部揃った完全に有効な
モジュールになりうるため避ける）。"
  (let* ((text (stablehlo-fixture "matmul"))
         (bound (max 1 (- (length text) 2)))
         (n (1+ (mod (abs cut) bound))))
    (subseq text 0 n)))

(defun %mismatched-result-type (dim)
  "宣言した返り値の型と実際に返す値の型が食い違う関数。DIM が 2 だと
たまたま一致してしまうので、その場合だけ 5 にずらす。"
  (let ((mismatched (if (= dim 2) 5 dim)))
    (format nil "func.func @main(%a: tensor<2x3xf32>) -> tensor<~Ax3xf32> {~%  func.return %a : tensor<2x3xf32>~%}"
            mismatched)))

(defun %unknown-op-in-known-dialect (suffix)
  "stablehlo 方言自体は登録されているが、その中に存在しない演算名を使う。
方言プレフィックスを常に stablehlo に固定するのは、IREE 自体のバグ
（登録されていない『方言』の演算が最適化パイプラインの途中
（FormDispatchRegions.cpp）でセグフォルトする、issue #5 で見つけた別の
問題）を default テストで踏まないため。方言が登録済みで演算名だけが
未知の場合は、パース時にきれいな :error 診断（iree-compile-error）で
止まることを別途確認済み（下の具体例1）。"
  (format nil "func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {~%  %0 = \"stablehlo.~A\"(%a) : (tensor<4xf32>) -> tensor<4xf32>~%  func.return %0 : tensor<4xf32>~%}"
          (if (zerop (length suffix)) "nabla_unknown" suffix)))

(defun %mismatched-braces (count)
  "開き括弧に対して閉じ括弧の数が合わない、構造的に壊れた入力。"
  (concatenate 'string "func.func @main() {" (make-string (mod count 5) :initial-element #\))))

(define-iree-test compiler/compile-stablehlo/malformed-text-signals-compile-error
    "MLIR として文法や意味の誤った StableHLO をコンパイルすると、少なくとも
1つの :error 診断を含む iree-compile-error が signal され、プロセスは
落ちない。check-it の (string) 単体だと [A-Za-z0-9] の短い文字列しか
出さず、開き括弧の直後で構文エラーになるパターンしか踏まないので、
構造的に異なる壊れ方（切り詰め・型の不一致・未知の演算・括弧の不整合）を
OR で混ぜる。方言プレフィックスは常に登録済み（stablehlo / func）にして、
未登録の方言によるセグフォルト（issue #5 の別バグ、下のコメント参照）を
避ける。

issue #68 の回帰: これらはどれも（プロセスを壊さない）普通のコンパイル
失敗であって #DE のような非局所脱出ではないので、コンパイラを poisoned に
してはならない。テストの最後で NABLA.IREE::*COMPILER-POISON-REASON* が
NIL のままであることを確かめる。"
  (skip-unless-iree :library :compiler)
  (is (check-it (generator (or (map #'%garbage-after-open-brace (string))
                                (map #'%truncated-valid-module (integer 0 500))
                                (map #'%mismatched-result-type (integer 0 10))
                                (map #'%unknown-op-in-known-dialect (string :max-length 12))
                                (map #'%mismatched-braces (integer 0 10))))
                (lambda (text)
                  (handler-case
                      (progn (compile-stablehlo text) nil)
                    (iree-compile-error (condition)
                      (some (lambda (diagnostic) (eq (car diagnostic) :error))
                            (iree-compile-error-diagnostics condition)))))
                :regression-id compiler/compile-stablehlo/malformed-text-signals-compile-error
                :regression-file (regression-path "iree-compiler-malformed" :package "NABLA.IREE.TESTS")))
  (is (null nabla.iree::*compiler-poison-reason*)
      "ordinary (non-crashing) malformed compiles should never poison the compiler (issue #68): ~A"
      nabla.iree::*compiler-poison-reason*)
  ;; 具体例1: 未知の演算。
  (handler-case
      (progn
        (compile-stablehlo "func.func @main() {
  \"stablehlo.bogus\"() : () -> ()
  func.return
}")
        (fiveam:fail "unknown op ~S should have signalled iree-compile-error" "stablehlo.bogus"))
    (iree-compile-error (condition)
      (is (some (lambda (diagnostic) (search "unregistered operation 'stablehlo.bogus'" (cdr diagnostic)))
                (iree-compile-error-diagnostics condition)))))
  ;; 具体例2: 返り値の型不一致。
  (handler-case
      (progn
        (compile-stablehlo "func.func @main(%a: tensor<2x3xf32>) -> tensor<3x3xf32> {
  func.return %a : tensor<2x3xf32>
}")
        (fiveam:fail "return type mismatch should have signalled iree-compile-error"))
    (iree-compile-error (condition)
      (is (some (lambda (diagnostic) (search "doesn't match function result type" (cdr diagnostic)))
                (iree-compile-error-diagnostics condition))))))

(define-iree-test compiler/compile-stablehlo/opt-level-flag-changes-output
    "--iree-opt-level=O3 を渡すと、既定（フラグなし）や --iree-opt-level=O0 と
比べて、コンパイル結果の vmfb のバイト列が変わる。session ごとに
ireeCompilerSessionSetFlags で渡した --iree-opt-level が実際に効いている
ことの回帰テスト。

背景（PR #20 で見つかった回帰）: LLVM のシグナルハンドラ登録を
ireeCompilerSetupGlobalCL(installSignalHandlers=true) 経由で行っていた版は、
CompilerDriver.cpp の GlobalInit::usesCommandLine を真にする副作用により、
Session::Session がセッション作成時に一度だけ
OptionsBinder::global().applyOptimizationDefaults() を適用し、
Invocation::runPipeline は !usesCommandLine のときしか
session.binder.applyOptimizationDefaults() を呼ばなくなる。結果として
--iree-opt-level が黙って無視され、default / O3 / O0 が全部同じバイト列を
生成していた（scratchpad/adv/ev5f/ で計測: いずれも 9893 バイトで一致。
iree-compile CLI 直叩きの O3 は 9917 バイトで異なる）。
signals.lisp の %register-llvm-signal-handlers が
ireeCompilerSetupGlobalCL を呼ばなくなった今、このテストが再び通ることを
確かめる。"
  (skip-unless-iree :library :compiler)
  (let* ((text (stablehlo-fixture "matmul"))
         (default (compile-stablehlo text))
         (o3 (compile-stablehlo text
                                 :flags (append (compile-flags :local) (list "--iree-opt-level=O3"))))
         (o0 (compile-stablehlo text
                                 :flags (append (compile-flags :local) (list "--iree-opt-level=O0")))))
    (is (not (equalp default o3))
        "--iree-opt-level=O3 produced byte-identical output to the default: opt-level flags are being ignored")
    (is (not (equalp o3 o0))
        "--iree-opt-level=O3 and --iree-opt-level=O0 produced byte-identical output: opt-level flags are being ignored")))

(define-iree-test compiler/compile-stablehlo/repeated-compiles-are-stable
    "同じ StableHLO を繰り返しコンパイルしても、結果のバイト列は毎回同じで、
メモリ使用量（RSS）が際限なく増え続けない。"
  (skip-unless-iree :library :compiler)
  (let* ((text (stablehlo-fixture "matmul"))
         (first (compile-stablehlo text)))
    (dotimes (i 19)
      (is (equalp first (compile-stablehlo text))))
    (let ((rss-before (%vmrss-kb)))
      (dotimes (i 20)
        (compile-stablehlo text))
      (let ((rss-after (%vmrss-kb)))
        (when (and rss-before rss-after)
          (is (< (- rss-after rss-before) (* 64 1024))
              "RSS grew by ~A KB over 20 compiles (before=~A after=~A)"
              (- rss-after rss-before) rss-before rss-after))))))

(define-iree-test compiler/compile-stablehlo/organic-gc-pressure-does-not-crash
    "compile-stablehlo を何度も呼んで IREE の永続ワーカースレッドプールが
できたあと、確保のしきい値で自動的に走る通常の GC（明示的な SB-EXT:GC
呼び出しはしない）が起きても SBCL プロセスは落ちない。これは issue #5 の
フレーキーなクラッシュ（SB-EXT:GC を明示的に呼んだときに限って 'no SP
known for thread' で確実に落ちる、compiler.lisp 冒頭のコメント参照）の
再発を、nabla の実際の使い方に近い形で検知する回帰テスト。有効な入力と
（安全な方言の）壊れた入力を混ぜ、コンパイルの合間に大量に consing して
自動 GC を何度も誘発する。

issue #68 の回帰: 150回のうちどれも普通の（プロセスを壊さない）成功・失敗
なので、コンパイラが poisoned にならないことも確かめる。"
  (skip-unless-iree :library :compiler)
  (let ((valid (stablehlo-fixture "matmul"))
        (malformed (%unknown-op-in-known-dialect "not_a_real_op")))
    (dotimes (i 150)
      ;; 自動 GC を誘発するための、意味のない確保。
      (dotimes (k 500) (make-array 1000))
      (if (evenp i)
          (is (plusp (length (compile-stablehlo valid))))
          (handler-case
              (progn (compile-stablehlo malformed)
                     (fiveam:fail "malformed input unexpectedly compiled"))
            (iree-compile-error () nil)))))
  (is (null nabla.iree::*compiler-poison-reason*)
      "150 rounds of ordinary compiles/failures should never poison the compiler (issue #68): ~A"
      nabla.iree::*compiler-poison-reason*))

(define-iree-test compiler/compile-stablehlo/poisoned-state-signals-clear-error
    "issue #68 の回帰テスト（compiler.lisp の *compiler-poison-reason* 冒頭の
コメント参照）。NABLA.IREE::*COMPILER-POISON-REASON*（defvar、動的スコープの
特殊変数）を LET で汚染状態に束縛すると、compile-stablehlo は FFI に
一切触れずに IREE-COMPILE-ERROR（:phase :poisoned、MESSAGE に理由を含む）を
signal する。LET の動的エクステントを抜ければ元の（NIL の）値に自動的に
戻るので、実際に #DE を起こす経路を通さずに、この契約だけをこのプロセス
自身の中で安全に・決定的に検査できる（他のテストを汚染しない）。エクステント
を抜けたあとは普通のコンパイルが成功することも確かめる。"
  (skip-unless-iree :library :compiler)
  (let ((nabla.iree::*compiler-poison-reason* "synthetic poison for compiler/compile-stablehlo/poisoned-state-signals-clear-error"))
    (handler-case
        (progn
          (compile-stablehlo (stablehlo-fixture "matmul"))
          (fiveam:fail "compile-stablehlo succeeded although *compiler-poison-reason* was bound"))
      (iree-compile-error (c)
        (is (eq :poisoned (iree-compile-error-phase c)))
        (is (search "synthetic poison for compiler/compile-stablehlo/poisoned-state-signals-clear-error"
                    (iree-compile-error-message c))
            "poisoned の IREE-COMPILE-ERROR の MESSAGE に理由が含まれていない: ~A"
            (iree-compile-error-message c)))))
  ;; LET を抜けたので *compiler-poison-reason* は NIL に戻っており、普通の
  ;; コンパイルが成功する。
  (is (plusp (length (compile-stablehlo (stablehlo-fixture "matmul"))))))

(define-iree-test compiler/compile-stablehlo/keeps-sbcl-signal-handlers
    "compile-stablehlo（と ensure-compiler-loaded）を呼んでも、SBCL が GC の
stop-the-world に使う SIGUSR2 と、エラートラップに使う SIGILL / SIGTRAP /
SIGSEGV のハンドラは変わらない。LLVM は初回の Pipeline で自前のハンドラを
sigaction で登録するが、nabla.iree はそれを元に戻す（signals.lisp 冒頭の
コメント参照）。ハンドラのアドレスを比べる直接の回帰テスト。"
  (skip-unless-iree :library :compiler)
  (let ((signals '((12 . "SIGUSR2") (4 . "SIGILL") (5 . "SIGTRAP") (11 . "SIGSEGV")))
        (before nil))
    (dolist (s signals) (push (cons (car s) (nabla.iree::%signal-handler-address (car s))) before))
    (compile-stablehlo (stablehlo-fixture "matmul"))
    (handler-case (compile-stablehlo (%unknown-op-in-known-dialect "not_a_real_op"))
      (iree-compile-error () nil))
    (dolist (s signals)
      (is (= (cdr (assoc (car s) before)) (nabla.iree::%signal-handler-address (car s)))
          "~A のハンドラが compile-stablehlo で書き換えられた" (cdr s)))))

(define-iree-test signals/ensure-compiler-loaded/registers-llvm-signal-handlers
    "ensure-compiler-loaded を呼んだあとは
nabla.iree::*llvm-signal-handlers-registered-p* が真になっている
（%register-llvm-signal-handlers が %call-with-world-stopped の窓の中で
SIGUSR2 の処分の変化を実際に観測した印。library.lisp / signals.lisp
参照）。third_party/iree.lock で固定した IREE 3.11.0 では
ireeCompilerOutputOpenMembuffer を呼ぶだけで実際に登録するので、warm-up
コンパイルへのフォールバック（%warm-up-compiler）を経由せずにここが真に
なる。"
  (skip-unless-iree :library :compiler)
  (nabla.iree::ensure-compiler-loaded)
  (is-true nabla.iree::*llvm-signal-handlers-registered-p*))

(defun %run-signal-registration-check-child ()
  "真っさらな子 SBCL プロセスで、ensure-compiler-loaded の呼び出しの前後で
SIGUSR2 のハンドラのアドレスを比べ、
\"BEFORE=<addr> AFTER=<addr> REGISTERED=<T/NIL>\" の1行を標準出力に印字して
終了する。REGISTERED は *llvm-signal-handlers-registered-p*
（%register-llvm-signal-handlers が窓の中で検出した、実際に登録が起きた
という印）。BEFORE と AFTER が一致することは、LLVM に一時的に奪われた
SIGUSR2 のハンドラが ensure-compiler-loaded から戻る頃には SBCL のものに
戻っていることを示す。他のテストが既に ensure-compiler-loaded 済みの
このプロセス自身では BEFORE が pristine な SBCL のハンドラだと確認できない
ため、%run-with-missing-iree-home と同じ理由で別プロセスを使う。"
  (let* ((forms
           (list "(require :asdf)"
                 "(asdf:load-system \"nabla/iree\")"
                 "(in-package :nabla.iree)"
                 "(let ((before (%signal-handler-address sb-unix:sigusr2)))
                    (ensure-compiler-loaded)
                    (format t \"BEFORE=~A AFTER=~A REGISTERED=~A~%\"
                            before (%signal-handler-address sb-unix:sigusr2)
                            *llvm-signal-handlers-registered-p*)
                    (sb-ext:exit :code 0))"))
         (args (list* "--non-interactive" "--disable-debugger"
                      (loop for form in forms append (list "--eval" form))))
         (env (append (%forward-env-vars (list* "NABLA_IREE_HOME" *child-sbcl-forwarded-env-vars*))
                      (list (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))
         (output (make-string-output-stream))
         (process (sb-ext:run-program "sbcl" args
                                       :search t :environment env
                                       :output output :error output)))
    (values (sb-ext:process-exit-code process) (get-output-stream-string output))))

(define-iree-test signals/ensure-compiler-loaded/registration-detected-and-reverted-in-fresh-process
    "真っさらな子プロセスで ensure-compiler-loaded を呼ぶと、
(1) *llvm-signal-handlers-registered-p* が真になり（%register-llvm-signal-handlers
が窓の中で SIGUSR2 の処分の変化を実際に観測した）、かつ
(2) ensure-compiler-loaded の前後で外から見える SIGUSR2 のハンドラの
アドレスが変わらない（LLVM に奪われたハンドラが SBCL のものへ戻っている）
ことを確かめる。IREE-SetupGlobalCL 経由だった旧版でも通っていた性質だが、
新しい ireeCompilerOutputOpenMembuffer 経由の登録がこれを壊していないことの
回帰テスト。"
  (skip-unless-iree :library :compiler)
  (multiple-value-bind (exit-code output) (%run-signal-registration-check-child)
    (is (= 0 exit-code) "child process exited ~D, output:~%~A" exit-code output)
    (is (search "REGISTERED=T" output)
        "child did not report REGISTERED=T, output:~%~A" output)
    (let ((before (parse-integer output :start (+ 7 (search "BEFORE=" output)) :junk-allowed t))
          (after (parse-integer output :start (+ 6 (search "AFTER=" output)) :junk-allowed t)))
      (is (integerp before) "could not parse BEFORE from output:~%~A" output)
      (is (integerp after) "could not parse AFTER from output:~%~A" output)
      (is (= before after)
          "SIGUSR2 handler address before (~A) and after (~A) ensure-compiler-loaded ~
differ: not restored to SBCL's own handler. output:~%~A"
          before after output))))

(fiveam:test (signals/call-with-world-stopped/returns-thunk-value-and-rejects-nesting :suite :nabla.medium)
  "%call-with-world-stopped は THUNK の戻り値をそのまま返し、既に
sb-sys:without-gcing の中（*gc-inhibit* が真）から呼ぶとエラーになる
（signals.lisp の %call-with-world-stopped docstring 参照。二重にロックを
取り合うと進めなくなるための安全策）。IREE の共有ライブラリは要らない。"
  (is (eql :ok (nabla.iree::%call-with-world-stopped (lambda () :ok))))
  (signals error
    (sb-sys:without-gcing
      (nabla.iree::%call-with-world-stopped (lambda () :unreachable)))))

(define-iree-test compiler/compile-stablehlo/explicit-full-gc-does-not-crash
    "compile-stablehlo のあとに SB-EXT:GC :FULL T を明示的に呼んでも SBCL
プロセスは落ちず、trivial-garbage の finalizer も走る。issue #5 の
'garbage_collect: no SP known for thread' クラッシュ（LLVM が SIGUSR2 の
ハンドラを乗っ取ることが原因。signals.lisp 参照）の回帰テスト。修正前は
最初の明示的 GC でほぼ確実に落ちた。finalizer は finalizer スレッドで
非同期に走るので、短い間だけ待つ。"
  (skip-unless-iree :library :compiler)
  (let ((valid (stablehlo-fixture "matmul"))
        (malformed (%unknown-op-in-known-dialect "not_a_real_op"))
        (finalized nil))
    (dotimes (i 6)
      (if (evenp i)
          (is (plusp (length (compile-stablehlo valid))))
          (handler-case (compile-stablehlo malformed)
            (iree-compile-error () nil)))
      (trivial-garbage:finalize (make-array 16) (lambda () (setf finalized t)))
      (sb-ext:gc :full t)
      (sb-ext:gc))
    (loop repeat 100 until finalized do (sleep 0.01))
    (is-true finalized "trivial-garbage の finalizer が明示的な full GC のあとに走らなかった")))

(fiveam:test (compiler/compile-flags/targets-are-deterministic :suite :nabla.small)
  "compile-flags は同じ TARGET に対して毎回同じフラグのリストを返し、
:local の先頭は StableHLO 入力を指定するフラグで、未知の TARGET はエラーになる。"
  (is (equal (compile-flags :local) (compile-flags :local)))
  (is (string= "--iree-input-type=stablehlo" (first (compile-flags :local))))
  (is (equal (compile-flags :cuda :cuda-arch "sm_80") (compile-flags :cuda :cuda-arch "sm_80")))
  (is (member "--iree-cuda-target=sm_80" (compile-flags :cuda :cuda-arch "sm_80") :test #'string=))
  (signals error (compile-flags :nope)))

(define-iree-test compiler/compiler-revision/mentions-locked-commit
    "compiler-revision の文字列は third_party/iree.lock で固定したコミット
ハッシュを含む。"
  (skip-unless-iree :library :compiler)
  (is (search (%locked-iree-commit) (compiler-revision))))

;; %child-source-registry は tests/iree/support.lisp（このファイルより先に
;; ロードされる）で共有定義している。

(defun %run-with-missing-iree-home (missing-home)
  "MISSING-HOME を NABLA_IREE_HOME として渡した、真っさらな子 SBCL
プロセスで ensure-compiler-loaded を呼び、その終了コードを返す
（0 = IREE-LIBRARY-NOT-FOUND が signal された、それ以外 = 想定外）。
グローバル初期化はプロセスにつき1回だけという仕様そのものにより、この
プロセス自身では（他のテストが一度でもロードに成功していると）
ensure-compiler-loaded の実際のロードは検証できないため、別プロセスで
確かめる。"
  (let* (;; sbcl の --eval は1つの文字列に複数フォームを詰めても2つめ以降を
         ;; 評価してくれない（require の効果が次の read に間に合わない）ので、
         ;; run-tests.sh と同じように --eval を1フォームずつ分ける。
         (forms (list "(require :asdf)"
                      "(asdf:load-system \"nabla/iree\")"
                      "(handler-case (progn (nabla.iree::ensure-compiler-loaded) (sb-ext:exit :code 2)) (nabla.iree:iree-library-not-found () (sb-ext:exit :code 0)) (error (c) (format *error-output* \"unexpected error: ~A~%\" c) (sb-ext:exit :code 3)))"))
         (args (list* "--non-interactive"
                      (loop for form in forms append (list "--eval" form))))
         ;; sb-ext:run-program に :environment を渡さなければ現在のプロセスの
         ;; 環境をそのまま複製してくれる（マニュアル参照）が、SBCL には環境
         ;; 全体を読み出す標準の手段が無いので、代わりに sbcl・ASDF・CFFI の
         ;; 動作に関わりうる変数を明示的に転送し、NABLA_IREE_HOME と
         ;; CL_SOURCE_REGISTRY だけ上書きする。
         (env (append (%forward-env-vars *child-sbcl-forwarded-env-vars*)
                      (list (format nil "NABLA_IREE_HOME=~A" (namestring missing-home))
                            (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))
         (process (sb-ext:run-program "sbcl" args
                                       :search t :environment env
                                       :output *standard-output* :error *standard-output*)))
    (sb-ext:process-exit-code process)))

(fiveam:test (library/ensure-compiler-loaded/missing-home-signals-library-not-found :suite :nabla.medium)
  "存在しないディレクトリを NABLA_IREE_HOME として渡すと、ensure-compiler-loaded
が実際に IREE-LIBRARY-NOT-FOUND を signal する。真っさらな子 SBCL
プロセスで確かめる（このプロセス自身では、他のテストが一度でもロードに
成功していると検証できないため）。"
  (let ((missing-home (merge-pathnames "nabla-iree-definitely-missing-home/" (nabla.iree::iree-home))))
    (is (not (probe-file missing-home)))
    (is (= 0 (%run-with-missing-iree-home missing-home))
        "child process should have signalled IREE-LIBRARY-NOT-FOUND (exit 0)")))

(defun %run-cold-start-race-child ()
  "真っさらな子 SBCL プロセスで、issue #5 のクラッシュを最も踏みやすい
シナリオ（初回の compile-stablehlo ―― ロード・グローバル初期化・LLVM の
シグナルハンドラ登録がすべてまだの状態 ―― と同時に、別スレッドが
明示的な SB-EXT:GC を回し続ける）を再現し、子プロセスの終了コードを返す。

修正前（%register-llvm-signal-handlers が無く、%warm-up-compiler の
保護付きコンパイルだけに頼っていた版）はこのシナリオでほぼ確実に落ちるか
ハングした（advisor の計測: 30 トライアル中 crash 25 / hang 5。
scratchpad/adv/ 参照）。修正後は、LLVM の登録がコンパイルを1つも走らせずに
世界を止めた状態で先に済むので、この競合の隙間が無くなる。

子プロセスは `timeout` でくるみ、修正が壊れて再びハングするようになっても
このテスト自身が止まらないようにする（-k 5 180: 180秒で TERM、応じなければ
5秒後に KILL）。真っさらなプロセスが要る理由は %run-with-missing-iree-home
と同じ（このプロセス自身では、他のテストが既に ensure-compiler-loaded を
済ませていると検証にならない）。"
  (let* ((forms
           (list "(require :asdf)"
                 "(asdf:load-system \"nabla/iree\")"
                 "(in-package :nabla.iree)"
                 "(defvar *stop* nil)"
                 "(defvar *sink* nil)"
                 ;; advisor の cold-race-child.lisp と同じ形: 20分の1の確率で
                 ;; 明示的 GC を挟みながら、止まるまで確保し続ける。
                 "(defvar *conser* (sb-thread:make-thread (lambda () (loop until *stop* do (setf *sink* (make-array 20000)) (when (zerop (random 20)) (sb-ext:gc)))) :name \"cold-start-race-conser\"))"
                 "(sleep 0.2)"
                 "(compile-stablehlo \"func.func @main() { return }\")"
                 "(dotimes (i 5) (compile-stablehlo \"func.func @main() { return }\"))"
                 "(setf *stop* t)"
                 "(sb-thread:join-thread *conser*)"
                 "(sb-ext:exit :code 0)"))
         (sbcl-args (list* "--non-interactive" "--disable-debugger"
                            (loop for form in forms append (list "--eval" form))))
         (args (list* "-k" "5" "180" "sbcl" sbcl-args))
         ;; sb-ext:run-program に :environment を渡さなければ現在のプロセスの
         ;; 環境をそのまま複製してくれるが、SBCL には環境全体を読み出す標準の
         ;; 手段が無いので、%run-with-missing-iree-home と同じ組み立て方で
         ;; 明示的に転送する（今回は NABLA_IREE_HOME も現在の値のまま渡す）。
         (env (append (%forward-env-vars (list* "NABLA_IREE_HOME" *child-sbcl-forwarded-env-vars*))
                      (list (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))
         (process (sb-ext:run-program "timeout" args
                                       :search t :environment env
                                       :output *standard-output* :error *standard-output*)))
    (sb-ext:process-exit-code process)))

(fiveam:test (signals/cold-start/concurrent-gc-during-first-compile-does-not-crash :suite :nabla.large)
  "issue #5 の回帰テスト（large。子 SBCL プロセスを3回起動するので数十秒
かかる）。初回の compile-stablehlo と同時に別スレッドが明示的な GC を
回し続けても、子プロセスが落ちず・ハングもせず終了コード0で終わることを、
複数トライアル確かめる。修正前はこのシナリオでほぼ確実に再現した
（%run-cold-start-race-child のコメントと scratchpad/adv/ 参照）。"
  (block iree-test
    (skip-unless-iree :library :compiler)
    (dotimes (trial 3)
      (is (eql 0 (%run-cold-start-race-child))
          "trial ~D/3: 子 SBCL が、別スレッドが GC を回している間の初回 ~
compile-stablehlo で落ちたかハングした（issue #5 の回帰。signals.lisp / ~
library.lisp 参照）"
          (1+ trial)))))
