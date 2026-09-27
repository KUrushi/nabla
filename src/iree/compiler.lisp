;;;; StableHLO のテキストを IREE でコンパイルして vmfb のバイト列を得る、
;;;; nabla.iree の公開 API。
;;;;
;;;; アルゴリズム（すべて $NABLA_IREE_HOME/include/iree/compiler/embedding_api.h
;;;; の埋め込み C API どおり）:
;;;;
;;;;   ensure-compiler-loaded
;;;;   -> SessionCreate -> SessionSetFlags（失敗なら phase :flags）
;;;;   -> InvocationCreate -> EnableCallbackDiagnostics（診断を集める）
;;;;   -> SourceWrapBuffer -> ParseSource（失敗なら phase :parse）
;;;;   -> Pipeline（失敗なら phase :compile）
;;;;   -> OutputOpenMembuffer -> OutputVMBytecode（失敗なら phase :output）
;;;;   -> OutputMapMemory -> バイト列にコピー
;;;;   -> unwind-protect で OutputDestroy, InvocationDestroy,
;;;;      SourceDestroy, SessionDestroy の順に後始末
;;;;
;;;; セッションと invocation は呼び出しごとに新しく作る（セッション自体は
;;;; スレッドセーフではないため、これが compile-stablehlo をスレッドセーフに
;;;; している）。
;;;;
;;;; 注意（issue #5 で踏んだクラッシュの根本原因）: llvm-cpu ターゲットが
;;;; 実行ファイルを直列化する際、ireeCompilerInvocationPipeline の初回実行
;;;; 中に LLVM が自前のシグナルハンドラをプロセスに（1回だけ）登録し、SBCL
;;;; が GC の stop-the-world に使う SIGUSR2 のハンドラを上書きしてしまう
;;;; （sigaction を使った LD_PRELOAD トレースで確認済み。詳しい仕組みは
;;;; signals.lisp 冒頭のコメント参照）。放っておくと、SBCL には常にいる
;;;; finalizer スレッドなどを別スレッドが GC で止めようとした瞬間に
;;;; "no SP known for thread" で確実に落ちる。そのため IREE を呼ぶ公開関数は
;;;; すべて with-lisp-signal-handlers-preserved（signals.lisp）で包む。
;;;;
;;;; LLVM の登録そのものは、ensure-compiler-loaded がロードロックを持った
;;;; 制御された1点で、他の全 Lisp スレッドを止めた状態で済ませる
;;;; （signals.lisp の %register-llvm-signal-handlers と
;;;; %call-with-world-stopped、library.lisp の ensure-compiler-loaded から
;;;; 呼ぶ）。コンパイルもセッションも1つも作らず
;;;; ireeCompilerOutputOpenMembuffer/Destroy を呼ぶだけなので、初回コンパイル
;;;; を待たずに済み、コンパイル中に別スレッドが GC を始める競合の隙間が
;;;; 無くなる（ireeCompilerSetupGlobalCL は使わない。usesCommandLine を真に
;;;; してしまい、セッションごとの --iree-opt-level を無視させる回帰を生む
;;;; ため。詳しくは signals.lisp 冒頭）。
;;;; この版の IREE で ireeCompilerOutputOpenMembuffer だけでは登録しなかった
;;;; 場合だけ、最後の手段として最小のモジュールを1つコンパイルする旧方式
;;;; （%warm-up-compiler）に落ちる（この経路には、その最初のコンパイル中に
;;;; 他スレッドが GC を始める競合の隙間がまだ残るので、
;;;; ensure-compiler-loaded が警告を出す）。
;;;; 将来 LLVM を呼びうる FFI エントリポイント（PJRT など）も
;;;; with-lisp-signal-handlers-preserved で包むこと。

(in-package #:nabla.iree)

(defun compiler-api-version ()
  "ireeCompilerGetAPIVersion の結果を (values major minor) にして返す。
上位16ビットがメジャー、下位16ビットがマイナー。"
  (ensure-compiler-loaded)
  (let ((raw (%compiler-get-api-version)))
    (values (ash raw -16) (logand raw #xFFFF))))

(defun compiler-revision ()
  "IREE コンパイラのビルドリビジョン文字列を返す（ireeCompilerGetRevision）。"
  (ensure-compiler-loaded)
  (or (%compiler-get-revision) ""))

(defvar *warned-missing-embedded-linker-p* nil
  "%embedded-linker-flags が iree-lld 不在の警告を出したかどうか。プロセスに
つき1回だけ警告する。")

(defun %embedded-linker-path ()
  "NABLA_IREE_HOME/bin/iree-lld があればその pathname を、無ければ NIL を
返す。llvm-cpu ターゲットは実行可能な小さな ELF を作るのに毎回リンカを
呼ぶ（IREE の embedding API に、これを避けてプロセス内でリンクする手段は
無い。§compile-flags のコメント参照）。scripts/build-iree.sh が PyPI
ホイール（third_party/iree.lock）に同梱の iree-lld をここにインストール
していれば、その固定コミット版を使う。"
  (let ((path (merge-pathnames "bin/iree-lld" (iree-home))))
    (and (probe-file path) path)))

(defun %embedded-linker-flags ()
  "llvm-cpu 向けの --iree-llvmcpu-embedded-linker-path フラグを、
iree-lld があるときだけ1要素のリストにして返す。無いときは空リストを返し、
プロセスにつき1回だけ、IREE 自身のリンカ探索に任せる旨を警告する
（iree-compile 自身が execve でそれを起動するので、nabla はここでは
サブプロセスを起動しない。CLAUDE.md がサブプロセスでの起動を避けたいのは
nabla 自身が明示的にそれをしないことで、iree-compile の内部実装までは制御
できない）。IREE は --iree-llvmcpu-embedded-linker-path が無いとき、
libIREECompiler.so と同じディレクトリの iree-lld → 実行ファイルと同じ
ディレクトリの iree-lld → PATH 上の iree-lld / lld / ld.lld の順に探す
（IREE の EmbeddedLinkerTool、findTool 相当）。ここで探すのは
NABLA_IREE_HOME/bin/iree-lld だけなので、それが無いとき実際にどれが選ばれる
かは環境依存（システムに何も無ければコンパイルは失敗する）。"
  (let ((path (%embedded-linker-path)))
    (cond
      (path (list (format nil "--iree-llvmcpu-embedded-linker-path=~A" (namestring path))))
      (t (unless *warned-missing-embedded-linker-p*
           (warn "NABLA_IREE_HOME/bin/iree-lld が見つからないので、リンカの選択を ~
IREE 自身の探索（libIREECompiler.so の隣 → 実行ファイルの隣 → PATH 上の ~
iree-lld / lld / ld.lld の順）に任せる。scripts/build-iree.sh が ~
third_party/iree.lock のホイールに同梱の iree-lld を NABLA_IREE_HOME/bin/ に ~
インストールするようになれば、この警告は出なくなる。")
           (setf *warned-missing-embedded-linker-p* t))
         nil))))

(defun compile-flags (target &key cuda-arch)
  "TARGET（:local または :cuda）向けの iree-compile 相当のフラグをリストで返す。
:local は CLAUDE.md / verify-iree.sh と同じ CPU 向けのレシピ
（llvm-cpu、target-cpu=host。生成される vmfb はこのため実行するマシンに
依存する）。NABLA_IREE_HOME/bin/iree-lld があれば
--iree-llvmcpu-embedded-linker-path でそれを明示し、無ければ何も指定せず
IREE 自身のリンカ探索に任せる（%embedded-linker-flags 参照。llvm-cpu の
実行ファイル直列化には常に何らかのリンカが要り、embedding API にはこれを
プロセス内で行う手段が無い）。:cuda は CUDA-ARCH（例: \"sm_80\"）を渡すと
--iree-cuda-target=CUDA-ARCH を追加する。TARGET がこれ以外なら型エラーを
signal する。呼び出しごとに新しいリストを作るが、内容は決定的。

注意（将来 jit キャッシュを作るとき向け）: :local のフラグは
NABLA_IREE_HOME/bin/iree-lld の有無というファイルシステムの状態に依存する。
CLAUDE.md の jit キャッシュキー（関数の同一性 + aval + 静的引数 +
コンパイルターゲット）はこれを含まないので、そのままだとキャッシュキーに
現れない「隠れた入力」になる。iree-lld を後から追加・削除する運用がある間は、
jit の実装側でこれをキーに含めるか、プロセス起動時に固定するかを決めること。"
  (ecase target
    (:local (append (list "--iree-input-type=stablehlo"
                           "--iree-hal-target-device=local"
                           "--iree-hal-local-target-device-backends=llvm-cpu"
                           "--iree-llvmcpu-target-cpu=host")
                     (%embedded-linker-flags)))
    (:cuda (append (list "--iree-input-type=stablehlo"
                          "--iree-hal-target-device=cuda")
                    (when cuda-arch
                      (list (format nil "--iree-cuda-target=~A" cuda-arch)))))))

;; ParseSource / Pipeline の失敗中に集める診断。callback は「invocation の
;; 破棄までどのスレッドからでも」呼ばれうる (embedding_api.h) ので、実行中の
;; compile-stablehlo 呼び出しを、動的束縛ではなく整数のクッキーで識別し、
;; ミューテックスで守ったハッシュ表に集める。

(defvar *diagnostics-lock* (sb-thread:make-mutex :name "nabla-iree-diagnostics"))
(defvar *diagnostics-table* (make-hash-table)
  "クッキー（整数）-> 集めている診断のリスト（逆順）。")
(defvar *next-diagnostics-cookie* 0)

(defun %diagnostics-begin ()
  (sb-thread:with-mutex (*diagnostics-lock*)
    (let ((cookie (incf *next-diagnostics-cookie*)))
      (setf (gethash cookie *diagnostics-table*) nil)
      cookie)))

(defun %diagnostics-push (cookie severity text)
  (sb-thread:with-mutex (*diagnostics-lock*)
    (push (cons severity text) (gethash cookie *diagnostics-table*))))

(defun %diagnostics-end (cookie)
  (sb-thread:with-mutex (*diagnostics-lock*)
    (prog1 (nreverse (gethash cookie *diagnostics-table*))
      (remhash cookie *diagnostics-table*))))

(cffi:defcallback %diagnostic-callback :void
    ((severity :int) (message :pointer) (message-size :size) (user-data :pointer))
  (let ((cookie (cffi:pointer-address user-data))
        (text (cffi:foreign-string-to-lisp message :count message-size :encoding :utf-8)))
    (%diagnostics-push cookie (%diagnostic-severity-keyword severity) text)))

(defun %copy-membuffer-to-octets (contents size)
  "CONTENTS（foreign :pointer）から SIZE バイトを読み、新しい
(simple-array (unsigned-byte 8) (*)) にコピーして返す。vmfb は数 MB になり
うるので、1バイトずつ mem-aref する代わりに memcpy を1回呼ぶ。"
  (let ((bytes (make-array size :element-type '(unsigned-byte 8))))
    (sb-sys:with-pinned-objects (bytes)
      (cffi:foreign-funcall "memcpy"
                             :pointer (sb-sys:vector-sap bytes)
                             :pointer contents
                             :size size
                             :pointer))
    bytes))

(defun %session-set-flags (session flags)
  (let ((argc (length flags)))
    (cffi:with-foreign-object (argv :pointer argc)
      (let ((c-strings nil))
        (unwind-protect
             (progn
               (loop for flag in flags
                     for i from 0
                     do (let ((c-string (cffi:foreign-string-alloc flag)))
                          (push c-string c-strings)
                          (setf (cffi:mem-aref argv :pointer i) c-string)))
               (let ((error (%compiler-session-set-flags session argc argv)))
                 (unless (cffi:null-pointer-p error)
                   (let ((message (%compiler-error-get-message error)))
                     (%compiler-error-destroy error)
                     (error 'iree-compile-error :phase :flags :message message)))))
          (dolist (c-string c-strings)
            (cffi:foreign-string-free c-string)))))))

(defun compile-stablehlo (text &key (flags (compile-flags :local)) (source-name "nabla.mlir"))
  "StableHLO の TEXT を IREE でコンパイルし、vmfb のバイト列を
(simple-array (unsigned-byte 8) (*)) として返す。FLAGS は iree-compile 相当の
コマンドライン引数のリスト（既定は (compile-flags :local)）。

コンパイルが失敗すると IREE-COMPILE-ERROR を signal する。フラグの設定・
構文解析・パイプラインの実行・出力のどの段階で失敗したかが PHASE に、
MLIR の診断（あれば）が DIAGNOSTICS に入る。共有ライブラリが見つからない
ときは IREE-LIBRARY-NOT-FOUND を signal する。

呼び出しごとに新しいセッションと invocation を作るので、複数スレッドから
並行に呼んでよい。

本体は WITH-ALL-FLOAT-TRAPS-MASKED（float-traps.lisp）で包む（issue #53）。
LLVM のコード生成は、Pipeline を呼んだこの Lisp スレッドの上で直接走ることが
あり、かつ初回コンパイル時に LLVM が内部で作るスレッドプールもこの時点の
呼び出し元スレッドの MXCSR を引き継ぐ（ガイダンス(3)）。

ただし、ゼロサイズの contracting 次元を持つ dot_general が起こす
DIVISION-BY-ZERO はこのマスクでは防げない（float-traps.lisp 冒頭の
コメント参照。x86 の整数除算命令による #DE で、マスクビットが無いため）。
その対策は %compile-stablehlo の Pipeline 呼び出しのすぐ側にある
ARITHMETIC-ERROR のハンドリング（IREE-COMPILE-ERROR への変換）。"
  (ensure-compiler-loaded)
  (with-lisp-signal-handlers-preserved
    (with-all-float-traps-masked
      (%compile-stablehlo text flags source-name))))

(defparameter *warm-up-module* "func.func @main() { return }"
  "ensure-compiler-loaded が %register-llvm-signal-handlers のフォールバック
としてコンパイルする最小のモジュール。中身は空でも Pipeline は llvm-cpu の
直列化パスまで走り、LLVM のシグナルハンドラ登録（signals.lisp 冒頭の
コメント参照）がここで起きる。")

(defun %warm-up-compiler ()
  "LLVM の「プロセスにつき1回」のシグナルハンドラ登録を済ませる、
最後の手段のフォールバック。ensure-compiler-loaded は通常
%register-llvm-signal-handlers（signals.lisp。世界を止めてコンパイル無しで
登録する）でこれを済ませ、その版の IREE で ireeCompilerOutputOpenMembuffer
だけでは登録しなかったときだけこちらを呼ぶ。ここは実際に最小モジュールを
コンパイルするので、
その間に別の Lisp スレッドが GC を始めると登録前の LLVM のハンドラと
競合する隙間が残る（呼び出し元の ensure-compiler-loaded が警告する）。
ensure-compiler-loaded から（ロードロックを持ったまま）呼ぶので、
ensure-compiler-loaded を再度呼び出す compile-stablehlo は経由しない。

本体は WITH-ALL-FLOAT-TRAPS-MASKED で包む（compile-stablehlo と同じ理由。
issue #53）。"
  (with-lisp-signal-handlers-preserved
    (with-all-float-traps-masked
      (%compile-stablehlo *warm-up-module* (compile-flags :local) "nabla-warm-up.mlir"))))

(defun %compile-stablehlo (text flags source-name)
  "compile-stablehlo の本体。ensure-compiler-loaded 済みで、かつ
with-lisp-signal-handlers-preserved の中から呼ぶこと（LLVM がここで
SIGUSR2 などのハンドラを上書きしうる。compiler.lisp 冒頭のコメント参照）。"
  (let (session invocation source output)
    (unwind-protect
         (progn
           (setf session (%compiler-session-create))
           (%session-set-flags session flags)
           (setf invocation (%compiler-invocation-create session))
           (let ((cookie (%diagnostics-begin)))
             (unwind-protect
                  ;; ireeCompilerSourceWrapBuffer は TEXT のバイト列をコピーせず、
                  ;; 渡したバッファをそのまま「包む」だけ（ヘッダの言う「ソースの
                  ;; 処理が終わるまでソースを生かしておく」の裏側）。そのため
                  ;; ParseSource（と、安全のためそれ以降の処理すべて）を
                  ;; with-foreign-string の動的エクステントの外に出してはいけない
                  ;; ——外に出すと、解放済みのメモリを読むことになり、無関係な
                  ;; 文字化けとして構文エラーが出る（実際に踏んだバグ）。
                  (cffi:with-foreign-string ((buffer length) text :encoding :utf-8)
                    (%compiler-invocation-enable-callback-diagnostics
                     invocation 0 (cffi:callback %diagnostic-callback) (cffi:make-pointer cookie))
                    (cffi:with-foreign-object (out-source :pointer)
                      (let ((error (%compiler-source-wrap-buffer
                                    session source-name buffer length t out-source)))
                        (unless (cffi:null-pointer-p error)
                          (let ((message (%compiler-error-get-message error)))
                            (%compiler-error-destroy error)
                            (error 'iree-compile-error :phase :parse :message message
                                                        :diagnostics (%diagnostics-end cookie)))))
                      (setf source (cffi:mem-ref out-source :pointer)))
                    (unless (%compiler-invocation-parse-source invocation source)
                      (error 'iree-compile-error :phase :parse
                                                  :diagnostics (%diagnostics-end cookie)))
                    ;; issue #53: ゼロサイズの contracting 次元を持つ dot_general
                    ;; （例 tensor<2x0xf32> x tensor<0x3xf32>）は、この固定コミットの
                    ;; IREE では ireeCompilerInvocationPipeline の内部（MLIR の
                    ;; タイリング関連パスと見られる）で本物の整数 0 除算
                    ;; （x86 の idiv 系命令。実験で確認: WITH-ALL-FLOAT-TRAPS-MASKED
                    ;; でも glibc の fedisableexcept でも防げない——MXCSR は
                    ;; 浮動小数点例外だけを制御し、整数の 0 除算 (#DE) には
                    ;; マスクビットが存在しないため）を起こし、SBCL の
                    ;; SIGFPE ハンドラ経由で DIVISION-BY-ZERO の Lisp
                    ;; コンディションとして飛んでくる。プロセスは落ちない
                    ;; （SIGFPE は正しく Lisp コンディションに変換され、
                    ;; 通常の非局所脱出で戻ってこられる）ので、ここで
                    ;; ARITHMETIC-ERROR（DIVISION-BY-ZERO や
                    ;; FLOATING-POINT-OVERFLOW 等のスーパークラス）を捕まえ、
                    ;; 他のフェーズと同じ IREE-COMPILE-ERROR（:phase :compile）に
                    ;; 変換する。これにより BACKEND-COMPILE の呼び出し側は、
                    ;; どんな失敗でも常に IREE-COMPILE-ERROR という1種類の
                    ;; コンディションだけを見ればよくなる（生の
                    ;; DIVISION-BY-ZERO が漏れ出ない）。実際に正しい vmfb を
                    ;; 得られるようにする根本修正は IREE 側のバグなので、
                    ;; nabla 側では対応できない（follow-up 課題）。
                    (handler-case
                        (unless (%compiler-invocation-pipeline invocation +compiler-pipeline-std+)
                          (error 'iree-compile-error :phase :compile
                                                      :diagnostics (%diagnostics-end cookie)))
                      (arithmetic-error (condition)
                        (error 'iree-compile-error :phase :compile
                               :message (format nil "Pipeline 実行中に Lisp の算術エラーが発生した ~
（IREE/LLVM 側の内部エラーの可能性が高い。詳細: ~A）" condition)
                               :diagnostics (%diagnostics-end cookie))))
                    (cffi:with-foreign-object (out-output :pointer)
                      (let ((error (%compiler-output-open-membuffer out-output)))
                        (unless (cffi:null-pointer-p error)
                          (let ((message (%compiler-error-get-message error)))
                            (%compiler-error-destroy error)
                            (error 'iree-compile-error :phase :output :message message
                                                        :diagnostics (%diagnostics-end cookie)))))
                      (setf output (cffi:mem-ref out-output :pointer)))
                    (let ((error (%compiler-invocation-output-vm-bytecode invocation output)))
                      (unless (cffi:null-pointer-p error)
                        (let ((message (%compiler-error-get-message error)))
                          (%compiler-error-destroy error)
                          (error 'iree-compile-error :phase :output :message message
                                                      :diagnostics (%diagnostics-end cookie)))))
                    (cffi:with-foreign-objects ((out-contents :pointer) (out-size :uint64))
                      (let ((error (%compiler-output-map-memory output out-contents out-size)))
                        (unless (cffi:null-pointer-p error)
                          (let ((message (%compiler-error-get-message error)))
                            (%compiler-error-destroy error)
                            (error 'iree-compile-error :phase :output :message message
                                                        :diagnostics (%diagnostics-end cookie)))))
                      (%copy-membuffer-to-octets (cffi:mem-ref out-contents :pointer)
                                                  (cffi:mem-ref out-size :uint64))))
               ;; 正常終了時のクッキーの後始末（例外経路では上の各所ですでに
               ;; %diagnostics-end 済みなので二重に消しても実害はない）。
               (sb-thread:with-mutex (*diagnostics-lock*)
                 (remhash cookie *diagnostics-table*)))))
      (when output (%compiler-output-destroy output))
      (when invocation (%compiler-invocation-destroy invocation))
      (when source (%compiler-source-destroy source))
      (when session (%compiler-session-destroy session)))))
