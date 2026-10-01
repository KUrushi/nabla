;;;; IREE / LLVM のシグナルハンドラ登録を、制御された1点で済ませる。
;;;;
;;;; 背景（issue #5 のクラッシュの根本原因、SBCL の SIGUSR2 を LLVM が
;;;; 乗っ取る仕組み）と、保存・復元の道具（with-lisp-signal-handlers-preserved、
;;;; %call-with-world-stopped）の汎用の説明、世界を止める手法に残る課題は
;;;; nabla/ffi-support（src/ffi-support/signals.lisp 冒頭）が正本。ここには
;;;; IREE 固有の部分だけを書く。
;;;;
;;;; LLVM（libIREECompiler.so の中）の登録は、llvm::sys::RemoveFileOnSignal /
;;;; llvm::sys::AddSignalHandler を最初に呼んだ経路——たとえば
;;;; ireeCompilerInvocationPipeline が llvm-cpu ターゲットを直列化する際に
;;;; 実行ファイルを llvm::ToolOutputFile で開く経路（CompilerDriver.cpp の
;;;; Output::openFile/openFD → ToolOutputFile のコンストラクタ →
;;;; CleanupInstaller → sys::RemoveFileOnSignal）——のどれかで初めて起きる
;;;; （NumRegisteredSignals という atomic な「プロセスにつき1回」ガードが
;;;; あり、2回目以降は何もしない）。
;;;;
;;;; 対策:
;;;;
;;;; 1. 登録を、ロード直後の「制御された1点」で済ませる。
;;;;    かつてはここで ireeCompilerSetupGlobalCL(1, {"nabla"}, NULL, true) を
;;;;    呼んでいた（installSignalHandlers=true が
;;;;    llvm::sys::PrintStackTraceOnErrorSignal 経由で登録を起こす）が、重大な
;;;;    回帰を生んだ（2026-09、PR #20 のレビューで発見。次の段落）のでやめた。
;;;;    現在は代わりに ireeCompilerOutputOpenMembuffer(&out) →
;;;;    ireeCompilerOutputDestroy(out) を呼ぶ（compiler.lisp の
;;;;    compile-stablehlo が実際の vmfb 出力に使っているのと同じ関数。
;;;;    compiler-ffi.lisp の %compiler-output-open-membuffer をそのまま使い、
;;;;    新しい CFFI バインディングは増やしていない）。Output::openMembuffer は、
;;;;    このビルドの前提である Linux + glibc 2.27 以降では memfd_create(2) で
;;;;    ディスクに一切触れない匿名の fd を作り、それを Output::openFD 経由で
;;;;    llvm::ToolOutputFile として開く。ToolOutputFile のコンストラクタは
;;;;    CleanupInstaller を経由して llvm::sys::RemoveFileOnSignal を呼ぶので、
;;;;    セッションを作らず・コンパイルも走らせず・ireeCompilerSetupGlobalCL
;;;;    にも触れず・ディスクにも触れずに、同じ RegisterHandlers を起こせる
;;;;    （実ファイルを開く ireeCompilerOutputOpenFile を使わない理由: 世界を
;;;;    止めた窓の中で open(2) の実ディスク I/O が起きるのを避けるため。
;;;;    TMPDIR が満杯・読み取り専用・低速だと、その間ずっと世界が止まった
;;;;    ままになりうる）。ireeCompilerOutputDestroy は CleanupInstaller の
;;;;    デストラクタで DontRemoveFileOnSignal を行うので、Lisp 側で後始末は
;;;;    要らない。SIGUSR2 の処分がこの呼び出しの前後（%call-with-world-stopped
;;;;    の窓の中）で変わること（=登録が実際に起きたこと）を
;;;;    %signal-handler-address のアドレス比較で確認し、その結果を
;;;;    *llvm-signal-handlers-registered-p* に残す。以後同じプロセスで何度
;;;;    呼んでも SIGUSR2 の処分はもう変わらない。
;;;;
;;;;    ireeCompilerSetupGlobalCL(installSignalHandlers=true) をやめた理由
;;;;    （重大な回帰）: この呼び出しは CompilerDriver.cpp の
;;;;    GlobalInit::usesCommandLine を真にする副作用を持つ。
;;;;    Session::Session は usesCommandLine が真だと、セッション作成時に
;;;;    一度だけ OptionsBinder::global().applyOptimizationDefaults() を
;;;;    適用し、Invocation::runPipeline は !usesCommandLine のときしか
;;;;    session.binder.applyOptimizationDefaults() を呼ばない
;;;;    （runPipeline 冒頭、resetDefaults の scope_exit と対）。つまり
;;;;    usesCommandLine が真になった瞬間から、そのプロセスで以後作る
;;;;    すべてのセッションについて、ireeCompilerSessionSetFlags で渡した
;;;;    --iree-opt-level（や他の opt-level 依存の既定値）が黙って無視される
;;;;    ようになる。default / --iree-opt-level=O3 / --iree-opt-level=O0 が
;;;;    matmul フィクスチャに対して全く同じバイト列の vmfb を生成すること
;;;;    （iree-compile CLI 直叩きの O3 とはサイズが異なる）で確認済み。
;;;;    ireeCompilerOutputOpenMembuffer 経由の登録は usesCommandLine に
;;;;    一切触れないので、この問題が起きない
;;;;    （compiler-test.lisp の opt-level-flag-changes-output が回帰テスト）。
;;;;
;;;; 2. それでも IREE を呼ぶ公開関数の本体は with-lisp-signal-handlers-preserved
;;;;    で包んでおく（多重防御。LLVM は NumRegisteredSignals ≠ 0 を見て
;;;;    再登録しないので、通常はここで差分は出ない）。
;;;;
;;;; この THUNK 固有のリスク: %register-llvm-signal-handlers の THUNK は
;;;; ireeCompilerOutputOpenMembuffer/Destroy を呼ぶので、memfd_create(2) や
;;;; LLVM 内部の std::string のコピーで malloc が起きうる。世界を止めている間に
;;;; 他スレッドが握った malloc アリーナのロックを待つと詰まりうる（GC の
;;;; stop-the-world には無かった種類のリスク）。openMembuffer は Lisp 側から
;;;; 渡す文字列や foreign バッファを必要としない（引数は out_output 用の
;;;; ポインタ1個だけ）ので Lisp 側の malloc は発生せず、実測ではプロセスに
;;;; つき1回・数ミリ秒で終わる。その他の残る課題（止めている間に届く
;;;; プロセス向けシグナル、GC ロックの餓死）は ffi-support 側に書いてある。

(in-package #:nabla.iree)

;; x86-64 Linux の stack_t（sigaltstack の引数）は 24 バイト。余裕をもって 32。
(defconstant +stack-t-size+ 32)

;;; LLVM のシグナルハンドラ登録

(defvar *llvm-signal-handlers-registered-p* nil
  "%register-llvm-signal-handlers が %call-with-world-stopped の窓の中で
SIGUSR2 の処分の変化を実際に観測したら真。窓の中でしか分からないこと
（LLVM のハンドラは窓を出る前に SBCL のものへ戻される）なので、外からは
これでしか「登録が本当に起きたか」を確認できない。compiler-test.lisp の
signals/ensure-compiler-loaded/registers-llvm-signal-handlers が見る。")

(defun %register-llvm-signal-handlers ()
  "LLVM の「プロセスにつき1回」のシグナルハンドラ登録を、他の Lisp スレッドを
止めた状態で ireeCompilerOutputOpenMembuffer(...) →
ireeCompilerOutputDestroy(...) により済ませ、SBCL のシグナルの処分（と、
この スレッドの sigaltstack）を元に戻す。ensure-compiler-loaded から、
ireeCompilerGlobalInitialize の直後に、ロードロックを持ったまま、最初の
セッションを作る前に1回だけ呼ぶ（ファイル冒頭のコメント参照。
ireeCompilerSetupGlobalCL は使わない: usesCommandLine を真にしてしまい、
セッションごとの --iree-opt-level を黙って無視させる重大な回帰を生む。
実ファイルを開く ireeCompilerOutputOpenFile も使わない: このビルドの前提
（Linux + glibc 2.27 以降）では ireeCompilerOutputOpenMembuffer が
memfd_create(2) を使うので、ディスク I/O を世界が止まった窓の中に持ち込ま
ずに同じ登録を起こせる）。

登録が実際に起きた（SIGUSR2 の処分が ireeCompilerOutputOpenMembuffer の
前後で変わった）なら T、変わらなかった（この IREE 版ではこれだけでは
登録しない）なら NIL を返す。*llvm-signal-handlers-registered-p* にも同じ
値を残す。

世界を止めている間に行うのは foreign 呼び出しだけ（openMembuffer は
out_output 用のポインタ以外に引数を取らないので、Lisp 側の文字列や
バッファをここで新たに確保する必要が無い）。保存用バッファは止める前に
確保する。ireeCompilerOutputOpenMembuffer が失敗した場合や、書き戻しに
失敗したシグナルがあった場合は、世界を再開したあとで（Lisp のロックを
取ってよくなってから）まとめて warn する。"
  (let ((registered nil)
        (restore-failed nil)
        (open-error nil))
    (cffi:with-foreign-objects ((saved :uint8 (* +nsig+ +sigaction-size+))
                                (altstack :uint8 +stack-t-size+)
                                (out-output :pointer))
      (let ((before (%signal-handler-address sb-unix:sigusr2)))
        (%call-with-world-stopped
         (lambda ()
           (%save-signal-dispositions saved)
           (cffi:foreign-funcall "sigaltstack"
                                 :pointer (cffi:null-pointer) :pointer altstack :int)
           (let ((error (%compiler-output-open-membuffer out-output)))
             ;; ireeCompilerOutputOpenMembuffer は成功・失敗にかかわらず
             ;; *out-output に Output オブジェクトを作って返すので、
             ;; エラーの有無によらず必ず Destroy する（さもないとリークする）。
             (%compiler-output-destroy (cffi:mem-ref out-output :pointer))
             (unless (cffi:null-pointer-p error)
               (setf open-error error)))
           (setf registered (/= before (%signal-handler-address sb-unix:sigusr2)))
           (setf restore-failed (%restore-signal-dispositions saved))
           (cffi:foreign-funcall "sigaltstack"
                                 :pointer altstack :pointer (cffi:null-pointer) :int)))))
    (setf *llvm-signal-handlers-registered-p* registered)
    (%warn-on-failed-signal-restore restore-failed)
    (when open-error
      (let ((message (%compiler-error-get-message open-error)))
        (%compiler-error-destroy open-error)
        (warn "ireeCompilerOutputOpenMembuffer が失敗した: ~A ~
（LLVM のシグナルハンドラ登録が済んでいない可能性がある。~
ensure-compiler-loaded は registered=NIL を見て warm-up コンパイルに ~
フォールバックする）" message)))
    registered))
