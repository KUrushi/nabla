;;;; SBCL のシグナルハンドラを、IREE / LLVM の呼び出しから守る。
;;;;
;;;; 背景（issue #5 のクラッシュの根本原因）: libIREECompiler.so の中の LLVM は
;;;; 「プロセスにつき1回」自前のシグナルハンドラを sigaction で SIGHUP /
;;;; SIGINT / SIGTERM / SIGUSR1 / SIGUSR2 / SIGILL / SIGTRAP / SIGABRT / SIGFPE /
;;;; SIGBUS / SIGSEGV / SIGQUIT / SIGSYS / SIGXCPU / SIGXFSZ（と SIGPIPE）に
;;;; 登録する（llvm/lib/Support/Unix/Signals.inc の RegisterHandlers、
;;;; SA_NODEFER|SA_RESETHAND|SA_ONSTACK。NumRegisteredSignals という atomic な
;;;; 「プロセスにつき1回」ガードがあり、2回目以降の呼び出しは何もしない）。
;;;; この登録は、llvm::sys::RemoveFileOnSignal / llvm::sys::AddSignalHandler を
;;;; 最初に呼んだ経路——たとえば ireeCompilerInvocationPipeline が llvm-cpu
;;;; ターゲットを直列化する際に実行ファイルを llvm::ToolOutputFile で開く
;;;; （CompilerDriver.cpp の Output::openFile/openFD → llvm::ToolOutputFile の
;;;; コンストラクタ（file 版・fd 版のどちらも）→ CleanupInstaller →
;;;; sys::RemoveFileOnSignal）——のどれかで初めて起きる。
;;;;
;;;; Linux の SBCL は SIGUSR2 を SIG_STOP_FOR_GC（GC の stop-the-world で他の
;;;; スレッドを止める合図）に使う。LLVM のハンドラは SA_ONSTACK で代替
;;;; スタック上で走り、元のハンドラを戻してから raise() で同じシグナルを
;;;; 送り直す。すると SBCL のハンドラは「割り込まれた SP が代替スタック上に
;;;; ある」コンテキストを記録し、GC は制御スタックの範囲内にある SP を
;;;; 見つけられず "garbage_collect: no SP known for thread" で落ちる
;;;; （gencgc.c の conservative_stack_scan）。SA_RESETHAND のせいで
;;;; "User defined signal 2" でプロセスがそのまま死ぬこともある。
;;;; SIGILL / SIGTRAP（SBCL の内部エラー trap）や SIGSEGV（ガードページ）も
;;;; 同様に乗っ取られる。
;;;;
;;;; 対策は2段構え:
;;;;
;;;; 1. LLVM の登録を、ロード直後の「制御された1点」で済ませる。
;;;;    かつてはここで ireeCompilerSetupGlobalCL(1, {"nabla"}, NULL, true) を
;;;;    呼んでいた（installSignalHandlers=true が
;;;;    llvm::sys::PrintStackTraceOnErrorSignal 経由で登録を起こす）が、これは
;;;;    重大な回帰を生んだ（2026-09、PR #20 のレビューで発見。次の段落）ので
;;;;    やめた。現在は代わりに ireeCompilerOutputOpenMembuffer(&out) →
;;;;    ireeCompilerOutputDestroy(out) を呼ぶ（compiler.lisp の
;;;;    compile-stablehlo が実際の vmfb 出力に使っているのと同じ関数。
;;;;    compiler-ffi.lisp の %compiler-output-open-membuffer をそのまま使い、
;;;;    新しい CFFI バインディングは増やしていない）。CompilerDriver.cpp の
;;;;    Output::openMembuffer は、このビルドの前提である Linux + glibc 2.27
;;;;    以降では memfd_create(2) でディスクに一切触れない匿名の fd を作り、
;;;;    それを Output::openFD 経由で llvm::ToolOutputFile として開く。
;;;;    ToolOutputFile のコンストラクタ（ファイルパス版・fd 版のどちらも）は
;;;;    CleanupInstaller を経由して llvm::sys::RemoveFileOnSignal を呼ぶので、
;;;;    セッションを1つも作らず・コンパイルも1つも走らせず・
;;;;    ireeCompilerSetupGlobalCL にもまったく触れず・ディスクにも触れずに、
;;;;    同じ RegisterHandlers を起こせる（実ファイルを開く
;;;;    ireeCompilerOutputOpenFile を使わない理由: 世界を止めた窓の中で
;;;;    open(2) の実ディスク I/O が起きるのを避けるため。TMPDIR が満杯・
;;;;    読み取り専用・低速なファイルシステムだと、その間ずっと世界が止まった
;;;;    ままになりうる）。ireeCompilerOutputDestroy は Output の破棄時に
;;;;    CleanupInstaller のデストラクタで DontRemoveFileOnSignal を行う
;;;;    （memfd の fd 自体は openFD が keep() 済みなので消されないが、消す
;;;;    べき実ファイルパスも元々無い）ので、Lisp 側で後始末は要らない。
;;;;    実際に SIGUSR2 の処分がこの呼び出しの前後（%call-with-world-stopped
;;;;    の窓の中）で変わること（=登録が実際に起きたこと）を
;;;;    %signal-handler-address のアドレス比較で確認し、その結果を
;;;;    *llvm-signal-handlers-registered-p* に残す。以後同じプロセスで何度
;;;;    呼んでも（NumRegisteredSignals の atomic ガードにより）SIGUSR2 の
;;;;    処分はもう変わらない。
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
;;;;    ようになる。HEAD の状態で default / --iree-opt-level=O3 /
;;;;    --iree-opt-level=O0 が matmul フィクスチャに対して全く同じバイト数・
;;;;    バイト列の vmfb を生成すること（iree-compile CLI 直叩きの O3 とは
;;;;    サイズが異なる）で確認済み。かつてこの箇所にあった「実害はない」
;;;;    というコメントおよび PR の説明は誤りだった。
;;;;    ireeCompilerOutputOpenMembuffer 経由の登録は usesCommandLine に一切
;;;;    触れないので、この問題が起きない
;;;;    （compiler-test.lisp の opt-level-flag-changes-output が回帰テスト）。
;;;;
;;;;    その呼び出しを %call-with-world-stopped で包む: SBCL 自身の GC
;;;;    （sb-kernel::sub-gc、src/code/gc.lisp）とまったく同じ手順で、GC ロック
;;;;    （try_acquire_gc_lock）を取ってから gc_stop_the_world で他の全 Lisp
;;;;    スレッドを SBCL 自身の SIG_STOP_FOR_GC ハンドラの中に止め、その間に
;;;;    処分の保存 → OutputOpenMembuffer/OutputDestroy → 処分の復元 を行い、
;;;;    gc_start_the_world で動かし直す。世界が止まっている間は
;;;;      - gc_stop_the_world を呼べるのは all_threads_lock を持つ自分だけなので、
;;;;        誰も SIGUSR2 を送れない（SBCL の runtime で SIGUSR2 を送るのは
;;;;        thread.c の gc_stop_the_world だけ）。
;;;;      - 他の Lisp スレッドは（外部呼び出し中のものも含めて）シグナル
;;;;        ハンドラの中で sem_wait しており Lisp コードを実行しないので、
;;;;        SIGILL / SIGTRAP / SIGSEGV / SIGFPE を同期的に起こすこともない。
;;;;      - 再開（gc_start_the_world）は state_sem のセマフォで行われ、
;;;;        シグナルは送られない。
;;;;    つまり「ハンドラが LLVM のものになっている瞬間」にシグナルを受け取る
;;;;    スレッドは存在しない。
;;;;
;;;; 2. それでも IREE を呼ぶ公開関数の本体は with-lisp-signal-handlers-preserved
;;;;    で包んでおく（多重防御。LLVM は NumRegisteredSignals ≠ 0 を見て
;;;;    再登録しないので、通常はここで差分は出ない）。
;;;;
;;;; 残る課題（許容している、または解決していないリスク）:
;;;;
;;;; 1. 世界を止めている間、他のスレッドが握ったままの外部ロック
;;;;    （libc の malloc アリーナのロックなど）を %call-with-world-stopped の
;;;;    THUNK が待つとデッドロックしうる。SBCL 自身の GC の stop-the-world は
;;;;    GC 中に malloc を一切行わないので、このクラスのロック待ちはそもそも
;;;;    起きない——つまり「GC の stop-the-world が元々受け入れているリスクを
;;;;    そのまま引き継ぐ」わけではない。%register-llvm-signal-handlers の
;;;;    THUNK は ireeCompilerOutputOpenMembuffer/Destroy を呼ぶので、
;;;;    memfd_create(2) 自体や、LLVM 内部の std::string へのコピー
;;;;    （CleanupInstaller / ToolOutputFile 自身が行う）で malloc が起きうる。
;;;;    これは THUNK 固有の新しいリスク（小さいとはいえ GC の
;;;;    stop-the-world には無かった種類のリスク）である。openMembuffer は
;;;;    Lisp 側から渡す文字列や foreign バッファを一切必要としない
;;;;    （引数は out_output 用のポインタ1個だけ）ので、この THUNK では
;;;;    Lisp 側の malloc/free はもともと発生せず、避けるべき対象は LLVM
;;;;    自身が内部で行う malloc だけになる。実測ではこの呼び出し全体で
;;;;    プロセスにつき1回・数ミリ秒で終わり、他スレッドが同時に malloc
;;;;    アリーナのロックを長く握り続けるような負荷は scratchpad/adv/ の
;;;;    tight シナリオでも観測していない。
;;;; 2. 世界が止まっている間にプロセスへ届いたシグナル（kill(1) や
;;;;    デバッガからの SIGINT / SIGTERM など、GC の SIGUSR2 以外の
;;;;    プロセス向けシグナル）は、それを受け取るスレッドが THUNK の中の
;;;;    どの時点にいるかによって、登録前の SBCL のハンドラにも、登録後の
;;;;    LLVM のハンドラにも渡りうる。ireeCompilerSetupGlobalCL 版でも同じ
;;;;    性質を持っていたが、これまで明記していなかったのでここに書いておく。
;;;;    実運用でこれらのシグナルをプロセスに送ることは通常なく、窓の長さも
;;;;    数ミリ秒なので確率は低いが、解消してはいない。
;;;; 3. 他の Lisp スレッドが sb-ext:gc を connectionless に呼び続けている
;;;;    （tight loop で明示的 GC を繰り返す）と、try_acquire_gc_lock が
;;;;    ずっと取れず %call-with-world-stopped の外側の loop が回り続ける
;;;;    「餓死（starvation）」がありうる。GC ロックは公平性を保証しないので、
;;;;    理論上は解消しない。実測では極端な continuous-GC 負荷でも数十 ms 以内
;;;;    に registration まで進んだ（scratchpad/adv/ の tight シナリオ参照）が、
;;;;    ロック取得に上限時間や優先度を設けてはいない。

(in-package #:nabla.iree)

;; x86-64 Linux（glibc）の struct sigaction の大きさ。sizeof(struct sigaction)
;; = 152 だが、余裕をもって 160 バイトずつ確保する。
(defconstant +sigaction-size+ 160)

;; Linux のシグナル番号は 1..64（_NSIG = 65）。
(defconstant +nsig+ 65)

;; x86-64 Linux の stack_t（sigaltstack の引数）は 24 バイト。余裕をもって 32。
(defconstant +stack-t-size+ 32)

(defparameter *skipped-signals* '(9 19 32 33)
  "sigaction で読み書きしないシグナル番号。SIGKILL(9) と SIGSTOP(19) は
sigaction 自体が EINVAL で拒否する。32 と 33 は glibc の pthread 実装が
内部で使う（NPTL の SIGCANCEL / SIGSETXID）ので、EINVAL で無害に失敗するとは
いえ触らないほうがよく、意図を明確にするため他の除外番号と一緒に明示的に
飛ばす。")

(defun %save-signal-dispositions (buffer)
  "プロセスの全シグナルの処分（struct sigaction）を BUFFER
（+nsig+ × +sigaction-size+ バイトの foreign メモリ）に読み出す。

読み出す前に BUFFER 全体をゼロクリアする: *skipped-signals* や、まれに
sigaction(2) 自体が失敗したシグナルのぶんは書き込まれずに残るので、
ゼロにしておかないと %restore-signal-dispositions が不定値のままの
struct sigaction で復元してしまう（ゼロ埋めなら handler=SIG_DFL・
mask=空・flags=0 になり、少なくとも復元先が壊れたバイト列にはならない）。"
  (cffi:foreign-funcall "memset" :pointer buffer :int 0
                                 :size (* +nsig+ +sigaction-size+) :pointer)
  (loop for signo from 1 below +nsig+
        unless (member signo *skipped-signals*)
          do (cffi:foreign-funcall "sigaction"
                                   :int signo
                                   :pointer (cffi:null-pointer)
                                   :pointer (cffi:inc-pointer buffer (* signo +sigaction-size+))
                                   :int)))

(defun %restore-signal-dispositions (buffer)
  "%save-signal-dispositions で BUFFER に保存した処分をすべて書き戻し、
書き戻しに失敗したシグナル番号のリスト（昇順、通常は空）を返す。

ここでは warn しない: %register-llvm-signal-handlers はこれを
%call-with-world-stopped の THUNK（世界が止まっている間）から呼ぶことが
あり、warn は Lisp のロック（*standard-output* のストリームロックなど）を
取りうるので、そこで呼ぶとデッドロックしうる。失敗を知らせたい呼び出し側は
戻り値を %warn-on-failed-signal-restore に渡す（世界を再開したあとで）。"
  (let ((failed nil))
    (loop for signo from 1 below +nsig+
          unless (member signo *skipped-signals*)
            do (let ((rc (cffi:foreign-funcall "sigaction"
                                                :int signo
                                                :pointer (cffi:inc-pointer buffer (* signo +sigaction-size+))
                                                :pointer (cffi:null-pointer)
                                                :int)))
                 (unless (zerop rc) (push signo failed))))
    (nreverse failed)))

(defun %warn-on-failed-signal-restore (failed)
  "FAILED（%restore-signal-dispositions の戻り値）が空でなければ、まとめて
1回 warn する。書き戻しに失敗したまま黙っていると、IREE / LLVM 呼び出し前の
SBCL のシグナルハンドラに戻せていないことに誰も気づけない
（sigaction(2) 自体の呼び出し失敗は稀だが、*skipped-signals* 以外で起きたら
偶然ではないはずなので知らせる価値がある）。世界を止めている間には
絶対に呼ばないこと（%restore-signal-dispositions の docstring 参照）。"
  (when failed
    (warn "sigaction によるシグナル処分の復元が ~D 個のシグナルで失敗した ~
（番号: ~{~D~^ ~}）。IREE / LLVM を呼ぶ前の SBCL のシグナルハンドラに ~
戻せていない可能性がある"
          (length failed) failed)))

(defmacro with-lisp-signal-handlers-preserved (&body body)
  "BODY を実行し、その前後でプロセスのシグナルハンドラが変わっていたら
元に戻す。IREE コンパイラ（LLVM）を呼ぶすべての公開関数の本体をこれで
包む（ファイル冒頭のコメント参照）。"
  (let ((buffer (gensym "SIGACTIONS")))
    `(cffi:with-foreign-object (,buffer :uint8 (* +nsig+ +sigaction-size+))
       (%save-signal-dispositions ,buffer)
       (unwind-protect (progn ,@body)
         (%warn-on-failed-signal-restore (%restore-signal-dispositions ,buffer))))))

(defun %signal-handler-address (signo)
  "SIGNO の現在のハンドラ（sa_handler）のアドレスを整数で返す。テストが
「IREE を呼んでも SBCL のハンドラが変わらないこと」を確かめるのに使う。"
  (cffi:with-foreign-object (buffer :uint8 +sigaction-size+)
    (cffi:foreign-funcall "sigaction" :int signo :pointer (cffi:null-pointer) :pointer buffer :int)
    (cffi:mem-ref buffer :uint64 0)))

;;; stop-the-world

(defun %call-with-world-stopped (thunk)
  "他のすべての Lisp スレッドを SBCL の GC と同じ仕組み（SIG_STOP_FOR_GC）で
止めた状態で THUNK を呼び、その値を返す。

手順は SBCL 2.2.9 の sb-kernel::sub-gc（src/code/gc.lisp）の写し:
without-interrupts の中で、without-gcing に入り、try_acquire_gc_lock
（thread.c の in_gc_lock、非ブロッキング）が取れたら gc_stop_the_world →
release_gc_lock → THUNK → gc_start_the_world。取れなければ別スレッドの GC が
世界を止めようとしているので、いったん without-gcing を抜けて（そこで自分が
止められ、GC が終わると再開する）やり直す。without-gcing を抜けるときに
*gc-pending* が立っていれば SBCL が通常どおり GC を走らせる。

THUNK に課す条件（守らないとデッドロックしうる）:
  - Lisp のロックを取らない、ストリームに書かない、Lisp のコードを大量に
    走らせない。実質、foreign 呼び出しとポインタ演算だけにする。
    止まっているスレッドが持っているロック（ストリームのロック、malloc の
    アリーナのロック等）を待つと、そのスレッドは再開されないので詰まる。
  - 止まっている間にシグナルハンドラを差し替えても、戻して THUNK から
    帰るまで、シグナルを受け取るスレッドは存在しない（ファイル冒頭参照）。
    THUNK が終わるまでに必ず SBCL のハンドラに戻すこと。

既に without-gcing の中（*gc-inhibit* が真）で呼ぶとロックの取り合いで
進めなくなるので、エラーにする。"
  (when sb-kernel:*gc-inhibit*
    (error "%call-with-world-stopped は without-gcing の中からは呼べない"))
  (sb-sys:without-interrupts
    (loop
      (sb-sys:without-gcing
        (when (eql 1 (sb-alien:alien-funcall
                      (sb-alien:extern-alien "try_acquire_gc_lock" (function sb-alien:int))))
          ;; sub-gc と同じく、世界が止まったら GC ロックは手放してよい:
          ;; 世界が止まっている間は誰も try_acquire_gc_lock を呼べない。
          (sb-kernel::gc-stop-the-world)
          (sb-alien:alien-funcall
           (sb-alien:extern-alien "release_gc_lock" (function sb-alien:void)))
          (return
            (unwind-protect (funcall thunk)
              (sb-kernel::gc-start-the-world))))))))

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
