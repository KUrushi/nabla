;;;; SBCL のシグナルハンドラを、IREE / LLVM の呼び出しから守る。
;;;;
;;;; 背景（issue #5 のクラッシュの根本原因）: libIREECompiler.so の中の LLVM は
;;;; 「プロセスにつき1回」自前のシグナルハンドラを sigaction で SIGHUP /
;;;; SIGINT / SIGTERM / SIGUSR1 / SIGUSR2 / SIGILL / SIGTRAP / SIGABRT / SIGFPE /
;;;; SIGBUS / SIGSEGV / SIGQUIT / SIGSYS / SIGXCPU / SIGXFSZ（と SIGPIPE）に
;;;; 登録する（llvm/lib/Support/Unix/Signals.inc の RegisterHandlers、
;;;; SA_NODEFER|SA_RESETHAND|SA_ONSTACK）。この登録は、最初の
;;;; ireeCompilerInvocationPipeline（llvm-cpu の直列化で ToolOutputFile →
;;;; RemoveFileOnSignal）か、ireeCompilerSetupGlobalCL(installSignalHandlers=true)
;;;; （PrintStackTraceOnErrorSignal）のどちらか先に走ったほうで起きる。
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
;;;;    ireeCompilerSetupGlobalCL(1, {"nabla"}, NULL, true) を呼ぶと、コンパイル
;;;;    せずに数回の sigaction だけで登録が終わる。その呼び出しを
;;;;    %call-with-world-stopped で包む: SBCL 自身の GC（sb-kernel::sub-gc、
;;;;    src/code/gc.lisp）とまったく同じ手順で、GC ロック
;;;;    （try_acquire_gc_lock）を取ってから gc_stop_the_world で他の全 Lisp
;;;;    スレッドを SBCL 自身の SIG_STOP_FOR_GC ハンドラの中に止め、その間に
;;;;    処分の保存 → SetupGlobalCL → 処分の復元 を行い、gc_start_the_world で
;;;;    動かし直す。世界が止まっている間は
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
;;;;    THUNK が待つとデッドロックする。SBCL 自身の GC の stop-the-world も
;;;;    同じ制約を受け入れている（GC 中に malloc するコードは元々ない）ので、
;;;;    同じクラスのリスクとして許容する。実測では ireeCompilerSetupGlobalCL
;;;;    自体の所要時間はプロセスにつき1回・約 2.8ms で、ここでロックを長く
;;;;    握り続けるわけではない。
;;;; 2. 他の Lisp スレッドが sb-ext:gc を connectionless に呼び続けている
;;;;    （tight loop で明示的 GC を繰り返す）と、try_acquire_gc_lock が
;;;;    ずっと取れず %call-with-world-stopped の外側の loop が回り続ける
;;;;    「餓死（starvation）」がありうる。GC ロックは公平性を保証しないので、
;;;;    理論上は解消しない。実測では極端な continuous-GC 負荷でも数十 ms 以内
;;;;    に registration まで進んだ（scratchpad/adv/ の tight シナリオ参照）が、
;;;;    ロック取得に上限時間や優先度を設けてはいない。
;;;; 3. ireeCompilerSetupGlobalCL(usesCommandLine=true) は LLVM の
;;;;    cl::ParseCommandLineOptions 相当の経路を通り、プロセス内のコマンド
;;;;    ライン風フラグ（LLVM の -mllvm 相当を含む）をここで一度だけ解釈する。
;;;;    現状はどのフラグも渡していない（argv は "nabla" の1要素のみ）ので
;;;;    実害はないが、将来ここにセッションごとに変えたいオプション
;;;;    （例: 将来のデバッグ用フラグ）を足したくなったときは、この呼び出しが
;;;;    プロセスにつき1回しか効かないことを踏まえて設計し直す必要がある。

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
  "%save-signal-dispositions で BUFFER に保存した処分をすべて書き戻す。

書き戻しに失敗したシグナルがあれば、まとめて1回 warn する。失敗したまま
黙っていると、IREE / LLVM 呼び出し前の SBCL のシグナルハンドラに戻せて
いないことに誰も気づけない（sigaction(2) 自体の呼び出し失敗は稀だが、
*skipped-signals* 以外で起きたら偶然ではないはずなので知らせる価値がある）。"
  (let ((failed nil))
    (loop for signo from 1 below +nsig+
          unless (member signo *skipped-signals*)
            do (let ((rc (cffi:foreign-funcall "sigaction"
                                                :int signo
                                                :pointer (cffi:inc-pointer buffer (* signo +sigaction-size+))
                                                :pointer (cffi:null-pointer)
                                                :int)))
                 (unless (zerop rc) (push signo failed))))
    (when failed
      (warn "sigaction によるシグナル処分の復元が ~D 個のシグナルで失敗した ~
（番号: ~{~D~^ ~}）。IREE / LLVM を呼ぶ前の SBCL のシグナルハンドラに ~
戻せていない可能性がある"
            (length failed) (nreverse failed)))))

(defmacro with-lisp-signal-handlers-preserved (&body body)
  "BODY を実行し、その前後でプロセスのシグナルハンドラが変わっていたら
元に戻す。IREE コンパイラ（LLVM）を呼ぶすべての公開関数の本体をこれで
包む（ファイル冒頭のコメント参照）。"
  (let ((buffer (gensym "SIGACTIONS")))
    `(cffi:with-foreign-object (,buffer :uint8 (* +nsig+ +sigaction-size+))
       (%save-signal-dispositions ,buffer)
       (unwind-protect (progn ,@body)
         (%restore-signal-dispositions ,buffer)))))

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

(cffi:defcfun ("ireeCompilerSetupGlobalCL" %compiler-setup-global-cl) :void
  (argc :int)
  (argv :pointer)
  (banner :pointer)
  (install-signal-handlers %bool))

(defvar *llvm-signal-handlers-registered-p* nil
  "%register-llvm-signal-handlers が ireeCompilerSetupGlobalCL を呼んだら真。
ireeCompilerSetupGlobalCL は2回呼ぶと abort() するので、ensure-compiler-loaded
が途中で失敗してやり直されても2回目を呼ばないための印。")

(defun %register-llvm-signal-handlers ()
  "LLVM の「プロセスにつき1回」のシグナルハンドラ登録を、他の Lisp スレッドを
止めた状態で ireeCompilerSetupGlobalCL(1, {\"nabla\"}, NULL, true) により
済ませ、SBCL のシグナルの処分（と、この スレッドの sigaltstack）を元に戻す。
ensure-compiler-loaded から、ireeCompilerGlobalInitialize の直後に、ロード
ロックを持ったまま、最初のセッションを作る前に1回だけ呼ぶ。

登録が実際に起きた（SIGUSR2 の処分が SetupGlobalCL の前後で変わった）なら
T、変わらなかった（この IREE 版では SetupGlobalCL が登録しない）なら NIL を
返す。2回目以降の呼び出しは何もせず NIL を返す。

世界を止めている間に行うのは foreign 呼び出しだけで、Lisp 側の割り付け
（argv の文字列、保存用バッファ）はすべて止める前に済ませる。"
  (when *llvm-signal-handlers-registered-p*
    (return-from %register-llvm-signal-handlers nil))
  (cffi:with-foreign-string (arg0 "nabla")
    (cffi:with-foreign-objects ((argv :pointer 1)
                                (saved :uint8 (* +nsig+ +sigaction-size+))
                                (altstack :uint8 +stack-t-size+))
      (setf (cffi:mem-aref argv :pointer 0) arg0)
      (let ((before (%signal-handler-address sb-unix:sigusr2)))
        (%call-with-world-stopped
         (lambda ()
           (%save-signal-dispositions saved)
           (cffi:foreign-funcall "sigaltstack"
                                 :pointer (cffi:null-pointer) :pointer altstack :int)
           (setf *llvm-signal-handlers-registered-p* t)
           (%compiler-setup-global-cl 1 argv (cffi:null-pointer) t)
           (let ((changed (/= before (%signal-handler-address sb-unix:sigusr2))))
             (%restore-signal-dispositions saved)
             (cffi:foreign-funcall "sigaltstack"
                                   :pointer altstack :pointer (cffi:null-pointer) :int)
             changed)))))))
