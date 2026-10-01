;;;; 外部ライブラリ（C の FFI）の呼び出しから SBCL のシグナルハンドラを守る
;;;; 汎用の道具（issue #79）。
;;;;
;;;; 背景（issue #5 のクラッシュの根本原因）: 外部ライブラリの中には、
;;;; プロセスにつき1回、自前のシグナルハンドラを sigaction で
;;;; SIGHUP / SIGINT / SIGTERM / SIGUSR1 / SIGUSR2 / SIGILL / SIGTRAP /
;;;; SIGABRT / SIGFPE / SIGBUS / SIGSEGV など多数に登録し直すものがある
;;;; （LLVM の llvm/lib/Support/Unix/Signals.inc の RegisterHandlers が実例。
;;;; SA_NODEFER|SA_RESETHAND|SA_ONSTACK）。
;;;;
;;;; Linux の SBCL は SIGUSR2 を SIG_STOP_FOR_GC（GC の stop-the-world で
;;;; 他のスレッドを止める合図）に使う。ライブラリのハンドラは SA_ONSTACK で
;;;; 代替スタック上で走り、元のハンドラを戻してから raise() で同じシグナルを
;;;; 送り直す。すると SBCL のハンドラは「割り込まれた SP が代替スタック上に
;;;; ある」コンテキストを記録し、GC は制御スタックの範囲内にある SP を
;;;; 見つけられず "garbage_collect: no SP known for thread" で落ちる
;;;; （gencgc.c の conservative_stack_scan）。SIGILL / SIGTRAP（SBCL の内部
;;;; エラー trap）や SIGSEGV（ガードページ）も同様に乗っ取られる。
;;;;
;;;; ここにあるもの:
;;;;
;;;; - with-lisp-signal-handlers-preserved: 本体の前後でシグナルの処分を
;;;;   保存・復元する（多重防御）。
;;;; - %call-with-world-stopped: SBCL の GC と同じ手順で他の全 Lisp スレッドを
;;;;   止めた「制御された1点」で THUNK を呼ぶ。この窓の中でシグナルの処分を
;;;;   保存 → 外部ライブラリに登録させる → 復元すれば、ハンドラが
;;;;   ライブラリのものになっている瞬間にシグナルを受け取るスレッドは存在しない。
;;;;   世界が止まっている間は
;;;;     - gc_stop_the_world を呼べるのは all_threads_lock を持つ自分だけなので、
;;;;       誰も SIGUSR2 を送れない（SBCL の runtime で SIGUSR2 を送るのは
;;;;       thread.c の gc_stop_the_world だけ）。
;;;;     - 他の Lisp スレッドは（外部呼び出し中のものも含めて）シグナル
;;;;       ハンドラの中で sem_wait しており Lisp コードを実行しないので、
;;;;       SIGILL / SIGTRAP / SIGSEGV / SIGFPE を同期的に起こすこともない。
;;;;     - 再開（gc_start_the_world）は state_sem のセマフォで行われ、
;;;;       シグナルは送られない。
;;;;
;;;; 個々のライブラリ固有の登録手順（何を呼べば登録が起きるか）と、その
;;;; 根拠は各ライブラリのシステムに置く（例: src/ 以下の各ライブラリのシステムの signals.lisp）。
;;;;
;;;; %call-with-world-stopped を使うときに残る課題（許容している、または
;;;; 解決していないリスク）:
;;;;
;;;; 1. 世界を止めている間、他のスレッドが握ったままの外部ロック
;;;;    （libc の malloc アリーナのロックなど）を THUNK が待つとデッドロック
;;;;    しうる。SBCL 自身の GC の stop-the-world は GC 中に malloc を一切
;;;;    行わないので、このクラスのロック待ちはそもそも起きない——つまり
;;;;    THUNK が外部ライブラリを呼ぶ場合、GC の stop-the-world には無かった
;;;;    種類のリスクを持ち込む。THUNK は foreign 呼び出しとポインタ演算だけに
;;;;    し、Lisp 側で malloc / free を発生させない（保存用バッファは止める前に
;;;;    確保する）こと。
;;;; 2. 世界が止まっている間にプロセスへ届いたシグナル（kill(1) やデバッガ
;;;;    からの SIGINT / SIGTERM など、GC の SIGUSR2 以外のプロセス向け
;;;;    シグナル）は、受け取るスレッドが THUNK のどの時点にいるかによって、
;;;;    登録前の SBCL のハンドラにも、登録後のライブラリのハンドラにも
;;;;    渡りうる。窓の長さは通常数ミリ秒なので確率は低いが、解消してはいない。
;;;; 3. 他の Lisp スレッドが sb-ext:gc を tight loop で呼び続けると、
;;;;    try_acquire_gc_lock がずっと取れず外側の loop が回り続ける「餓死」が
;;;;    ありうる。GC ロックは公平性を保証しないので理論上は解消しない。
;;;;    ロック取得に上限時間や優先度は設けていない。

(in-package #:nabla.ffi-support)

;; x86-64 Linux（glibc）の struct sigaction の大きさ。sizeof(struct sigaction)
;; = 152 だが、余裕をもって 160 バイトずつ確保する。
(defconstant +sigaction-size+ 160)

;; Linux のシグナル番号は 1..64（_NSIG = 65）。
(defconstant +nsig+ 65)

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

ここでは warn しない: 呼び出し側はこれを
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
1回 warn する。書き戻しに失敗したまま黙っていると、外部ライブラリ（LLVM など）の呼び出し前の
SBCL のシグナルハンドラに戻せていないことに誰も気づけない
（sigaction(2) 自体の呼び出し失敗は稀だが、*skipped-signals* 以外で起きたら
偶然ではないはずなので知らせる価値がある）。世界を止めている間には
絶対に呼ばないこと（%restore-signal-dispositions の docstring 参照）。"
  (when failed
    (warn "sigaction によるシグナル処分の復元が ~D 個のシグナルで失敗した ~
（番号: ~{~D~^ ~}）。外部ライブラリ（LLVM など）を呼ぶ前の SBCL のシグナルハンドラに ~
戻せていない可能性がある"
          (length failed) failed)))

(defmacro with-lisp-signal-handlers-preserved (&body body)
  "BODY を実行し、その前後でプロセスのシグナルハンドラが変わっていたら
元に戻す。シグナルハンドラを書き換えうる外部ライブラリ（LLVM など）を呼ぶすべての公開関数の本体をこれで
包む（ファイル冒頭のコメント参照）。"
  (let ((buffer (gensym "SIGACTIONS")))
    `(cffi:with-foreign-object (,buffer :uint8 (* +nsig+ +sigaction-size+))
       (%save-signal-dispositions ,buffer)
       (unwind-protect (progn ,@body)
         (%warn-on-failed-signal-restore (%restore-signal-dispositions ,buffer))))))

(defun %signal-handler-address (signo)
  "SIGNO の現在のハンドラ（sa_handler）のアドレスを整数で返す。テストが
「外部ライブラリを呼んでも SBCL のハンドラが変わらないこと」を確かめるのに使う。"
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
