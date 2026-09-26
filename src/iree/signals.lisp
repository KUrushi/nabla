;;;; SBCL のシグナルハンドラを、IREE / LLVM の呼び出しから守る。
;;;;
;;;; 背景（issue #5 のクラッシュの根本原因）: ireeCompilerInvocationPipeline
;;;; の中で llvm-cpu ターゲットが実行ファイルを直列化するとき、IREE は一時
;;;; ファイルを llvm::ToolOutputFile で開く。そのコンストラクタは
;;;; llvm::sys::RemoveFileOnSignal を呼び、LLVM は「プロセスにつき1回」
;;;; 自前のシグナルハンドラを sigaction で SIGHUP / SIGINT / SIGTERM /
;;;; SIGUSR2 / SIGILL / SIGTRAP / SIGABRT / SIGFPE / SIGBUS / SIGSEGV /
;;;; SIGQUIT / SIGSYS / SIGXCPU / SIGXFSZ（と SIGUSR1）に登録する
;;;; （llvm/lib/Support/Unix/Signals.inc の RegisterHandlers）。
;;;;
;;;; Linux の SBCL は SIGUSR2 を SIG_STOP_FOR_GC（GC の stop-the-world で他の
;;;; スレッドを止める合図）に使う。LLVM のハンドラは SA_ONSTACK で代替
;;;; スタック上で走り、元のハンドラを戻してから raise() で同じシグナルを
;;;; 送り直す。すると SBCL のハンドラは「割り込まれた SP が代替スタック上に
;;;; ある」コンテキストを記録し、GC は制御スタックの範囲内にある SP を
;;;; 見つけられず "garbage_collect: no SP known for thread" で落ちる
;;;; （gencgc.c の conservative_stack_scan）。SBCL には常に finalizer
;;;; スレッドがいるので、明示的な GC は最初のコンパイル以降ほぼ確実に落ちる。
;;;;
;;;; 対策: IREE を呼ぶ前にプロセスのシグナルの処分（disposition）を保存し、
;;;; 呼び出しのあとで戻す。LLVM 側は「もう登録した」と記憶したままなので
;;;; （NumRegisteredSignals ≠ 0）、以後の呼び出しで再登録することはない。
;;;; 残る隙間は「最初の Pipeline 実行中」だけで、その間に別の Lisp スレッドが
;;;; GC を始めると同じクラッシュが起きうる。そのため ensure-compiler-loaded
;;;; はロード直後に最小のモジュールを1つコンパイルして（同じ保護の下で）
;;;; LLVM の登録を済ませてしまう（compiler.lisp の %warm-up-compiler）。

(in-package #:nabla.iree)

;; x86-64 Linux（glibc）の struct sigaction の大きさ。sizeof(struct sigaction)
;; = 152 だが、余裕をもって 160 バイトずつ確保する。
(defconstant +sigaction-size+ 160)

;; Linux のシグナル番号は 1..64（_NSIG = 65）。SIGKILL(9) と SIGSTOP(19) は
;; sigaction で読めない（EINVAL）ので飛ばす。32 と 33 は glibc 内部用で
;; これも EINVAL になるが、読み書きとも失敗するだけで害はない。
(defconstant +nsig+ 65)

(defun %save-signal-dispositions (buffer)
  "プロセスの全シグナルの処分（struct sigaction）を BUFFER
（+nsig+ × +sigaction-size+ バイトの foreign メモリ）に読み出す。"
  (loop for signo from 1 below +nsig+
        unless (member signo '(9 19))
          do (cffi:foreign-funcall "sigaction"
                                   :int signo
                                   :pointer (cffi:null-pointer)
                                   :pointer (cffi:inc-pointer buffer (* signo +sigaction-size+))
                                   :int)))

(defun %restore-signal-dispositions (buffer)
  "%save-signal-dispositions で BUFFER に保存した処分をすべて書き戻す。"
  (loop for signo from 1 below +nsig+
        unless (member signo '(9 19))
          do (cffi:foreign-funcall "sigaction"
                                   :int signo
                                   :pointer (cffi:inc-pointer buffer (* signo +sigaction-size+))
                                   :pointer (cffi:null-pointer)
                                   :int)))

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
