;;;; backend: 実行系（コンパイル・実行を行うバックエンド）を core から
;;;; 切り離す抽象プロトコル（issue #9）。
;;;;
;;;; core（このファイル）は「実行系」という概念だけを知り、どの実行系が
;;;; あるかは知らない。実行系は BACKEND を継承したクラスと、下の総称関数の
;;;; メソッドを実装する別のシステムが提供する。CLAUDE.md
;;;; 「設計上の約束」の「実行系は backend プロトコルの裏に置く」を参照。
;;;;
;;;; 総称関数の名前に "compile" / "load" / "invoke" をそのまま使わないのは、
;;;; Common Lisp の CL:COMPILE / CL:LOAD と衝突し、(:use #:nabla) する
;;;; パッケージを壊すため。issue #9 の文言より CL との整合を優先する
;;;; （契約 §2 / §8）。

(in-package #:nabla)

(defclass backend ()
  ()
  (:documentation
   "実行系の抽象基底クラス。MAKE-BACKEND が返すインスタンスは
これを継承する。BACKEND 自体を MAKE-INSTANCE することは想定していない。"))

(defgeneric make-backend (kind &key)
  (:documentation
   "KIND（実行系ごとに決めるキーワード）に対応する BACKEND のインスタンスを
新しく作って返す。実装するシステムが (EQL KIND) に特化したメソッドを
追加する。対応する実装が一つもロードされていなければ、下の既定メソッドが
BACKEND-NOT-AVAILABLE を signal する。

通常は MAKE-BACKEND を直接呼ばず、プロセスにつき1つのインスタンスを共有
する FIND-BACKEND を使う。"))

(defmethod make-backend (kind &key &allow-other-keys)
  (error 'backend-not-available :kind kind))

(defvar *backends* (make-hash-table :test 'eq)
  "KIND（keyword）から、FIND-BACKEND が共有する BACKEND インスタンスへの表。")

(defvar *backends-lock* (sb-thread:make-mutex :name "nabla-backends")
  "*BACKENDS* を守るロック。FIND-BACKEND だけがこれを取る。")

(defun find-backend (kind)
  "KIND ごとに1つだけ作る、プロセス寿命の共有 BACKEND インスタンスを返す。

初めて呼ばれた KIND については (MAKE-BACKEND KIND) して *BACKENDS* に
記録する。2回目以降は同じインスタンスをそのまま返す（EQ で比較できる）。
MAKE-BACKEND が既定のオプションで失敗したときに signal した条件は、
そのまま呼び出し元に伝わる（*BACKENDS* には何も記録しないので、次回の
FIND-BACKEND でまた作り直しを試みる）。"
  (sb-thread:with-mutex (*backends-lock*)
    (or (gethash kind *backends*)
        (setf (gethash kind *backends*) (make-backend kind)))))

(defgeneric backend-target (backend)
  (:documentation
   "BACKEND が実行するターゲットを表すキーワードを返す（:local / :cuda の
ような、実行系ごとの値。フェイクの実行系なら :fake）。JIT キャッシュの
キーの一部として使われることを想定している。"))

(defgeneric backend-fingerprint (backend)
  (:documentation
   "BACKEND-COMPILE に渡す TEXT 以外で、その出力を左右するものすべてを
文字列のリストにして返す（ターゲット、GPU の世代、解決済みのコンパイル
フラグ、実行系のバージョンなど）。jit / コンパイル結果のディスクキャッシュ
のキーの構成要素として使う。"))

(defgeneric backend-compile (backend text)
  (:documentation
   "StableHLO の TEXT を BACKEND 向けにコンパイルし、
(SIMPLE-ARRAY (UNSIGNED-BYTE 8) (*)) のバイト列（コンパイル済みモジュール）
にして返す。コンパイルが失敗すれば BACKEND-ERROR の subtype を signal する。

ディスクキャッシュ（jit キャッシュ）は、この総称関数に :AROUND メソッドを
足す形で実装する（BACKEND そのものを変更しない）。"))

(defgeneric backend-load (backend octets)
  (:documentation
   "BACKEND-COMPILE が返したような OCTETS（コンパイル済みモジュールの
バイト列）を BACKEND に読み込み、不透明な module オブジェクトを返す。
module の実際の型は BACKEND の実装ごとに違う。BACKEND-INVOKE と
BACKEND-UNLOAD にそのまま渡すこと以外の使い道は保証しない。"))

(defgeneric backend-unload (backend module)
  (:documentation
   "BACKEND-LOAD が返した MODULE を解放する。解放済みの MODULE に対する
2回目以降の呼び出しは何もしない（idempotent）。"))

(defgeneric backend-invoke (backend module function-name &rest arrays)
  (:documentation
   "BACKEND-LOAD が返した MODULE の中の、FUNCTION-NAME（\"main\" のような
モジュール名を含まない関数名）で指定した関数を、ARRAYS（TO-DEVICE で
作った device array のリスト）を入力にして呼び出す。出力の device array を
多値で返す（出力が無ければ (VALUES)）。"))

(defgeneric to-device (array backend &key dtype)
  (:documentation
   "ARRAY（Lisp の simple-array）を BACKEND 上にコピーし、その BACKEND
実装ごとの device array を返す。DTYPE を渡すと、ARRAY の要素型との対応を
そのつもりで確かめる（ARRAY-AVAL 経由。矛盾すれば DTYPE-MISMATCH を
signal する）。"))

(defgeneric to-host (device-array)
  (:documentation
   "TO-DEVICE や BACKEND-INVOKE の出力である DEVICE-ARRAY の内容を、
その AVAL と同じ shape・要素型を持つ、新しい多次元 simple-array にコピーして
返す。"))

(defgeneric device-array-aval (device-array)
  (:documentation
   "DEVICE-ARRAY（TO-DEVICE や BACKEND-INVOKE の出力）の形状と dtype を表す
NABLA:AVAL を返す。"))

(define-condition backend-error (error)
  ()
  (:documentation
   "BACKEND-COMPILE / BACKEND-LOAD / BACKEND-INVOKE など、実行系の呼び出しが
signal するすべてのエラーの root コンディション。実装ごとのエラーは
これの subtype にする。"))

(define-condition backend-not-available (backend-error)
  ((kind :initarg :kind :reader backend-not-available-kind))
  (:report
   (lambda (condition stream)
     (format stream
             "~S に対応する backend を提供するシステムがロードされていない。~
そのシステムを ASDF でロードしてから MAKE-BACKEND / FIND-BACKEND を~
呼び直すこと。"
             (backend-not-available-kind condition))))
  (:documentation
   "MAKE-BACKEND / FIND-BACKEND に渡した KIND に対応する実装（対応する
(EQL KIND) の MAKE-BACKEND メソッド）が一つもロードされていないときに
signal する。KIND は渡されたキーワードそのもの。"))
