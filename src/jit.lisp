;;;; jit: TRACEABLE-FUNCTION をコンパイル・実行系にかけて実行する
;;;; インメモリのコンパイルキャッシュ付き関数に変換する（issue #34、wave 4 j1）。
;;;;
;;;; パイプラインは trace -> graph -> emit -> compile -> load -> execute の
;;;; 6段階に分け、それぞれ %JIT-TRACE / EMIT-STABLEHLO / %JIT-COMPILE-AND-LOAD /
;;;; %JIT-EXECUTE という別々の関数のままにする（phase 2 の grad / phase 3 の
;;;; vmap は %JIT-TRACE と EMIT-STABLEHLO の間に graph -> graph の書き換えを
;;;; 挟む変換として書けるようにするため。design tab 参照）。
;;;;
;;;; このファイルは実行系の実装を一切知らない（core は backend プロトコルの
;;;; 裏だけを見る、issue #9 の設計）。既定の実行系は、このファイルではなく、
;;;; それを実装する側のシステムのロード時に *DEFAULT-BACKEND* へ設定される。

(in-package #:nabla)

;;; --- 条件 ---

(define-condition jit-error (error)
  ((format-control :initarg :format-control :reader jit-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil :reader jit-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "jit: ~?"
             (jit-error-format-control condition)
             (jit-error-format-arguments condition))))
  (:documentation
   "JIT / JITTED-FUNCTION の呼び出しが、使い方の誤りを検出したときに
signal する。TRACING-ERROR と同じ形（FORMAT-CONTROL・FORMAT-ARGUMENTS）。"))

(defun %jit-error (format-control &rest format-arguments)
  "JIT-ERROR を signal する（FORMAT-CONTROL・FORMAT-ARGUMENTS を JIT-ERROR に
そのまま渡す小さなヘルパー）。"
  (error 'jit-error :format-control format-control :format-arguments format-arguments))

;;; --- 既定の backend ---

(defvar *default-backend* nil
  "JIT が BACKEND を明示的に指定されなかったときに使う既定値。

NIL（未設定）、BACKEND の KIND（キーワード。呼び出しのたびに FIND-BACKEND
で解決する）、または BACKEND のインスタンスのいずれか。

実行系を実装するシステムをロードすると、この変数がまだ NIL のときに
限り、そのシステムが自分の KIND を設定する（ユーザーが明示的に設定した
値を上書きしない。core であるこのファイルは、実行系の名前を知らない）。")

(defun %jit-resolve-backend (jitted)
  "JITTED-FUNCTION の呼び出しに使う BACKEND インスタンスを決める。

JITTED 自身の :BACKEND（keyword なら FIND-BACKEND で解決、インスタンス
ならそのまま） > *DEFAULT-BACKEND*（同様に解決） > どちらも無ければ
JIT-ERROR、の優先順位。呼び出しのたびに解決するので、*DEFAULT-BACKEND*
を束縛し直せば次の呼び出しから backend を切り替えられる。"
  (flet ((resolve (value)
           (etypecase value
             (backend value)
             (keyword (find-backend value)))))
    (cond
      ((%jitted-function-backend jitted) (resolve (%jitted-function-backend jitted)))
      (*default-backend* (resolve *default-backend*))
      (t (%jit-error "*DEFAULT-BACKEND* が未設定で、JIT にも :BACKEND を渡していない")))))

;;; --- JITTED-FUNCTION ---

(defclass jitted-function (sb-mop:funcallable-standard-object)
  ((fn :initarg :fn :reader %jitted-function-fn)
   (static-positions :initarg :static-positions :reader %jitted-function-static-positions)
   (backend :initarg :backend :reader %jitted-function-backend))
  (:metaclass sb-mop:funcallable-standard-class)
  (:documentation
   "JIT が返す funcallable なオブジェクト。FN は元の TRACEABLE-FUNCTION、
STATIC-POSITIONS は静的引数の0始まりの位置（昇順）のリスト、BACKEND は
JIT に渡された :BACKEND（NIL・keyword・BACKEND インスタンスのいずれか。
実際に使う BACKEND は呼び出しのたびに %JIT-RESOLVE-BACKEND が決める）。
export しない（ハイラムの法則。TYPEP で判定する必要も今のところ無い）。"))

(defun %jit-static-positions-valid-p (positions arity)
  "POSITIONS（jit の :STATIC-ARGS）が ARITY に対して妥当かを返す。妥当とは
各要素が 0 <= p < ARITY を満たす整数で、重複が無いこと。"
  (and (every (lambda (p) (and (integerp p) (<= 0 p) (< p arity))) positions)
       (= (length positions) (length (remove-duplicates positions)))))

(defun jit (fn &key static-args backend)
  "FN（WITH-TRACING / DEFJIT が作った TRACEABLE-FUNCTION）を、呼ぶたびに
必要なら1回だけコンパイルしてから実行する JITTED-FUNCTION にする。

STATIC-ARGS は FN の仮引数リストへの0始まりの位置のリスト。そこにある
引数はトレース時に本体へ普通の Lisp の値として渡され（クロージャに
閉じ込められる）、EQUAL で比較できる値（数値・シンボル・文字列・その
リストなど）でなければならない。1 と 1.0 は異なる静的引数として扱う。
配列を静的引数にすると、キャッシュの比較は EQUAL（＝配列は同一性）に
なることに注意する。

BACKEND は NIL（呼び出し時に *DEFAULT-BACKEND* を使う）・BACKEND の
KIND（keyword）・BACKEND のインスタンスのいずれか。

FN が TRACEABLE-FUNCTION でない、STATIC-ARGS が不正（整数でない・範囲外・
重複）なら JIT-ERROR を signal する。"
  (unless (typep fn 'traceable-function)
    (%jit-error "FN は TRACEABLE-FUNCTION でなければならない（WITH-TRACING か DEFJIT で作った関数を渡すこと）: ~S" fn))
  (let ((arity (length (traceable-function-lambda-list fn))))
    (unless (%jit-static-positions-valid-p static-args arity)
      (%jit-error "STATIC-ARGS が不正（0以上 ~D 未満の整数で、重複が無いこと）: ~S" arity static-args)))
  (let ((instance (make-instance 'jitted-function
                                  :fn fn
                                  :static-positions (sort (copy-list static-args) #'<)
                                  :backend backend)))
    (sb-mop:set-funcallable-instance-function
     instance (lambda (&rest args) (%jit-call instance args)))
    instance))

;;; --- %jit-trace: 静的引数の合成 ---

(defun %jit-merge-args (arity static-positions static-values dynamic-values)
  "静的引数と動的引数を、元の（0始まりの）位置の順に並べ直した ARITY 個の
リストにして返す。STATIC-POSITIONS は昇順、STATIC-VALUES はそれと対応する
値のリスト、DYNAMIC-VALUES は残りの位置を出現順に埋める値のリスト。"
  (let ((result (make-array arity))
        (dynamic-values dynamic-values))
    (loop for position in static-positions
          for value in static-values
          do (setf (aref result position) value))
    (dotimes (position arity)
      (unless (member position static-positions)
        (setf (aref result position) (pop dynamic-values))))
    (coerce result 'list)))

(defun %jit-trace (fn avals static-positions static-values)
  "FN（TRACEABLE-FUNCTION）を、STATIC-POSITIONS の引数を STATIC-VALUES に
固定した上で、残りの引数を AVALS（動的引数の分だけの長さ）でトレースし、
GRAPH を返す。

STATIC-POSITIONS の引数はトレース対象の本体には普通の Lisp の値として
渡る（クロージャに閉じ込める。トレーサにはしない）ので、本体の中で
その値を使った分岐や形状計算がそのまま働く。"
  (let* ((arity (length (traceable-function-lambda-list fn)))
         (dynamic-lambda-list (loop repeat (- arity (length static-positions)) collect (gensym "DYN")))
         (wrapper (%make-traceable-function
                   dynamic-lambda-list
                   (lambda (&rest dynamic-args)
                     (apply (%traceable-function-function fn)
                            (%jit-merge-args arity static-positions static-values dynamic-args))))))
    (trace-to-graph wrapper avals)))

;;; --- %jit-compile-and-load ---

(defun %jit-compile-and-load (backend text)
  "TEXT を BACKEND-COMPILE してから BACKEND-LOAD した module を返す。"
  (backend-load backend (backend-compile backend text)))

;;; --- 引数から aval を求める ---

(defun %jit-argument-aval (value)
  "VALUE（配列または device array）から AVAL を求める。VALUE が CL の配列
（simple-array）なら ARRAY-AVAL、そうでなければ DEVICE-ARRAY-AVAL を使う。

VALUE が生の (UNSIGNED-BYTE 16) 配列（bf16/f16）で dtype を推論できなければ
（ARRAY-AVAL が DTYPE-MISMATCH を signal すれば）、bf16/f16 は TO-DEVICE で
device array にしてから渡すよう案内する JIT-ERROR に変換する。"
  (if (arrayp value)
      (handler-case (array-aval value)
        (dtype-mismatch ()
          (%jit-error "bf16/f16 は TO-DEVICE で device array にしてから渡すこと: ~S" value)))
      (device-array-aval value)))

;;; --- %jit-execute ---

(defun %jit-to-device-argument (backend value aval)
  "VALUE（配列または device array）を、必要なら BACKEND 上の device array に
してから返す。VALUE がすでに CL の配列でなければ（device array なら）
そのまま返す。"
  (if (arrayp value)
      (to-device value backend :dtype (aval-dtype aval))
      value))

(defun %jit-execute (backend module arrays avals)
  "ARRAYS（配列または device array のリスト。AVALS はそれぞれの aval）を
必要なら BACKEND 上の device array にしてから MODULE を BACKEND-INVOKE し、
結果を TO-HOST した多値で返す（出力が無ければ (VALUES)）。"
  (let* ((device-arrays (mapcar (lambda (value aval) (%jit-to-device-argument backend value aval))
                                 arrays avals))
         (results (multiple-value-list (apply #'backend-invoke backend module "main" device-arrays))))
    (values-list (mapcar #'to-host results))))

;;; --- キャッシュ ---

(defvar *jit-cache* (make-hash-table :test 'eq :weakness :key)
  "TRACEABLE-FUNCTION（jit に渡した FN そのもの）から、その関数のキャッシュ
エントリを持つ内側の EQUAL ハッシュ表への表。:WEAKNESS :KEY なので、FN が
どこからも参照されなくなれば GC がこのエントリごと回収する（ただし
BACKEND-UNLOAD は呼ばれない。既知の制約、follow-up は J1.5 のコメント参照）。")

(defvar *jit-cache-lock* (sb-thread:make-mutex :name "nabla-jit-cache")
  "*JIT-CACHE* の読み書きと、キャッシュミス時のコンパイルを保護する粗い
ロック。コンパイルはこのロックを持ったまま行う（同じ関数への同時呼び出しが
二重にコンパイルしないようにするための単純な選択。詳しい並行性が要る
ようになったら、より細かいロックに分けるのを follow-up にする）。")

(defvar *jit-miss-count* 0
  "キャッシュミスして実際にトレース・コンパイルした回数（内部の統計。
テストが nb::*jit-miss-count* で読む）。")

(defstruct (%jit-entry (:constructor %make-jit-entry (graph text module)))
  "*JIT-CACHE* の内側の表に入るエントリ。GRAPH は phase 2 の grad や
use-eager リスタート、診断が必要とするので保持する。"
  graph
  text
  module)

(defun %jit-cache-key (backend avals static-values)
  "*JIT-CACHE* の内側（EQUAL）の表に使うキーを返す。AVALS は AVAL のリスト
（構造体は EQUAL が EQ 相当になってしまうため、(shape dtype) のリストに
正規化する）。BACKEND インスタンス自体は EQUAL の中で EQ 比較される。
BACKEND-FINGERPRINT も含めるのは、同じ BACKEND インスタンスでもコンパイル
フラグ等が呼び出しごとに変わりうるため（design tab の「ターゲット（GPU
世代を含む）」の文言）。"
  (list backend
        (backend-fingerprint backend)
        (mapcar (lambda (aval) (list (aval-shape aval) (aval-dtype aval))) avals)
        static-values))

(defun %jit-cache-entry-count (fn)
  "FN（jit に渡した TRACEABLE-FUNCTION）が *JIT-CACHE* に持つエントリの数を
返す（無ければ0）。内部テスト用。"
  (let ((table (gethash fn *jit-cache*)))
    (if table (hash-table-count table) 0)))

(defun %jit-cache-forget (fn)
  "FN の *JIT-CACHE* エントリをすべて捨てる。各エントリの MODULE を、その
キーに含まれる BACKEND で BACKEND-UNLOAD してから REMHASH する。捨てた
エントリの数を返す。"
  (sb-thread:with-mutex (*jit-cache-lock*)
    (let ((table (gethash fn *jit-cache*)))
      (if (null table)
          0
          (let ((count (hash-table-count table)))
            (maphash (lambda (key entry)
                       (backend-unload (first key) (%jit-entry-module entry)))
                     table)
            (remhash fn *jit-cache*)
            count)))))

(defun %jit-cache-lookup-or-compile (fn key backend graph-thunk)
  "*JIT-CACHE* から FN・KEY に対応するエントリを引く。無ければ GRAPH-THUNK
（引数無しの関数で、GRAPH を返す）を呼んでトレースし、EMIT-STABLEHLO ->
%JIT-COMPILE-AND-LOAD（BACKEND 上に）した結果を新しいエントリとして
書き込む。呼び出し全体を *JIT-CACHE-LOCK* で保護する。"
  (sb-thread:with-mutex (*jit-cache-lock*)
    (let ((table (or (gethash fn *jit-cache*)
                      (setf (gethash fn *jit-cache*) (make-hash-table :test 'equal)))))
      (or (gethash key table)
          (progn
            (incf *jit-miss-count*)
            (let* ((graph (funcall graph-thunk))
                   (text (emit-stablehlo graph))
                   (module (%jit-compile-and-load backend text)))
              (setf (gethash key table) (%make-jit-entry graph text module))))))))

;;; --- %jit-call ---

(defun %jit-call (jitted args)
  "JITTED（JITTED-FUNCTION）の呼び出しの糸口。引数の個数を確かめ、backend を
解決し、静的引数を切り出し、動的引数から aval を求め、キャッシュを引いて
（無ければコンパイルして）%JIT-EXECUTE する。"
  (let* ((fn (%jitted-function-fn jitted))
         (static-positions (%jitted-function-static-positions jitted))
         (arity (length (traceable-function-lambda-list fn))))
    (unless (= (length args) arity)
      (%jit-error "引数の個数 ~D が関数の引数の個数 ~D と一致しない" (length args) arity))
    (let ((backend (%jit-resolve-backend jitted))
          (static-values nil)
          (dynamic-values nil))
      (loop for position from 0
            for arg in args
            do (if (member position static-positions)
                   (push arg static-values)
                   (push arg dynamic-values)))
      (setf static-values (nreverse static-values)
            dynamic-values (nreverse dynamic-values))
      (let* ((avals (mapcar #'%jit-argument-aval dynamic-values))
             (key (%jit-cache-key backend avals static-values))
             (entry (%jit-cache-lookup-or-compile
                     fn key backend
                     (lambda () (%jit-trace fn avals static-positions static-values)))))
        (%jit-execute backend (%jit-entry-module entry) dynamic-values avals)))))
