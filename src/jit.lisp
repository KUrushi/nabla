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

動的引数と戻り値は PyTree（リスト・SIMPLE-VECTOR・DEFPYTREE で登録した型）でもよい。
葉は配列・device array・実数（rank 0 の配列として扱う）で、
要素型 T の SIMPLE-VECTOR は配列ではなく節（葉に分かれる）なので、ベクタの配列を渡すときは
型特化した配列（nabla の配列）にする。キャッシュのキーには葉の
aval と動的引数の treedef が入る（木の形が違えば別のコンパイル）。戻り値は FN の戻り値と
同じ形（多値は多値、木は同じ形の木）で、葉は host 配列。

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

(defun %jit-trace (fn avals static-positions static-values in-treedef)
  "FN（TRACEABLE-FUNCTION）を、STATIC-POSITIONS の引数を STATIC-VALUES に
固定した上で、残りの動的引数の木（IN-TREEDEF は動的引数の木のリストの treedef）の
葉を AVALS（葉の個数だけの長さ）でトレースし、(VALUES GRAPH 戻り値の treedef) を返す。
戻り値の treedef は FN の戻り値（多値）をリストにした形で、%UNFLATTEN-RESULTS が使う。

STATIC-POSITIONS の引数はトレース対象の本体には普通の Lisp の値として
渡る（クロージャに閉じ込める。トレーサにはしない）ので、本体の中で
その値を使った分岐や形状計算がそのまま働く。"
  (let* ((arity (length (traceable-function-lambda-list fn)))
         (merged (lambda (&rest dynamic-args)
                   (apply (%traceable-function-function fn)
                          (%jit-merge-args arity static-positions static-values dynamic-args)))))
    (multiple-value-bind (flat out-treedef) (%flatten-function merged in-treedef)
      (let ((graph (trace-to-graph
                    (%make-traceable-function (loop repeat (length avals) collect (gensym "LEAF"))
                                              flat)
                    avals)))
        (values graph (funcall out-treedef))))))

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

(defun %jit-leaf-array (leaf)
  "引数の木の葉 LEAF が Lisp の実数なら rank 0 の配列（DOUBLE-FLOAT は :F64、それ以外は :F32。
grad と同じ）にして返す。配列と device array はそのまま返し、それ以外は JIT-ERROR。"
  (cond
    ((realp leaf) (%scalar-array leaf (if (typep leaf 'double-float) :f64 :f32)))
    ((and (arrayp leaf) (not (stringp leaf))) leaf)
    ;; device array は backend ごとのクラスなので、DEVICE-ARRAY-AVAL のメソッドがあるかで判定する
    ((and (not (stringp leaf)) (compute-applicable-methods #'device-array-aval (list leaf))) leaf)
    (t (%jit-error "引数の木の葉は配列・device array・実数でなければならない: ~S" leaf))))

(defun %jit-to-device-argument (backend value aval)
  "VALUE（配列または device array）を、必要なら BACKEND 上の device array に
してから返す。VALUE がすでに CL の配列でなければ（device array なら）
そのまま返す。"
  (if (arrayp value)
      (to-device value backend :dtype (aval-dtype aval))
      value))

(defun %jit-execute (backend module arrays avals out-treedef)
  "ARRAYS（配列または device array のリスト。AVALS はそれぞれの aval）を
必要なら BACKEND 上の device array にしてから MODULE を BACKEND-INVOKE し、
結果を TO-HOST して OUT-TREEDEF の形の戻り値（多値。出力が無ければ (VALUES)）にして返す。"
  (let* ((device-arrays (mapcar (lambda (value aval) (%jit-to-device-argument backend value aval))
                                 arrays avals))
         (results (multiple-value-list (apply #'backend-invoke backend module "main" device-arrays))))
    (%unflatten-results out-treedef (mapcar #'to-host results))))

;;; --- キャッシュ ---
;;;
;;; 2段の表にする（issue #71）。外側の *JIT-CACHE* は TRACEABLE-FUNCTION から
;;; その関数専用の %JIT-FUNCTION-CACHE への弱参照の表で、*JIT-CACHE-LOCK* は
;;; この外側の表の読み書きだけを守る（トレースやコンパイルの間は持たない）。
;;; 関数ごとのキャッシュは自分のロックを持ち、コンパイル中のキーを PENDING に
;;; 記録する。同じキーを後から引いたスレッドはその完了を待ち、別のキー・別の
;;; 関数のコンパイルは並行して進む。
;;;
;;; module の解放: 関数ごとのキャッシュを作るとき、FN に finalizer を登録する。
;;; finalizer は %JIT-FUNCTION-CACHE だけを捕まえ（FN を捕まえると FN が永遠に
;;; 回収されない）、FN が GC されたらキャッシュ中の module をすべて
;;; BACKEND-UNLOAD する。%JIT-CACHE-FORGET（DEFJIT の再定義を含む）は、その場で
;;; 同じことをする。

(defvar *jit-cache* (make-hash-table :test 'eq :weakness :key)
  "TRACEABLE-FUNCTION（jit に渡した FN そのもの）から、その関数の
%JIT-FUNCTION-CACHE への表。:WEAKNESS :KEY なので、FN がどこからも参照
されなくなれば GC がこのエントリごと回収し、FN の finalizer が module を
BACKEND-UNLOAD する。")

(defvar *jit-cache-lock* (sb-thread:make-mutex :name "nabla-jit-cache")
  "*JIT-CACHE*（外側の表）の読み書きと *JIT-MISS-COUNT* の更新だけを守る
ロック。トレース・コンパイルの間は持たない（関数ごとのロックは
%JIT-FUNCTION-CACHE-LOCK）。")

(defvar *jit-miss-count* 0
  "キャッシュミスして実際にトレース・コンパイルした回数（内部の統計。
テストが nb::*jit-miss-count* で読む）。")

(defstruct (%jit-entry (:constructor %make-jit-entry (graph text module out-treedef)))
  "%JIT-FUNCTION-CACHE の ENTRIES に入るエントリ。GRAPH は phase 2 の grad や
診断が必要とするので保持する。OUT-TREEDEF は戻り値（多値）をリストにした木の treedef
で、実行結果の葉から戻り値を作るのに使う（キーには入らない: 同じ入力の形からは
トレースが同じ戻り値の形を作る）。"
  graph
  text
  module
  out-treedef)

(defstruct (%jit-function-cache (:constructor %make-jit-function-cache ()))
  "1つの TRACEABLE-FUNCTION のキャッシュ。ENTRIES はキー（%JIT-CACHE-KEY）から
%JIT-ENTRY への EQUAL の表、PENDING はコンパイル中のキーから、それを
コンパイルしているスレッドへの表。LOCK が ENTRIES と PENDING を守り、
READY はコンパイルが終わる（成功・失敗を問わない）たびに通知される。
FN への参照は持たない（FN の finalizer がこの構造体を捕まえるため）。"
  (lock (sb-thread:make-mutex :name "nabla-jit-function-cache"))
  (ready (sb-thread:make-waitqueue))
  (entries (make-hash-table :test 'equal))
  (pending (make-hash-table :test 'equal)))

(defun %jit-cache-key (backend avals static-values treedef)
  "%JIT-FUNCTION-CACHE の ENTRIES（EQUAL）に使うキーを返す。AVALS は動的引数の
葉の AVAL のリスト（構造体は EQUAL が EQ 相当になってしまうため、(shape dtype) の
リストに正規化する）、TREEDEF は動的引数の木のリストの treedef（#199。同じ aval の
葉でも、木の形が違えば別のエントリにする。treedef の spec は EQUAL で比べられる）。
BACKEND インスタンス自体は EQUAL の中で EQ 比較される。
BACKEND-FINGERPRINT も含めるのは、同じ BACKEND インスタンスでもコンパイル
フラグ等が呼び出しごとに変わりうるため（design tab の「ターゲット（GPU
世代を含む）」の文言）。"
  (list backend
        (backend-fingerprint backend)
        (mapcar (lambda (aval) (list (aval-shape aval) (aval-dtype aval))) avals)
        (treedef-spec treedef)
        static-values))

(defun %jit-unload-entries (cache)
  "CACHE（%JIT-FUNCTION-CACHE）の全エントリの MODULE を、キーに含まれる
BACKEND で BACKEND-UNLOAD してから ENTRIES を空にし、捨てた数を返す。
CACHE のロックを持って呼ぶこと。"
  (let ((entries (%jit-function-cache-entries cache)))
    (prog1 (hash-table-count entries)
      (maphash (lambda (key entry) (backend-unload (first key) (%jit-entry-module entry)))
               entries)
      (clrhash entries))))

(defun %jit-function-cache (fn &key (create t))
  "FN の %JIT-FUNCTION-CACHE を返す。無ければ、CREATE が真なら作って
*JIT-CACHE* に登録し、FN が GC されたときに module を解放する finalizer を
FN に登録する。CREATE が偽なら NIL を返す。"
  (sb-thread:with-mutex (*jit-cache-lock*)
    (or (gethash fn *jit-cache*)
        (when create
          (let ((cache (%make-jit-function-cache)))
            ;; CACHE だけを捕まえる（FN を捕まえない）。finalizer thread で
            ;; 走るので、BACKEND-UNLOAD のエラーでそのスレッドを止めない。
            (trivial-garbage:finalize
             fn (lambda ()
                  (ignore-errors
                   (sb-thread:with-mutex ((%jit-function-cache-lock cache))
                     (%jit-unload-entries cache)))))
            (setf (gethash fn *jit-cache*) cache))))))

(defun %jit-cache-entry-count (fn)
  "FN（jit に渡した TRACEABLE-FUNCTION）が *JIT-CACHE* に持つエントリの数を
返す（無ければ0）。内部テスト用。"
  (let ((cache (%jit-function-cache fn :create nil)))
    (if cache
        (sb-thread:with-mutex ((%jit-function-cache-lock cache))
          (hash-table-count (%jit-function-cache-entries cache)))
        0)))

(defun %jit-cache-forget (fn)
  "FN のキャッシュエントリをすべて捨てる。各エントリの MODULE を、その
キーに含まれる BACKEND で BACKEND-UNLOAD する。捨てたエントリの数を返す。
その時点でコンパイル中のキーは捨てない（コンパイルが終われば普通に載る）。"
  (let ((cache (%jit-function-cache fn :create nil)))
    (if cache
        (sb-thread:with-mutex ((%jit-function-cache-lock cache))
          (%jit-unload-entries cache))
        0)))

;;; --- jit-compile-error（issue #34、wave 4 j2） ---
;;;
;;; %JIT-COMPILE-AND-LOAD は、コンパイル・ロードが BACKEND-ERROR を signal
;;; したときに、その eqn を逆引きしてから JIT-COMPILE-ERROR に変換する
;;; （GRAPH を引数に追加している）。%JIT-CALL は、キャッシュミス時の経路を
;;; RESTART-CASE（USE-EAGER・RECOMPILE）で包む。

(define-condition jit-compile-error (jit-error)
  ((condition :initarg :condition :reader jit-compile-error-condition)
   (graph :initarg :graph :reader jit-compile-error-graph)
   (eqn :initarg :eqn :initform nil :reader jit-compile-error-eqn)
   (eqn-index :initarg :eqn-index :initform nil :reader jit-compile-error-eqn-index))
  (:report
   (lambda (condition stream)
     (format stream "jit: コンパイルに失敗した: ~A~@[~%原因の eqn (~D): ~A ~S -> ~S~]"
             (jit-compile-error-condition condition)
             (jit-compile-error-eqn-index condition)
             (and (jit-compile-error-eqn condition)
                  (primitive-name (eqn-prim (jit-compile-error-eqn condition))))
             (and (jit-compile-error-eqn condition)
                  (mapcar #'var-aval (eqn-invars (jit-compile-error-eqn condition))))
             (and (jit-compile-error-eqn condition)
                  (var-aval (first (eqn-outvars (jit-compile-error-eqn condition))))))))
  (:documentation
   "%JIT-COMPILE-AND-LOAD が BACKEND-COMPILE / BACKEND-LOAD から受け取った
BACKEND-ERROR を包み直したもの。CONDITION は元のコンディション、GRAPH は
コンパイルしようとした graph、EQN・EQN-INDEX は
GRAPH-EQN-FOR-DIAGNOSTIC が (PRINC-TO-STRING CONDITION) から逆引きできた
原因の eqn とその GRAPH-EQNS 中の位置（逆引きできなければ両方 NIL）。

JIT-ERROR のサブタイプなので、この上に USE-EAGER・RECOMPILE の2つの
リスタートが %JIT-CALL の中で使える（restart-case の :REPORT を見る）。"))

(defun %jit-compile-and-load (backend text graph)
  "TEXT を BACKEND-COMPILE してから BACKEND-LOAD した module を返す。
BACKEND-COMPILE・BACKEND-LOAD のどちらかが BACKEND-ERROR を signal したら、
GRAPH-EQN-FOR-DIAGNOSTIC で GRAPH の中の原因の eqn を探し（見つからなければ
NIL・NIL）、JIT-COMPILE-ERROR に変換して signal し直す。"
  (handler-case (backend-load backend (backend-compile backend text))
    (backend-error (c)
      (multiple-value-bind (eqn index) (graph-eqn-for-diagnostic graph (princ-to-string c))
        (error 'jit-compile-error :condition c :graph graph :eqn eqn :eqn-index index)))))

(defun %jit-claim-or-wait (cache key)
  "CACHE（%JIT-FUNCTION-CACHE）で KEY のエントリがあればそれを返す。無ければ、
KEY をコンパイル中の別スレッドがいればその完了を待って引き直し、誰も
コンパイルしていなければ KEY を自分のスレッドのコンパイル中として登録して
NIL を返す（呼び出し元がコンパイルする）。自分のスレッドが同じ KEY を
コンパイル中なら（トレース中の本体がその関数自身を同じ引数の形で呼んだ）、
待つと永遠に終わらないので JIT-ERROR を signal する。

既知の制約: 2つのスレッドが互いのトレース中に相手の関数を呼ぶ（F の
トレースが G を、G のトレースが F を、どちらも具体的な配列で呼ぶ）と、
互いの完了を待ち続ける。このような相互再帰は1スレッドなら上の
JIT-ERROR になる誤りなので、スレッドをまたぐ待ちの循環までは検出しない。"
  (let ((entries (%jit-function-cache-entries cache))
        (pending (%jit-function-cache-pending cache))
        (self sb-thread:*current-thread*))
    (sb-thread:with-mutex ((%jit-function-cache-lock cache))
      (loop
        (let ((entry (gethash key entries)))
          (when entry (return entry)))
        (let ((owner (gethash key pending)))
          (cond ((null owner)
                 (setf (gethash key pending) self)
                 (return nil))
                ((eq owner self)
                 (%jit-error "関数のトレース・コンパイル中に、その関数自身を同じ引数の形で呼んだ"))
                (t
                 (sb-thread:condition-wait (%jit-function-cache-ready cache)
                                           (%jit-function-cache-lock cache)))))))))

(defun %jit-cache-lookup-or-compile (fn key backend graph-thunk)
  "FN のキャッシュから KEY に対応するエントリを引く。無ければ GRAPH-THUNK
（引数無しの関数で、(VALUES GRAPH 戻り値の treedef) を返す）を呼んでトレースし、EMIT-STABLEHLO ->
%JIT-COMPILE-AND-LOAD（BACKEND 上に、GRAPH 付きで診断できる形で）した
結果を新しいエントリとして書き込む。

トレース・コンパイルの間はどのロックも持たない（%JIT-CLAIM-OR-WAIT で
KEY をコンパイル中として登録するだけ）ので、別の関数や別のキーの jit は
並行して進み、トレース中の本体から別の jit した関数を呼べる。コンパイルが
JIT-COMPILE-ERROR などで非局所脱出しても、UNWIND-PROTECT が KEY の登録を
外して待っているスレッドを起こす（起きたスレッドは自分でコンパイルを
やり直す）。"
  (let ((cache (%jit-function-cache fn)))
    (or (%jit-claim-or-wait cache key)
        (let ((entry nil))
          (unwind-protect
               (progn
                 (sb-thread:with-mutex (*jit-cache-lock*)
                   (incf *jit-miss-count*))
                 (multiple-value-bind (graph out-treedef) (funcall graph-thunk)
                   (let* ((text (emit-stablehlo graph))
                          (module (%jit-compile-and-load backend text graph)))
                     (setf entry (%make-jit-entry graph text module out-treedef)))))
            (sb-thread:with-mutex ((%jit-function-cache-lock cache))
              (remhash key (%jit-function-cache-pending cache))
              (when entry
                (setf (gethash key (%jit-function-cache-entries cache)) entry))
              (sb-thread:condition-broadcast (%jit-function-cache-ready cache))))
          entry))))

(defun %jit-eager-fallback (graph leaves out-treedef)
  "GRAPH を LEAVES（動的引数の葉。配列または device array のリスト）に対して
EVAL-GRAPH で評価し、OUT-TREEDEF の形の戻り値（多値）にして返す（USE-EAGER リスタートの
本体。device array は先に TO-HOST してから渡す）。"
  (%unflatten-results
   out-treedef
   (multiple-value-list
    (apply #'eval-graph graph (mapcar (lambda (value) (if (arrayp value) value (to-host value))) leaves)))))

(defun %jit-call (jitted args)
  "JITTED（JITTED-FUNCTION）の呼び出しの糸口。引数の個数を確かめ、静的引数を
切り出し、動的引数（PyTree。#199）を flatten して葉から aval を求め、backend を解決し、
キャッシュ（キーは葉の aval と動的引数の treedef）を引いて（無ければコンパイルして）
%JIT-EXECUTE する。戻り値は FN の戻り値と同じ形の木（多値も同じ）。

動的引数の葉のどれかが TRACER なら（別の関数のトレース中に呼ばれたなら）、
コンパイルせずに FN の本体をそのトレースの中で呼ぶ（呼び出し元の graph に
展開する。JAX の jit の入れ子と同じ）。

キャッシュミスのコンパイルが JIT-COMPILE-ERROR を signal したときのために
2つのリスタートを提供する: USE-EAGER はこの呼び出しだけ、コンパイルに
失敗した GRAPH（この呼び出しが最後にトレースしたもの）を EVAL-GRAPH で評価して返す
（トレースし直さない。何もキャッシュしないので、次の呼び出しはまた同じ
コンパイルを試みる）。RECOMPILE はキャッシュの引き直しからやり直す
（コンパイルが直っていれば今度はキャッシュに載る）。やり直しはループで
行うので、ハンドラが何度 RECOMPILE を選んでもスタックは深く
ならない。

この2つのリスタートは %JIT-CACHE-LOOKUP-OR-COMPILE（キャッシュミスの
トレース・コンパイル経路。JIT-COMPILE-ERROR が起こりうる場所）だけを
囲む。%JIT-EXECUTE（キャッシュヒット後の TO-DEVICE / BACKEND-INVOKE /
TO-HOST）はこの restart-case の外にあるので、実行時のエラーには
USE-EAGER・RECOMPILE のどちらも提供されない（実行時エラーを
EVAL-GRAPH で読み替えたり、実行をやり直したりする意味が無いため）。"
  (let* ((fn (%jitted-function-fn jitted))
         (static-positions (%jitted-function-static-positions jitted))
         (arity (length (traceable-function-lambda-list fn))))
    (unless (= (length args) arity)
      (%jit-error "引数の個数 ~D が関数の引数の個数 ~D と一致しない" (length args) arity))
    (let ((static-values nil)
          (dynamic-values nil))
      (loop for position from 0
            for arg in args
            do (if (member position static-positions)
                   (push arg static-values)
                   (push arg dynamic-values)))
      (setf static-values (nreverse static-values)
            dynamic-values (nreverse dynamic-values))
      (multiple-value-bind (leaves in-treedef) (%flatten-arguments dynamic-values)
        (when (some (lambda (value) (typep value 'tracer)) leaves)
          (return-from %jit-call
            (apply (%traceable-function-function fn)
                   (%jit-merge-args arity static-positions static-values dynamic-values))))
        (let* ((backend (%jit-resolve-backend jitted))
               (leaves (mapcar #'%jit-leaf-array leaves))
               (avals (mapcar #'%jit-argument-aval leaves))
               (key (%jit-cache-key backend avals static-values in-treedef))
               ;; 最後にトレースした graph と戻り値の treedef。コンパイルに失敗したら、
               ;; それが USE-EAGER で評価する graph になる（トレースし直さない）。
               (traced-graph nil)
               (traced-out-treedef nil)
               (graph-thunk (lambda ()
                              (multiple-value-bind (graph out-treedef)
                                  (%jit-trace fn avals static-positions static-values in-treedef)
                                (setf traced-graph graph
                                      traced-out-treedef out-treedef)
                                (values graph out-treedef))))
               (entry (loop
                        (restart-case
                            (return (%jit-cache-lookup-or-compile fn key backend graph-thunk))
                          (use-eager ()
                            :report "この呼び出しだけ eager（eval-graph）で実行する"
                            (return-from %jit-call
                              (progn
                                (unless traced-graph (funcall graph-thunk))
                                (%jit-eager-fallback traced-graph leaves traced-out-treedef))))
                          (recompile ()
                            :report "もう一度コンパイルする"
                            nil)))))
          ;; 実行が終わるまで FN を生かしておく（FN が GC されると、その
          ;; finalizer が実行中の module を BACKEND-UNLOAD してしまう）。
          (sb-sys:with-pinned-objects (fn)
            (%jit-execute backend (%jit-entry-module entry) leaves avals
                          (%jit-entry-out-treedef entry))))))))

;;; --- defjit（issue #34、wave 4 j2） ---

(defun %defjit-install (name traceable static-args)
  "TRACEABLE を STATIC-ARGS 付きで JIT し（STATIC-ARGS が不正ならここで
JIT-ERROR になり、以前の定義はそのまま残る）、NAME（シンボル）の property
list の :%DEFJIT-TRACEABLE に TRACEABLE を記録し、以前そこに
TRACEABLE-FUNCTION があれば %JIT-CACHE-FORGET でその module を先に解放し
（DEFJIT の再定義のたびに、古いキャッシュを使い続けない。design tab の
「Lisp らしさ」）、NAME の FDEFINITION を JIT の結果にする。"
  (let ((jitted (jit traceable :static-args static-args))
        (old (get name '%defjit-traceable)))
    (when old (%jit-cache-forget old))
    (setf (get name '%defjit-traceable) traceable)
    (setf (fdefinition name) jitted)))

(defmacro defjit (name-and-options (&rest lambda-list) &body body)
  "NAME を、BODY を WITH-TRACING でトレース対象にしてから JIT した関数として
定義する（呼び出しのたびに必要なら1回だけコンパイルする通常の関数として
FUNCALL・(NAME ...) の両方で呼べる）。

NAME-AND-OPTIONS は NAME（シンボル）か (NAME &KEY STATIC-ARGS)。
STATIC-ARGS は評価され、JIT の :STATIC-ARGS と同じ意味（LAMBDA-LIST への
0始まりの位置のリスト）になる。例: (defjit (f :static-args '(1)) (a axis) ...)。

DEFJIT を再評価するたびに新しい TRACEABLE-FUNCTION が作られるので、以前の
評価が作ったキャッシュエントリは再利用されず、その module はその場で
解放される（%DEFJIT-INSTALL 参照）。ドキュメント文字列はサポートしない。
使う BACKEND は呼び出し時の *DEFAULT-BACKEND*。"
  (destructuring-bind (name &key static-args)
      (if (listp name-and-options) name-and-options (list name-and-options))
    `(progn
       (declaim (ftype function ,name))
       (%defjit-install ',name (with-tracing ,lambda-list ,@body) ,static-args)
       ',name)))
