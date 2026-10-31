;;;; pytree: PyTree（リスト・ベクタ・登録した型を入れ子にした木）の flatten / unflatten
;;;; （issue #195）。
;;;;
;;;; 決めたこと:
;;;;
;;;; - 既定の節はリスト（真リスト）と SIMPLE-VECTOR（要素型 T の1次元の単純な配列）だけ。
;;;;   NIL は子の無いリスト、つまり空の部分木で、葉にはならない。型特化した配列
;;;;   （(simple-array single-float (3)) などの1次元の配列も）・文字列・数値・トレーサは葉
;;;;   （nabla の配列は要素型 T を使わないので、配列の葉と節が紛れない）。
;;;; - 登録の無い型はすべて葉（JAX と同じ）。plist / alist / ハッシュ表や構造体を節にするには
;;;;   DEFPYTREE で明示的に登録する（(1 2 3) が葉のリストか plist かはコンスの型からは
;;;;   判別できないため。設計タブ）。登録はクラス名で引き、サブクラスには継承しない。
;;;; - 真リストでないコンス（(1 . 2) など）は PYTREE-ERROR。
;;;; - treedef は木の形を EQUAL で比べられるリスト（下の「spec」）で持つ。後続の jit の
;;;;   キャッシュのキーに入れるため（#199）。登録した型の補助データも spec に入るので、
;;;;   EQUAL で比べられる値（数値・文字・シンボル・文字列・パス名と、それらのコンス）に限る。
;;;;
;;;; 節の判定は %NODE-PARTS の1か所に集め、TREE-FLATTEN と公開の TREE-CHILDREN（#239。
;;;; PyTree の構造を内部表現を知らずに歩きたい nabla/nn のチェックポイント用）が共有する
;;;; （vmap の軸の木は NIL を葉とする別の規則で、これは使わない）。
;;;;
;;;; 登録した型の FLATTEN は、3つ目の値として子の名前（文字列のリスト）を返せる（#285 の前提）。
;;;; 名前は TREE-CHILDREN の4つ目の値で外に見せるだけで、spec（treedef）には入れない（jit の
;;;; キャッシュのキーも変わらない）。nabla/nn のチェックポイントが添字の代わりにパス名に使う
;;;; ので、nabla/data のように defmodule（nn）を使えない層でも、子の並びに依らないパス名を
;;;; 持てる。名前の誤りは定義の誤りとして、使うかどうかに依らず TREE-FLATTEN でも PYTREE-ERROR。
;;;;
;;;; spec の形（内部。export しない）:
;;;;   :leaf                       葉
;;;;   (:list child...)            リスト（NIL は (:list)）
;;;;   (:vector child...)          SIMPLE-VECTOR
;;;;   (:node type aux child...)   DEFPYTREE で登録した型 TYPE（補助データ AUX）

(in-package #:nabla)

(define-condition pytree-error (error)
  ((format-control :initarg :format-control :initform "" :reader pytree-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil
                     :reader pytree-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "PyTree エラー: ~?"
             (pytree-error-format-control condition)
             (pytree-error-format-arguments condition))))
  (:documentation
   "PyTree の操作（TREE-FLATTEN / TREE-UNFLATTEN / TREE-MAP / DEFPYTREE）に不正な木や引数
（真リストでないコンス、葉の個数が treedef と合わない、組み込みの型の再登録、EQUAL で
比べられない補助データなど）を渡したときに signal する。"))

(define-condition pytree-structure-mismatch (pytree-error)
  ()
  (:documentation
   "同じ構造であるべき木（TREE-MAP の複数の木など）の treedef が違うときに signal する。
PYTREE-ERROR の子。"))

(defun %pytree-error (control &rest arguments)
  (error 'pytree-error :format-control control :format-arguments arguments))

;;; --- treedef ---

(defstruct (treedef (:constructor %make-treedef (spec num-leaves))
                    (:copier nil)
                    (:predicate nil))
  "PyTree の形（葉を除いた構造）。TREE-FLATTEN / TREE-STRUCTURE が作り、TREE-UNFLATTEN が
葉を戻すのに使う。比較は TREEDEF=。"
  (spec :leaf :read-only t)
  (num-leaves 0 :type (integer 0) :read-only t))

(setf (documentation 'treedef-num-leaves 'function)
      "TREEDEF が持つ葉の個数（TREE-UNFLATTEN に渡す葉のリストの長さ）。")

(defun treedef= (a b)
  "treedef A と B が同じ形（節の種類・子の個数・登録した型とその補助データ）なら真。"
  (check-type a treedef)
  (check-type b treedef)
  (equal (treedef-spec a) (treedef-spec b)))

(defun %write-spec (spec stream)
  "SPEC を人が読む形で STREAM に書く（葉は *、リストは (..)、ベクタは #(..)、登録した型は
#<型 補助データ ..>）。PRINT-OBJECT とエラーメッセージ用。"
  (flet ((children (open specs)
           (write-string open stream)
           (loop for (child . more) on specs
                 do (%write-spec child stream)
                    (when more (write-char #\Space stream)))
           (write-char #\) stream)))
    (if (eq spec :leaf)
        (write-char #\* stream)
        (ecase (first spec)
          (:list (children "(" (rest spec)))
          (:vector (children "#(" (rest spec)))
          (:node (destructuring-bind (type aux &rest specs) (rest spec)
                   (format stream "#<~S ~S" type aux)
                   (loop for child in specs
                         do (write-char #\Space stream)
                            (%write-spec child stream))
                   (write-char #\> stream)))))))

(defmethod print-object ((treedef treedef) stream)
  (print-unreadable-object (treedef stream :type t)
    (%write-spec (treedef-spec treedef) stream)))

;;; --- 登録 ---

(defvar *pytree-node-types* (make-hash-table :test 'eq :synchronized t)
  "DEFPYTREE で登録した型名 → (FLATTEN . UNFLATTEN)。")

(defun %equal-comparable-p (value)
  "VALUE が EQUAL で中身を比べられる値（数値・文字・シンボル・文字列・ビットベクタ・
パス名と、それらのコンス）なら真。"
  (typecase value
    (cons (and (%equal-comparable-p (car value)) (%equal-comparable-p (cdr value))))
    ((or number character symbol string bit-vector pathname) t)
    (t nil)))

(defun %register-pytree-node (type flatten unflatten)
  (unless (and (symbolp type) type (find-class type nil))
    (%pytree-error "DEFPYTREE の型はクラス名でなければならない: ~S" type))
  ;; 葉と決めた型（数値・配列・文字列・トレーサ・treedef など）を節として登録できると、
  ;; どの木の中のその値も節になってしまう（登録は大域的）ので拒む
  (when (or (eq type t)
            (subtypep type '(or sequence array number character symbol function condition
                             tracer treedef)))
    (%pytree-error "組み込みの型・葉と決めた型 ~S は DEFPYTREE で登録できない（リストと SIMPLE-VECTOR は既定の節）" type))
  (unless (and (functionp flatten) (functionp unflatten))
    (%pytree-error "DEFPYTREE の :FLATTEN と :UNFLATTEN は関数でなければならない: ~S ~S"
                   flatten unflatten))
  (setf (gethash type *pytree-node-types*) (cons flatten unflatten))
  type)

(defmacro defpytree (type &key flatten unflatten)
  "クラス名 TYPE のオブジェクトを PyTree の節として登録する（Flax / JAX の
register_pytree_node に相当）。TYPE の直接のインスタンスだけが節になり、サブクラスには
継承しない。同じ TYPE を再び登録すると上書きする。

FLATTEN は (lambda (object) ...) で、子のリストと補助データの2値を返す。子は flatten の
順に葉が並ぶ部分木。補助データは treedef に入る静的な値で、EQUAL で比べられるもの
（数値・文字・シンボル・文字列・パス名と、それらのコンス）でなければならない（そうで
なければ TREE-FLATTEN が PYTREE-ERROR）。
FLATTEN は3つ目の値として子の名前のリストを返してもよい（返さないか NIL なら名前なし）。
名前は子と同じ個数の、空でなく \".\" を含まない、互いに異なる文字列でなければならず、
そうでなければ TREE-FLATTEN / TREE-CHILDREN が PYTREE-ERROR。名前は TREE-CHILDREN の
4つ目の値になり（nabla/nn のチェックポイントは添字の代わりにこれをパス名に使う）、treedef には
入らない（名前だけが違っても同じ形）。そのため名前は型と補助データだけで決まらなければならない
（UNFLATTEN で作り直した節も同じ名前を返す。フィールド名の定数のリストにする）。
UNFLATTEN は (lambda (aux children) ...) で、補助データと子のリストから TYPE の
オブジェクトを作る。

リスト・SIMPLE-VECTOR などの組み込みの型と、葉と決めた型（数値・配列・文字列・
シンボル・関数・コンディション・トレーサ・treedef）は登録できない（PYTREE-ERROR）。
FLATTEN が真リストでない子を返したときも PYTREE-ERROR。"
  `(%register-pytree-node ',type ,flatten ,unflatten))

(defun %pytree-node-entry (object)
  (gethash (class-name (class-of object)) *pytree-node-types*))

;;; --- flatten / unflatten ---

(defun %proper-list-p (object)
  "OBJECT が真リスト（NIL で終わり、循環しない）なら真。"
  (and (listp object) (handler-case (list-length object) (type-error () nil)) t))

(defun %check-child-names (type names children)
  "登録した型 TYPE の FLATTEN が返した子の名前 NAMES を検査する（NIL は名前なしで常によい）。"
  (when names
    (unless (%proper-list-p names)
      ;; 名前そのものは印字しない（循環リストの印字は終わらない）
      (%pytree-error "~S の FLATTEN が返した子の名前は真リストでなければならない" type))
    (unless (= (length names) (length children))
      (%pytree-error "~S の FLATTEN が返した子の名前 ~S は、子（~D 個）と同じ個数でなければならない"
                     type names (length children)))
    (loop for (name . rest) on names
          do (unless (and (stringp name) (plusp (length name)) (not (find #\. name)))
               (%pytree-error "~S の子の名前 ~S は、空でなく \".\" を含まない文字列でなければならない"
                              type name))
             (when (member name rest :test #'equal)
               (%pytree-error "~S の子の名前 ~S が重複している" type name)))))

(defun %node-parts (tree)
  "TREE が節なら (VALUES 子のリスト 補助データ 真 種類 子の名前)、葉なら (VALUES NIL NIL NIL NIL NIL)。
種類は :LIST・:VECTOR・登録した型名。子の名前は登録した型の FLATTEN の3つ目の値（無ければ NIL）。
TREE-FLATTEN と TREE-CHILDREN が共有する節の判定（vmap の軸の木は NIL を葉とする別の規則）。
真リストでないコンスと、不正な FLATTEN の結果（子の名前を含む）は PYTREE-ERROR。"
  (cond
    ((listp tree)
     (unless (%proper-list-p tree)
       ;; 木そのものは印字しない（循環リストの印字は終わらない）
       (%pytree-error "真リストでない（ドット対や循環を含む）コンスは PyTree にできない"))
     (values tree nil t :list))
    ((simple-vector-p tree)
     (values (coerce tree 'list) nil t :vector))
    (t
     (let ((entry (%pytree-node-entry tree)))
       (if (null entry)
           (values nil nil nil nil nil)
           (multiple-value-bind (children aux names) (funcall (car entry) tree)
             (let ((type (class-name (class-of tree))))
               (unless (%proper-list-p children)
                 (%pytree-error "~S の FLATTEN が子のリストを返さなかった: ~S" type children))
               (unless (%equal-comparable-p aux)
                 (%pytree-error "~S の補助データは EQUAL で比べられる値でなければならない: ~S"
                                type aux))
               (%check-child-names type names children)
               (values children aux t type names))))))))

(defun tree-children (object)
  "OBJECT が PyTree の節なら (VALUES 子のリスト 補助データ T 子の名前)、葉なら (VALUES NIL NIL NIL NIL)。
節の判定は TREE-FLATTEN と完全に同じ（リスト・SIMPLE-VECTOR・DEFPYTREE で登録した型が節、
それ以外は葉）。子は TREE-FLATTEN が葉を集める順の直下の要素で、これを再帰でたどった葉の
列は TREE-LEAVES と一致する。補助データは treedef に入る静的な値（DEFPYTREE の FLATTEN の2つ目の
値。リストと SIMPLE-VECTOR は NIL）。子が0個の節（空のベクタ・子を返さない登録型）と葉は、
3つ目の値で区別する。NIL は TREE-FLATTEN と同じく子の無いリストの節（空の部分木）なので
(VALUES NIL NIL T NIL)。子の名前は DEFPYTREE の FLATTEN が3つ目の値で返した文字列のリスト
（子と同じ順。チェックポイントのパス名用で、treedef には入らない。FLATTEN の定数のリストそのもの
でありうるので、破壊的に変更しない）で、名前を返さない登録型とリスト・SIMPLE-VECTOR は NIL。
真リストでないコンスや不正な FLATTEN（不正な子の名前を含む）は TREE-FLATTEN と同じく PYTREE-ERROR。木の中身（型ごとの内部表現）を知らずに木の構造を歩くための公開 API。"
  (multiple-value-bind (children aux node-p kind names) (%node-parts object)
    (declare (ignore kind))
    (values children aux node-p names)))

(defun %flatten-into (tree collect)
  "TREE の spec を返す。葉は左から順に COLLECT（1引数の関数）に渡す。"
  (multiple-value-bind (children aux node-p kind) (%node-parts tree)
    (if (not node-p)
        (progn (funcall collect tree) :leaf)
        (let ((specs (mapcar (lambda (child) (%flatten-into child collect)) children)))
          (case kind
            ((:list :vector) (cons kind specs))
            (t (list* :node kind aux specs)))))))

(defun tree-flatten (tree)
  "TREE の葉を左から順に並べたリストと、TREE の形（treedef）の2値を返す。
リストと SIMPLE-VECTOR は先頭から、DEFPYTREE で登録した型は FLATTEN が返す子の順に
たどる。NIL は空の部分木で葉にならない。登録の無い型のオブジェクト（配列・数値・
トレーサなど）は葉。真リストでないコンスは PYTREE-ERROR。"
  (let* ((leaves '())
         (spec (%flatten-into tree (lambda (leaf) (push leaf leaves))))
         (ordered (reverse leaves)))
    (values ordered (%make-treedef spec (length ordered)))))

(defun tree-leaves (tree)
  "TREE の葉を、TREE-FLATTEN と同じ順に並べたリスト。"
  (values (tree-flatten tree)))

(defun tree-structure (tree)
  "TREE の形（treedef）。TREE-FLATTEN の2つ目の値。"
  (nth-value 1 (tree-flatten tree)))

(defun %unflatten-spec (spec leaves)
  "SPEC の木を LEAVES（リスト）の先頭から葉を取って作る。木と、残りの葉の2値を返す。"
  (if (eq spec :leaf)
      (values (first leaves) (rest leaves))
      (flet ((children (specs)
               (loop for child in specs
                     collect (multiple-value-bind (subtree rest) (%unflatten-spec child leaves)
                               (setf leaves rest)
                               subtree))))
        (ecase (first spec)
          (:list (let ((tree (children (rest spec)))) (values tree leaves)))
          (:vector (let ((tree (coerce (children (rest spec)) 'simple-vector)))
                     (values tree leaves)))
          (:node
           (destructuring-bind (type aux &rest specs) (rest spec)
             ;; spec の型は TREE-FLATTEN の時点で登録されていた（登録を消す API は無い）
             (let ((tree (funcall (cdr (gethash type *pytree-node-types*)) aux (children specs))))
               (values tree leaves))))))))

(defun tree-unflatten (treedef leaves)
  "TREEDEF の形の木を、葉のリスト LEAVES から作る（TREE-FLATTEN の逆）。LEAVES の長さが
(TREEDEF-NUM-LEAVES TREEDEF) と違えば PYTREE-ERROR。リストと SIMPLE-VECTOR の節は
新しく作り、登録した型の節はその UNFLATTEN で作る。"
  (check-type treedef treedef)
  (unless (and (listp leaves) (null (cdr (last leaves))))
    (%pytree-error "TREE-UNFLATTEN の葉は真リストでなければならない: ~S" leaves))
  (unless (= (length leaves) (treedef-num-leaves treedef))
    (%pytree-error "葉の個数 ~D が treedef ~S の葉の個数 ~D と合わない"
                   (length leaves) treedef (treedef-num-leaves treedef)))
  (values (%unflatten-spec (treedef-spec treedef) leaves)))

(defun tree-map (function tree &rest more-trees)
  "TREE の葉ごとに FUNCTION を呼んだ結果を葉とする、TREE と同じ形の木を返す。
MORE-TREES を渡すと、FUNCTION は各木の同じ位置の葉を引数に呼ばれる（MORE-TREES は
TREE と同じ treedef でなければならず、違えば PYTREE-STRUCTURE-MISMATCH）。
FUNCTION は葉の数だけ、TREE-FLATTEN の葉の順に呼ばれる。"
  (multiple-value-bind (leaves treedef) (tree-flatten tree)
    (let ((more-leaves
            (loop for other in more-trees
                  collect (multiple-value-bind (other-leaves other-treedef) (tree-flatten other)
                            (unless (treedef= treedef other-treedef)
                              (error 'pytree-structure-mismatch
                                     :format-control "TREE-MAP の木の形が違う: ~S と ~S"
                                     :format-arguments (list treedef other-treedef)))
                            other-leaves))))
      (tree-unflatten treedef
                      ;; MORE-LEAVES は自分で作ったリストなので、先頭から POP して進める。
                      (loop for leaf in leaves
                            collect (apply function leaf
                                           (loop for cell on more-leaves
                                                 collect (pop (car cell)))))))))

;;; --- 関数の引数・戻り値を PyTree にする部品（内部。export しない） ---
;;;
;;; jit / grad / vmap / 制御構造（#199〜#201）が共有する。方針: 木を受けて木を返す関数 F を、
;;; 引数の木を flatten して葉のリストにし、葉だけを受けて葉だけを返す関数（今までのトレースの
;;; 経路がそのまま使える形）に変えてトレースする。F の戻り値（多値）は「値のリスト」という
;;; 1つの木として flatten するので、多値で返す関数も、木を1つ返す関数も同じ扱いになる
;;; （(values a b) と (list a b) は treedef が違う: 前者は (:list * *)、後者は (:list (:list * *))）。

(defun %flatten-arguments (args)
  "引数の木のリスト ARGS を、(VALUES 葉のリスト treedef) にする（TREE-FLATTEN と同じ。
名前で意図を示すためのもの）。葉のリストは引数を先頭から順に flatten して連結したもの、
treedef は ARGS 全体の形（jit のキャッシュのキーに入れる）。treedef に ARGS の個数も
入っているので、個数の違う呼び出しとは区別される。引数ごとの葉の個数が要るときは
(LENGTH (TREE-LEAVES ARG)) を引数ごとに求める。"
  (tree-flatten args))

(defun %flatten-function (fn in-treedef)
  "FN（引数の木を受け取る関数）を、葉だけを受け取る関数にして返す。IN-TREEDEF は引数の木の
リストの treedef（%FLATTEN-ARGUMENTS の2つ目の値）。返す関数は葉を IN-TREEDEF で引数に
戻して FN を呼び、FN の戻り値（多値）をリストにして flatten した葉を多値で返す。

2つ目の値は引数なしの関数で、返す関数が最後に走ったときの戻り値の treedef（値のリストの
形。まだ走っていなければ NIL）を返す。トレースの前後で読む。"
  (let ((out-treedef nil))
    (values
     (lambda (&rest leaves)
       (multiple-value-bind (out-leaves treedef)
           (tree-flatten (multiple-value-list (apply fn (tree-unflatten in-treedef leaves))))
         (setf out-treedef treedef)
         (values-list out-leaves)))
     (lambda () out-treedef))))

(defun %unflatten-results (out-treedef leaves)
  "%FLATTEN-FUNCTION の戻り値の treedef OUT-TREEDEF と葉のリスト LEAVES から、元の関数の
戻り値を作って多値で返す（値が1つなら1つの木、複数なら多値、0個なら (VALUES)）。"
  (values-list (tree-unflatten out-treedef leaves)))
