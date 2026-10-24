;;;; defmodule: モジュール = パラメータの PyTree ＋ 純粋関数（issue #202）。
;;;;
;;;; 決めたこと:
;;;;
;;;; - モジュールは構造体で、スロットは3種類: :param（学習する葉・サブモジュール・それらの
;;;;   リスト。PyTree の子）、:state（学習しない状態。BatchNorm の統計など。PyTree の子だが
;;;;   MAP-STATE で一括して扱える）、:static（層の幅などのハイパーパラメータ。葉でなく
;;;;   treedef の補助データ。EQUAL で比べられる値に限る）。子の順は宣言順。
;;;; - 構造体は定義した時点で DEFPYTREE に登録される。スロットは DEFMODULE のフォームが
;;;;   すべて知っているので、MOP でなく定義時（ロード時）に作る表（*DEFINED-MODULES*）で引く（継承は持たない）。
;;;;   この表は MODULE-SLOTS として公開する（チェックポイントのパス名 "encoder.layers.0.w"
;;;;   のようなスロット名を後続の issue #206 が組み立てるのに使う）。
;;;; - 呼び出しの名前は CALL（総称関数）。APPLY は CL のシンボルで、CALL は CL に無い
;;;;   （CALL-METHOD / CALL-NEXT-METHOD だけ）ので nabla.nn の中で衝突しない。Flax の apply は
;;;;   「変数を渡して呼ぶ」意味だが、nabla のモジュールは自分がパラメータを持つので (call module x)。
;;;; - init は PRNG キーを受け取って構造体を返す関数 INIT-<name>。サブモジュールへは
;;;;   SPLIT-KEYS で分けたキーを渡す（同じキーからは常に同じパラメータ）。

(in-package #:nabla.nn)

(defvar *defined-modules* (make-hash-table :test 'eq :synchronized t)
  "モジュールのクラス名 → ((スロット名 . 種類) ...)（宣言順）と、コンストラクタの名前の対。")

(defun module-slots (module)
  "MODULE（defmodule で定義したクラス名か、そのインスタンス）のスロットを宣言順の
\(スロット名 . 種類) の新しいリストで返す。種類は :param / :state / :static。defmodule で定義した
モジュールでなければ NIL。チェックポイントのパス名のように、スロット名が要る後続の処理用。"
  (let ((entry (gethash (if (symbolp module) module (class-name (class-of module))) *defined-modules*)))
    (mapcar (lambda (slot) (cons (car slot) (cdr slot))) (car entry))))

(defun split-keys (key n)
  "PRNG キー KEY から独立な N 個のキー（それぞれ :u32 の (2)）のリストを作る。
\(nabla:split key n) の行を取り出したもので、同じ KEY・N からは常に同じ。eager 専用
\(モジュールの init はホストで実行する)。KEY を split-keys したら元の KEY では乱数を引かない。"
  (let ((rows (nabla:split key n)))
    (loop for i below n
          collect (make-array 2 :element-type '(unsigned-byte 32)
                                :initial-contents (list (aref rows i 0) (aref rows i 1))))))

(defun %module-entry (module)
  (gethash (class-name (class-of module)) *defined-modules*))

(defun %module-flatten (module)
  "DEFPYTREE の FLATTEN: 子は :param / :state のスロットの値（宣言順）、補助データは :static の値。"
  (let ((slots (car (%module-entry module))))
    (values (loop for (name . kind) in slots
                  unless (eq kind :static) collect (slot-value module name))
            (loop for (name . kind) in slots
                  when (eq kind :static) collect (slot-value module name)))))

(defun %module-rebuild (type function)
  "TYPE のコンストラクタを、スロット名ごとに (FUNCTION 名 種類) の値で呼ぶ。"
  (destructuring-bind (slots . constructor) (gethash type *defined-modules*)
    (apply constructor
           (loop for (name . kind) in slots
                 append (list (intern (symbol-name name) :keyword) (funcall function name kind))))))

(defun %module-unflatten (type aux children)
  (%module-rebuild type (lambda (name kind)
                          (declare (ignore name))
                          (if (eq kind :static) (pop aux) (pop children)))))

(defun %map-registered-children (function tree &rest more-trees)
  "TREE が defmodule でない defpytree の節なら、直下の子（と MORE-TREES の同じ位置の子）に FUNCTION をかけた
子で組み立て直した同じ形の節を返す（補助データはそのまま）。葉なら TREE そのもの。MAP-STATE と
APPLY-UPDATES が、自前の構造体で包んだモデルの中の :state まで歩くのに使う。"
  (multiple-value-bind (children aux node-p) (nabla:tree-children tree)
    (declare (ignore aux))
    (if node-p
        (let ((new-children (apply #'mapcar function children
                                   (mapcar (lambda (other) (nth-value 0 (nabla:tree-children other))) more-trees))))
          ;; 子の形は変わらないので、元の treedef に新しい子の葉を並べて戻す
          (nabla:tree-unflatten (nabla:tree-structure tree) (mapcan #'nabla:tree-leaves new-children)))
        tree)))

(defun map-state (function tree)
  "TREE（モジュールを入れ子にしたリスト・ベクタ・モジュール）の :state のスロットの葉すべてに
FUNCTION をかけた同じ形の木を返す。リスト・simple-vector・defmodule のモジュールに加えて、defpytree で
登録した別の節（モデルを包む自前の構造体など）の子もたどる（issue #260）。:param と :static のスロットは
（サブモジュールの中の :state を除いて）そのまま共有する。FUNCTION は葉を葉に写す（形を変える関数は
登録した節の中では組み立て直せない）。学習しない状態を grad から外すのに使う:
\(map-state #'nabla:stop-gradient model)。"
  (cond ((listp tree) (mapcar (lambda (child) (map-state function child)) tree))
        ((simple-vector-p tree)
         (map 'simple-vector (lambda (child) (map-state function child)) tree))
        ((%module-entry tree)
         (%module-rebuild (class-name (class-of tree))
                          (lambda (name kind)
                            (let ((value (slot-value tree name)))
                              (case kind
                                (:state (nabla:tree-map function value))
                                (:param (map-state function value))
                                (t value))))))
        (t (%map-registered-children (lambda (child) (map-state function child)) tree))))

(defgeneric call (module &rest arguments)
  (:documentation "MODULE を ARGUMENTS に適用する。defmodule の (:call (self ...) body) が
メソッドを定義する。モジュールは自分のパラメータを持つ純粋関数で、状態を書き換えない。"))

(defun %parse-module-slots (name slot-specs)
  (let ((slots (loop for spec in slot-specs
                     for (slot kind) = (let ((spec (if (consp spec) spec (list spec))))
                                         (if (cdr spec) spec (list (first spec) :param)))
                     do (unless (and (symbolp slot) slot (member kind '(:param :state :static))
                                     (or (atom spec) (<= (length spec) 2)))
                          (error "defmodule ~S: スロットは 名前 か (名前 [:param|:state|:static]): ~S" name spec))
                     collect (cons slot kind))))
    (let ((names (mapcar #'car slots)))
      (unless (= (length names) (length (remove-duplicates names)))
        (error "defmodule ~S: スロット名が重複している: ~S" name names)))
    slots))

(defmacro defmodule (name slot-specs &rest clauses)
  "モジュール NAME を定義する。SLOT-SPECS は (スロット名 種類) のリスト（種類は :param
\(学習するパラメータ・サブモジュール・そのリスト。省略可。`(w)` も同じ)、:state（学習しない状態）、
:static（層の幅などのハイパーパラメータ。EQUAL で比べられる値に限り、treedef の補助データに入る））。

生成するもの:
- 構造体 NAME とコンストラクタ MAKE-NAME（スロットのキーワード引数、すべて必須）、
  アクセサ NAME-<slot>。構造体は定義した時点で PyTree に登録される。葉は :param / :state の
  スロットの宣言順で、サブモジュールは入れ子の節になる。
- (:init (key . lambda-list) body...) があれば関数 INIT-NAME。先頭の引数は PRNG キーで、本体は
  モジュールを返す。サブモジュールへは SPLIT-KEYS で分けたキーを渡す。
- (:call (self . lambda-list) body...) があれば CALL のメソッド。lambda-list は必須引数の
  シンボルだけ、本体は nabla:with-tracing の本体として書く（コードウォークされ、CL の + * などが
  配列にもトレーサにも使える。setq は禁止。SELF は閉包で捕まえる）。このため grad / vmap / jit の
  トレースの中で呼べる。同じ defmodule フォームを再評価せずに構造体だけ再定義すると CALL の
  メソッドは古いクラスのまま残るので、フォーム全体を評価し直すこと。
- 述語 NAME-P と複写関数は作らない（スロット名 P とアクセサが衝突しないため）。
- :static の値は EQUAL で比べられるものに限る（関数は不可。活性化関数はシンボルで持つ。
  1e-5 と 1d-5 は別の treedef になる）。
- (:init ...) と (:call ...) の節はそれぞれ1つまで。

この形以外の節は展開時にエラーにする。"
  (let ((slots (%parse-module-slots name slot-specs))
        (constructor (intern (concatenate 'string "MAKE-" (symbol-name name)) *package*))
        (init-name (intern (concatenate 'string "INIT-" (symbol-name name)) *package*)))
    (let ((heads (mapcar #'first clauses)))
      (unless (= (length heads) (length (remove-duplicates heads)))
        (error "defmodule ~S: (:init ...) と (:call ...) の節はそれぞれ1つまで: ~S" name heads)))
    `(progn
       (defstruct (,name (:constructor ,constructor
                             (&key ,@(loop for (slot) in slots
                                           collect `(,slot (error "~S: ~S は必須" ',constructor ',slot)))))
                         (:copier nil) (:predicate nil))
         ,@(mapcar #'car slots))
       (setf (gethash ',name *defined-modules*) (cons ',slots #',constructor))
       (nabla:defpytree ,name
         :flatten #'%module-flatten
         :unflatten (lambda (aux children) (%module-unflatten ',name aux children)))
       ,@(loop for clause in clauses
               collect (destructuring-bind (&optional head ll &rest body) clause
                         (unless (and (member head '(:init :call)) (consp ll) (symbolp (first ll)))
                           (error "defmodule ~S: 節は (:init (key ...) body) か (:call (self ...) body): ~S"
                                  name clause))
                         (case head
                           (:init `(defun ,init-name ,ll ,@body))
                           ;; 本体は with-tracing の本体と同じ（CL の + * などがトレーサと配列に使える）
                           (:call (let ((arguments (gensym "ARGUMENTS")))
                                    `(defmethod call ((,(first ll) ,name) &rest ,arguments)
                                       (apply (nabla:with-tracing ,(rest ll) ,@body) ,arguments))))
                           (t (error "defmodule ~S: 未知の節: ~S" name clause)))))
       ',name)))
