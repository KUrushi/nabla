;;;; PyTree の公開 API（defpytree / tree-flatten / tree-unflatten / tree-leaves /
;;;; tree-structure / tree-map / treedef= / tree-children。issue #195・#239、small）。
;;;;
;;;; 守らせる性質:
;;;; - 往復: (tree-unflatten (tree-structure x) (tree-leaves x)) は x と同じ形・同じ葉。
;;;; - 葉の順: 節の葉は、子の葉を先頭から順に連結したもの（リスト・ベクタ・登録した型）。
;;;;   NIL は空の部分木で葉を持たない。
;;;; - tree-map: 形を保ち、f を葉の数だけ flatten の順に呼ぶ。複数の木は葉ごとに対応させ、
;;;;   形の違う木は PYTREE-STRUCTURE-MISMATCH。
;;;; - treedef=: 葉だけを変えた木とは等しく、形を変えた木とは等しくない。
;;;; - 葉: 型特化した配列・文字列・登録の無い構造体は、そのまま（EQ で）葉になる。
;;;; - 不正な入力（真リストでないコンス、葉の個数違い、組み込みの型の登録、EQUAL で比べ
;;;;   られない補助データ）は PYTREE-ERROR。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defmacro def-pytree-property (name docstring (&rest vars) gen &body body)
  "GEN の値を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test ,name ,docstring
     (is (check-it ,gen
                   (lambda (value) (destructuring-bind ,vars value ,@body))
                   :regression-id ,name
                   :regression-file (regression-path
                                     ,(format nil "pytree-~(~A~)"
                                              (substitute #\- #\/ (subseq (string name)
                                                                          (length "pytree/"))))))
         ,(format nil "~A の性質が成り立たなかった" name))))

;;; --- 往復と葉の順 ---

(def-pytree-property pytree/flatten-roundtrip
  "unflatten(structure(x), leaves(x)) は x と同じ形・同じ葉になる。"
  (tree) (generator (tuple (pytree-spec)))
  (multiple-value-bind (leaves treedef) (nb:tree-flatten tree)
    (and (= (length leaves) (nb:treedef-num-leaves treedef))
         (pytree-equal tree (nb:tree-unflatten treedef leaves)))))

(def-pytree-property pytree/leaves-and-structure-agree-with-flatten
  "tree-leaves と tree-structure は tree-flatten の2つの値と同じ。"
  (tree) (generator (tuple (pytree-spec)))
  (multiple-value-bind (leaves treedef) (nb:tree-flatten tree)
    (and (equal leaves (nb:tree-leaves tree))
         (nb:treedef= treedef (nb:tree-structure tree)))))

(def-pytree-property pytree/leaves-concatenate-over-children
  "リスト・ベクタ・登録した型の葉は、子の葉を先頭から順に連結したもの。"
  (a b tag) (generator (tuple (pytree-spec) (pytree-spec) (uniform-integer :lo 0 :hi 2)))
  (let ((expected (append (nb:tree-leaves a) (nb:tree-leaves b))))
    (and (equal expected (nb:tree-leaves (list a b)))
         (equal expected (nb:tree-leaves (vector a b)))
         (equal expected (nb:tree-leaves (make-pytree-test-pair a b tag))))))

(def-pytree-property pytree/nil-is-an-empty-subtree
  "NIL は葉を持たない部分木で、往復で NIL に戻る。"
  (tree) (generator (tuple (pytree-spec)))
  (let ((padded (list nil tree (vector nil))))
    (and (equal (nb:tree-leaves tree) (nb:tree-leaves padded))
         (pytree-equal padded (nb:tree-unflatten (nb:tree-structure padded)
                                                 (nb:tree-leaves padded))))))

(def-pytree-property pytree/arrays-strings-and-unregistered-objects-are-leaves
  "型特化した配列・文字列・登録の無い構造体はそのまま葉になる（中をたどらない）。"
  (spec) (generator (tuple (array-spec)))
  (let* ((array (make-random-array spec))
         (string "abc")
         (object (nb:make-aval '(2) :f32))
         (leaves (nb:tree-leaves (list array (vector string) object))))
    (and (= 3 (length leaves))
         (eq array (first leaves))
         (eq string (second leaves))
         (eq object (third leaves)))))

;;; --- tree-map ---

(def-pytree-property pytree/tree-map-identity-preserves-the-tree
  "tree-map #'identity は形と葉を保つ。"
  (tree) (generator (tuple (pytree-spec)))
  (pytree-equal tree (nb:tree-map #'identity tree)))

(def-pytree-property pytree/tree-map-calls-function-in-flatten-order
  "tree-map の f は葉の数だけ、tree-flatten の葉の順に呼ばれ、結果は同じ形の木になる。"
  (tree) (generator (tuple (pytree-spec)))
  (let* ((seen '())
         (mapped (nb:tree-map (lambda (leaf) (push leaf seen) (* 2 leaf)) tree)))
    (and (equal (nb:tree-leaves tree) (reverse seen))
         (equal (mapcar (lambda (leaf) (* 2 leaf)) (nb:tree-leaves tree))
                (nb:tree-leaves mapped))
         (nb:treedef= (nb:tree-structure tree) (nb:tree-structure mapped)))))

(def-pytree-property pytree/tree-map-pairs-leaves-of-several-trees
  "複数の木の tree-map は同じ位置の葉を引数にする。"
  (tree) (generator (tuple (pytree-spec)))
  (flet ((combine (a b) (+ (* 1000 a) b)))
    (let ((shifted (nb:tree-map #'1+ tree)))
      (equal (mapcar #'combine (nb:tree-leaves tree) (nb:tree-leaves shifted))
             (nb:tree-leaves (nb:tree-map #'combine tree shifted))))))

(def-pytree-property pytree/tree-map-rejects-trees-of-different-structure
  "形の違う木を tree-map に渡すと PYTREE-STRUCTURE-MISMATCH。"
  (tree) (generator (tuple (pytree-spec)))
  (handler-case (progn (nb:tree-map #'+ tree (list tree)) nil)
    (nb:pytree-structure-mismatch () t)))

;;; --- treedef= ---

(def-pytree-property pytree/treedef=-ignores-leaves-and-sees-structure
  "葉だけを変えた木とは treedef= が真、包んだ木・別の TAG の木とは偽。"
  (tree tag) (generator (tuple (pytree-spec) (uniform-integer :lo 0 :hi 2)))
  (let ((structure (nb:tree-structure tree)))
    (and (nb:treedef= structure (nb:tree-structure (nb:tree-map #'1+ tree)))
         (not (nb:treedef= structure (nb:tree-structure (list tree))))
         (not (nb:treedef= structure (nb:tree-structure (vector tree))))
         (not (nb:treedef= (nb:tree-structure (make-pytree-test-pair tree nil tag))
                           (nb:tree-structure (make-pytree-test-pair tree nil (1+ tag))))))))

;;; --- 不正な入力 ---

(def-pytree-property pytree/unflatten-rejects-a-wrong-number-of-leaves
  "葉の個数が treedef と違うと tree-unflatten が PYTREE-ERROR。"
  (tree) (generator (tuple (pytree-spec)))
  (multiple-value-bind (leaves treedef) (nb:tree-flatten tree)
    (flet ((rejects-p (wrong)
             (handler-case (progn (nb:tree-unflatten treedef wrong) nil)
               (nb:pytree-error () t))))
      (and (rejects-p (cons 0 leaves))
           (or (null leaves) (rejects-p (rest leaves)))))))

(test pytree/invalid-input-is-reported
  "真リストでないコンス、組み込みの型の登録、EQUAL で比べられない補助データは PYTREE-ERROR。"
  (signals nb:pytree-error (nb:tree-flatten '(1 . 2)))
  (signals nb:pytree-error (nb:tree-flatten (list 1 (list* 2 3 4))))
  (signals nb:pytree-error (nb:tree-unflatten (nb:tree-structure '(1 2)) '(1 . 2)))
  ;; 循環リストは終わらずに回り続けるのではなく PYTREE-ERROR
  (signals nb:pytree-error (nb:tree-flatten (let ((l (list 1 2))) (setf (cddr l) l) l)))
  (signals nb:pytree-error (nb:defpytree cons :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree simple-vector :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree no-such-class-for-pytree-test
                             :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:tree-flatten (make-pytree-test-pair 1 2 (make-array 2 :initial-element 0)))))

(test pytree/builtin-types-cannot-be-registered
  "リスト・ベクタ・配列などの組み込みの型は DEFPYTREE で登録できない。"
  (signals nb:pytree-error (nb:defpytree t :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree cons :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree null :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree list :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree sequence :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree vector :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree simple-vector :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree array :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree simple-array :flatten #'identity :unflatten #'identity))
  ;; 葉として扱うと決めた型（数値・型特化した配列・文字列・トレーサ・treedef）も登録できない。
  ;; 登録できると、どの木の中の数値も節になってしまう
  (signals nb:pytree-error (nb:defpytree fixnum :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree single-float :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree string :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree nb::tracer :flatten #'identity :unflatten #'identity))
  (signals nb:pytree-error (nb:defpytree nb:treedef :flatten #'identity :unflatten #'identity)))

;; DEFPYTREE をテストの実行中に呼ぶための型（ロード時に登録済みの PYTREE-TEST-PAIR と違い、
;; 登録そのものの振る舞いを確かめる）。
(defstruct (pytree-test-box (:constructor make-pytree-test-box (contents)))
  contents)

(test pytree/defpytree-registers-and-overrides-a-node-type
  "DEFPYTREE で登録した型は節になり、再登録は前の登録を上書きする。FLATTEN が子のリストを
返さない、関数でない FLATTEN / UNFLATTEN は PYTREE-ERROR。"
  (let ((box (make-pytree-test-box (list 1 2))))
    (nb:defpytree pytree-test-box
      :flatten (lambda (b) (values (list (pytree-test-box-contents b)) nil))
      :unflatten (lambda (aux children) (declare (ignore aux))
                   (make-pytree-test-box (first children))))
    (is (equal '(1 2) (nb:tree-leaves box)))
    (is (equal '(10 20) (pytree-test-box-contents (nb:tree-map (lambda (x) (* 10 x)) box))))
    (nb:defpytree pytree-test-box
      :flatten (lambda (b) (values (list (first (pytree-test-box-contents b))) :first))
      :unflatten (lambda (aux children) (declare (ignore aux))
                   (make-pytree-test-box (list (first children)))))
    (is (equal '(1) (nb:tree-leaves box)))
    (nb:defpytree pytree-test-box
      :flatten (lambda (b) (values (coerce (pytree-test-box-contents b) 'vector) nil))
      :unflatten (lambda (aux children) (declare (ignore aux)) children))
    (signals nb:pytree-error (nb:tree-flatten box))
    ;; 真リストでない子のリストも PYTREE-ERROR（素の TYPE-ERROR にしない）
    (nb:defpytree pytree-test-box
      :flatten (lambda (b) (values (cons (pytree-test-box-contents b) 2) nil))
      :unflatten (lambda (aux children) (declare (ignore aux)) children))
    (signals nb:pytree-error (nb:tree-flatten box))
    (signals nb:pytree-error (nb:defpytree pytree-test-box :flatten nil :unflatten #'list))
    (signals nb:pytree-error (nb:defpytree pytree-test-box :flatten #'list :unflatten 'list))))

(def-pytree-property pytree/treedefs-and-mismatch-conditions-can-be-printed
  "treedef と、形の不一致のコンディションは（どんな形の木でも）印字できる。"
  (tree) (generator (tuple (pytree-spec)))
  (handler-case (progn (nb:tree-map #'+ tree (list tree)) nil)
    (nb:pytree-structure-mismatch (condition)
      (and (plusp (length (prin1-to-string (nb:tree-structure tree))))
           (plusp (length (princ-to-string condition)))))))

;;; --- tree-children（issue #239） ---

(defstruct (pytree-test-empty (:constructor make-pytree-test-empty ()))
  "子を1つも持たない、DEFPYTREE で登録した節（葉と node-p で区別できることの確認用）。")

(nb:defpytree pytree-test-empty
  :flatten (lambda (object) (declare (ignore object)) (values '() :empty-aux))
  :unflatten (lambda (aux children) (declare (ignore aux children)) (make-pytree-test-empty)))

(defun %children-leaves (tree)
  "TREE-CHILDREN だけで木をたどって集めた葉のリスト（左から順）。"
  (multiple-value-bind (children aux node-p) (nb:tree-children tree)
    (declare (ignore aux))
    (if node-p
        (loop for child in children append (%children-leaves child))
        (list tree))))

(def-pytree-property pytree/children-walk-gives-tree-leaves
  "tree-children だけで再帰的にたどった葉の列は tree-leaves と一致する。"
  (tree) (generator (tuple (pytree-spec)))
  (equal (%children-leaves tree) (nb:tree-leaves tree)))

(def-pytree-property pytree/children-node-p-agrees-with-flatten
  "全ての部分木で、node-p は treedef が葉の treedef（葉1つの木の形）と違うことと一致し、
葉は (values nil nil nil) になる。"
  (tree) (generator (tuple (pytree-spec)))
  (let ((leaf-treedef (nb:tree-structure 0)))
    (labels ((agrees (node)
               (multiple-value-bind (children aux node-p) (nb:tree-children node)
                 (if node-p
                     (and (not (nb:treedef= (nb:tree-structure node) leaf-treedef))
                          (every #'agrees children))
                     (and (nb:treedef= (nb:tree-structure node) leaf-treedef)
                          (null children) (null aux))))))
      (agrees tree))))

(def-pytree-property pytree/children-are-the-direct-children
  "リスト・ベクタ・登録した型の子は、構成した順の直下の要素で、aux は登録した型の補助データ。"
  (a b tag) (generator (tuple (pytree-spec) (pytree-spec) (uniform-integer :lo 0 :hi 2)))
  (and (equal (multiple-value-list (nb:tree-children (list a b))) (list (list a b) nil t))
       (equal (multiple-value-list (nb:tree-children (vector a b))) (list (list a b) nil t))
       (equal (multiple-value-list (nb:tree-children (make-pytree-test-pair a b tag)))
              (list (list a b) tag t))))

(test pytree/children-of-empty-nodes-and-leaves
  "子が0個の節（NIL・空ベクタ・空の登録型）は node-p が真、葉は3値とも NIL で区別できる。
NIL は tree-flatten と同じく空のリスト節。"
  (is (equal '(nil nil t) (multiple-value-list (nb:tree-children nil))))
  (is (equal '(nil nil t) (multiple-value-list (nb:tree-children (vector)))))
  (is (equal '(nil :empty-aux t)
             (multiple-value-list (nb:tree-children (make-pytree-test-empty)))))
  (dolist (leaf (list 3 "str" :key 'sym 1.5d0 (make-array 3 :element-type 'single-float)))
    (is (equal '(nil nil nil) (multiple-value-list (nb:tree-children leaf))))))

(test pytree/children-rejects-improper-lists
  "真リストでないコンスは tree-flatten と同じく PYTREE-ERROR。"
  (signals nb:pytree-error (nb:tree-children (cons 1 2)))
  (signals nb:pytree-error (nb:tree-children (list* 1 2 3))))
