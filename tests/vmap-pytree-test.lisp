;;;; vmap の引数・戻り値・in-axes / out-axes の PyTree（issue #200、small）。
;;;;
;;;; 守らせる性質:
;;;; - ランダムな木（リスト・ベクタ・登録した型）を受けて木を返す f の vmap は、ランダムな
;;;;   in-axes / out-axes（木の接頭辞。部分木まるごとの整数・NIL を含む）でも、葉ごとに
;;;;   切り出して f を適用し積み直したものと一致し、戻り値の treedef は f の戻り値と同じ。
;;;; - 接頭辞は構造が合わなければ vmap-error（長さ・種類・補助データの違い、葉に整数・NIL
;;;;   以外）。既存の整数・NIL・引数ごとのリストの呼び出しは変わらない。
;;;; - f がリストを返せる（多値と木の混在）。
;;;; 期待値は葉ごとに展開した軸（生成器が接頭辞と同時に作る）から、vmap に依存せず計算する。
;;;; IREE 経由の合成は tests/iree/vmap-pytree-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- 接頭辞の生成器（接頭辞と、葉ごとに展開した軸を同時に作る） ---

(defun %vp-next (state)
  "LCG。STATE（1要素のリスト）を進め、0 以上の整数を返す。"
  (setf (first state) (mod (+ (* (first state) 1103515245) 12345) 2147483648))
  (floor (first state) 65536))

(defun %vp-pick (state choices)
  (nth (mod (%vp-next state) (length choices)) choices))

(defun %vp-prefix (tree state choose &optional (counter (list 0)))
  "TREE（リスト・ベクタ・PYTREE-TEST-PAIR の入れ子。葉は何でもよい）の接頭辞を作る。
(VALUES 接頭辞 葉ごとに展開した軸のリスト) を返す。各部分木は確率 1/3 で整数か NIL の1つに
まとめる。CHOOSE は (葉の通し番号のリスト) → 軸 を返す関数で、まとめる値と葉の値を決める
（その葉たちに許される軸を返す）。COUNTER は葉の通し番号。"
  (let* ((count (length (nb:tree-leaves tree)))
         (first-leaf (first counter)))
    (flet ((indices () (loop for i from first-leaf below (+ first-leaf count) collect i)))
      (cond
        ((and (plusp count) (or (not (or (listp tree) (simple-vector-p tree) (typep tree 'pytree-test-pair)))
                                (zerop (mod (%vp-next state) 3))))
         (let ((axis (funcall choose (indices))))
           (incf (first counter) count)
           (values axis (make-list count :initial-element axis))))
        ((zerop count) (values tree '()))
        (t
         (let ((expanded '()))
           (flet ((child (c)
                    (multiple-value-bind (p e) (%vp-prefix c state choose counter)
                      (setf expanded (append expanded e))
                      p)))
             (values (cond ((listp tree) (mapcar #'child tree))
                           ((simple-vector-p tree) (map 'simple-vector #'child tree))
                           (t (make-pytree-test-pair (child (pytree-test-pair-left tree))
                                                     (child (pytree-test-pair-right tree))
                                                     (pytree-test-pair-tag tree))))
                     expanded))))))))

(defun %vp-in-prefix (spec state)
  "SPEC の葉の軸（NIL / 0 / -1）の接頭辞と展開。展開が全部 NIL ならバッチする引数が無いので
全体を 0 にする。"
  (multiple-value-bind (prefix expanded)
      (%vp-prefix spec state (lambda (indices) (declare (ignore indices)) (%vp-pick state '(nil 0 -1))))
    (if (some #'identity expanded)
        (values prefix expanded)
        (values 0 (make-list (length expanded) :initial-element 0)))))

(defun %vp-inner-shape (n)
  (ecase (mod n 3) (0 '()) (1 '(3)) (2 '(2 3))))

(defun %vp-array-tree (spec in-expanded size)
  "SPEC の葉 n を、IN-EXPANDED の軸（NIL / 0 / -1）で長さ SIZE のバッチ軸を持つ f64 配列にした木。
(VALUES 木 非負に直した軸のリスト) を返す。"
  (let ((axes '()) (k 0))
    (let ((tree (nb:tree-map
                 (lambda (n)
                   (let* ((axis (nth k in-expanded))
                          (inner (%vp-inner-shape n))
                          (position (and axis (if (minusp axis) (length inner) axis))))
                     (incf k)
                     (setf axes (append axes (list position)))
                     (make-random-array
                      (make-array-spec (if position
                                           (append (subseq inner 0 position) (list size) (nthcdr position inner))
                                           inner)
                                       :f64)
                      :seed (+ 1000 n k))))
                 spec)))
      (values tree axes))))

(defun %vp-reference (f tree in-axes out-expanded size)
  "TREE の葉を IN-AXES（非負か NIL）で切り出して F を適用し、出力の葉を OUT-EXPANDED の軸
（NIL / 0 / -1。-1 は出力の最後の軸）に積み直した木。"
  (multiple-value-bind (leaves treedef) (nb:tree-flatten tree)
    (let* ((results (loop for i below size
                          collect (nb:tree-leaves
                                   (funcall f (nb:tree-unflatten
                                               treedef
                                               (loop for leaf in leaves for axis in in-axes
                                                     collect (if axis (slice-along-axis leaf axis i) leaf)))))))
           (out-treedef (nb:tree-structure (funcall f tree))))
      (nb:tree-unflatten
       out-treedef
       (loop for axis in out-expanded for k from 0
             collect (let ((column (mapcar (lambda (r) (nth k r)) results)))
                       (cond ((null axis) (first column))
                             (t (stack-along-axis
                                 column
                                 (if (minusp axis) (array-rank (first column)) axis))))))))))

(defun %vp-trees-close-p (actual expected)
  (and (nb:treedef= (nb:tree-structure actual) (nb:tree-structure expected))
       (every (lambda (a e) (allclose a e :dtype :f64))
              (nb:tree-leaves actual) (nb:tree-leaves expected))))

(test vmap-pytree/matches-per-slice-application-with-prefix-axes
  "ランダムな木を受けて木を返す f の vmap は、ランダムな接頭辞の in-axes / out-axes でも、葉ごとの
切り出し・適用・積み直しと一致し、戻り値の treedef が f の戻り値と同じ。"
  (is (check-it (generator (tuple (pytree-spec) (uniform-integer :lo 1 :hi 1000000) (uniform-integer :lo 1 :hi 3)))
                (lambda (case)
                  (destructuring-bind (spec seed size) case
                    ;; 葉の無い木にはバッチする引数が無い（vmap-error になる）ので対象外
                    (or (null (nb:tree-leaves spec))
                     (let ((state (list seed)))
                      (multiple-value-bind (in-prefix in-expanded) (%vp-in-prefix spec state)
                        (multiple-value-bind (tree in-axes) (%vp-array-tree spec in-expanded size)
                          (multiple-value-bind (out-prefix out-expanded)
                              (%vp-prefix spec state
                                          (lambda (indices)
                                            (if (notany (lambda (i) (nth i in-axes)) indices)
                                                (%vp-pick state '(nil 0 -1))
                                                (%vp-pick state '(0 -1)))))
                            (let ((f (pytree-map-function)))
                              (%vp-trees-close-p
                               (funcall (nb:vmap f :in-axes (list in-prefix) :out-axes (list out-prefix)) tree)
                               (%vp-reference f tree in-axes out-expanded size))))))))))
                :regression-id vmap-pytree/matches-per-slice-application-with-prefix-axes
                :regression-file (regression-path "vmap-pytree-matches-per-slice"))))

;;; --- 例: 接頭辞・多引数・木の戻り値 ---

(defun %vp-vec (&rest xs)
  (make-array (length xs) :element-type 'double-float :initial-contents (mapcar (lambda (x) (float x 1d0)) xs)))

(defparameter *vp-add* (nb:with-tracing (a b) (+ a b)))
(defparameter *vp-mul* (nb:with-tracing (a b) (* a b)))

(defun %vp-function (arity function)
  "FUNCTION（トレーサの木を受ける普通の関数）を ARITY 引数の TRACEABLE-FUNCTION にする。"
  (nb::%make-traceable-function (loop repeat arity collect (gensym "ARG")) function))

(test vmap-pytree/mixed-batched-and-unbatched-arguments-and-tree-result
  "木の引数の葉をすべてバッチし (in-axes ((0 0) nil))、スカラー y は共有する。リストを返す f の結果は
同じ形のリストで、バッチに依存しない葉 y も out-axes 0 で複製される。"
  (let* ((f (%vp-function 2 (lambda (tree y)
                              (destructuring-bind (a b) tree
                                (list (funcall *vp-add* a y) (list (funcall *vp-mul* b y) y))))))
         (result (funcall (nb:vmap f :in-axes '((0 0) nil))
                          (list (%vp-vec 1 2 3) (%vp-vec 10 20 30)) 5d0)))
    (destructuring-bind (r1 (r2 r3)) result
      (is (allclose r1 (%vp-vec 6 7 8) :dtype :f64))
      (is (allclose r2 (%vp-vec 50 100 150) :dtype :f64))
      (is (allclose r3 (%vp-vec 5 5 5) :dtype :f64)))))

(test vmap-pytree/prefix-nil-shares-a-whole-subtree
  "引数1つの木 (a (b c)) に in-axes ((0 nil)): a だけをバッチして (b c) の部分木全体を共有する。
out-axes ((0 nil)) は、バッチに依存しない2つ目の出力を複製せずそのまま返す。"
  (let* ((a (%vp-vec 1 2 3)) (b (%vp-vec 4 5 6)) (c (%vp-vec 7 8))
         (f (%vp-function 1 (lambda (tree)
                              (destructuring-bind (x (y z)) tree
                                (list (funcall *vp-add* x (nb:reduce-sum y)) (nb:reduce-sum z))))))
         (result (funcall (nb:vmap f :in-axes '((0 nil)) :out-axes '((0 nil))) (list a (list b c)))))
    (destructuring-bind (r1 r2) result
      (is (allclose r1 (%vp-vec 16 17 18) :dtype :f64))
      (is (allclose r2 (nb:reduce-sum c) :dtype :f64)))))

(test vmap-pytree/out-axes-prefix-applies-to-whole-returned-subtree
  "out-axes の整数は返した木の部分木まるごとに効く: (1) は ((1 (1 1))) と同じ。"
  (let* ((x (make-random-array (make-array-spec '(2 3) :f64) :seed 1))
         (f (%vp-function 1 (lambda (x) (list (funcall *vp-add* x x) (list (funcall *vp-mul* x x) x)))))
         (result (funcall (nb:vmap f :out-axes '(1)) x))
         (expected (funcall (nb:vmap f :out-axes '((1 (1 1)))) x)))
    (is (nb:treedef= (nb:tree-structure result) (nb:tree-structure expected)))
    (is (every (lambda (a e) (allclose a e :dtype :f64)) (nb:tree-leaves result) (nb:tree-leaves expected)))
    (is (equal '(3 2) (array-dimensions (first result))))))

(test vmap-pytree/multiple-values-still-use-one-out-axis-per-value
  "多値の f の out-axes は値ごとのリスト（従来どおり）。整数 1 つなら全値共通。"
  (let* ((x (make-random-array (make-array-spec '(2 3) :f64) :seed 3))
         (f (nb:with-tracing (x) (values (+ x x) (* x x)))))
    (multiple-value-bind (a b) (funcall (nb:vmap f :out-axes '(1 0)) x)
      (is (equal '(3 2) (array-dimensions a)))
      (is (equal '(2 3) (array-dimensions b))))
    (multiple-value-bind (a b) (funcall (nb:vmap f :out-axes 1) x)
      (is (equal '(3 2) (array-dimensions a)))
      (is (equal '(3 2) (array-dimensions b))))))

;;; --- 接頭辞が構造に合わないとき ---

(test vmap-pytree/mismatched-axes-prefix-signals-vmap-error
  "in-axes / out-axes の接頭辞が木の構造に合わなければ vmap-error（リストの長さ・リストとベクタ・
葉より深い接頭辞・整数/NIL 以外の葉・補助データの違い）。"
  (let ((tree (list (%vp-vec 1 2) (list (%vp-vec 3 4) (%vp-vec 5 6))))
        (f (%vp-function 1 (lambda (tree) (list (first tree))))))
    ;; 構造が合わないものは呼び出し時に報告する
    (signals nb:vmap-error (funcall (nb:vmap f :in-axes '((0 0 0))) tree))
    (signals nb:vmap-error (funcall (nb:vmap f :in-axes '((0 (0)))) tree))
    (signals nb:vmap-error (funcall (nb:vmap f :in-axes '(#(0 0))) tree))
    (signals nb:vmap-error (funcall (nb:vmap f :in-axes '((0 (0 (0))))) tree))
    (signals nb:vmap-error (funcall (nb:vmap f :out-axes '((0 0))) tree))
    ;; 整数・NIL 以外の葉は作った時点で報告する
    (signals nb:vmap-error (nb:vmap f :in-axes '((0 (:a 0)))))
    (signals nb:vmap-error (nb:vmap f :in-axes '(1.5)))
    (signals nb:vmap-error (nb:vmap f :out-axes '((0 "x"))))
    ;; 引数の個数と合わない引数ごとのリストも作った時点で報告する
    (signals nb:vmap-error (nb:vmap f :in-axes '(0 0))))
  ;; 登録した型は、型と補助データ（tag）が同じ接頭辞だけを受ける
  (let ((g (%vp-function 1 (lambda (p) (list p))))
        (pair (make-pytree-test-pair (%vp-vec 1 2) (%vp-vec 3 4) 1)))
    (signals nb:vmap-error
      (funcall (nb:vmap g :in-axes (list (make-pytree-test-pair 0 0 2))) pair))
    (signals nb:vmap-error (funcall (nb:vmap g :in-axes '((0 0))) pair))
    (is (typep (first (funcall (nb:vmap g :in-axes (list (make-pytree-test-pair 0 nil 1))
                                         :out-axes (list (list (make-pytree-test-pair 0 0 1))))
                               pair))
               'pytree-test-pair))))

;; 補助データは EQUAL で比べる（EQ ではない）: 別々に作った同じ中身のリストの tag を受け付ける。
(test vmap-pytree/registered-node-prefix-compares-auxiliary-data-with-equal
  "登録した型の接頭辞は、補助データが EQUAL なら（別のオブジェクトでも）受け付ける。"
  (let ((g (%vp-function 1 (lambda (p) (list p))))
        (pair (make-pytree-test-pair (%vp-vec 1 2) (%vp-vec 3 4) (list :k 1))))
    (is (typep (first (funcall (nb:vmap g :in-axes (list (make-pytree-test-pair 0 nil (list :k 1))))
                               pair))
               'pytree-test-pair))
    (signals nb:vmap-error
      (funcall (nb:vmap g :in-axes (list (make-pytree-test-pair 0 nil (list :k 2)))) pair))))

(test vmap-pytree/nil-out-axes-for-batched-output-is-still-an-error
  "バッチされた出力に out-axes nil は従来どおり vmap-error。"
  (let ((x (%vp-vec 1 2 3))
        (f (%vp-function 2 (lambda (a b) (list (funcall *vp-add* a a) (funcall *vp-add* b b))))))
    (signals nb:vmap-error (funcall (nb:vmap f :in-axes '(0 nil) :out-axes '((nil nil))) x x))
    (destructuring-bind (ra rb) (funcall (nb:vmap f :in-axes '(0 nil) :out-axes '((0 nil))) x (%vp-vec 1 2))
      (is (allclose ra (%vp-vec 2 4 6) :dtype :f64))
      (is (allclose rb (%vp-vec 2 4) :dtype :f64)))))

;;; --- 軸の木の複写・トップレベルの型・壊れた登録 ---

(test vmap-pytree/axes-are-copied-when-vmap-is-called
  "vmap した後で in-axes / out-axes（リスト・ベクタ・登録した型の入れ子）を書き換えても、vmap した関数の結果は変わらない。"
  (let* ((f (%vp-function 1 (lambda (tree) (list (first tree) (second tree)))))
         (in-axes (list (list 0 (vector nil))))
         (tree (list (%vp-vec 1 2 3) (vector (%vp-vec 4 5)))))
    (let* ((g (nb:vmap (%vp-function 1 (lambda (tree) (list (first tree) (second tree))))
                       :in-axes in-axes :out-axes '((0 #(0)))))
           (before (funcall g tree)))
      ;; 利用者が後から軸の木を壊す
      (setf (first (first in-axes)) :broken
            (svref (second (first in-axes)) 0) 0)
      (is (nb:treedef= (nb:tree-structure before) (nb:tree-structure (funcall g tree))))
      (is (every (lambda (a e) (allclose a e :dtype :f64))
                 (nb:tree-leaves (funcall g tree)) (nb:tree-leaves before))))
    (let* ((out (list (list 0 0)))
           (g2 (nb:vmap f :out-axes out))
           (x (list (make-random-array (make-array-spec '(2 3) :f64) :seed 1)
                    (make-random-array (make-array-spec '(2 3) :f64) :seed 2)))
           (r1 (funcall g2 x)))
      (setf (first (first out)) 1)
      (is (equal '(2 3) (array-dimensions (first r1))))
      (is (equal '(2 3) (array-dimensions (first (funcall g2 x))))))))

(test vmap-pytree/top-level-in-axes-must-be-an-axis-or-a-per-argument-list
  "整数・NIL・リスト以外の in-axes（ベクタなど）は、引数が常にリストなので vmap を呼んだ時点で vmap-error。"
  (let ((f (%vp-function 2 (lambda (a b) (list a b)))))
    (signals nb:vmap-error (nb:vmap f :in-axes (vector 0 0)))
    (signals nb:vmap-error (nb:vmap f :in-axes "ab"))
    (is (functionp (nb:vmap f :in-axes '(0 nil))))))

(defstruct (vp-broken-node (:constructor make-vp-broken-node ())))
(nb:defpytree vp-broken-node
  :flatten (lambda (node) (declare (ignore node)) (values 5 nil))
  :unflatten (lambda (aux children) (declare (ignore aux children)) (make-vp-broken-node)))

(test vmap-pytree/registered-axes-node-with-a-non-list-flatten-is-a-vmap-error
  "登録した型の flatten が子のリストを返さないとき、軸の木に渡すと pytree-error ではなく vmap-error。"
  (signals nb:vmap-error
    (nb:vmap (%vp-function 1 (lambda (a) a)) :in-axes (list (make-vp-broken-node)))))
