;;;; jit / grad / value-and-grad の引数と戻り値の PyTree（issue #199、small）。
;;;;
;;;; 守らせる性質:
;;;; - grad / value-and-grad: 勾配は微分した引数と同じ treedef を持ち、葉ごとに f64 の
;;;;   中心差分と一致する。argnums がリストなら引数ごとに木が返る。値は f の結果と一致する。
;;;; - 木の中に浮動小数点でない葉があれば autodiff-error。f がスカラー1つ以外を返せば
;;;;   grad-requires-scalar-output（葉が1つでも木で包んで返すと拒否する）。
;;;; - jit のキャッシュ: 同じ treedef・aval の2回目はコンパイルせず、aval が同じでも
;;;;   treedef が違えばコンパイルする（フェイク backend）。戻り値の木の形が保たれる。
;;;; IREE 経由の jit(f)(tree) = f(tree) は tests/iree/pytree-function-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defmacro def-pytree-function-property (name docstring (&rest vars) gen &body body)
  "GEN の値を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test ,name ,docstring
     (is (check-it ,gen
                   (lambda (value) (destructuring-bind ,vars value ,@body))
                   :regression-id ,name
                   :regression-file (regression-path
                                     ,(format nil "pytree-function-~(~A~)"
                                              (substitute #\- #\/ (subseq (string name)
                                                                          (length "pytree-function/"))))))
         ,(format nil "~A の性質が成り立たなかった" name))))

(defun %has-leaves-p (spec)
  (plusp (length (nb:tree-leaves spec))))

;;; --- grad ---

(def-pytree-function-property pytree-function/grad-has-argument-treedef-and-matches-central-difference
  "木を受ける f の grad は引数と同じ treedef の木を返し、葉ごとに f64 の中心差分と一致する。"
  (spec) (generator (tuple (pytree-spec)))
  (or (not (%has-leaves-p spec))
      (let* ((tree (pytree-array-tree spec :f64))
             (treedef (nb:tree-structure tree))
             (grads (funcall (nb:grad (pytree-loss-function)) tree))
             (expected (central-difference-gradient (pytree-loss-oracle treedef)
                                                    (nb:tree-leaves tree))))
        (and (nb:treedef= treedef (nb:tree-structure grads))
             (%results-close-p (nb:tree-leaves grads) expected
                               :rtol *autodiff-rtol* :atol *autodiff-atol*)))))

(def-pytree-function-property pytree-function/value-and-grad-matches-grad-and-oracle
  "value-and-grad は (f の値, grad と同じ木) を返す。"
  (spec) (generator (tuple (pytree-spec)))
  (or (not (%has-leaves-p spec))
      (let* ((tree (pytree-array-tree spec :f64))
             (loss (pytree-loss-function)))
        (multiple-value-bind (value grads) (funcall (nb:value-and-grad loss) tree)
          (and (allclose value (funcall loss tree) :dtype :f64)
               (nb:treedef= (nb:tree-structure tree) (nb:tree-structure grads))
               (%results-close-p (nb:tree-leaves grads)
                                 (nb:tree-leaves (funcall (nb:grad loss) tree))
                                 :rtol *autodiff-rtol* :atol *autodiff-atol*))))))

(def-pytree-function-property pytree-function/grad-with-list-argnums-returns-one-tree-per-argument
  "argnums がリストなら、引数ごとにその引数と同じ treedef の木のリストを返す。木でない引数
（配列）の勾配は配列のまま。"
  (spec-a spec-b) (generator (tuple (pytree-spec) (pytree-spec)))
  (or (not (and (%has-leaves-p spec-a) (%has-leaves-p spec-b)))
      (let* ((a (pytree-array-tree spec-a :f64))
             (b (pytree-array-tree spec-b :f64))
             (x (make-random-array (make-array-spec '(3) :f64) :seed 7))
             (loss (pytree-loss-function))
             (f (nb::%make-traceable-function
                 (list (gensym "A") (gensym "X") (gensym "B"))
                 (lambda (a x b)
                   (funcall *pytree-add-op*
                            (funcall *pytree-add-op* (funcall loss a) (nb:reduce-sum (funcall *pytree-leaf-op* x)))
                            (funcall loss b)))))
             (grads (funcall (nb:grad f :argnums '(0 1 2)) a x b)))
        (and (= 3 (length grads))
             (nb:treedef= (nb:tree-structure a) (nb:tree-structure (first grads)))
             (arrayp (second grads))
             (equal '(3) (array-dimensions (second grads)))
             (nb:treedef= (nb:tree-structure b) (nb:tree-structure (third grads)))
             ;; a に依存する勾配は、a だけを渡した grad と一致する
             (%results-close-p (nb:tree-leaves (first grads))
                               (nb:tree-leaves (funcall (nb:grad loss) a))
                               :rtol *autodiff-rtol* :atol *autodiff-atol*)))))

(test pytree-function/grad-accepts-lisp-numbers-and-nested-registered-nodes
  "Lisp の実数の葉と、defpytree で登録した型の節を含む木でも grad が動く。"
  (let* ((tree (make-pytree-test-pair (list 0.5d0) (vector 1.5d0) 3))
         (grads (funcall (nb:grad (pytree-loss-function)) tree)))
    (is (nb:treedef= (nb:tree-structure tree) (nb:tree-structure grads)))
    (is (typep grads 'pytree-test-pair))
    (is (= 3 (pytree-test-pair-tag grads)))))

(test pytree-function/grad-rejects-non-float-leaf-in-differentiated-tree
  "微分する木に整数の葉があれば autodiff-error。微分しない引数の中なら構わない。"
  (let ((ints (make-array '(2) :element-type '(signed-byte 32) :initial-element 1))
        (floats (make-random-array (make-array-spec '(2) :f64) :seed 1))
        (f (nb::%make-traceable-function
            (list (gensym "A") (gensym "B"))
            (lambda (a b) (declare (ignore a)) (nb:reduce-sum (funcall *pytree-leaf-op* b))))))
    (signals nb:autodiff-error (funcall (nb:grad f :argnums 0) (list ints) floats))
    (is (arrayp (funcall (nb:grad f :argnums 1) (list ints) floats)))))

(test pytree-function/grad-requires-a-bare-scalar-output
  "f が葉1つでも木（リスト）で包んで返すと grad-requires-scalar-output。"
  (let ((f (nb::%make-traceable-function
            (list (gensym "X"))
            (lambda (x) (list (nb:reduce-sum x)))))
        (x (make-random-array (make-array-spec '(2) :f64) :seed 2)))
    (signals nb:grad-requires-scalar-output (funcall (nb:grad f) x))))

;;; --- jit のキャッシュと戻り値の木（フェイク backend） ---

(defun %pair-add-function (&key (wrap #'list))
  "木の葉2つを足した結果を WRAP で包んで返す TRACEABLE-FUNCTION。"
  (nb::%make-traceable-function
   (list (gensym "TREE"))
   (lambda (tree)
     (destructuring-bind (a b) (nb:tree-leaves tree)
       (funcall wrap (funcall *pytree-add-op* a b))))))

(test pytree-function/jit-same-treedef-hits-different-treedef-misses
  "同じ treedef・aval の2回目の呼び出しはコンパイルしない。葉の aval が同じでも treedef が
違えば（リストとベクタ、入れ子の違い）コンパイルする。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (jf (nb:jit (%pair-add-function) :backend backend))
         (a (make-random-array (make-array-spec '(2) :f32) :seed 1))
         (b (make-random-array (make-array-spec '(2) :f32) :seed 2)))
    (funcall jf (list a b))
    (is (= 1 (fake-backend-compile-count backend)))
    (funcall jf (list b a))
    (is (= 1 (fake-backend-compile-count backend)))
    (funcall jf (vector a b))
    (is (= 2 (fake-backend-compile-count backend)))
    (funcall jf (list (list a) b))
    (is (= 3 (fake-backend-compile-count backend)))
    (funcall jf (list a b))
    (funcall jf (vector a b))
    (is (= 3 (fake-backend-compile-count backend)))))

(test pytree-function/jit-registered-node-aux-data-is-part-of-the-key
  "defpytree の補助データ（tag）が違えば別のコンパイル。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (jf (nb:jit (%pair-add-function) :backend backend))
         (a (make-random-array (make-array-spec '(2) :f32) :seed 1))
         (b (make-random-array (make-array-spec '(2) :f32) :seed 2)))
    (funcall jf (make-pytree-test-pair a b 0))
    (funcall jf (make-pytree-test-pair b a 0))
    (is (= 1 (fake-backend-compile-count backend)))
    (funcall jf (make-pytree-test-pair a b 1))
    (is (= 2 (fake-backend-compile-count backend)))))

(test pytree-function/jit-returns-the-tree-the-function-returned
  "f がリスト・ベクタ・登録した型で包んで返した木は、その形のまま戻る。複数値で返せば
多値のまま、1つの木で返せば1つの木で戻る。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (a (make-random-array (make-array-spec '(2) :f32) :seed 1))
         (b (make-random-array (make-array-spec '(2) :f32) :seed 2))
         (expected (funcall *pytree-add-op* a b)))
    (let ((result (funcall (nb:jit (%pair-add-function :wrap (lambda (x) (vector (list x))))
                                   :backend backend)
                           (list a b))))
      (is (simple-vector-p result))
      (is (allclose (first (aref result 0)) expected :dtype :f32)))
    (let ((result (funcall (nb:jit (%pair-add-function :wrap (lambda (x) (make-pytree-test-pair x nil 5)))
                                   :backend backend)
                           (list a b))))
      (is (typep result 'pytree-test-pair))
      (is (= 5 (pytree-test-pair-tag result)))
      (is (null (pytree-test-pair-right result))))
    (is (allclose (funcall (nb:jit (%pair-add-function :wrap #'identity) :backend backend) (list a b))
                  expected :dtype :f32))))

(test pytree-function/jit-number-leaves-become-rank-0-arrays
  "jit の葉の Lisp の実数は rank 0 の配列として扱う（double-float は :f64、それ以外は :f32）。"
  (is (eq :f64 (nb:aval-dtype (nb:array-aval (nb::%jit-leaf-array 1d0)))))
  (is (eq :f32 (nb:aval-dtype (nb:array-aval (nb::%jit-leaf-array 1.0f0)))))
  (is (eq :f32 (nb:aval-dtype (nb:array-aval (nb::%jit-leaf-array 3)))))
  (let ((array (make-random-array (make-array-spec '(2) :f32) :seed 1)))
    (is (eq array (nb::%jit-leaf-array array)))))

(test pytree-function/jit-called-while-tracing-inlines-tree-arguments
  "別の関数のトレース中に、トレーサを葉に持つ木を jit した関数へ渡すと、コンパイルせず
呼び出し元の graph に展開する（結果は木を直接 f に渡したものと一致する）。"
  (let* ((nb:*compile-cache-directory* nil)
         (backend (nb:make-backend :fake))
         (jf (nb:jit (%pair-add-function) :backend backend))
         (outer (nb::%make-traceable-function
                 (list (gensym "A") (gensym "B"))
                 (lambda (a b) (first (funcall jf (list a b))))))
         (avals (list (nb:make-aval '(2) :f32) (nb:make-aval '(2) :f32)))
         (graph (nb:trace-to-graph outer avals))
         (a (make-random-array (make-array-spec '(2) :f32) :seed 1))
         (b (make-random-array (make-array-spec '(2) :f32) :seed 2)))
    (is (= 0 (fake-backend-compile-count backend)))
    (is (allclose (nb:eval-graph graph a b) (funcall *pytree-add-op* a b) :dtype :f32))))

(test pytree-function/grad-error-names-the-argument-with-a-non-float-leaf
  "微分する木に整数の葉があるときのエラーは、その引数の位置を含む。"
  (let ((ints (make-array '(2) :element-type '(signed-byte 32) :initial-element 1))
        (floats (make-random-array (make-array-spec '(2) :f64) :seed 1))
        (f (nb::%make-traceable-function
            (list (gensym "A") (gensym "B"))
            (lambda (a b) (declare (ignore a)) (nb:reduce-sum (funcall *pytree-leaf-op* b))))))
    (handler-case (funcall (nb:grad f :argnums '(1 0)) floats (list ints))
      (nb:autodiff-error (condition)
        (is (search "引数 1" (princ-to-string condition))))
      (:no-error (&rest results)
        (declare (ignore results))
        (fail "autodiff-error にならなかった")))))

(test pytree-function/jit-rejects-leaves-that-are-not-arrays-or-numbers
  "jit の引数の木の葉がキーワード・文字列などなら、jit-error（葉を示す）で拒否する。"
  (let* ((nb:*compile-cache-directory* nil)
         (jf (nb:jit (%pair-add-function) :backend (nb:make-backend :fake)))
         (a (make-random-array (make-array-spec '(2) :f32) :seed 1)))
    (signals nb:jit-error (funcall jf (list a :oops)))
    (signals nb:jit-error (funcall jf (list a "text")))))
