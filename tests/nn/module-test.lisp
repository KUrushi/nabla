;;;; defmodule（issue #202、small）。
;;;;
;;;; 守らせる性質:
;;;; - 往復: モジュール（入れ子のサブモジュール・リスト・:state を含む）の
;;;;   tree-flatten → tree-unflatten は同じクラス・同じスロット・同じ葉のモジュールを作る。
;;;; - 葉の順と形: 葉は :param / :state のスロットの宣言順。:static は葉でなく treedef に入り、
;;;;   :static だけが違うモジュールは treedef= でない。
;;;; - init: 同じキーなら同じパラメータ、違うキーなら違うパラメータ。サブモジュールのキーは
;;;;   split で分かれていて、すべて別の値になる。
;;;; - tree-map はクラスとスロット名を保つ。
;;;; - call: :call の本体が呼ばれる。
;;;; - map-state: :state のスロットの葉だけに関数をかけ、入れ子の中でも効く。
;;;; - module-slots: 宣言順の (スロット名 . 種類)。

(in-package #:nabla.nn.tests)

(defmodule tm-dense ((w :param) (b :param) (act :static))
  (:init (key n act)
   (destructuring-bind (kw kb) (split-keys key 2)
     (make-tm-dense :w (nabla:normal kw (list n)) :b (nabla:uniform kb (list n)) :act act)))
  (:call (self scale)
   (* scale (aref (tm-dense-w self) 0))))

(defmodule tm-net ((inp :param) (layers :param) (stats :state) (tag :static))
  (:init (key n depth)
   (let ((keys (split-keys key (+ 2 depth))))
     (make-tm-net :inp (init-tm-dense (first keys) n :relu)
                  :layers (loop for k in (cddr keys) collect (init-tm-dense k n :tanh))
                  :stats (list (nabla:uniform (second keys) (list n)))
                  :tag (list :depth depth)))))

(defun same-leaves-p (a b)
  (let ((la (nabla:tree-leaves a)) (lb (nabla:tree-leaves b)))
    (and (= (length la) (length lb)) (every #'equalp la lb))))

(defmacro def-module-property (name docstring (&rest vars) &body body)
  "(seed n depth) を VARS に分配束縛して BODY（真偽）を PBT で確かめる。"
  `(test (,name :suite :nabla.small)
     ,docstring
     (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000)
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 0 :hi 3)))
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.NN.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

(def-module-property module/flatten-roundtrip
  "入れ子のモジュールの flatten → unflatten は同じクラス・同じ葉・同じ静的値のモジュールを作る。"
  (seed n depth)
  (let ((net (init-tm-net (nabla:prng-key seed) n depth)))
    (multiple-value-bind (leaves treedef) (nabla:tree-flatten net)
      (let ((back (nabla:tree-unflatten treedef leaves)))
        (and (typep back 'tm-net)
             (equal (tm-net-tag back) (list :depth depth))
             (eq (tm-dense-act (tm-net-inp back)) :relu)
             (= (length (tm-net-layers back)) depth)
             (same-leaves-p net back)
             ;; 葉は inp の w b、layers の (w b) ...、stats の順
             (= (length leaves) (+ 2 (* 2 depth) 1)))))))

(def-module-property module/init-is-deterministic
  "同じキーで init すると同じパラメータ、違うキーなら違うパラメータになる。"
  (seed n depth)
  (let ((a (init-tm-net (nabla:prng-key seed) n depth))
        (b (init-tm-net (nabla:prng-key seed) n depth))
        (c (init-tm-net (nabla:prng-key (1+ seed)) n depth)))
    (and (same-leaves-p a b)
         (not (same-leaves-p a c)))))

(def-module-property module/submodule-keys-are-distinct
  "サブモジュールのパラメータはすべて別の値になる（キーが split で分かれている）。"
  (seed n depth)
  (let* ((net (init-tm-net (nabla:prng-key seed) n depth))
         (ws (cons (tm-dense-w (tm-net-inp net))
                   (mapcar #'tm-dense-w (tm-net-layers net)))))
    (= (length ws) (length (remove-duplicates ws :test #'equalp)))))

(def-module-property module/tree-map-preserves-class-and-statics
  "tree-map は同じクラス・同じ静的値・同じ treedef のモジュールを返し、葉に関数をかける。"
  (seed n depth)
  (let* ((net (init-tm-net (nabla:prng-key seed) n depth))
         (doubled (nabla:tree-map (lambda (x) (map-into (copy-seq x) (lambda (v) (* 2 v)) x)) net)))
    (and (typep doubled 'tm-net)
         (equal (tm-net-tag doubled) (tm-net-tag net))
         (nabla:treedef= (nabla:tree-structure net) (nabla:tree-structure doubled))
         (every (lambda (a b) (every (lambda (x y) (= (* 2 x) y)) a b))
                (nabla:tree-leaves net) (nabla:tree-leaves doubled)))))

(def-module-property module/map-state-touches-only-state
  "map-state は :state のスロットの葉だけを変え、:param と :static は EQ のまま残す。"
  (seed n depth)
  (let* ((net (init-tm-net (nabla:prng-key seed) n depth))
         (zeroed (map-state (lambda (x) (make-array (length x) :element-type 'single-float
                                                                :initial-element 0.0))
                            net)))
    (and (eq (tm-dense-w (tm-net-inp zeroed)) (tm-dense-w (tm-net-inp net)))
         (equal (tm-net-tag zeroed) (tm-net-tag net))
         (every (lambda (x) (every #'zerop x)) (tm-net-stats zeroed))
         (not (equalp (tm-net-stats zeroed) (tm-net-stats net))))))

(def-module-property module/static-slots-are-in-the-treedef
  "静的値が違うモジュールは treedef= でなく、葉だけが違うモジュールは treedef= になる。"
  (seed n depth)
  (let* ((net (init-tm-net (nabla:prng-key seed) n depth))
         (other (init-tm-net (nabla:prng-key (1+ seed)) n depth))
         (retagged (nabla:tree-unflatten (nabla:tree-structure net) (nabla:tree-leaves net))))
    (setf (tm-net-tag retagged) (list :depth (+ depth 1)))
    (and (nabla:treedef= (nabla:tree-structure net) (nabla:tree-structure other))
         (not (nabla:treedef= (nabla:tree-structure net) (nabla:tree-structure retagged))))))

;;; --- 例ベース ---

(test (module/call-dispatches-to-the-call-body :suite :nabla.small)
  (let ((layer (init-tm-dense (nabla:prng-key 3) 2 :relu)))
    (is (= (* 3 (aref (tm-dense-w layer) 0)) (call layer 3)))))

(test (module/slots-are-listed-in-declaration-order :suite :nabla.small)
  (is (equal '((inp . :param) (layers . :param) (stats . :state) (tag . :static))
             (module-slots 'tm-net)))
  (is (equal (module-slots 'tm-net)
             (module-slots (init-tm-net (nabla:prng-key 0) 1 1)))))

(test (module/non-module-has-no-slots :suite :nabla.small)
  (is (null (module-slots 'cons)))
  (is (null (module-slots 42))))

(test (module/split-keys-matches-split :suite :nabla.small)
  (let* ((key (nabla:prng-key 9))
         (rows (nabla:split key 3))
         (keys (split-keys key 3)))
    (is (= 3 (length keys)))
    (loop for k in keys for i from 0
          do (is (equalp k (make-array 2 :element-type '(unsigned-byte 32)
                                         :initial-contents (list (aref rows i 0) (aref rows i 1))))))))

(test (module/rejects-bad-definitions :suite :nabla.small)
  (signals error (macroexpand-1 '(defmodule bad-kind ((w :weird)))))
  (signals error (macroexpand-1 '(defmodule bad-dup ((w :param) (w :state)))))
  (signals error (macroexpand-1 '(defmodule bad-clause ((w :param)) (:foo)))))

(test (module/unequal-static-is-a-pytree-error :suite :nabla.small)
  (let ((layer (make-tm-dense :w (vector) :b (vector) :act (make-hash-table))))
    (signals nabla:pytree-error (nabla:tree-flatten layer))))

(test (module/rejects-malformed-clauses :suite :nabla.small)
  (signals error (macroexpand-1 '(defmodule bad-call ((w :param)) (:call))))
  (signals error (macroexpand-1 '(defmodule bad-init ((w :param)) (:init () nil))))
  (signals error (macroexpand-1 '(defmodule bad-slot ((w :state :frozen))))))

(defun mentions-p (name form)
  "FORM（の入れ子）に、名前が NAME のシンボルがあれば真。"
  (cond ((symbolp form) (and form (string= (symbol-name form) name)))
        ((consp form) (or (mentions-p name (car form)) (mentions-p name (cdr form))))))

(test (module/expansion-defines-struct-registration-and-names :suite :nabla.small)
  (let ((expansion (macroexpand-1 '(defmodule demo-m ((a :param) b (c :state) (d :static))
                                    (:init (key) key)
                                    (:call (self) self)))))
    (is (eq 'progn (first expansion)))
    ;; 構造体・登録・init・call の4つのフォームと戻り値
    (is (equal '("DEFSTRUCT" "SETF" "DEFPYTREE" "DEFUN" "DEFMETHOD" "QUOTE")
               (mapcar (lambda (form) (symbol-name (first form))) (rest expansion)))
        "展開が期待したフォームの並びでない")
    (is (mentions-p "MAKE-DEMO-M" expansion))
    (is (mentions-p "INIT-DEMO-M" expansion))))

(test (module/accepts-every-slot-kind-and-bare-symbols :suite :nabla.small)
  (dolist (spec '(((a)) ((a :param)) ((a :state)) ((a :static)) (a) ((a :param) b)))
    (is (consp (macroexpand-1 `(defmodule slot-kind-demo ,spec))))))

(defun run-module-tests ()
  "このファイルのテストだけを実行する（変更した行への mutation testing の --test-form 用）。"
  (let ((names (loop for symbol being the symbols of (find-package '#:nabla.nn.tests)
                     when (and (eq (symbol-package symbol) (find-package '#:nabla.nn.tests))
                               (eql 0 (search "MODULE/" (symbol-name symbol)))
                               (fiveam:get-test symbol))
                       collect symbol)))
    (fiveam:run! names)))

;;; --- :call は with-tracing の本体（grad / vmap の中でも使える） ---

(defmodule tm-affine ((w :param) (b :param) (stats :state))
  (:call (self x)
   (+ (* x (tm-affine-w self)) (+ (tm-affine-b self) (tm-affine-stats self)))))

(defun tm-affine-example (seed n)
  (destructuring-bind (k1 k2 k3) (split-keys (nabla:prng-key seed) 3)
    (make-tm-affine :w (nabla:normal k1 (list n)) :b (nabla:normal k2 (list n))
                    :stats (nabla:normal k3 (list n)))))

(defun close-p (a b)
  (and (= (array-total-size a) (array-total-size b))
       (loop for i below (array-total-size a)
             always (<= (abs (- (row-major-aref a i) (row-major-aref b i))) 1e-5))))

(def-module-property module/call-uses-array-arithmetic-eagerly
  "(+ (* x w) ...) の :call は配列に対して eager に計算できる。"
  (seed n depth)
  (declare (ignore depth))
  (let* ((m (tm-affine-example seed n))
         (x (nabla:normal (nabla:prng-key (1+ seed)) (list n)))
         (y (call m x)))
    (loop for i below n
          always (<= (abs (- (aref y i)
                             (+ (* (aref x i) (aref (tm-affine-w m) i))
                                (+ (aref (tm-affine-b m) i) (aref (tm-affine-stats m) i)))))
                     1e-5))))

(def-module-property module/grad-through-call-matches-analytic
  "grad (reduce-sum (call m x)) の x についての勾配は w（モジュールを閉包で捕まえる）。"
  (seed n depth)
  (declare (ignore depth))
  (let* ((m (tm-affine-example seed n))
         (x (nabla:normal (nabla:prng-key (1+ seed)) (list n)))
         (g (funcall (nabla:grad (nabla:with-tracing (x) (nabla:reduce-sum (call m x)))) x)))
    (close-p g (tm-affine-w m))))

(def-module-property module/vmap-through-call-matches-per-row
  "vmap (call m x) は各行を単独に call した結果と一致する。"
  (seed n depth)
  (let* ((m (tm-affine-example seed n))
         (rows (1+ depth))
         (x (nabla:normal (nabla:prng-key (1+ seed)) (list rows n)))
         (y (funcall (nabla:vmap (nabla:with-tracing (x) (call m x))) x)))
    (loop for r below rows
          always (let ((row (make-array n :element-type 'single-float)))
                   (dotimes (j n) (setf (aref row j) (aref x r j)))
                   (let ((expected (call m row)))
                     (loop for j below n
                           always (<= (abs (- (aref y r j) (aref expected j))) 1e-5)))))))

(def-module-property module/state-gets-no-gradient
  "map-state #'stop-gradient をかけたモジュールの :state の葉への勾配は 0、:param の勾配は 0 でない。"
  (seed n depth)
  (declare (ignore depth))
  (let* ((m (tm-affine-example seed n))
         (treedef (nabla:tree-structure m))
         (x (nabla:normal (nabla:prng-key (1+ seed)) (list n)))
         (grads (funcall (nabla:grad
                          (nabla:with-tracing (w b s)
                            (nabla:reduce-sum
                             (call (map-state #'nabla:stop-gradient
                                              (nabla:tree-unflatten treedef (list w b s)))
                                   x)))
                          :argnums '(0 1 2))
                         (tm-affine-w m) (tm-affine-b m) (tm-affine-stats m))))
    (and (every #'zerop (third grads))
         (notevery #'zerop (second grads))
         (close-p (first grads) x))))

(test (module/module-slots-returns-fresh-conses :suite :nabla.small)
  (let ((slots (module-slots 'tm-net)))
    (setf (cdr (first slots)) :static)
    (is (equal '((inp . :param) (layers . :param) (stats . :state) (tag . :static))
               (module-slots 'tm-net)))))

(test (module/rejects-duplicate-clauses :suite :nabla.small)
  (signals error (macroexpand-1 '(defmodule dup-init (w) (:init (k) 1) (:init (k) 2))))
  (signals error (macroexpand-1 '(defmodule dup-call (w) (:call (self) 1) (:call (self) 2)))))

(test (module/slot-named-p-has-no-predicate-clash :suite :nabla.small)
  (is (consp (macroexpand-1 '(defmodule pm ((p :param) (q :static))))))
  (is (null (fboundp 'pm-p)))
  (is (null (fboundp 'tm-net-p))))
