;;;; mutate-tests.lisp -- mutate-form の性質

(in-package #:nabla.mutate.tests)

(in-suite :nabla-mutate)

(defparameter *arith-ops* '(+ - * /))
(defparameter *boundary-ops* '(< <= > >=))

(defun %count-diffs (a b)
  "同じ形をしている（はずの）A と B のリーフの違いを数える。"
  (cond
    ((and (consp a) (consp b))
     (+ (%count-diffs (car a) (car b)) (%count-diffs (cdr a) (cdr b))))
    ((equal a b) 0)
    (t 1)))

(test mutate-form-arith-swap-changes-exactly-one-node-and-is-involutive
  (is (check-it
       (generator (tuple (integer 0 3) (integer -20 20) (integer -20 20)))
       (lambda (input)
         (destructuring-bind (op-index a b) input
           (let ((form (list (nth op-index *arith-ops*) a b)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :arith-swap)
               (and applied
                    (not (equal mutated form))
                    (= 1 (%count-diffs form mutated))
                    (equal form (nabla.mutate:mutate-form mutated :arith-swap))))))))))

(test mutate-form-boundary-changes-exactly-one-node-and-is-involutive
  (is (check-it
       (generator (tuple (integer 0 3) (integer -20 20) (integer -20 20)))
       (lambda (input)
         (destructuring-bind (op-index a b) input
           (let ((form (list (nth op-index *boundary-ops*) a b)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :boundary)
               (and applied
                    (not (equal mutated form))
                    (= 1 (%count-diffs form mutated))
                    (equal form (nabla.mutate:mutate-form mutated :boundary))))))))))

(defun %build-nested-arith (branch)
  "check-it が生成した BRANCH（整数、または (op-index a b) のリスト）を
実際の Lisp フォームに組み立てる。整数はそのままリーフとして返す。"
  (if (integerp branch)
      branch
      (destructuring-bind (op-index a b) branch
        (list (nth op-index *arith-ops*) a b))))

(test mutate-form-arith-swap-first-applicable-node-on-nested-tree
  "入れ子になった算術式でも、mutate-form は前順で最初に見つかった
ノード（外側の演算子）だけを書き換え、内側の枝には触らない。"
  (is (check-it
       (generator (tuple (integer 0 3)
                          (or (integer -10 10)
                              (tuple (integer 0 3) (integer -10 10) (integer -10 10)))
                          (integer -10 10)))
       (lambda (input)
         (destructuring-bind (op-index left-branch right) input
           (let* ((left (%build-nested-arith left-branch))
                  (form (list (nth op-index *arith-ops*) left right)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :arith-swap)
               (and applied
                    (not (equal mutated form))
                    ;; 前順なので、外側のノードだけが変わり、内側の枝
                    ;; （LEFT が入れ子の算術式でも）はそのまま残る。
                    (equal left (second mutated))
                    (equal right (third mutated))
                    (= 1 (%count-diffs form mutated))
                    (equal form (nabla.mutate:mutate-form mutated :arith-swap))))))))))

(test mutate-form-branch-swap-changes-exactly-one-node-and-is-involutive
  (is (check-it
       (generator (tuple (integer -10 10) (integer -10 10) (integer 0 20)))
       (lambda (input)
         (destructuring-bind (c a width) input
           (let* ((b (+ a 1 width))
                  (form (list 'if (list '> c 0) a b)))
             (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form form :branch-swap)
               (and applied
                    (not (equal mutated form))
                    (equal form (nabla.mutate:mutate-form mutated :branch-swap))))))))))

(test mutate-form-constant-never-identity
  (is (check-it
       (generator (integer -50 50))
       (lambda (n)
         (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form n :constant)
           (and applied (not (equal mutated n))))))))

(test mutate-form-skips-arid-nodes
  "arid node（ここでは format）の中の算術は変異させない。"
  (multiple-value-bind (mutated applied)
      (nabla.mutate:mutate-form '(format t "~A" (+ 1 2)) :arith-swap)
    (is-false applied)
    (is (equal mutated '(format t "~A" (+ 1 2))))))

(test mutate-form-no-match-returns-unapplied
  (multiple-value-bind (mutated applied) (nabla.mutate:mutate-form '(list 1 2 3) :branch-swap)
    (is-false applied)
    (is (equal mutated '(list 1 2 3)))))

;;; 箇所ごとの粒度（issue #70）と、足した演算子の性質

(defun %distinct-p (list)
  (= (length list) (length (remove-duplicates list :test #'equal))))

(test mutation-sites-yields-one-mutant-per-applicable-node
  "(progn e1 ... en) の各 ei が算術式なら、:arith-swap は ei ごとに
1つ、合計 n 個の変異体を作る。どれも元と1リーフだけ違い、互いに
異なり、MUTATE-FORM の SITE 番目と一致する。"
  (is (check-it
       (generator (list (integer 0 3) :min-length 1 :max-length 6))
       (lambda (op-indices)
         (let* ((form (cons 'progn (mapcar (lambda (i) (list (nth i *arith-ops*) 'x 'y))
                                           op-indices)))
                (sites (nabla.mutate:mutation-sites form :arith-swap)))
           (and (= (length op-indices) (length sites))
                (%distinct-p sites)
                (every (lambda (m) (= 1 (%count-diffs form m))) sites)
                (loop for m in sites
                      for k from 0
                      always (equal m (nabla.mutate:mutate-form form :arith-swap k)))
                (not (nth-value 1 (nabla.mutate:mutate-form
                                   form :arith-swap (length sites))))))))))

(test negate-condition-is-involutive-on-if-when-unless
  (is (check-it
       (generator (tuple (integer 0 2) (integer -5 5)))
       (lambda (input)
         (destructuring-bind (head-index n) input
           (let ((form (ecase head-index
                         (0 (list 'if (list '> 'x n) 'a 'b))
                         (1 (list 'when (list '> 'x n) 'a))
                         (2 (list 'unless (list '> 'x n) 'a)))))
             (multiple-value-bind (mutated applied)
                 (nabla.mutate:mutate-form form :negate-condition)
               (and applied
                    (not (equal mutated form))
                    (equal form (nabla.mutate:mutate-form mutated :negate-condition))))))))))

(test negate-condition-makes-one-site-per-cond-clause-except-t
  (is (check-it
       (generator (integer 1 5))
       (lambda (n)
         (let* ((clauses (append (loop for i below n collect (list (list '= 'x i) i))
                                 (list (list t -1))))
                (form (cons 'cond clauses))
                (sites (nabla.mutate:mutation-sites form :negate-condition)))
           (and (= n (length sites))
                (%distinct-p sites)
                (every (lambda (m) (equal (car (last m)) '(t -1))) sites)))))))

(test delete-form-drops-one-non-last-body-form
  "(progn f1 ... fn) から、最後以外のフォームを1つずつ消した n-1 個の
変異体を作る。最後のフォーム（返り値）は残す。"
  (is (check-it
       (generator (integer 1 6))
       (lambda (n)
         (let* ((body (loop for i below n collect (list 'f i)))
                (form (cons 'progn body))
                (sites (nabla.mutate:mutation-sites form :delete-form)))
           (and (= (1- n) (length sites))
                (%distinct-p sites)
                (every (lambda (m)
                         (and (= (length m) n)
                              (equal (car (last m)) (car (last body)))
                              (subsetp (cdr m) body :test #'equal)))
                       sites)))))))

(test delete-form-keeps-docstring-and-declarations
  (let ((form '(defun f (x) "doc" (declare (ignorable x)) (g x) (h x))))
    (is (equal '((defun f (x) "doc" (declare (ignorable x)) (h x)))
               (nabla.mutate:mutation-sites form :delete-form)))))

(test off-by-one-shifts-integer-constants-both-ways
  (is (check-it
       (generator (integer -50 50))
       (lambda (n)
         (equal (list (1+ n) (1- n))
                (nabla.mutate:mutation-sites n :off-by-one))))))

(test member-drop-removes-one-literal-element
  (is (check-it
       (generator (integer 2 6))
       (lambda (n)
         (let* ((items (loop for i below n collect (intern (format nil "K~D" i) :keyword)))
                (form (list 'member 'x (list 'quote items)))
                (sites (nabla.mutate:mutation-sites form :member-drop)))
           (and (= n (length sites))
                (%distinct-p sites)
                (every (lambda (m)
                         (= (1- n) (length (second (third m)))))
                       sites)))))))

(test equality-swap-replaces-equal-with-eq
  (is (equal '((eq a b)) (nabla.mutate:mutation-sites '(equal a b) :equality-swap)))
  (is (null (nabla.mutate:mutation-sites '(list a b) :equality-swap))))

(test string-constant-empties-literals-but-not-docstrings
  "文字列リテラル（StableHLO の演算名など）は \"\" に置き換えるが、
docstring と arid node（error / format）の中の文字列には触らない。"
  (is (check-it
       (generator (list (integer 0 3) :min-length 1 :max-length 4))
       (lambda (indices)
         (let* ((names (mapcar (lambda (i) (nth i '("add" "subtract" "multiply" "divide")))
                               indices))
                (form `(defun f (x) "doc"
                         (error "bad ~A" x)
                         (list ,@names)))
                (sites (nabla.mutate:mutation-sites form :string-constant)))
           (and (= (length names) (length sites))
                (every (lambda (m) (= 1 (%count-diffs form m))) sites)
                (every (lambda (m) (equal "doc" (fourth m))) sites)
                (every (lambda (m) (equal (fifth form) (fifth m))) sites)))))))

(test string-constant-skips-report-and-documentation-strings
  "restart-case の :report や :documentation の文字列は、利用者向けの説明で
あってテストの抜けを教えないので、\"\" にしない。"
  (is (null (nabla.mutate:mutation-sites
             '(restart-case (f) (retry () :report "もう一度" (g)))
             :string-constant)))
  (is (null (nabla.mutate:mutation-sites '(list :documentation "doc") :string-constant))))

(test string-constant-skips-message-like-strings
  "空白・非 ASCII・~ を含む文字列（自前のエラー関数に渡すメッセージや
format の制御文字列）は、出力するトークン（演算名など）ではないので
\"\" にしない。"
  (is (null (nabla.mutate:mutation-sites '(%my-error "FN が不正: ~S" fn) :string-constant)))
  (is (null (nabla.mutate:mutation-sites '(%my-error "bad value") :string-constant)))
  (is (null (nabla.mutate:mutation-sites '(%my-error "x=~A") :string-constant)))
  (is (equal '((emit "")) (nabla.mutate:mutation-sites '(emit "stablehlo.add") :string-constant))))
