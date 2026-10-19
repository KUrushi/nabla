;;;; cond* / while-loop / scan の PyTree の carry・xs・ys・operands を IREE（local）で
;;;; jit した end-to-end テスト（issue #201、medium）。
;;;;
;;;; 守らせる性質: 木を受けて木を返す関数（内側で cond* / while-loop / scan を木のまま使う）を
;;;; jit した結果は、jit しない eager の結果と同じ treedef で、葉ごとに f32 の許容誤差内で一致する。
;;;; (jit (grad f)) が木の carry の scan を通る場合も同じ。IREE はコンパイルが遅いので、
;;;; PBT の試行回数は小さく抑え、木の葉の個数も制限する（IREE 3.11 の入力 162 個の制限）。

(in-package #:nabla.iree.tests)

(defparameter *pc-scale* (nb:with-tracing (x) (* 0.9 x)))
(defparameter *pc-step* (nb:with-tracing (c s) (tanh (+ c s))))
(defparameter *pc-inc* (nb:with-tracing (x) (+ x 1.0)))
(defparameter *pc-lt* (nb:with-tracing (a b) (< a b)))
(defparameter *pc-add* (nb:with-tracing (a b) (+ a b)))

(defun %pc-scale-tree (tree)
  (nb:tree-map (lambda (x) (funcall *pc-scale* x)) tree))

(defun %pc-tree (seed)
  "(登録した節 #(a) (b c)) の木。a は [3]、b は [2 3]、c はスカラー。"
  (make-pytree-test-pair
   (vector (make-random-array (make-array-spec '(3) :f32) :seed seed))
   (list (make-random-array (make-array-spec '(2 3) :f32) :seed (+ seed 1))
         (make-random-array (make-array-spec '() :f32) :seed (+ seed 2)))
   5))

(defun %pc-jit-matches-eager-p (f backend args)
  (let ((jf (nb:jit f :backend backend)))
    (unwind-protect
         (let ((actual (multiple-value-list (apply jf args)))
               (expected (multiple-value-list (apply f args))))
           (and (= (length actual) (length expected))
                (every (lambda (a e) (%leaves-close-p a e :f32)) actual expected)))
      (nb::%jit-cache-forget f))))

(define-iree-test pytree-control/jit-while-loop-with-tree-carry-matches-eager
    "木の carry（カウンタと入れ子の木）の while-loop を jit した結果は eager と一致する
（limit は cond が閉包で捕まえる。0回で終わる場合を含む）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil)
        (f (nb::%make-traceable-function
            (list 'limit 'tree)
            (lambda (limit tree)
              (nb:while-loop
               (lambda (c) (funcall *pc-lt* (first c) limit))
               (lambda (c) (list (funcall *pc-inc* (first c)) (%pc-scale-tree (second c))))
               (list (nb::%scalar-array 0.0 :f32) tree))))))
    (dolist (limit '(0.0 1.0 3.0))
      (is (%pc-jit-matches-eager-p f backend (list (nb::%scalar-array limit :f32) (%pc-tree 11)))
          "limit ~A で jit と eager の while-loop が一致しなかった" limit))
    (gc-and-run-finalizers)))

(define-iree-test pytree-control/jit-cond-with-tree-operands-matches-eager
    "木の operands と木の戻り値の cond* を jit した結果は、pred が真でも偽でも eager と一致する。
（枝は入力を素通しで返さない。IREE 3.11 は、cond の枝が入力をそのまま返す出力を6個ほど
持つと実行時に落ちる。PyTree とは無関係の既存の問題で、フラットな引数でも再現する。）"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (nb:*compile-cache-directory* nil)
        (then (nb::%make-traceable-function
               (list 'a 'b) (lambda (a b) (list (%pc-scale-tree a) (%pc-scale-tree b)))))
        (else (nb::%make-traceable-function
               (list 'a 'b) (lambda (a b) (list (%pc-scale-tree (%pc-scale-tree a)) (%pc-scale-tree b))))))
    (let ((f (nb::%make-traceable-function
              (list 'p 'a 'b)
              (lambda (p a b) (nb:cond* (funcall *pc-lt* p 0.5) then else a b)))))
      (dolist (p '(0.0 1.0))
        (is (%pc-jit-matches-eager-p f backend
                                     (list (nb::%scalar-array p :f32) (%pc-tree 21) (list (%pc-tree 31))))
            "p ~A で jit と eager の cond* が一致しなかった" p))
      (gc-and-run-finalizers))))

(defun %pc-scan-body (carry x)
  (let* ((s (nb:reduce-sum (svref (first x) 0)))
         (new (nb:tree-map (lambda (c) (funcall *pc-step* c s)) carry)))
    (values new (list (%pc-scale-tree x) new))))

(defparameter *pc-scan-f* (nb::%make-traceable-function (list 'carry 'x) #'%pc-scan-body))

(defun %pc-xs (length seed)
  (list (vector (make-random-array (make-array-spec (list length 2) :f32) :seed seed))))

(define-iree-test pytree-control/jit-scan-with-tree-carry-xs-ys-matches-eager
    "木の init / xs / ys の scan を jit した結果は、reverse でも eager と同じ treedef・同じ値になる。
init はランダムな木（リスト・ベクタ・登録した節）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 4)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (pytree-spec) (uniform-integer :lo 0 :hi 1)))
                  (lambda (case)
                    (destructuring-bind (spec reverse) case
                      (let ((init (pytree-array-tree spec :f32 :max-leaves 4)))
                        (or (null (nb:tree-leaves init))
                            (let ((f (nb::%make-traceable-function
                                      (list 'init 'xs)
                                      (lambda (init xs)
                                        (nb:scan *pc-scan-f* init xs :reverse (= reverse 1))))))
                              (%pc-jit-matches-eager-p f backend (list init (%pc-xs 3 7))))))))
                  :regression-id pytree-control/jit-scan-with-tree-carry-xs-ys-matches-eager
                  :regression-file (regression-path "iree-pytree-control-jit-scan"
                                                    :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test pytree-control/jit-grad-through-tree-carry-scan-matches-eager-grad
    "木の carry の scan を通る (jit (grad f)) は、jit しない grad と同じ treedef で葉ごとに一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (xs (%pc-xs 3 9))
         (f (nb::%make-traceable-function
             (list 'init)
             (lambda (init)
               (multiple-value-bind (carry ys) (nb:scan *pc-scan-f* init xs)
                 (reduce *pc-add*
                         (mapcar #'nb:reduce-sum (nb:tree-leaves (list carry ys)))))))))
    (let ((g (nb:grad f))
          (init (%pc-tree 41)))
      (let ((jg (nb:jit g :backend backend)))
        (unwind-protect
             (is (%leaves-close-p (funcall jg init) (funcall g init) :f32))
          (nb::%jit-cache-forget g))))
    (gc-and-run-finalizers)))
