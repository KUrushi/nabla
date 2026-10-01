;;;; f64 と :i1 を jit で IREE の local backend に通す end-to-end テスト
;;;; （issue #72）。to-device / to-host / invoke が f64 と :i1 を扱えるように
;;;; なったことで、op 対応表（docs/stablehlo-ops.md）の全 dtype が jit で
;;;; 実行できることを確かめる。
;;;;
;;;; tests/iree/jit-test.lisp と同じく、distinct な (関数・shape・dtype) の
;;;; 組は1回の IREE コンパイルになるので、試行回数・shape は小さく抑える。
;;;; 試行ごとに新しい TRACEABLE-FUNCTION を作るので、*JIT-CACHE* のエントリは
;;;; 毎回 %JIT-CACHE-FORGET で消す（jit-test.lisp の %JIT-PBT-MATCHES-EAGER-P
;;;; のコメント参照）。

(in-package #:nabla.iree.tests)

(defun %jit-dtype-matches-eager-p (backend f arrays compare)
  "F（TRACEABLE-FUNCTION）を BACKEND 上で JIT して ARRAYS に適用した結果と、
F を eager に直接呼んだ結果を COMPARE（2引数の述語: jit の結果、eager の
結果）で比べた真偽値を返す。終わったら F の jit キャッシュを消す。"
  (unwind-protect
       (funcall compare (apply (nb:jit f :backend backend) arrays) (apply f arrays))
    (nb::%jit-cache-forget f)))

(define-iree-test jit-dtype/f64-elementwise-dot-reduce-matches-eager
    "f64 の引数を取り、エレメントワイズ（+ max neg）・dot・reduce-sum を
組み合わせた関数を IREE 上で jit した結果は、eager 実装の結果と
allclose :dtype :f64（rtol = atol = 1e-12）で一致する（issue #72）。
IREE は既定で f64 を f32 に落とす（--iree-input-demote-f64-to-f32）ので、
その落とし込みが効いていればこの許容誤差では一致しない。exp / log / tanh
は jit-dtype/f64-transcendentals-match-eager で別に確かめる。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 8)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 1 :hi 4)
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (m k n seed) args
                      (%jit-dtype-matches-eager-p
                       backend
                       (nb:with-tracing (a w b)
                         (nb:reduce-sum (max (+ (nb:dot a w) b) (- b)) :axes '(1)))
                       (list (make-random-array (make-array-spec (list m k) :f64) :seed seed)
                             (make-random-array (make-array-spec (list k n) :f64) :seed (+ seed 1))
                             (make-random-array (make-array-spec (list m n) :f64) :seed (+ seed 2)))
                       (lambda (actual expected)
                         (and (eq :f64 (nb:array-dtype actual))
                              (allclose actual expected :dtype :f64))))))
                  :regression-id jit-dtype/f64-elementwise-dot-reduce-matches-eager
                  :regression-file (regression-path "iree-jit-dtype-f64" :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test jit-dtype/f64-transcendentals-match-eager
    "f64 の exp / log / tanh を IREE 上で jit した結果（多値の3つすべて）は、
eager 実装の結果と allclose :dtype :f64（rtol = atol = 1e-12）で一致する。
llvm-cpu はこれらの f64 版を多項式近似せず libm の呼び出しとして残すので、
embedded linker（-nostdlib）ではリンクできない。BACKEND-COMPILE が
%needs-libm-p でそれを見つけて system library（dlopen で読み込む共有
ライブラリ）を作らせ、プロセスの libm に解決させることを確かめる。
log の引数は定義域内（:positive）に限る。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 8)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (array-spec :dtypes '(:f64) :max-rank 3 :max-dim 5)
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (spec seed) args
                      (let ((f (nb:with-tracing (a p) (values (exp a) (log p) (tanh a))))
                            (a (make-random-array spec :seed seed))
                            (p (make-random-array spec :seed (+ seed 1) :domain :positive)))
                        (unwind-protect
                             (every (lambda (actual expected) (allclose actual expected :dtype :f64))
                                    (multiple-value-list (funcall (nb:jit f :backend backend) a p))
                                    (multiple-value-list (funcall f a p)))
                          (nb::%jit-cache-forget f)))))
                  :regression-id jit-dtype/f64-transcendentals-match-eager
                  :regression-file (regression-path "iree-jit-dtype-f64-transcendental" :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test jit-dtype/comparison-result-matches-eager-as-bit-array
    "f32 / f64 の引数から比較（<）の結果（:i1）を返す関数を IREE 上で jit
すると、eager 実装と同じ BIT 配列（equalp）が返る（issue #72）。比較の
両辺は1回の IEEE 演算（+ と -）だけで作るので、IREE と eager で丸めが
食い違って比較結果が反転することはない。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 8)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64) :max-rank 3 :max-dim 5)
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (spec seed) args
                      (%jit-dtype-matches-eager-p
                       backend
                       (nb:with-tracing (a b) (< (+ a b) (- a b)))
                       (list (make-random-array spec :seed seed)
                             (make-random-array spec :seed (+ seed 1)))
                       (lambda (actual expected)
                         (and (eq :i1 (nb:array-dtype actual))
                              (equalp actual expected))))))
                  :regression-id jit-dtype/comparison-result-matches-eager-as-bit-array
                  :regression-file (regression-path "iree-jit-dtype-i1-output" :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))

(define-iree-test jit-dtype/i1-argument-selects-like-eager
    ":i1（BIT 配列）を引数に取り、where で選ぶ関数を IREE 上で jit すると、
eager 実装と同じ f32 配列（equalp。選ぶだけで丸めは入らない）が返る
（issue #72。:i1 が to-device を通って invoke の入力になれる）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 8)
        (nb:*compile-cache-directory* nil))
    (is (check-it (generator (tuple (array-spec :dtypes '(:f32) :max-rank 3 :max-dim 5)
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (args)
                    (destructuring-bind (spec seed) args
                      (%jit-dtype-matches-eager-p
                       backend
                       (nb:with-tracing (c a b) (nb:where c a b))
                       (list (make-random-array (make-array-spec (array-spec-shape spec) :i1) :seed seed)
                             (make-random-array spec :seed (+ seed 1))
                             (make-random-array spec :seed (+ seed 2)))
                       #'equalp)))
                  :regression-id jit-dtype/i1-argument-selects-like-eager
                  :regression-file (regression-path "iree-jit-dtype-i1-input" :package "NABLA.IREE.TESTS")))
    (gc-and-run-finalizers)))
