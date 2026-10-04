;;;; PRNG の公開 API と、バッチ次元つき rng-bit-generator の IREE（local）での実行
;;;; （issue #136、medium）。
;;;;
;;;; - バッチ次元つきの状態（stablehlo.while で行ごとに rng_bit_generator を回す
;;;;   StableHLO）の出力が、eager とビット単位で一致する。
;;;; - jit した uniform / normal / split / fold-in が eager と一致する（整数の split / fold-in は
;;;;   ビット単位、浮動小数点は許容誤差つき。f32 は exp / log の実装の差と乗加算の融合があり得るので
;;;;   rtol 1e-4。normal の裾（erf の逆関数が大きくなる所）で差が増幅される）。
;;;; - vmap でバッチしたキーの jit。

(in-package #:nabla.iree.tests)

(defun %prng-iree-states (shape seed)
  "SHAPE（末尾が 2）の ui64 の状態配列（全部ランダムな 64 ビット）。"
  (let ((rs (sb-ext:seed-random-state seed))
        (states (make-array shape :element-type '(unsigned-byte 64))))
    (dotimes (i (array-total-size states) states)
      (setf (row-major-aref states i) (random (expt 2 64) rs)))))

(defun %prng-iree-batched-graph (state-shape shape dtype)
  (nb:trace-to-graph (nb:with-tracing (s) (nb::rng-bit-generator s :shape shape :dtype dtype))
                     (list (nb:make-aval state-shape :u64))))

(defun %prng-iree-run (backend graph &rest arrays)
  "GRAPH を IREE にコンパイルして ARRAYS で実行し、出力のホスト配列のリストを返す。"
  (let ((module (nabla:backend-load
                 backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect
         (let ((devices (mapcar (lambda (a) (to-device a backend :dtype (nb:array-dtype a))) arrays)))
           (unwind-protect
                (let ((outputs (multiple-value-list
                                (apply #'nabla:backend-invoke backend module "main" devices))))
                  (unwind-protect (mapcar #'to-host outputs)
                    (mapc #'release-device-array outputs)))
             (mapc #'release-device-array devices)))
      (nabla:backend-unload backend module))))

(define-iree-test iree/prng/batched-rng-bit-generator-matches-eager
  "バッチ次元つきの状態（行数 1・2・5 と2段の (2 3)）× 形 × :u32 / :u64 で、IREE の出力
（新しい状態・ビット）が eager とビット単位で一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree)))
    (dolist (state-shape '((1 2) (2 2) (5 2) (2 3 2)))
      (dolist (shape '(() (1) (3) (4) (2 3) (3 3)))
        (dolist (dtype '(:u32 :u64))
          (let* ((graph (%prng-iree-batched-graph state-shape shape dtype))
                 (states (%prng-iree-states state-shape 3))
                 (iree (%prng-iree-run backend graph states))
                 (eager (multiple-value-list (nb:eval-graph graph states))))
            (is (and (equalp (first iree) (first eager)) (equalp (second iree) (second eager)))
                "状態の shape ~S・bits の shape ~S・dtype ~S: IREE が eager と一致しなかった"
                state-shape shape dtype)))))))

;; ビットのバッファは本体で optimization_barrier を通して in-place に書き換える（issue #159 と同じ
;; 回避）。入力のデバイス配列や前の呼び出しの出力と記憶を共有してしまうと、2回目の呼び出しで
;; 結果が変わる。それを同じデバイス配列で2回呼んで確かめる。
(define-iree-test iree/prng/batched-rng-same-device-input-twice-gives-identical-bits
  "同じデバイス上の状態で、バッチされた rng-bit-generator のモジュールを2回呼ぶと、2回とも
同じ（eager と一致する）状態とビットを返し、入力の状態も書き換わらない。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (states (%prng-iree-states '(5 2) 11))
         (graph (%prng-iree-batched-graph '(5 2) '(3 4) :u32))
         (eager (multiple-value-list (nb:eval-graph graph states)))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect
         (let ((device (to-device states backend :dtype :u64)))
           (unwind-protect
                (flet ((call ()
                         (let ((outputs (multiple-value-list
                                         (nabla:backend-invoke backend module "main" device))))
                           (unwind-protect (mapcar #'to-host outputs)
                             (mapc #'release-device-array outputs)))))
                  (let* ((first-call (call))
                         (second-call (call)))
                    (is (equalp first-call second-call) "2回目の呼び出しの結果が1回目と違う")
                    (is (equalp first-call eager) "1回目の呼び出しの結果が eager と違う")
                    (is (equalp (to-host device) states) "入力の状態が書き換わった")))
             (release-device-array device)))
      (nabla:backend-unload backend module))))

(define-iree-test iree/prng/two-batched-rngs-on-the-same-state-do-not-share-a-buffer
  "同じ状態からバッチされた rng-bit-generator を2回呼ぶ graph（ビットのバッファの型も同じ）でも、
2つの出力がそれぞれ eager と一致する。ビットのバッファの初期値の barrier が CSE でまとめられると、
2つのループが同じバッファに書き込む（scan の ys と同じ。docs/stablehlo-ops.md）。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (states (%prng-iree-states '(4 2) 12))
         (graph (nb:trace-to-graph
                 (nb:with-tracing (s)
                   (multiple-value-bind (s1 b1) (nb::rng-bit-generator s :shape '(5) :dtype :u32)
                     (multiple-value-bind (s2 b2) (nb::rng-bit-generator s1 :shape '(5) :dtype :u32)
                       (multiple-value-bind (s3 b3) (nb::rng-bit-generator s :shape '(5) :dtype :u32)
                         (values s2 b1 b2 s3 b3)))))
                 (list (nb:make-aval '(4 2) :u64))))
         (iree (%prng-iree-run backend graph states))
         (eager (multiple-value-list (nb:eval-graph graph states))))
    (is (equalp iree eager) "同じ状態から2回・続けて1回呼んだバッチされた rng が eager と一致しなかった")))

(defun %prng-iree-jit-matches-eager (name fn args &key (rtol 1d-4) (atol 1d-5) exact)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (jitted (nb:jit fn :backend backend))
         (compiled (apply jitted args))
         (eager (apply fn args)))
    (unwind-protect
         (if exact
             (is (equalp compiled eager) "~A: jit の結果が eager と一致しなかった" name)
             (is (allclose compiled eager :dtype (nb:array-dtype eager) :rtol rtol :atol atol)
                 "~A: jit の結果が eager と許容誤差内で一致しなかった" name))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))

(define-iree-test iree/prng/jit-matches-eager
  "jit した uniform / normal（f32 / f64）、split、fold-in（整数のデータ・トレースされたデータ）が eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((key (nb:prng-key 2026)))
    (%prng-iree-jit-matches-eager "split" (nb:with-tracing (k) (nb:split k 5)) (list key) :exact t)
    (%prng-iree-jit-matches-eager "fold-in" (nb:with-tracing (k) (nb:fold-in k 9)) (list key) :exact t)
    (%prng-iree-jit-matches-eager "fold-in トレースされたデータ"
                                  (nb:with-tracing (k i) (nb:fold-in k i))
                                  (list key (make-array '() :element-type '(unsigned-byte 32) :initial-element 77))
                                  :exact t)
    (%prng-iree-jit-matches-eager "uniform f32" (nb:with-tracing (k) (nb:uniform k '(7 5)))
                                  (list key) :rtol 1d-5 :atol 1d-6)
    (%prng-iree-jit-matches-eager "uniform f32 範囲つき"
                                  (nb:with-tracing (k) (nb:uniform k '(9) :minval -3 :maxval 5))
                                  (list key) :rtol 1d-5 :atol 1d-6)
    (%prng-iree-jit-matches-eager "uniform f64"
                                  (nb:with-tracing (k) (nb:uniform k '(6) :dtype :f64 :minval 2 :maxval 3))
                                  (list key) :rtol 1d-12 :atol 1d-12)
    (%prng-iree-jit-matches-eager "normal f32" (nb:with-tracing (k) (nb:normal k '(64 4)))
                                  (list key))
    (%prng-iree-jit-matches-eager "normal f64" (nb:with-tracing (k) (nb:normal k '(33) :dtype :f64))
                                  (list key) :rtol 1d-9 :atol 1d-9)))

(define-iree-test iree/prng/f64-erf-inv-tails-match-eager
  "normal の :f64 が使う erf の逆関数（倍精度用の3区間の近似）を、全区間（w < 6.25、< 16、
それ以上）にまたがる x で jit したものが eager と一致する。裾は乱数では引けないので内部関数を直接呼ぶ。"
  (skip-unless-iree :library :both)
  (let ((x (make-array 8 :element-type 'double-float
                         :initial-contents (list 0d0 0.5d0 -0.9d0 0.998d0 (- 1 1d-5) (- (- 1 1d-7))
                                                 (- 1 1d-12) (- 1 (scale-float 1d0 -53))))))
    (%prng-iree-jit-matches-eager "erf-inv f64"
                                  (nb:with-tracing (v) (nb::%prng-dispatch (list v) #'nb::%prng-erf-inv))
                                  (list x) :rtol 1d-12 :atol 1d-12)))

(define-iree-test iree/prng/jit-is-deterministic-across-calls
  "同じキーでの jit 呼び出しは何度でも同じ値、別のキーなら別の値になる。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (jitted (nb:jit (nb:with-tracing (k) (nb:normal k '(16))) :backend backend))
         (a (funcall jitted (nb:prng-key 1)))
         (b (funcall jitted (nb:prng-key 1)))
         (c (funcall jitted (nb:prng-key 2))))
    (unwind-protect
         (progn (is (equalp a b))
                (is (not (equalp a c))))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))

(define-iree-test iree/prng/jit-vmap-over-keys-matches-per-key-calls
  "(jit (vmap f)) でキーをバッチした結果が、各キーで単独に eager で呼んだ結果と一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (nb:*compile-cache-directory* nil)
         (f (nb:with-tracing (k) (nb:uniform k '(4 3))))
         (jitted (nb:jit (nb:vmap f) :backend backend))
         (keys (nb:split (nb:prng-key 9) 5))
         (batched (funcall jitted keys))
         (expected (first (reference-vmap f (list keys)))))
    (unwind-protect
         (is (allclose batched expected :dtype :f32))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jitted)))
    (gc-and-run-finalizers)))
