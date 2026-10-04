;;;; nb:scan の StableHLO（stablehlo.while + dynamic_slice / dynamic_update_slice）を
;;;; IREE の local backend でコンパイル・実行し、eager（eval-graph）と比べる
;;;; medium テスト（issue #132）。graph ごとに1回コンパイルし、PBT の各試行で再利用する。

(in-package #:nabla.iree.tests)

(defun %scan-iree-arrays (graph seed)
  (loop for var in (nb:graph-invars graph) for i from 0
        for aval = (nb:var-aval var)
        collect (make-random-array (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval))
                                   :seed (+ seed i))))

(defun %scan-iree-matches-eager-p (backend module graph seed)
  (let* ((arrays (%scan-iree-arrays graph seed))
         (avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
         (device-arrays nil)
         (results nil))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a aval) (to-device a backend :dtype (nb:aval-dtype aval)))
                                       arrays avals))
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e var)
                           (let ((host (to-host r)))
                             (and (equal (array-dimensions host) (array-dimensions e))
                                  (if (eq (nb:aval-dtype (nb:var-aval var)) :i1)
                                      (equalp host e)
                                      (allclose host e :dtype (nb:aval-dtype (nb:var-aval var)))))))
                         results expected (nb:graph-outvars graph)))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(defmacro %with-scan-iree-check ((graph name) message)
  `(let* ((backend (nabla:find-backend :iree))
          (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo ,graph)))))
     (unwind-protect
          (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                        (lambda (seed) (%scan-iree-matches-eager-p backend module ,graph seed))
                        :regression-id ,(intern (string-upcase (format nil "scan/iree-~A" name)))
                        :regression-file (regression-path ,(format nil "iree-scan-~A" name)
                                                          :package "NABLA.IREE.TESTS"))
              ,message)
       (nabla:backend-unload backend module))))

(defun %scan-iree-mixed-graph (length reverse)
  "carry = (h:f32 [3], c:i32)、xs = (u:[3], v:[2])、ys = (h*u, v, c) の scan を含む graph。"
  (nb::trace-to-graph
   (nb:with-tracing (h c u v)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    (let ((h (first carry)) (c (second carry)) (u (first x)) (v (second x)))
                      (values (list (tanh (+ (* h 0.5) u)) (+ c 1))
                              (list (* h u) v c))))
                  (list h c) (list u v) :length length :reverse reverse)
       (values (first carry) (second carry) (first ys) (second ys) (third ys))))
   (list (nb:make-aval '(3) :f32) (nb:make-aval '() :i32)
         (nb:make-aval (list length 3) :f32) (nb:make-aval (list length 2) :f32))))

(define-iree-test scan/iree-mixed-forward-matches-eager
    "carry / xs / ys の形と dtype（f32 と i32）が混ざった scan が、IREE の実行結果と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 4 nil)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-mixed-reverse-matches-eager
    "reverse の scan（添字 length-1 から 0）も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 3 t)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の reverse scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-length-one-matches-eager
    "長さ 1 の scan も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 1 nil)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の長さ 1 の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-length-zero-matches-eager
    "長さ 0 の scan（carry は素通し、ys は先頭の軸が 0 の空の配列）も IREE が受け付け、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-mixed-graph 0 nil)))
    (%with-scan-iree-check (graph "matches-lisp-loop") "IREE の長さ 0 の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-without-xs-matches-eager
    "xs の無い scan（length だけ指定）も IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 x
                                 (values (list (* (first carry) 0.9)) (list (first carry))))
                               (list h) '() :length 5 :reverse t)
                    (values (first carry) (first ys))))
                (list (nb:make-aval '(2 2) :f32)))))
    (%with-scan-iree-check (graph "without-xs") "IREE の xs 無しの scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-closure-consts-match-eager
    "本体が閉包で捕まえた外側の値（consts）を使う scan も IREE と eager で一致する。
scan の結果を使う後続の演算も含める。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h0 xs w)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (tanh (+ (* (first carry) w) (first x)))))
                                   (values (list h) (list (* h w)))))
                               (list h0) (list xs) :reverse t)
                    carry
                    (+ (* (first ys) 2.0) 1.0)))
                (list (nb:make-aval '(3) :f32) (nb:make-aval '(4 3) :f32) (nb:make-aval '(3) :f32)))))
    (%with-scan-iree-check (graph "closure") "IREE の consts つき scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-bf16-carry-and-i1-ys-match-eager
    "bf16 の carry と、x の符号を調べる i1 の ys（IREE は bf16 の演算を融合して丸めの回数が eager と
違いうるので、i1 は丸めに左右されない入力の比較にする）（ys バッファの 0 の初期値のリテラルが dtype ごとに違う）も
IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h u)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (+ (first carry) (first x))))
                                   (values (list h) (list h (< (first x) 0.0)))))
                               (list h) (list u))
                    (values (first carry) (first ys) (second ys))))
                (list (nb:make-aval '(3) :bf16) (nb:make-aval '(4 3) :bf16)))))
    (%with-scan-iree-check (graph "bf16-i1") "IREE の bf16 / i1 の scan の結果が eager と一致しなかった")))

;;; ---- IREE 3.11 の Stream AffinityAnalysis のクラッシュの回避（optimization_barrier） ----

(define-iree-test scan/iree-repeated-jit-compiles-do-not-crash
    "while のカウンタと ys の0初期値が constant のままだと、IREE 3.11 の Stream の
AffinityAnalysis が非決定的に落ちる（docs/stablehlo-ops.md）。carry が2つ以上（rank 1 以上を
含む）と ys を持つ scan を、ディスクキャッシュ無しで5回別々にコンパイルして実行しても
落ちず、eager と一致する。"
  (skip-unless-iree :library :both)
  (let ((nb:*compile-cache-directory* nil)
        (backend (nabla:find-backend :iree)))
    (dotimes (i 5)
      (let* ((f (nb:with-tracing (h s xs)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (first carry)) (s (second carry)) (u (first x)))
                                   (values (list (+ (* h 0.5) u) (+ (* s 0.9) 0.9))
                                           (list (* h 0.5)))))
                               (list h s) (list xs) :reverse t)
                    (values (first carry) (second carry) (first ys)))))
             (h (make-random-array (make-array-spec '(3) :f32) :seed i))
             (s (make-random-array (make-array-spec '() :f32) :seed (+ i 10)))
             (xs (make-random-array (make-array-spec '(4 3) :f32) :seed (+ i 20)))
             (jitted (nb:jit f :backend backend))
             (actual (multiple-value-list (funcall jitted h s xs)))
             (expected (multiple-value-list (funcall f h s xs))))
        (is (every (lambda (a e) (allclose a e :dtype :f32)) actual expected))))))

;;; ---- ys の書き込みの性能（issue #159） ----

(define-iree-test scan/iree-same-typed-ys-do-not-share-a-buffer
    "同じ形と dtype の ys が2つある scan（jvp / vjp の scan はこの形になる）も IREE と eager で一致する。
ys のバッファは in-place に書き換えるので、0 の初期値が CSE で1つのバッファにまとめられると、
2つの ys が同じ値になってしまう（issue #159）。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h xs)
                  (multiple-value-bind (carry ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (+ (first carry) (first x))))
                                   (values (list h) (list h (- h)))))
                               (list h) (list xs))
                    (values (first carry) (first ys) (second ys))))
                (list (nb:make-aval '(3) :f32) (nb:make-aval '(4 3) :f32)))))
    (%with-scan-iree-check (graph "same-typed-ys") "IREE の同じ型の ys が2つある scan の結果が eager と一致しなかった")))


(defun %scan-ys-cost-graph (length width with-ys)
  "carry h:f32 [WIDTH] を LENGTH 回 tanh(h)+h で更新する scan の graph。WITH-YS なら各 h を ys に積んで返す。"
  (nb::trace-to-graph
   (nb:with-tracing (h)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (carry x)
                    x
                    (let ((h (+ (tanh (first carry)) (first carry))))
                      (values (list h) (if with-ys (list h) '()))))
                  (list h) '() :length length)
       (if with-ys (values (first carry) (first ys)) (first carry))))
   (list (nb:make-aval (list width) :f32))))

(defun %scan-ys-cost-best-ms (backend graph repeats)
  "GRAPH をコンパイルし、REPEATS 回の backend-invoke（結果はデバイスに置いたまま）の最短時間（ミリ秒）。"
  (let ((module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
        (input (to-device (make-random-array (make-array-spec (nb:aval-shape (nb:var-aval (first (nb:graph-invars graph))))
                                                              :f32)
                                             :seed 0)
                          backend :dtype :f32)))
    (unwind-protect
         (loop repeat repeats
               minimize (let* ((start (get-internal-real-time))
                               (results (multiple-value-list
                                         (nabla:backend-invoke backend module "main" input)))
                               (end (get-internal-real-time)))
                          (mapc #'release-device-array results)
                          (/ (* 1000.0 (- end start)) internal-time-units-per-second)))
      (release-device-array input)
      (nabla:backend-unload backend module))))

(define-iree-test/large scan/iree-ys-write-cost-is-linear-in-length
    "ys を積む scan の実行時間は、ys を積まない同じ scan と同程度（ys の書き込みが1行ぶんの
コピーで済み、ys のバッファ全体を毎ステップコピーしない）。IREE 3.11 は while の carry を
本体で使うたびに丸ごとコピーするので、何もしないと長さ 1000 × 幅 1024 で約 50 倍遅い
（docs/stablehlo-ops.md、issue #159）。タイミングの揺れで落ちないよう、最短時間の比較に
5 倍 + 100 ms の余裕を持たせる。実行時間はマシンの負荷で揺れるので large に置き、既定のスイートでは
出力の形を検査する scan/emits-ys-buffers-through-barriers（tests/scan-test.lisp）で守る。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (without-ys (%scan-ys-cost-best-ms backend (%scan-ys-cost-graph 1000 1024 nil) 3))
         (with-ys (%scan-ys-cost-best-ms backend (%scan-ys-cost-graph 1000 1024 t) 3)))
    (is (<= with-ys (+ (* 5 without-ys) 100))
        "ys あり ~,1F ms、ys なし ~,1F ms（許容は ys なしの 5 倍 + 100 ms）" with-ys without-ys)))

;;; ---- issue #166 (f): rank 0 の x_t、rank 2 の y_t、入れ子の scan ----

(define-iree-test scan/iree-rank0-xs-matches-eager
    "xs の形が (L) で x_t が rank 0 になる scan（carry と y_t も rank 0）が、順方向・逆方向とも
IREE と eager で一致する（dynamic_slice の結果を rank 0 に reshape する経路）。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h xs)
                  (multiple-value-bind (fwd-carry fwd-ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (+ (* (first carry) 0.5) (first x))))
                                   (values (list h) (list (* h (first x))))))
                               (list h) (list xs))
                    (multiple-value-bind (rev-carry rev-ys)
                        (nb:scan (nb:with-tracing (carry x)
                                   (let ((h (+ (* (first carry) 0.5) (first x))))
                                     (values (list h) (list (* h (first x))))))
                                 (list h) (list xs) :reverse t)
                      (values (first fwd-carry) (first fwd-ys) (first rev-carry) (first rev-ys)))))
                (list (nb:make-aval '() :f32) (nb:make-aval '(5) :f32)))))
    (%with-scan-iree-check (graph "rank0-xs") "IREE の rank 0 の x_t の scan の結果が eager と一致しなかった")))

(define-iree-test scan/iree-rank2-ys-matches-eager
    "y_t が rank 2（ys が rank 3）になる scan が、順方向・逆方向とも IREE と eager で一致する
（dynamic_update_slice の添字が rank 2 の y_t の分だけ 0 で埋まる経路）。"
  (skip-unless-iree :library :both)
  (let ((graph (nb::trace-to-graph
                (nb:with-tracing (h xs)
                  (multiple-value-bind (fwd-carry fwd-ys)
                      (nb:scan (nb:with-tracing (carry x)
                                 (let ((h (tanh (+ (first carry) (first x)))))
                                   (values (list h) (list h (* (first carry) (first x))))))
                               (list h) (list xs))
                    (multiple-value-bind (rev-carry rev-ys)
                        (nb:scan (nb:with-tracing (carry x)
                                   (let ((h (tanh (+ (first carry) (first x)))))
                                     (values (list h) (list h (* (first carry) (first x))))))
                                 (list h) (list xs) :reverse t)
                      (values (first fwd-carry) (first fwd-ys) (second fwd-ys)
                              (first rev-carry) (first rev-ys) (second rev-ys)))))
                (list (nb:make-aval '(2 3) :f32) (nb:make-aval '(4 2 3) :f32)))))
    (%with-scan-iree-check (graph "rank2-ys") "IREE の rank 2 の y_t の scan の結果が eager と一致しなかった")))

(defun %scan-iree-nested-graph (outer-reverse inner-reverse)
  "外側の scan の本体の中で内側の scan を回す graph。内側の本体は外側の carry h と、
graph の引数 w を閉包で捕まえる。"
  (nb::trace-to-graph
   (nb:with-tracing (h0 xss w)
     (multiple-value-bind (carry ys)
         (nb:scan (nb:with-tracing (outer-carry xs)
                    (let ((h (first outer-carry)))
                      (multiple-value-bind (inner-carry inner-ys)
                          (nb:scan (nb:with-tracing (carry x)
                                     (let ((g (tanh (+ (* (first carry) w) (first x) h))))
                                       (values (list g) (list (* g w)))))
                                   (list h) (list (first xs)) :reverse inner-reverse)
                        (values (list (first inner-carry)) (list (first inner-ys))))))
                  (list h0) (list xss) :reverse outer-reverse)
       (values (first carry) (first ys))))
   (list (nb:make-aval '(3) :f32) (nb:make-aval '(4 2 3) :f32) (nb:make-aval '(3) :f32))))

(define-iree-test scan/iree-nested-forward-reverse-matches-eager
    "順方向の外側の scan の中で逆方向の内側の scan を回す入れ子の scan（内側は外側の carry と
外側の引数を閉包で捕まえる）が、IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-nested-graph nil t)))
    (%with-scan-iree-check (graph "nested") "IREE の入れ子の scan（外側 順・内側 逆）の結果が eager と一致しなかった")))

(define-iree-test scan/iree-nested-reverse-forward-matches-eager
    "逆方向の外側の scan の中で順方向の内側の scan を回す入れ子の scan も、IREE と eager で一致する。"
  (skip-unless-iree :library :both)
  (let ((graph (%scan-iree-nested-graph t nil)))
    (%with-scan-iree-check (graph "nested") "IREE の入れ子の scan（外側 逆・内側 順）の結果が eager と一致しなかった")))
