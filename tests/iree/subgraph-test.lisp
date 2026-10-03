;;;; サブグラフを持つ複数出力の eqn の medium テスト（issue #127）。
;;;;
;;;; テスト専用の高階プリミティブ %TEST-CALL-SUBGRAPH（tests/support/
;;;; subgraph-primitive.lisp）を含む graph を emit-stablehlo し、IREE の
;;;; local backend でコンパイル・実行した結果が、本体を直接評価した結果と
;;;; 一致することを確かめる。StableHLO は stablehlo.case の1枝のリージョン
;;;; （理由は subgraph-primitive.lisp の冒頭）。コンパイルは graph ごとに1回。

(in-package #:nabla.iree.tests)

(defun %subgraph-iree-matches-direct-p (backend module graph direct-graph seed)
  "GRAPH（高階プリミティブを含む）をコンパイルした MODULE を IREE で実行した結果が、
DIRECT-GRAPH（本体を直接呼ぶ graph）の eval-graph の結果と一致するか。"
  (let* ((avals (mapcar #'nb:var-aval (nb:graph-invars graph)))
         (arrays (loop for aval in avals for i from 0
                       collect (make-random-array
                                (make-array-spec (nb:aval-shape aval) (nb:aval-dtype aval))
                                :seed (+ seed i))))
         (device-arrays nil)
         (results nil))
    (unwind-protect
         (progn
           (setf device-arrays (mapcar (lambda (a) (to-device a backend :dtype :f32)) arrays))
           (setf results (multiple-value-list
                          (apply #'nabla:backend-invoke backend module "main" device-arrays)))
           (let ((expected (multiple-value-list (apply #'nb:eval-graph direct-graph arrays)))
                 (via-eager (multiple-value-list (apply #'nb:eval-graph graph arrays))))
             (and (= (length results) (length expected))
                  (every (lambda (r e) (allclose (to-host r) e :dtype :f32)) results expected)
                  ;; eager（eval-graph）の高階プリミティブも、直接呼んだ結果と一致する。
                  (every (lambda (v e) (allclose v e :dtype :f32)) via-eager expected))))
      (dolist (r results) (release-device-array r))
      (dolist (da device-arrays) (release-device-array da)))))

(defmacro %with-subgraph-iree-check ((backend graph direct name) message)
  "GRAPH を1回だけコンパイル・ロードし、PBT の各試行でその module を再利用して
%SUBGRAPH-IREE-MATCHES-DIRECT-P を確かめる。"
  `(let ((module (nabla:backend-load ,backend (nabla:backend-compile ,backend (nb:emit-stablehlo ,graph)))))
     (unwind-protect
          (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                        (lambda (seed) (%subgraph-iree-matches-direct-p ,backend module ,graph ,direct seed))
                        :regression-id ,(intern (string-upcase (format nil "subgraph/iree-~A" name)))
                        :regression-file (regression-path ,(format nil "iree-subgraph-~A" name)
                                                          :package "NABLA.IREE.TESTS"))
              ,message)
       (nabla:backend-unload ,backend module))))

(defparameter *subgraph-iree-avals*
  (list (nb:make-aval '(3 5) :f32) (nb:make-aval '(3 5) :f32)))

(define-iree-test subgraph/iree-multiple-outputs-match-direct-body
    "複数出力の本体を %TEST-CALL-SUBGRAPH 経由で IREE 実行した結果は、本体を直接
評価した結果と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (graph (nb::trace-to-graph
                (nb:with-tracing (x y)
                  (let ((r (test-call-subgraph (nb:with-tracing (a b) (values (+ a b) (* a b))) x y)))
                    (values (first r) (second r))))
                *subgraph-iree-avals*))
        (direct (nb::trace-to-graph (nb:with-tracing (a b) (values (+ a b) (* a b)))
                                    *subgraph-iree-avals*)))
    (%with-subgraph-iree-check (backend graph direct "multiple-outputs")
      "IREE の実行結果が本体を直接評価した結果と一致しなかった")))

(define-iree-test subgraph/iree-closure-captured-inputs-match-direct-body
    "本体が閉包で捕まえた外側の値（追加の入力に持ち上げたもの）も、IREE の実行結果が
本体を直接評価した結果と一致する。後続の演算が結果を使う graph でも動く。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (graph (nb::trace-to-graph
                (nb:with-tracing (x y)
                  (let ((r (test-call-subgraph (nb:with-tracing (a) (+ (* a y) y)) x)))
                    (- (first r))))
                *subgraph-iree-avals*))
        (direct (nb::trace-to-graph (nb:with-tracing (x y) (- (+ (* x y) y)))
                                    *subgraph-iree-avals*)))
    (%with-subgraph-iree-check (backend graph direct "closure-captured")
      "IREE の実行結果が本体を直接評価した結果と一致しなかった")))

(define-iree-test subgraph/iree-nested-regions-match-direct-body
    "入れ子のリージョン（本体の中でさらに高階プリミティブを呼ぶ）も IREE が受け付け、
直接評価した結果と一致する。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (graph (nb::trace-to-graph
                (nb:with-tracing (x y)
                  (first (test-call-subgraph
                          (nb:with-tracing (a)
                            (first (test-call-subgraph (nb:with-tracing (b) (+ b y)) a)))
                          x)))
                *subgraph-iree-avals*))
        (direct (nb::trace-to-graph (nb:with-tracing (x y) (+ x y)) *subgraph-iree-avals*)))
    (%with-subgraph-iree-check (backend graph direct "nested-regions")
      "IREE の実行結果が本体を直接評価した結果と一致しなかった")))

;;; ---- while 形のリージョン: carry はブロック引数、閉包で捕まえた値は外側の名前 ----

(define-iree-test subgraph/iree-while-region-mixes-block-args-and-outer-names
    "stablehlo.while の cond / body が、carry をブロック引数、閉包で捕まえた値を外側の
SSA 名として使うリージョンを IREE が受け付け、eager の結果と一致する。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (avals (list (nb:make-aval '() :f32) (nb:make-aval '() :f32)))
         (graph (nb::trace-to-graph
                 (nb:with-tracing (x s)
                   (test-while-capture (nb:with-tracing (c) (< c (+ s 10.0)))
                                       (nb:with-tracing (c) (+ c s))
                                       x))
                 avals))
         (module (nabla:backend-load backend (nabla:backend-compile backend (nb:emit-stablehlo graph)))))
    (unwind-protect
         (is (check-it (generator (uniform-integer :lo 0 :hi 100000))
                       (lambda (seed)
                         ;; 刻み s は 0.5 以上にして、反復回数を有限に保つ。
                         (let* ((x (make-random-array (make-array-spec '() :f32) :seed seed :domain :positive))
                                (s (make-array '() :element-type 'single-float
                                                   :initial-element
                                                   (+ 0.5f0 (row-major-aref
                                                             (make-random-array (make-array-spec '() :f32)
                                                                                :seed (1+ seed) :domain :positive)
                                                             0)))))
                           (with-device-arrays ((dx (to-device x backend :dtype :f32))
                                                (ds (to-device s backend :dtype :f32)))
                             (with-device-arrays ((result (nabla:backend-invoke backend module "main" dx ds)))
                               (allclose (to-host result) (nb:eval-graph graph x s) :dtype :f32)))))
                       :regression-id subgraph/iree-while-region
                       :regression-file (regression-path "iree-subgraph-while-region"
                                                         :package "NABLA.IREE.TESTS"))
             "IREE の while の結果が eager と一致しなかった")
      (nabla:backend-unload backend module))))
