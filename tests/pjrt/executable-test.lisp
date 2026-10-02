;;;; PJRT の backend プロトコルの後半（backend-compile / backend-load /
;;;; backend-invoke / backend-unload / backend-fingerprint）と、jit を
;;;; :pjrt で動かす end-to-end のテスト（issue #87）。
;;;; tests/iree/backend-test.lisp・stablehlo-test.lisp・jit-test.lisp の
;;;; PJRT 版。
;;;;
;;;; backend-compile / load / invoke は生の CFFI のオーケストレーションなので
;;;; mutation testing の対象外（tools/mutate/README.md）。fingerprint は純粋な
;;;; 文字列の組み立てで、下の fingerprint テストが中身を検査する。

(in-package #:nabla.pjrt.tests)

(defun %run-fixture (backend fixture arrays &key (dtype :f32))
  "FIXTURE（tests/fixtures/stablehlo/ の名前）を BACKEND でコンパイル・ロード・
実行し、最初の出力を to-host した配列を返す。"
  (let ((module (nabla:backend-load
                 backend (nabla:backend-compile backend (stablehlo-fixture fixture)))))
    (unwind-protect
         (let ((inputs (mapcar (lambda (a) (nabla:to-device a backend :dtype dtype)) arrays)))
           (unwind-protect
                (let ((result (apply #'nabla:backend-invoke backend module "main" inputs)))
                  (unwind-protect (nabla:to-host result)
                    (release-device-array result)))
             (mapc #'release-device-array inputs)))
      (nabla:backend-unload backend module))))

(define-pjrt-test executable/fixtures/add-matches-reference
  "add.mlir（4x8 の要素ごとの加算）を PJRT で実行した結果は reference-add と
allclose :f32 で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt)))
    (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                  (lambda (seed)
                    (let ((a (make-random-array (make-array-spec '(4 8) :f32) :seed seed))
                          (b (make-random-array (make-array-spec '(4 8) :f32) :seed (1+ seed))))
                      (allclose (%run-fixture backend "add" (list a b))
                                (reference-add a b) :dtype :f32)))
                  :regression-id executable/fixtures/add-matches-reference
                  :regression-file (regression-path "pjrt-executable-add"
                                                    :package "NABLA.PJRT.TESTS")))))

(define-pjrt-test executable/fixtures/matmul-matches-reference
  "matmul.mlir（2x3 · 3x2）を PJRT で実行した結果は reference-matmul と一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt)))
    (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                  (lambda (seed)
                    (let ((a (make-random-array (make-array-spec '(2 3) :f32) :seed seed))
                          (b (make-random-array (make-array-spec '(3 2) :f32) :seed (1+ seed))))
                      (allclose (%run-fixture backend "matmul" (list a b))
                                (reference-matmul a b) :dtype :f32)))
                  :regression-id executable/fixtures/matmul-matches-reference
                  :regression-file (regression-path "pjrt-executable-matmul"
                                                    :package "NABLA.PJRT.TESTS")))))

(define-pjrt-test executable/fixtures/reduce-sum-matches-reference
  "reduce_sum.mlir（4x8 を dimension 1 で総和）を PJRT で実行した結果は
reference-reduce-sum と一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt)))
    (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                  (lambda (seed)
                    (let ((a (make-random-array (make-array-spec '(4 8) :f32) :seed seed)))
                      (allclose (%run-fixture backend "reduce_sum" (list a))
                                (reference-reduce-sum a 1) :dtype :f32)))
                  :regression-id executable/fixtures/reduce-sum-matches-reference
                  :regression-file (regression-path "pjrt-executable-reduce-sum"
                                                    :package "NABLA.PJRT.TESTS")))))

(defun %sequential-bf16-row-sums (a)
  "bf16 の (unsigned-byte 16) 配列 A（2次元）の各行を、左から1回の加算ごとに
bf16 へ丸めて累積した和を、double-float のベクタで返す。"
  (let* ((rows (array-dimension a 0)) (cols (array-dimension a 1))
         (result (make-array rows :element-type 'double-float)))
    (dotimes (i rows result)
      (let ((acc 0.0))
        (dotimes (j cols)
          (setf acc (nabla::bf16-bits->single-float
                     (nabla::single-float->bf16-bits
                      (+ acc (nabla::bf16-bits->single-float (aref a i j)))))))
        (setf (aref result i) (float acc 1d0))))))

(define-pjrt-test executable/fixtures/bf16-match-reference
  "add_bf16 / matmul_bf16 / reduce_sum_bf16 を PJRT で実行した結果は、
decode-array で戻した reference-* と bf16 の許容誤差で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt)))
    (multiple-value-bind (rtol atol) (dtype-tolerance :bf16)
      (flet ((run-bf16 (fixture &rest arrays)
               (decode-array (%run-fixture backend fixture arrays :dtype :bf16) :bf16))
             (dec (a) (decode-array a :bf16)))
        (let ((a (make-random-array (make-array-spec '(4 8) :bf16) :seed 100))
              (b (make-random-array (make-array-spec '(4 8) :bf16) :seed 101)))
          (is (allclose (run-bf16 "add_bf16" a b) (reference-add (dec a) (dec b))
                        :rtol rtol :atol atol)))
        (let ((a (make-random-array (make-array-spec '(2 3) :bf16) :seed 102))
              (b (make-random-array (make-array-spec '(3 2) :bf16) :seed 103)))
          (is (allclose (run-bf16 "matmul_bf16" a b) (reference-matmul (dec a) (dec b))
                        :rtol rtol :atol atol)))
        (let ((a (make-random-array (make-array-spec '(4 8) :bf16) :seed 104)))
          ;; XLA CPU は bf16 の総和を「1回の加算ごとに bf16 へ丸める逐次累積」で
          ;; 計算する（実測で 30 seed・120 要素すべてが完全一致した。IREE は f32 で
          ;; 累積して最後に1回丸める）。そのため参照も同じ順序で逐次丸める。
          (is (allclose (run-bf16 "reduce_sum_bf16" a) (%sequential-bf16-row-sums a)
                        :rtol rtol :atol atol)))))))

(define-pjrt-test executable/invoke/result-aval-matches-the-program
  "backend-invoke が返す device-array の aval は、プログラムの出力の形と dtype。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "reduce_sum")))))
    (unwind-protect
         (with-pjrt-arrays ((a (nabla:to-device
                                (make-random-array (make-array-spec '(4 8) :f32) :seed 1) backend)))
           (with-pjrt-arrays ((r (nabla:backend-invoke backend module "main" a)))
             (is (equalp (nabla:make-aval '(4) :f32) (nabla:device-array-aval r)))))
      (nabla:backend-unload backend module))))

;;; --- 多出力・エラー・解放 ---

(defparameter *two-outputs-text*
  "func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  %0 = stablehlo.add %a, %b : tensor<3xf32>
  %1 = stablehlo.multiply %a, %b : tensor<3xf32>
  func.return %0, %1 : tensor<3xf32>, tensor<3xf32>
}
")

(define-pjrt-test executable/invoke/returns-multiple-outputs
  "出力が2つのプログラムは、backend-invoke が2つの device-array を多値で返す。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (module (nabla:backend-load backend (nabla:backend-compile backend *two-outputs-text*)))
         (a (make-array 3 :element-type 'single-float :initial-contents '(1.0 2.0 3.0)))
         (b (make-array 3 :element-type 'single-float :initial-contents '(4.0 5.0 6.0))))
    (unwind-protect
         (with-pjrt-arrays ((da (nabla:to-device a backend)) (db (nabla:to-device b backend)))
           (multiple-value-bind (sum prod) (nabla:backend-invoke backend module "main" da db)
             (unwind-protect
                  (progn (is (allclose (nabla:to-host sum) (reference-add a b) :dtype :f32))
                         (is (allclose (nabla:to-host prod)
                                       (make-array 3 :element-type 'single-float
                                                     :initial-contents '(4.0 10.0 18.0))
                                       :dtype :f32)))
               (release-device-array sum)
               (release-device-array prod))))
      (nabla:backend-unload backend module))))

(define-pjrt-test executable/compile/invalid-text-signals-a-backend-error
  "StableHLO として不正なテキストは、プラグインのメッセージを持つ pjrt-error
（nabla:backend-error の子）になる。その後も backend は使える。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt)))
    (handler-case (nabla:backend-compile backend "this is not mlir")
      (nabla:backend-error (c)
        (is (typep c 'pjrt-error))
        (is (string= "PJRT_Client_Compile" (pjrt-error-context c)))
        (is (plusp (length (pjrt-error-message c)))))
      (:no-error (&rest values)
        (declare (ignore values))
        (fail "compiling garbage should have signalled")))
    (is (plusp (length (nabla:backend-compile backend (stablehlo-fixture "add")))))))

(define-pjrt-test executable/invoke/rejects-bad-calls
  "main 以外の関数名は error、unload 済みの module の invoke は
pjrt-object-released。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
    (with-pjrt-arrays ((a (nabla:to-device
                           (make-random-array (make-array-spec '(4 8) :f32) :seed 1) backend)))
      (signals error (nabla:backend-invoke backend module "other" a a))
      (nabla:backend-unload backend module)
      (signals pjrt-object-released (nabla:backend-invoke backend module "main" a a)))))

(define-pjrt-test executable/unload/is-idempotent
  "backend-unload は同じ module に2回（3回）呼んでもエラーにならない。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
    (nabla:backend-unload backend module)
    (finishes (nabla:backend-unload backend module))
    (finishes (nabla:backend-unload backend module))))

;;; --- unload せずに捨てた module の finalizer（issue #114）---

(defun %drop-modules (backend count)
  "COUNT 個の module を backend-load し、unload せずに捨てる。参照がスタックに
残らないよう別関数にしてある。"
  (declare (notinline nabla:backend-load))
  (let ((octets (nabla:backend-compile backend (stablehlo-fixture "add"))))
    (dotimes (i count)
      (nabla:backend-load backend octets))))

(define-pjrt-test executable/finalizer/frees-dropped-modules
  "unload せずに捨てた module は、GC と finalizer の後で破棄される（client-state の
生存数が元に近い値に戻る）。保守的なスタックルートで少数は残りうる。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (baseline (%live-buffers backend))
         (count 100))
    (%drop-modules backend count)
    (is (>= (- (%live-buffers backend) baseline) 1))
    (dotimes (i 3) (gc-and-run-finalizers))
    (is (<= (- (%live-buffers backend) baseline) 5)
        "~D modules still alive after GC" (- (%live-buffers backend) baseline))))

(defun %module-on-private-backend ()
  "専用クライアントで module を1つロードし、(values module client-state) を返す。
backend への参照はこの関数のフレームとともに消える。"
  (declare (notinline nabla:make-backend nabla:backend-load))
  (let* ((backend (nabla:make-backend :pjrt))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
    (values module
            (nabla.pjrt::%pjrt-client-state (nabla.pjrt::%pjrt-backend-client backend)))))

(define-pjrt-test executable/module/outlives-its-backend
  "所有者（backend / client）が消えた後でも module は invoke でき、クライアントは
module を unload した時点で初めて破棄される（LoadedExecutable が先）。"
  (skip-unless-pjrt :kind :cpu)
  (multiple-value-bind (module state) (%module-on-private-backend)
    (nabla.pjrt::%client-state-owner-gone state)
    (is (not (nabla.pjrt::client-state-destroyed-p state))
        "the client was destroyed while a module was still loaded")
    (let* ((client (nabla.pjrt::pjrt-module-client module))
           (device (first (nabla.pjrt::%pjrt-client-devices client)))
           (x (make-array '(4 8) :element-type 'single-float :initial-element 1.5))
           (a (nabla.pjrt::%client-to-device client device x :f32)))
      (unwind-protect
           (let ((out (nabla.pjrt::%module-invoke module device (list a a))))
             (is (allclose (nabla:to-host out) (reference-add x x) :dtype :f32))
             (release-device-array out))
        (release-device-array a))
      (is (not (nabla.pjrt::client-state-destroyed-p state))))
    (nabla:backend-unload (nabla:find-backend :pjrt) module)
    (is (nabla.pjrt::client-state-destroyed-p state))))

(define-pjrt-test executable/unload/then-gc-does-not-double-free
  "明示的に unload した module を捨てて GC しても、二重解放にならず、生存数は
元に戻っている。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (baseline (%live-buffers backend)))
    (flet ((churn ()
             (declare (notinline nabla:backend-load nabla:backend-unload))
             (dotimes (i 20)
               (let ((m (nabla:backend-load
                         backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
                 (nabla:backend-unload backend m)
                 (nabla:backend-unload backend m)))))
      (churn))
    (dotimes (i 3) (finishes (gc-and-run-finalizers)))
    (is (= baseline (%live-buffers backend)))))

;;; --- ランダムな graph（emit-stablehlo）が eval-graph と一致する ---

(defun %graph-invar-arrays (graph base-seed)
  (loop for invar in (nb:graph-invars graph)
        for i from 0
        collect (make-random-array
                 (make-array-spec (nb:aval-shape (nb:var-aval invar))
                                  (nb:aval-dtype (nb:var-aval invar)))
                 :seed (+ base-seed i))))

(defun %pjrt-matches-eval-graph-p (backend recipe base-seed)
  "RECIPE の graph を emit-stablehlo → PJRT で実行した結果が eval-graph と一致するか。
許容誤差の決め方は tests/iree/stablehlo-test.lisp と同じ（graph で使われた最も粗い
浮動小数点 dtype、bf16 / f16 は eqn 数で rtol を緩める）。"
  (let* ((graph (build-primitive-graph recipe))
         (module (nabla:backend-load
                  backend (nabla:backend-compile backend (nb:emit-stablehlo graph))))
         (host-arrays (%graph-invar-arrays graph base-seed))
         (out-aval (nb:var-aval (first (nb:graph-outvars graph))))
         (dtype (nb:aval-dtype out-aval))
         (tolerance-dtype (or (graph-worst-float-dtype graph) dtype))
         (n-eqns (length (nb:graph-eqns graph)))
         (inputs nil)
         (result nil))
    (unwind-protect
         (progn
           (setf inputs (mapcar (lambda (array invar)
                                  (nabla:to-device array backend
                                                   :dtype (nb:aval-dtype (nb:var-aval invar))))
                                host-arrays (nb:graph-invars graph)))
           (setf result (apply #'nabla:backend-invoke backend module "main" inputs))
           (let ((eager (apply #'nb:eval-graph graph host-arrays)))
             (multiple-value-bind (rtol atol) (dtype-tolerance tolerance-dtype)
               (and (equalp (nabla:device-array-aval result) out-aval)
                    (allclose (decode-array (nabla:to-host result) dtype)
                              (decode-array eager dtype)
                              :rtol (if (member tolerance-dtype '(:bf16 :f16))
                                        (* rtol (1+ n-eqns))
                                        rtol)
                              :atol atol)))))
      (when result (release-device-array result))
      (mapc #'release-device-array inputs)
      (nabla:backend-unload backend module))))

(define-pjrt-test executable/pbt/pjrt-matches-eval-graph
  "PRIMITIVE-GRAPH-RECIPE で作ったランダムな graph を emit-stablehlo → PJRT で
実行した結果は、eval-graph の結果と dtype ごとの許容誤差で一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let ((backend (nabla:find-backend :pjrt))
        (*num-trials* 50))
    (is (check-it (generator (tuple (primitive-graph-recipe :max-ops 3)
                                    (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (pair)
                    (destructuring-bind (recipe base-seed) pair
                      (%pjrt-matches-eval-graph-p backend recipe base-seed)))
                  :regression-id executable/pbt/pjrt-matches-eval-graph
                  :regression-file (regression-path "pjrt-matches-eval-graph"
                                                    :package "NABLA.PJRT.TESTS")))))

;;; --- jit ---

(nb:defjit %pjrt-jit-mlp (x w b)
  (nb:reduce-sum (tanh (+ (nb:dot x w) (nb:broadcast-in-dim b '(2 3) '(1)))) :axes '(1)))

(define-pjrt-test jit/pjrt-backend-matches-eager
  "(nb:jit f :backend :pjrt) と (nb:jit f :backend (find-backend :pjrt)) の
結果は、f を直接（eager に）呼んだ結果と一致する。2回目の呼び出しはキャッシュを
使い、同じ結果を返す。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((f (nb:with-tracing (x w b)
              (nb:reduce-sum (tanh (+ (nb:dot x w) (nb:broadcast-in-dim b '(2 3) '(1))))
                             :axes '(1))))
         (x (make-random-array (make-array-spec '(2 4) :f32) :seed 1))
         (w (make-random-array (make-array-spec '(4 3) :f32) :seed 2))
         (b (make-random-array (make-array-spec '(3) :f32) :seed 3))
         (expected (funcall f x w b)))
    (dolist (backend (list :pjrt (nabla:find-backend :pjrt)))
      (let ((jf (nb:jit f :backend backend)))
        (unwind-protect
             (progn (is (allclose (funcall jf x w b) expected :dtype :f32))
                    (is (allclose (funcall jf x w b) expected :dtype :f32)))
          (nb::%jit-cache-forget f))))))

(define-pjrt-test jit/defjit-runs-with-pjrt-as-default-backend
  "nabla:*default-backend* を :pjrt に束縛すると defjit の関数も PJRT で動き、
eager（トレース対象の関数を直接呼んだ結果）と一致する。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((x (make-random-array (make-array-spec '(2 4) :f32) :seed 1))
         (w (make-random-array (make-array-spec '(4 3) :f32) :seed 2))
         (b (make-random-array (make-array-spec '(3) :f32) :seed 3))
         (eager (funcall (get '%pjrt-jit-mlp 'nb::%defjit-traceable) x w b)))
    (let ((nabla:*default-backend* :pjrt))
      (is (allclose (%pjrt-jit-mlp x w b) eager :dtype :f32)))))

;;; --- fingerprint とディスクキャッシュ ---

(define-pjrt-test fingerprint/contains-plugin-sha256-and-api-version
  "backend-fingerprint は実装名 \"pjrt\"、プラグインの .so の中身から計算した
SHA-256（外部の sha256sum と一致）、PJRT API の版を含み、IREE の
fingerprint とは異なる。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((backend (nabla:find-backend :pjrt))
         (fingerprint (nabla:backend-fingerprint backend))
         (sha (ironclad:byte-array-to-hex-string
               (ironclad:digest-file :sha256 (plugin-path :cpu)))))
    (is (member "pjrt" fingerprint :test #'string=))
    (is (member (format nil "plugin-sha256=~A" sha) fingerprint :test #'string=))
    (multiple-value-bind (major minor) (plugin-api-version (load-plugin :cpu))
      (is (member (format nil "pjrt-api=~D.~D" major minor) fingerprint :test #'string=)))
    (is (equal fingerprint (nabla:backend-fingerprint backend)))
    (is (not (member "iree" fingerprint :test #'string=)))))

(defclass %other-plugin-backend (pjrt-backend) ()
  (:documentation "プラグインの sha256 だけが違う fingerprint を返す、テスト用の PJRT-BACKEND。"))

(defmethod nabla:backend-fingerprint ((backend %other-plugin-backend))
  (substitute-if "plugin-sha256=other" (lambda (s) (search "plugin-sha256=" s))
                 (call-next-method)))

(define-pjrt-test fingerprint/disk-cache-keys-differ-with-the-plugin-sha256
  "ディスクキャッシュのキーは fingerprint を含むので、プラグインの sha256 だけが
違う backend は同じテキストでも別の .module ファイルを作る。同じ backend の
2回目はそれを読んで同じバイト列を返す（ファイルは増えない）。"
  (skip-unless-pjrt :kind :cpu)
  (with-temporary-directory (dir)
    (let* ((nabla:*compile-cache-directory* dir)
           (backend (nabla:find-backend :pjrt))
           (other (make-instance '%other-plugin-backend
                                 :target (nabla:backend-target backend)
                                 :client (nabla.pjrt::%pjrt-backend-client backend)
                                 :device (nabla.pjrt::%pjrt-backend-device backend)))
           (text (stablehlo-fixture "add"))
           (first (nabla:backend-compile backend text))
           (second (nabla:backend-compile backend text)))
      (is (equalp first second))
      (flet ((module-count ()
               (length (directory (make-pathname :name :wild :type "module" :defaults dir)))))
        (is (= 1 (module-count)))
        (nabla:backend-compile other text)
        (is (= 2 (module-count)))))))

;; 注意: シリアライズした実行体のバイト列はコンパイルごとに非決定的なので、
;; 新しくコンパイルした結果同士を比べてはいけない（キャッシュから読んだ
;; バイト列との一致だけを比べる）。
(define-pjrt-test fingerprint/disk-cached-bytes-load-and-run
  "ディスクキャッシュから読んだ（2回目の）バイト列も backend-load → invoke
できて、結果は reference-add と一致する。"
  (skip-unless-pjrt :kind :cpu)
  (with-temporary-directory (dir)
    (let* ((nabla:*compile-cache-directory* dir)
           (backend (nabla:find-backend :pjrt))
           (a (make-random-array (make-array-spec '(4 8) :f32) :seed 0))
           (b (make-random-array (make-array-spec '(4 8) :f32) :seed 1)))
      (nabla:backend-compile backend (stablehlo-fixture "add"))
      (let ((module (nabla:backend-load
                     backend (nabla:backend-compile backend (stablehlo-fixture "add")))))
        (unwind-protect
             (with-pjrt-arrays ((da (nabla:to-device a backend)) (db (nabla:to-device b backend)))
               (with-pjrt-arrays ((r (nabla:backend-invoke backend module "main" da db)))
                 (is (allclose (nabla:to-host r) (reference-add a b) :dtype :f32))))
          (nabla:backend-unload backend module))))))

;;; --- シグナルの処分と浮動小数点モード ---

(defparameter *compile-signal-check-script*
  '("(require :asdf)"
    "(asdf:load-system \"nabla/pjrt\")"
    ;; 全シグナル（1..64）の処分を、「プラグインのロード・クライアント作成の後、
    ;; 最初のコンパイルの前」と「コンパイル・ロード・実行の後」で比べる。
    ;; 別スレッドが GC を回し続ける中で行う（GC の間に眠る）。
    "(defparameter *text* \"func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.tanh %a : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}\")"
    "(let* ((handlers (lambda () (loop for s from 1 below 65
                                      collect (nabla.ffi-support::%signal-handler-address s))))
           (before (funcall handlers))
           (backend (nabla:make-backend :pjrt))
           (modes-before (sb-int:get-floating-point-modes))
           (stop nil)
           (gc-thread (sb-thread:make-thread
                       (lambda () (loop until stop do (sb-ext:gc :full t) (sleep 0.01)))))
           (x (make-array 4 :element-type 'single-float :initial-element 0.5))
           (module (nabla:backend-load backend (nabla:backend-compile backend *text*))))
      (dotimes (i 100)
        (let* ((a (nabla:to-device x backend))
               (r (nabla:backend-invoke backend module \"main\" a)))
          (unless (< (abs (- (aref (nabla:to-host r) 0) (tanh 0.5))) 1e-5)
            (sb-ext:exit :code 3 :abort t))
          (nabla.pjrt:release-device-array r)
          (nabla.pjrt:release-device-array a)))
      (nabla:backend-unload backend module)
      (setf stop t)
      (sb-thread:join-thread gc-thread)
      (dotimes (i 5) (sb-ext:gc :full t))
      (format t \"CHANGED=~S~%\"
              (loop for b in before for a in (funcall handlers) for s from 1
                    unless (eql a b) collect s))
      (format t \"MODES-SAME=~S~%\" (equal modes-before (sb-int:get-floating-point-modes)))
      (sb-ext:exit :code 0))"))

(define-pjrt-test client-and-compile/leave-signal-handlers-and-float-modes-alone
  "真っさらな子プロセスで、プラグインのロード・クライアント作成（PJRT_Plugin_Initialize
/ PJRT_Client_Create）・to-device / to-host・最初のコンパイル（XLA が LLVM を
動かす）・ロード・実行を、別スレッドが GC を回し続ける中で行っても、全シグナルの処分が変わらず、
浮動小数点のモード（トラップのマスク）も元のままで、プロセスが落ちない。
処分を元に戻す with-lisp-signal-handlers-preserved で包んであるので、この検査が
通るのは「登録が起きない」か「起きても戻せた」のどちらか。起きていないことは
docs/pjrt-setup.md の「シグナルハンドラ」に実験を記録した。"
  (skip-unless-pjrt :kind :cpu)
  (let* ((args (list* "--non-interactive" "--disable-debugger"
                      (loop for form in *compile-signal-check-script*
                            append (list "--eval" form))))
         (env (append (%forward-env-vars (list* "NABLA_PJRT_HOME" *child-sbcl-forwarded-env-vars*))
                      (list (format nil "CL_SOURCE_REGISTRY=~A" (%child-source-registry)))))
         (output (make-string-output-stream))
         (process (sb-ext:run-program "timeout" (list* "-k" "5" "240" "sbcl" args)
                                      :search t :environment env
                                      :output output :error output))
         (text (get-output-stream-string output)))
    (is (= 0 (sb-ext:process-exit-code process)) "child exited ~D, output:~%~A"
        (sb-ext:process-exit-code process) text)
    (is (search "CHANGED=NIL" text) "signal dispositions changed, output:~%~A" text)
    (is (search "MODES-SAME=T" text) "floating point modes changed, output:~%~A" text)))
