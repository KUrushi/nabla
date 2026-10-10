;;;; nb:emit-stablehlo / nb::graph-eqn-for-diagnostic の性質（issue #33）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

;;; --- tensor-type-string: 5つの dtype すべて、rank 0 も込み ---

(test stablehlo/tensor-type-string-covers-all-dtypes
  "TENSOR-TYPE-STRING は5つの dtype すべてで正しい綴りを出す（i1 も含む。
primitive-test.lisp は f32/bf16 しか確かめていないので、ここで i1 と f16 の
rank 0 / rank ≥ 1 を補う）。"
  (is (string= "tensor<f32>" (nb::tensor-type-string (nb:make-aval '() :f32))))
  (is (string= "tensor<f64>" (nb::tensor-type-string (nb:make-aval '() :f64))))
  (is (string= "tensor<f16>" (nb::tensor-type-string (nb:make-aval '() :f16))))
  (is (string= "tensor<4xi1>" (nb::tensor-type-string (nb:make-aval '(4) :i1))))
  (is (string= "tensor<i1>" (nb::tensor-type-string (nb:make-aval '() :i1))))
  (is (string= "tensor<2x3xbf16>" (nb::tensor-type-string (nb:make-aval '(2 3) :bf16)))))

;;; --- 手書きグラフの golden text ---

(defun %build-add-add-graph ()
  "(x + [1.0 2.0]) + y という2eqnの graph を組み立てる（x, y: f32[2]）。
契約の golden text と一致するように、入力→定数→eqn1→eqn2 の順に var を作る
（%ASSIGN-VAR-NUMBERS の番号がそのまま出現順になるようにするため）。"
  (let* ((f32-2 (nb:make-aval '(2) :f32))
         (x (nb::make-var f32-2))
         (y (nb::make-var f32-2))
         (const-array (make-array 2 :element-type 'single-float :initial-contents '(1.0 2.0)))
         (const-var (nb::make-var (nb:array-aval const-array :f32)))
         (eqn0 (nb::make-eqn :add (list x const-var)))
         (tmp (first (nb:eqn-outvars eqn0)))
         (eqn1 (nb::make-eqn :add (list tmp y)))
         (out (first (nb:eqn-outvars eqn1))))
    (nb::make-graph (list x y) (list eqn0 eqn1) (list out) (list (cons const-var const-array)))))

(test stablehlo/golden-text-matches-hand-built-graph
  "手書きの2eqn graph を emit-stablehlo すると、契約の golden text と
string= で一致する。"
  (is (string=
       "module {
  func.func @main(%0: tensor<2xf32>, %1: tensor<2xf32>) -> (tensor<2xf32>) {
    %2 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
    %3 = stablehlo.add %0, %2 : tensor<2xf32> loc(\"eqn-0\")
    %4 = stablehlo.add %3, %1 : tensor<2xf32> loc(\"eqn-1\")
    func.return %4 : tensor<2xf32>
  }
}"
       (nb:emit-stablehlo (%build-add-add-graph)))))

;;; --- loc("eqn-N") の個数と番号 ---

(test stablehlo/loc-count-and-order-match-eqn-count
  "emit した text に現れる loc(\"eqn-N\") の個数は graph の eqn 数と一致し、
現れる N の集合は 0..eqn数-1 とちょうど一致する（reduce の2行 :emit を
含むレシピでも、両方の行に同じ N が付くだけで N の集合は増えない）。"
  (is (check-it (generator (primitive-graph-recipe :max-ops 4))
                (lambda (recipe)
                  (let* ((graph (build-primitive-graph recipe))
                         (text (nb:emit-stablehlo graph))
                         (n (length (nb:graph-eqns graph)))
                         (found '()))
                    (loop with start = 0
                          for pos = (search "loc(\"eqn-" text :start2 start)
                          while pos
                          do (let* ((digit-start (+ pos (length "loc(\"eqn-")))
                                    (digit-end (position-if-not #'digit-char-p text :start digit-start)))
                               (push (parse-integer text :start digit-start :end digit-end) found)
                               (setf start (or digit-end (length text)))))
                    (equal (sort (remove-duplicates found) #'<) (loop for i below n collect i))))
                :regression-id stablehlo/loc-count-and-order-match-eqn-count
                :regression-file (regression-path "stablehlo-loc-count-and-order-match-eqn-count"))))

;;; --- SSA numbering order: invars 優先、次いで定数 ---

(test stablehlo/ssa-numbers-args-before-constants
  "先頭 N 個の %0..%(N-1) は必ず引数（invars）に、その次（定数があれば）は
定数に振られる（%ASSIGN-VAR-NUMBERS の出現順の契約を emit-stablehlo 経由で
確かめる）。"
  (let* ((graph (%build-add-add-graph))
         (text (nb:emit-stablehlo graph)))
    (is (search "@main(%0: tensor<2xf32>, %1: tensor<2xf32>)" text))
    (is (search "%2 = stablehlo.constant" text))))

;;; --- 複数出力 ---

(test stablehlo/multi-output-return-lists-all-types
  "outvars が2つある graph の func.return は \"%a, %b : T, T\" の形になる。"
  (let* ((f32-2 (nb:make-aval '(2) :f32))
         (x (nb::make-var f32-2))
         (eqn (nb::make-eqn :neg (list x)))
         (y (first (nb:eqn-outvars eqn)))
         (graph (nb::make-graph (list x) (list eqn) (list x y) nil))
         (text (nb:emit-stablehlo graph)))
    (is (search "-> (tensor<2xf32>, tensor<2xf32>)" text))
    (is (search "func.return %0, %1 : tensor<2xf32>, tensor<2xf32>" text))))

;;; --- 出力の無い graph ---

(test stablehlo/no-outvars-omits-arrow-clause
  "outvars が0個の graph は \"->\" 節ごと省き、func.return は素のまま。"
  (let* ((f32-2 (nb:make-aval '(2) :f32))
         (x (nb::make-var f32-2))
         (graph (nb::make-graph (list x) nil nil nil))
         (text (nb:emit-stablehlo graph)))
    (is (search "func.func @main(%0: tensor<2xf32>) {" text))
    (is (search "func.return
" text))
    (is (not (search "->" text)))))

;;; --- 定数リテラル ---

(test stablehlo/f32-constant-round-trips-via-read-from-string
  "有限の f32/f64 定数は、READ-FROM-STRING した値が元の値と = で一致する
（丸め込み無しの round-trip）。"
  (is (check-it (generator (uniform-integer :lo (- (expt 2 31)) :hi (1- (expt 2 31))))
                (lambda (bits)
                  (let* ((value (sb-kernel:make-single-float bits)))
                    (or (sb-ext:float-nan-p value)
                        (sb-ext:float-infinity-p value)
                        (let ((lit (nb::%stablehlo-finite-float-literal value :f32)))
                          (and (find #\. lit)
                               (= value (let ((*read-default-float-format* 'single-float))
                                          (read-from-string lit))))))))
                :regression-id stablehlo/f32-constant-round-trips-via-read-from-string
                :regression-file (regression-path "stablehlo-f32-constant-round-trips"))))

(test stablehlo/f64-constant-round-trips-via-read-from-string
  (dolist (value '(0.0d0 -0.0d0 1.0d0 -1.0d0 3.141592653589793d0 1.0d10 1.0d-10))
    (let ((lit (nb::%stablehlo-finite-float-literal value :f64)))
      (is (find #\. lit))
      (is (= value (let ((*read-default-float-format* 'double-float)) (read-from-string lit)))))))

(test stablehlo/bf16-constant-is-hex-with-leading-zeros
  "bf16 の定数は常に4桁16進（\"0x0080\" のように先頭ゼロも省略しない）。"
  (is (string= "0x0080" (nb::%stablehlo-f16-literal 128)))
  (is (string= "0x0000" (nb::%stablehlo-f16-literal 0)))
  (is (string= "0xFFFF" (nb::%stablehlo-f16-literal #xFFFF))))

(test stablehlo/non-finite-f32-f64-are-hex-bit-patterns
  "NaN / +inf / -inf は10進では出さず、16進ビット列にする。"
  (is (string= "0x7F800000" (nb::%stablehlo-float-literal sb-ext:single-float-positive-infinity :f32)))
  (is (string= "0xFF800000" (nb::%stablehlo-float-literal sb-ext:single-float-negative-infinity :f32)))
  (is (= 8 (length (subseq (nb::%stablehlo-float-literal (sb-kernel:make-single-float #x7FC00000) :f32)
                            2))))
  (is (string= "0x7FF0000000000000"
               (nb::%stablehlo-float-literal sb-ext:double-float-positive-infinity :f64)))
  (is (= 16 (length (subseq (nb::%stablehlo-float-literal
                             (sb-kernel:make-double-float #x7FF80000 0) :f64)
                            2)))))

(test stablehlo/non-finite-literal-is-the-exact-bit-pattern
  "符号・ペイロードの任意の NaN と ±inf の :f32 / :f64 のリテラルは、その値のビット列を
そのまま（f32 は8桁、f64 は16桁の）16進にしたものになる。ペイロードの下位ビットまで
落とさない（上の例は下位ビットがすべて 0 の値だけなので、ビット列の下位を落とす
変異体が issue #70 の mutation testing で生き残っていた）。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1)
                                  (uniform-integer :lo 0 :hi (1- (expt 2 23)))
                                  (uniform-integer :lo 0 :hi (1- (expt 2 52)))))
                (lambda (args)
                  (destructuring-bind (sign payload32 payload64) args
                    (let ((bits32 (logior (ash sign 31) #x7F800000 payload32))
                          (bits64 (logior (ash sign 63) (ash #x7FF 52) payload64)))
                      (and (string= (format nil "0x~8,'0X" bits32)
                                    (nb::%stablehlo-float-literal
                                     (sb-kernel:make-single-float (if (logbitp 31 bits32) (- bits32 (expt 2 32)) bits32))
                                     :f32))
                           (string= (format nil "0x~16,'0X" bits64)
                                    (nb::%stablehlo-float-literal
                                     (sb-kernel:make-double-float
                                      (let ((hi (ash bits64 -32))) (if (logbitp 31 hi) (- hi (expt 2 32)) hi))
                                      (ldb (byte 32 0) bits64))
                                     :f64))))))
                :regression-id stablehlo/non-finite-literal-is-the-exact-bit-pattern
                :regression-file (regression-path "stablehlo-non-finite-literal-is-the-exact-bit-pattern"))))

(test stablehlo/rank-0-constant-has-no-brackets
  (let ((array (make-array '() :element-type 'single-float :initial-element 1.5f0)))
    (is (string= "dense<1.5>" (nb::%constant-literal array :f32)))))

(test stablehlo/rank-2-constant-nests-by-row-major-dims
  (let ((array (make-array '(2 2) :element-type 'single-float
                            :initial-contents '((1.0 2.0) (3.0 4.0)))))
    (is (string= "dense<[[1.0, 2.0], [3.0, 4.0]]>" (nb::%constant-literal array :f32)))))

(test stablehlo/zero-size-constant-is-dense-empty
  (let ((array (make-array '(2 0) :element-type 'single-float)))
    (is (string= "dense<>" (nb::%constant-literal array :f32)))))

(test stablehlo/i1-constant-is-true-false
  (is (string= "true" (nb::%stablehlo-element-literal 1 :i1)))
  (is (string= "false" (nb::%stablehlo-element-literal 0 :i1))))

;;; --- 複数行の :emit（reduce-sum/reduce-max）にも loc が付く ---

(test stablehlo/multi-line-emit-gets-loc-on-every-line
  "reduce-sum の :emit は2行返す。emit-stablehlo はその両方の行に
同じ loc(\"eqn-N\") を付ける。"
  (let* ((in (nb::make-var (nb:make-aval '(3) :f32)))
         (eqn (nb::make-eqn :reduce-sum (list in) :axes '(0)))
         (out (first (nb:eqn-outvars eqn)))
         (graph (nb::make-graph (list in) (list eqn) (list out) nil))
         (text (nb:emit-stablehlo graph))
         (loc-count (loop with start = 0 with n = 0
                           for pos = (search "loc(\"eqn-0\")" text :start2 start)
                           while pos do (incf n) (setf start (1+ pos))
                           finally (return n))))
    (is (= 2 loc-count))))

;;; --- :emit を持たないプリミティブ ---

(nb:defprimitive %test-no-emit ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager (lambda (arrays in-avals) (declare (ignore in-avals)) (first arrays)))

(test stablehlo/primitive-without-emit-signals-primitive-not-emittable
  (let* ((in (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-no-emit (list in)))
         (out (first (nb:eqn-outvars eqn)))
         (graph (nb::make-graph (list in) (list eqn) (list out) nil)))
    (signals nb:primitive-not-emittable (nb:emit-stablehlo graph))
    (handler-case (nb:emit-stablehlo graph)
      (nb:primitive-not-emittable (c) (is (eq :%test-no-emit (nb:primitive-not-emittable-name c)))))))

;;; --- graph-eqn-for-diagnostic ---

(test stablehlo/graph-eqn-for-diagnostic-finds-referenced-eqn
  (let* ((graph (%build-add-add-graph))
         (eqns (nb:graph-eqns graph)))
    (multiple-value-bind (eqn n)
        (nb::graph-eqn-for-diagnostic graph "<unknown>:0: error: loc(\"eqn-1\"): some diagnostic")
      (is (eq (second eqns) eqn))
      (is (= 1 n)))
    (multiple-value-bind (eqn n) (nb::graph-eqn-for-diagnostic graph "no loc here at all")
      (is (null eqn))
      (is (null n)))
    ;; 先頭の eqn（N = 0）も見つける（0 を範囲外と取り違える変異体が issue #70 の
    ;; mutation testing で生き残っていた）
    (multiple-value-bind (eqn n) (nb::graph-eqn-for-diagnostic graph "error: loc(\"eqn-0\"): x")
      (is (eq (first eqns) eqn))
      (is (eql 0 n)))
    (multiple-value-bind (eqn n) (nb::graph-eqn-for-diagnostic graph "loc(\"eqn-99\")")
      (is (null eqn))
      (is (null n)))))
