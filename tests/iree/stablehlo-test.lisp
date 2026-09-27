;;;; StableHLO emitter の medium テスト（issue #33、wave 3 s1）。
;;;;
;;;; emit-stablehlo が出した text を backend-compile/load/invoke した結果は、
;;;; 同じ graph を eval-graph で eager 評価した結果と dtype ごとの許容誤差で
;;;; 一致する（issue #33 の完了条件）。graph は tests/support/primitive-recipes.lisp
;;;; の PRIMITIVE-GRAPH-RECIPE（実プリミティブ上のランダムな graph）で作る。
;;;;
;;;; 許容誤差は graph の最終的な出力 dtype ではなく、GRAPH-WORST-FLOAT-DTYPE
;;;; が返す「graph 中で実際に使われた最も粗い浮動小数点 dtype」で決める。
;;;; 例えば bf16 の入力を演算してから f32 に convert する graph は、出力こそ
;;;; f32 だが値は bf16 の丸め誤差を引き継いでおり、出力 dtype の厳しい
;;;; 許容誤差で判定すると誤って失敗する（issue #33 のリグレッション）。
;;;;
;;;; bf16/f16 は IREE がエレメントワイズ演算を融合して1回だけ丸めるのに
;;;; 対し、eager は演算ごとに丸めるため、rtol を (1 + eqn数) 倍に緩める
;;;; （契約のピットフォール(7)）。distinct な graph = 1回のコンパイル
;;;; （約350ms）なので、:max-ops を小さく・試行回数を抑える（ピットフォール(8)）。

(in-package #:nabla.iree.tests)

(defun %stablehlo-invar-arrays (graph base-seed)
  "GRAPH-INVARS それぞれに、BASE-SEED から決定的に作った乱数配列を1つずつ
対応させたリストを返す。"
  (loop for invar in (nb:graph-invars graph)
        for i from 0
        collect (make-random-array
                 (make-array-spec (nb:aval-shape (nb:var-aval invar)) (nb:aval-dtype (nb:var-aval invar)))
                 :seed (+ base-seed i))))

(defun %stablehlo-iree-matches-eval-graph-p (backend recipe base-seed)
  "RECIPE から組み立てた graph を emit-stablehlo → backend-compile/load/invoke
した結果が、eval-graph の結果と一致するかどうかを返す。"
  (let* ((graph (build-primitive-graph recipe))
         (text (nb:emit-stablehlo graph))
         (module (nabla:backend-load backend (nabla:backend-compile backend text)))
         (host-arrays (%stablehlo-invar-arrays graph base-seed))
         (out-var (first (nb:graph-outvars graph)))
         (out-aval (nb:var-aval out-var))
         (dtype (nb:aval-dtype out-aval))
         ;; 許容誤差は出力の dtype ではなく、graph の中で実際に使われた
         ;; 最も粗い浮動小数点 dtype で決める（DECODE-ARRAY 自体は
         ;; バイト列を正しく解釈する必要があるので、それは引き続き
         ;; 出力の実際の dtype DTYPE を使う）。理由は
         ;; GRAPH-WORST-FLOAT-DTYPE の docstring と issue #33 のリグレッション
         ;; 参照。
         (tolerance-dtype (or (graph-worst-float-dtype graph) dtype))
         (n-eqns (length (nb:graph-eqns graph)))
         (device-arrays nil)
         (result nil))
    (unwind-protect
         (progn
           (setf device-arrays
                 (mapcar (lambda (array invar) (to-device array backend :dtype (nb:aval-dtype (nb:var-aval invar))))
                         host-arrays (nb:graph-invars graph)))
           (setf result (apply #'nabla:backend-invoke backend module "main" device-arrays))
           (let ((eager-result (apply #'nb:eval-graph graph host-arrays)))
             (multiple-value-bind (rtol atol) (dtype-tolerance tolerance-dtype)
               (and (equalp (device-array-aval result) out-aval)
                    (allclose (decode-array (to-host result) dtype)
                              (decode-array eager-result dtype)
                              :rtol (if (member tolerance-dtype '(:bf16 :f16)) (* rtol (1+ n-eqns)) rtol)
                              :atol atol)))))
      (when result (release-device-array result))
      (dolist (da device-arrays) (release-device-array da))
      (nabla:backend-unload backend module))))

(define-iree-test stablehlo/iree-matches-eval-graph
    "PRIMITIVE-GRAPH-RECIPE で作ったランダムな graph を emit-stablehlo →
backend-compile/load/invoke した結果は、同じ graph を eval-graph で評価した
結果と dtype ごとの許容誤差で一致する（issue #33 の完了条件）。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree))
        (*num-trials* 20))
    (is (check-it (generator (tuple (primitive-graph-recipe :max-ops 3)
                                     (uniform-integer :lo 0 :hi (1- (expt 2 31)))))
                  (lambda (pair)
                    (destructuring-bind (recipe base-seed) pair
                      (%stablehlo-iree-matches-eval-graph-p backend recipe base-seed)))
                  :regression-id stablehlo/iree-matches-eval-graph
                  :regression-file (regression-path "iree-stablehlo-matches-eval-graph" :package "NABLA.IREE.TESTS")))))

;;; --- IREE のコンパイル診断からどの eqn が原因かを逆引きできる ---

(nb:defprimitive %test-bad-reshape ()
  ;; ABSTRACT-EVAL は出力の shape を1要素だけ増やして返す。emit-stablehlo が
  ;; 書く func.return もこの（間違った）shape に揃うので、生成される
  ;; テキスト全体は SSA の型として自己矛盾しない。実際に矛盾するのは
  ;; stablehlo.reshape の検証器が見る「入出力の要素数」（2 → 3）だけなので、
  ;; IREE はパーサではなく検証器（この op に付いた loc）で拒否する。SSA の
  ;; 型そのものが自己矛盾する壊し方（例えば func.return の型だけ違える）
  ;; だと、診断は演算の loc ではなく MLIR パーサのトークン位置
  ;; （\"nabla.mlir:L:C\"）を指してしまい、GRAPH-EQN-FOR-DIAGNOSTIC で
  ;; 逆引きできなくなる。
  :abstract-eval (lambda (in-avals)
                   (let ((in (first in-avals)))
                     (nb:make-aval (append (nb:aval-shape in) '(2)) (nb:aval-dtype in))))
  :emit (lambda (in-names in-avals out-name out-aval)
          (declare (ignore in-avals))
          (format nil "~A = stablehlo.reshape ~A : (~A) -> ~A"
                  out-name (first in-names) (nb::tensor-type-string (first in-avals))
                  (nb::tensor-type-string out-aval))))

(define-iree-test stablehlo/diagnostic-maps-back-to-broken-eqn
    "わざと壊した eqn（要素数の合わない stablehlo.reshape）を含む graph を
backend-compile すると IREE-COMPILE-ERROR が signal され、その
(PRINC-TO-STRING condition) を GRAPH-EQN-FOR-DIAGNOSTIC に渡すと、壊れた
eqn そのものが返る。"
  (skip-unless-iree :library :both)
  (let* ((backend (nabla:find-backend :iree))
         (in (nb::make-var (nb:make-aval '(2) :f32)))
         (eqn (nb::make-eqn :%test-bad-reshape (list in)))
         (out (first (nb:eqn-outvars eqn)))
         (graph (nb::make-graph (list in) (list eqn) (list out) nil))
         (text (nb:emit-stablehlo graph)))
    (handler-case
        (progn
          (nabla:backend-compile backend text)
          (fiveam:fail "型が矛盾した StableHLO のコンパイルが成功してしまった"))
      (nabla.iree:iree-compile-error (c)
        (multiple-value-bind (found-eqn n) (nb::graph-eqn-for-diagnostic graph (princ-to-string c))
          (is (eq eqn found-eqn))
          (is (= 0 n)))))))
