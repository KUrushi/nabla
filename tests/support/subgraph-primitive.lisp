;;;; テスト専用の高階プリミティブ（issue #127）。
;;;;
;;;; %TEST-CALL-SUBGRAPH は params :BODY のサブグラフを1回呼ぶだけの恒等の
;;;; 高階プリミティブ。複数出力（契約 C1）とサブグラフ params（契約 C2）の
;;;; 両方を通すために、:multiple-outputs t にしてある。cond / while-loop / scan
;;;; が本物の高階プリミティブになる前の、土台（サブグラフの印字・StableHLO
;;;; リージョン出力・eval / inline / jvp / transpose の素通し）の検査に使う。
;;;;
;;;; StableHLO は、定数 i32 の 0 を添え字にした "stablehlo.case" の1枝として出す。
;;;; stablehlo.case は添え字が範囲外なら最後の枝を実行するので、枝が1つなら常に
;;;; その枝が実行される。stablehlo.if は枝が2つ要る（同じ本体を2回出すことになる）
;;;; ので、枝が1つで済む case にした。枝のリージョンは外側の SSA 値を直接参照
;;;; できるので、本体の invars は外側の入力の名前に結びつける
;;;; （%STABLEHLO-REGION-LINES の :ARG-NAMES）。IREE が受け付けることは
;;;; tests/iree/subgraph-test.lisp で確かめる。

(in-package #:nabla.tests.support)

(nabla:defprimitive %test-call-subgraph (:body)
  :multiple-outputs t
  :abstract-eval
  (lambda (in-avals &key body)
    (unless (equalp in-avals (mapcar #'nabla:var-aval (nabla:graph-invars body)))
      (error 'nabla:primitive-error :name :%test-call-subgraph :in-avals in-avals
             :format-control "入力の aval が本体の入力と一致しない: ~S / ~S"
             :format-arguments (list in-avals (mapcar #'nabla:var-aval (nabla:graph-invars body)))))
    (mapcar #'nabla:var-aval (nabla:graph-outvars body)))
  :emit
  (lambda (in-names in-avals out-names out-avals &key body)
    (declare (ignore in-avals))
    (let ((index-name (format nil "%idx_~A" (subseq (first out-names) 1))))
      (format nil "~A = stablehlo.constant dense<0> : tensor<i32>~%~{~A~^, ~} = \"stablehlo.case\"(~A) ({~%~{  ~A~^~%~}~%}) : (tensor<i32>) -> (~{~A~^, ~})"
              index-name
              out-names
              index-name
              (nabla::%stablehlo-region-lines body :arg-names in-names)
              (mapcar #'nabla::tensor-type-string out-avals))))
  :eager
  (lambda (arrays in-avals &key body)
    (declare (ignore in-avals))
    (multiple-value-list (apply #'nabla:eval-graph body arrays))))

(defun test-call-subgraph (body &rest args)
  "BODY（WITH-TRACING で作った TRACEABLE-FUNCTION）を、ARGS（トレーサ）の aval で
サブグラフにトレースし、%TEST-CALL-SUBGRAPH の eqn を足して、出力のトレーサの
リストを返す。BODY が閉包で捕まえた外側のトレーサは、eqn の invars の末尾に
足される（closure conversion。契約 C2）。トレース中にしか呼べない。"
  (multiple-value-bind (graph captured)
      (nabla::%trace-subgraph body (mapcar #'nabla::tracer-aval args))
    (nabla::%trace-eqn* :%test-call-subgraph (append args captured) :body graph)))

;;; ---- while 形のリージョン（ブロック引数と外側の名前の混在）の確認用 ----
;;;
;;; %TEST-WHILE-CAPTURE は carry を1つと、閉包で捕まえた値を1つ持つ。cond / body の
;;; サブグラフの invars は (carry captured)。StableHLO では carry をブロック引数
;;; (^bb0)、captured を外側の SSA 名にした stablehlo.while として出す
;;; （%STABLEHLO-REGION-LINES の :ARG-NAMES に (nil "%名前") を渡す形）。

(nabla:defprimitive %test-while-capture (:cond :body)
  :multiple-outputs t
  :abstract-eval
  (lambda (in-avals &key cond body)
    (declare (ignore cond))
    (unless (equalp in-avals (mapcar #'nabla:var-aval (nabla:graph-invars body)))
      (error 'nabla:primitive-error :name :%test-while-capture :in-avals in-avals
             :format-control "入力の aval が本体の入力と一致しない: ~S"
             :format-arguments (list in-avals)))
    (list (first in-avals)))
  :emit
  (lambda (in-names in-avals out-names out-avals &key cond body)
    (declare (ignore in-avals))
    (let ((arg-names (list nil (second in-names)))
          (type (nabla::tensor-type-string (first out-avals))))
      (format nil "~A = \"stablehlo.while\"(~A) ({~%~{  ~A~^~%~}~%}, {~%~{  ~A~^~%~}~%}) : (~A) -> ~A"
              (first out-names)
              (first in-names)
              (nabla::%stablehlo-region-lines cond :arg-names arg-names)
              (nabla::%stablehlo-region-lines body :arg-names arg-names)
              type type)))
  :eager
  (lambda (arrays in-avals &key cond body)
    (declare (ignore in-avals))
    (let ((carry (first arrays))
          (captured (second arrays)))
      (loop while (= 1 (row-major-aref (nabla:eval-graph cond carry captured) 0))
            do (setf carry (nabla:eval-graph body carry captured)))
      (list carry))))

(defun test-while-capture (cond-fn body-fn carry)
  "COND-FN / BODY-FN（どちらも WITH-TRACING で作った TRACEABLE-FUNCTION。引数は carry
1つ）を CARRY の aval でサブグラフにし、%TEST-WHILE-CAPTURE の eqn を足して、結果の
トレーサを返す。2つの本体が閉包で捕まえた外側のトレーサは同じ1つでなければならない。"
  (let ((avals (list (nabla::tracer-aval carry))))
    (multiple-value-bind (cond-graph cond-captured) (nabla::%trace-subgraph cond-fn avals)
      (multiple-value-bind (body-graph body-captured) (nabla::%trace-subgraph body-fn avals)
        (unless (and (= 1 (length cond-captured)) (equal cond-captured body-captured))
          (error "test-while-capture: cond と body は同じ外側の値を1つだけ捕まえること: ~S / ~S"
                 cond-captured body-captured))
        (first (nabla::%trace-eqn* :%test-while-capture (cons carry cond-captured)
                                   :cond cond-graph :body body-graph))))))
