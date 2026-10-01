;;;; nb::linearize-graph と、その土台の nb::partition-eqns-by-dependence の
;;;; 性質（issue #82）。
;;;;
;;;; 依存の判定は、テスト自身が別の実装（%TANGENT-DEPENDENT-VARS）で行う。
;;;; 分割関数が内部で作る印には頼らない。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %mixed-output-graph (graph)
  "GRAPH の出力を (最後の出力, 最初の invar, 最後の出力, 最初の定数（あれば）) に
作り直す。複数出力・重複した出力・invar そのもの・定数そのものが出力になる
場合を同時に覆う。"
  (let ((out (first (last (nb:graph-outvars graph))))
        (constant (car (first (nb:graph-constants graph)))))
    (nb::make-graph (nb:graph-invars graph) (nb:graph-eqns graph)
                    (append (list out (first (nb:graph-invars graph)) out)
                            (when constant (list constant)))
                    (nb:graph-constants graph))))

(defun %tangent-dependent-vars (eqns seeds)
  "SEEDS（var のリスト）から EQNS（順序どおり）を順にたどって、SEEDS に推移的に
依存する var の集合（EQ なハッシュ表）を返す。"
  (let ((table (make-hash-table :test 'eq)))
    (dolist (v seeds) (setf (gethash v table) t))
    (dolist (eqn eqns table)
      (when (some (lambda (v) (gethash v table)) (nb:eqn-invars eqn))
        (dolist (v (nb:eqn-outvars eqn)) (setf (gethash v table) t))))))

(defun %eqn-depends-p (eqn table)
  (some (lambda (v) (gethash v table)) (nb:eqn-invars eqn)))

(test linearize/partition-splits-by-dependence-on-seeds
  "partition-eqns-by-dependence は eqn を2つに分ける: 線形側の eqn はどれも seeds に
推移的に依存し、主値側の eqn はどれも依存しない。2つを合わせると元の eqn の集合で、
それぞれの中の順序は保たれる。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((graph (%mixed-output-graph (build-graph recipe)))
                         (jvp (nb::jvp-graph graph))
                         (n (length (nb:graph-invars graph)))
                         (seeds (nthcdr n (nb:graph-invars jvp)))
                         (table (%tangent-dependent-vars (nb:graph-eqns jvp) seeds)))
                    (multiple-value-bind (primal linear)
                        (nb::partition-eqns-by-dependence (nb:graph-eqns jvp) seeds)
                      (and (notany (lambda (e) (%eqn-depends-p e table)) primal)
                           (every (lambda (e) (%eqn-depends-p e table)) linear)
                           (equal primal (remove-if (lambda (e) (%eqn-depends-p e table))
                                                    (nb:graph-eqns jvp)))
                           (equal linear (remove-if-not (lambda (e) (%eqn-depends-p e table))
                                                        (nb:graph-eqns jvp)))))))
                :regression-id linearize/partition-splits-by-dependence-on-seeds
                :regression-file (regression-path "linearize-partition"))))

(test linearize/partition-fixed-example
  "x * y の jvp: 主値の eqn は x * y の1つ、残り（tx * y、x * ty、足し算）が線形側。"
  (let* ((aval (nb:make-aval '(2) :f64))
         (x (nb::make-var aval)) (y (nb::make-var aval))
         (mul (nb::make-eqn :%test-mul (list x y)))
         (graph (nb::make-graph (list x y) (list mul) (nb:eqn-outvars mul) '()))
         (jvp (nb::jvp-graph graph)))
    (multiple-value-bind (primal linear)
        (nb::partition-eqns-by-dependence (nb:graph-eqns jvp) (nthcdr 2 (nb:graph-invars jvp)))
      (flet ((names (eqns) (mapcar (lambda (e) (nb:primitive-name (nb:eqn-prim e))) eqns)))
        (is (equal '(:%test-mul) (names primal)))
        (is (equal '(:%test-mul :%test-mul :add) (names linear)))))))

(defun %linearize-nonzero (graph recipe)
  (loop for i below (length (nb:graph-invars graph)) collect (evenp (+ i (length recipe)))))

(test linearize/linear-part-depends-on-tangents-and-primal-part-does-not
  "linearize-graph の線形 graph の eqn はどれも接線の入力に推移的に依存し、
主値 graph には接線の入力が無い（入力は主値だけ）。入出力の個数と aval の規約も確かめる。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((graph (%mixed-output-graph (build-graph recipe)))
                         (n (length (nb:graph-invars graph)))
                         (m (length (nb:graph-outvars graph)))
                         (lin (nb::linearize-graph graph))
                         (linear (nb::linearization-linear-graph lin))
                         (primal (nb::linearization-primal-graph lin))
                         (n-res (nb::linearization-n-residuals lin))
                         (table (%tangent-dependent-vars (nb:graph-eqns linear)
                                                         (nthcdr n-res (nb:graph-invars linear)))))
                    (and (every (lambda (e) (%eqn-depends-p e table)) (nb:graph-eqns linear))
                         (= n (length (nb:graph-invars primal)))
                         (= (+ m n-res) (length (nb:graph-outvars primal)))
                         (= (+ n-res n) (length (nb:graph-invars linear)))
                         (= m (length (nb:graph-outvars linear)))
                         (= m (nb::linearization-n-outputs lin))
                         ;; 線形 graph の入力の前半（残差）の aval は、主値 graph の残差の出力と同じ。
                         (equalp (mapcar #'nb:var-aval (subseq (nb:graph-invars linear) 0 n-res))
                                 (mapcar #'nb:var-aval (nthcdr m (nb:graph-outvars primal))))
                         ;; 後半（接線）の aval は元の入力と同じ。出力は元の出力と同じ aval。
                         (equalp (mapcar #'nb:var-aval (nthcdr n-res (nb:graph-invars linear)))
                                 (mapcar #'nb:var-aval (nb:graph-invars graph)))
                         (equalp (mapcar #'nb:var-aval (nb:graph-outvars linear))
                                 (mapcar #'nb:var-aval (nb:graph-outvars graph))))))
                :regression-id linearize/linear-part-depends-on-tangents-and-primal-part-does-not
                :regression-file (regression-path "linearize-dependence"))))

(test linearize/composition-equals-jvp-graph
  "主値 graph → 線形 graph と順に評価した結果は、jvp-graph を評価した結果（主値 ++ 接線）
と一致する。nonzero で一部の入力の接線を落とした場合も同じ。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((graph (%mixed-output-graph (build-graph recipe)))
                         (nonzero (%linearize-nonzero graph recipe))
                         (m (length (nb:graph-outvars graph))))
                    (loop for nz in (list nil nonzero)
                          always
                          (let* ((lin (if nz (nb::linearize-graph graph :nonzero nz) (nb::linearize-graph graph)))
                                 (jvp (if nz (nb::jvp-graph graph :nonzero nz) (nb::jvp-graph graph)))
                                 (primals (%jvp-arrays graph))
                                 (tangents (let ((all (%jvp-arrays graph :tangent t)))
                                             (if nz (loop for tg in all for f in nz when f collect tg) all)))
                                 (primal-result (%jvp-eval (nb::linearization-primal-graph lin) primals))
                                 (residuals (nthcdr m primal-result))
                                 (linear-result (%jvp-eval (nb::linearization-linear-graph lin)
                                                           (append residuals tangents))))
                            (equalp (append (subseq primal-result 0 m) linear-result)
                                    (%jvp-eval jvp (append primals tangents)))))))
                :regression-id linearize/composition-equals-jvp-graph
                :regression-file (regression-path "linearize-composition"))))

(test linearize/results-are-well-formed-and-round-trip
  "主値 graph と線形 graph は check-graph と print → read → print の往復を満たし、
元の graph は変わらない。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((graph (%mixed-output-graph (build-graph recipe)))
                         (before (nb:print-graph graph))
                         (lin (nb::linearize-graph graph)))
                    (and (%jvp-round-trips-p (nb::linearization-primal-graph lin))
                         (%jvp-round-trips-p (nb::linearization-linear-graph lin))
                         (string= before (nb:print-graph graph)))))
                :regression-id linearize/results-are-well-formed-and-round-trip
                :regression-file (regression-path "linearize-well-formed"))))

(test linearize/linear-part-has-no-dead-code
  "線形 graph には DCE がかかっている（出力に効かない eqn が無い）。そのため、
どの残差も線形 graph の中で使われている。"
  (is (check-it (generator (graph-recipe :dtypes '(:f32 :f64) :binary-prims '(:%test-add :%test-mul)))
                (lambda (recipe)
                  (let* ((graph (%mixed-output-graph (build-graph recipe)))
                         (lin (nb::linearize-graph graph))
                         (linear (nb::linearization-linear-graph lin))
                         (n-res (nb::linearization-n-residuals lin))
                         (used (make-hash-table :test 'eq)))
                    (dolist (e (nb:graph-eqns linear))
                      (dolist (v (nb:eqn-invars e)) (setf (gethash v used) t)))
                    (dolist (v (nb:graph-outvars linear)) (setf (gethash v used) t))
                    (and (= (length (nb:graph-eqns linear))
                            (length (nb:graph-eqns (nb::dce-graph linear))))
                         (every (lambda (v) (gethash v used)) (subseq (nb:graph-invars linear) 0 n-res)))))
                :regression-id linearize/linear-part-has-no-dead-code
                :regression-file (regression-path "linearize-no-dead-code"))))
