;;;; vmap: vmap のテストが共通で使う参照実装（issue #125）。
;;;;
;;;; 守らせる性質は「vmap f の結果は、バッチ軸で切り出した各要素に f を eager で
;;;; 適用して、出力の軸の位置に積み直したものと一致する」。REFERENCE-VMAP が
;;;; その右辺を、vmap にも自動微分にも依存せずに作る（配列の切り出しと積み直しだけ）。
;;;; #128 / #129 / #140 のバッチ化ルールのテストも、これを期待値に使う。

(in-package #:nabla.tests.support)

(defun %vmap-subscripts (flat-index dims)
  "DIMS の配列の row-major の通し番号 FLAT-INDEX を添字のリストにする。"
  (let ((subscripts '()))
    (dolist (d (reverse dims) subscripts)
      (push (mod flat-index d) subscripts)
      (setf flat-index (floor flat-index d)))))

(defun slice-along-axis (array axis index)
  "ARRAY の軸 AXIS の INDEX 番目の要素（その軸を取り除いた配列）を返す。"
  (let* ((dims (array-dimensions array))
         (out-dims (append (subseq dims 0 axis) (nthcdr (1+ axis) dims)))
         (out (make-array out-dims :element-type (array-element-type array))))
    (dotimes (i (array-total-size out) out)
      (let ((s (%vmap-subscripts i out-dims)))
        (setf (row-major-aref out i)
              (apply #'aref array (append (subseq s 0 axis) (list index) (nthcdr axis s))))))))

(defun stack-along-axis (arrays axis)
  "同じ形の ARRAYS（リスト）を、新しい軸 AXIS に沿って積み重ねる。"
  (let* ((first-array (first arrays))
         (dims (array-dimensions first-array))
         (out-dims (append (subseq dims 0 axis) (list (length arrays)) (nthcdr axis dims)))
         (out (make-array out-dims :element-type (array-element-type first-array))))
    (dotimes (i (array-total-size out) out)
      (let ((s (%vmap-subscripts i out-dims)))
        (setf (row-major-aref out i)
              (apply #'aref (nth (nth axis s) arrays)
                     (append (subseq s 0 axis) (nthcdr (1+ axis) s))))))))

(defun %per-argument (spec count)
  (if (or (null spec) (integerp spec)) (make-list count :initial-element spec) spec))

(defun reference-vmap (fn args &key (in-axes 0) (out-axes 0))
  "FN（配列を受け取って配列を多値で返す関数。WITH-TRACING の関数も eager に呼べる）を、
ARGS（配列のリスト）のバッチ軸 IN-AXES（引数ごとの整数か NIL。整数・NIL 1つなら全引数共通）
で切り出した各要素に適用し、出力ごとに OUT-AXES（出力ごとの整数か NIL。1つなら共通）の
位置へ積み直した配列のリストを返す。IN-AXES が NIL の引数は、そのまま全要素に渡す。
OUT-AXES が NIL の出力は、最初の要素の結果をそのまま返す（バッチに依存しない出力）。
IN-AXES は非負の整数だけを受ける（負の軸は呼び出し側で正規化する）。"
  (let* ((in-axes (%per-argument in-axes (length args)))
         (size (or (loop for a in args for axis in in-axes
                         when axis return (array-dimension a axis))
                   (error "reference-vmap: IN-AXES に整数が1つも無い: ~S" in-axes)))
         (results (loop for i below size
                        collect (multiple-value-list
                                 (apply fn (loop for a in args for axis in in-axes
                                                 collect (if axis (slice-along-axis a axis i) a))))))
         (out-axes (%per-argument out-axes (length (first results)))))
    (loop for axis in out-axes for k from 0
          collect (if axis
                      (stack-along-axis (mapcar (lambda (r) (nth k r)) results) axis)
                      (nth k (first results))))))

(defun primitive-function (name params arity)
  "プリミティブ NAME（キーワード）を PARAMS（plist）で1回だけ呼ぶ ARITY 引数の
TRACEABLE-FUNCTION。トレーサが渡れば eqn を足し、配列だけなら eager 実装を呼ぶ。
形状演算・縮約・dot-general のバッチ化ルールのテスト（issue #129）が、params を
ランダムにした f を作るのに使う。"
  (nb::%make-traceable-function
   (loop for i below arity collect (intern (format nil "X~D" i)))
   (lambda (&rest args)
     (if (some (lambda (a) (typep a 'nb::tracer)) args)
         (apply #'nb::%trace-eqn name args params)
         (apply (nb::primitive-eager (nb::find-primitive name))
                args (mapcar #'nb:array-aval args) params)))))
