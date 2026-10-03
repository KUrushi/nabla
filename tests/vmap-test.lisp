;;;; nb:vmap の性質（issue #125）。
;;;;
;;;; 期待値は vmap に依存しない参照実装 REFERENCE-VMAP（tests/support/vmap.lisp。バッチ軸で
;;;; 切り出した各要素に f を eager で適用して積み直す）。f はバッチ化ルールを書いた
;;;; プリミティブ（add / broadcast-in-dim）だけで作る。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defparameter *vmap-add*
  (nb:with-tracing (x y) (+ x y)))

(defparameter *vmap-add-twice*
  (nb:with-tracing (x y) (+ (+ x y) x)))

(defun %vmap-inner-shape (rank seed)
  "SEED から決まる、各次元 1..4 の rank 次元の形。"
  (loop for i below rank collect (1+ (mod (floor seed (expt 5 i)) 4))))

(defun %vmap-axis-or-nil (code rank)
  "CODE から、0..RANK のバッチ軸か NIL（CODE mod (RANK+2) = RANK+1 のとき）を選ぶ。"
  (let ((v (mod code (+ rank 2))))
    (and (<= v rank) v)))

(defun %vmap-batched-array (inner-shape size axis seed)
  "INNER-SHAPE の形に、AXIS（NIL ならバッチ軸なし）の位置へ長さ SIZE の軸を足した f64 乱数配列。"
  (make-random-array
   (make-array-spec (if axis
                        (append (subseq inner-shape 0 axis) (list size) (nthcdr axis inner-shape))
                        inner-shape)
                    :f64)
   :seed seed))

(defun %vmap-case-property (f-for-shape out-rank-extra)
  "(rank size seed code-x code-y code-out) のケースを受け取り、F-FOR-SHAPE（inner-shape を
受けて2引数の関数を返す）の vmap が REFERENCE-VMAP と一致する性質（述語）を返す。
OUT-RANK-EXTRA は出力の rank が入力の rank に比べて何次元増えるか（出力の軸の選び方に使う）。"
  (lambda (case)
    (destructuring-bind (rank size seed code-x code-y code-out) case
      (let* ((inner (%vmap-inner-shape rank seed))
             (ax (%vmap-axis-or-nil code-x rank))
             (ay (if ax (%vmap-axis-or-nil code-y rank) (or (%vmap-axis-or-nil code-y rank) 0)))
             (out (mod code-out (+ rank out-rank-extra 1)))
             (f (funcall f-for-shape inner))
             (x (%vmap-batched-array inner size ax seed))
             (y (%vmap-batched-array inner size ay (1+ seed)))
             (expected (reference-vmap f (list x y) :in-axes (list ax ay) :out-axes out))
             (actual (funcall (nb:vmap f :in-axes (list ax ay) :out-axes out) x y)))
        (allclose actual (first expected) :dtype :f64)))))

(defun %vmap-case-generator ()
  (generator (tuple (uniform-integer :lo 0 :hi 3)
                    (uniform-integer :lo 1 :hi 4)
                    (uniform-integer :lo 0 :hi 10000)
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99))))

(test vmap/add-matches-per-slice-reference
  "add: in-axes（引数ごとに 0..rank か nil）と out-axes をランダムにした vmap が、
要素ごとに適用して積み直した参照実装と一致する（バッチされた側とされていない側の組を含む）。"
  (is (check-it (%vmap-case-generator)
                (%vmap-case-property (lambda (inner) (declare (ignore inner)) *vmap-add*) 0)
                :regression-id vmap/add-matches-per-slice-reference
                :regression-file (regression-path "vmap-skeleton"))))

(test vmap/add-chain-matches-per-slice-reference
  "add を2つ連ねた f（x が2回使われる）でも参照実装と一致する。"
  (is (check-it (%vmap-case-generator)
                (%vmap-case-property (lambda (inner) (declare (ignore inner)) *vmap-add-twice*) 0)
                :regression-id vmap/add-chain-matches-per-slice-reference
                :regression-file (regression-path "vmap-skeleton"))))

(test vmap/broadcast-in-dim-matches-per-slice-reference
  "broadcast-in-dim（先頭に長さ2の軸を足す）を通した2つの値の和。出力の rank は入力より1増える。"
  (is (check-it (%vmap-case-generator)
                (%vmap-case-property
                 (lambda (inner)
                   (let ((shape (cons 2 inner))
                         (dims (loop for i from 1 to (length inner) collect i)))
                     (nb:with-tracing (x y)
                       (+ (nb:broadcast-in-dim x shape dims)
                          (nb:broadcast-in-dim y shape dims)))))
                 1)
                :regression-id vmap/broadcast-in-dim-matches-per-slice-reference
                :regression-file (regression-path "vmap-skeleton"))))

;;; --- 固定の例: バッチされない値・out-axes・入れ子 ---

(defun %vec (&rest xs)
  (make-array (list (length xs)) :element-type 'double-float :initial-contents xs))

(defun %mat (rows)
  (make-array (list (length rows) (length (first rows))) :element-type 'double-float
                                                         :initial-contents rows))

(test vmap/unbatched-only-eqns-skip-the-rule-and-add-no-broadcast
  "バッチされていない入力だけの eqn（y + y）はバッチ化ルールを呼ばずに残り、出力 (nil) も
broadcast されない。add のルールは x + x の1回だけ呼ばれ、graph に broadcast-in-dim は無い。"
  (let* ((f (nb:with-tracing (x y) (values (+ x x) (+ y y))))
         (g (nb:vmap f :in-axes '(0 nil) :out-axes '(0 nil)))
         (add (nb::find-primitive :add))
         (original (nb::primitive-batch add))
         (calls 0))
    (setf (nb::primitive-batch add)
          (lambda (&rest args) (incf calls) (apply original args)))
    (unwind-protect
         (let ((graph (nb:trace-to-graph g (list (nb:make-aval '(3 2) :f64) (nb:make-aval '(2) :f64)))))
           (is (= 1 calls))
           (is (equal '(:add :add) (mapcar (lambda (e) (nb::primitive-name (nb::eqn-prim e)))
                                           (nb:graph-eqns graph)))))
      (setf (nb::primitive-batch add) original))
    (multiple-value-bind (a b) (funcall g (%mat '((1d0 2d0) (3d0 4d0) (5d0 6d0))) (%vec 1d0 2d0))
      (is (equalp (%mat '((2d0 4d0) (6d0 8d0) (10d0 12d0))) a))
      (is (equalp (%vec 2d0 4d0) b)))))

(test vmap/unbatched-output-is-broadcast-only-when-an-axis-is-requested
  "バッチに依存しない出力は、out-axes が整数なら複製され、nil ならそのまま返る。"
  (let ((f (nb:with-tracing (x y) (values (+ x x) (+ y y))))
        (xs (%mat '((1d0 2d0) (3d0 4d0))))
        (y (%vec 1d0 2d0)))
    (is (equalp (%mat '((2d0 4d0) (2d0 4d0)))
                (second (multiple-value-list (funcall (nb:vmap f :in-axes '(0 nil) :out-axes '(0 0)) xs y)))))
    (is (equalp (%mat '((2d0 2d0) (4d0 4d0)))
                (second (multiple-value-list (funcall (nb:vmap f :in-axes '(0 nil) :out-axes '(0 1)) xs y)))))))

(test vmap/negative-axes-count-from-the-end
  "in-axes / out-axes の負の値は末尾から数える（in-axes -1 は rank の最後の軸、
out-axes -2 は出力（バッチ軸を含む rank 2）の軸 0）。"
  (let ((x (%mat '((1d0 2d0 3d0) (4d0 5d0 6d0))))
        (y (%mat '((1d0 4d0 3d0) (3d0 6d0 3d0)))))
    ;; 軸 1 でバッチ: j 番目の和は (x[0,j]+y[0,j], x[1,j]+y[1,j])
    (is (equalp (%mat '((2d0 7d0) (6d0 11d0) (6d0 9d0)))
                (funcall (nb:vmap *vmap-add* :in-axes -1 :out-axes -2) x y)))))

(test vmap/nested-vmap-matches-doubly-sliced-reference
  "(vmap (vmap f)) は入れ子のバッチ軸を持つ: 外側 in-axes 1・内側 0 など、軸の組を変えても
2重に切り出した参照実装と一致する。"
  (let* ((x (make-random-array (make-array-spec '(3 2 4) :f64) :seed 11))
         (y (make-random-array (make-array-spec '(3 2 4) :f64) :seed 12))
         (inner (nb:vmap *vmap-add* :in-axes 0 :out-axes 0))
         (outer (nb:vmap inner :in-axes 1 :out-axes 2)))
    (is (allclose (funcall outer x y)
                  (first (reference-vmap inner (list x y) :in-axes 1 :out-axes 2))
                  :dtype :f64))
    (is (allclose (funcall (nb:vmap (nb:vmap *vmap-add*)) x y)
                  (funcall *vmap-add* x y)
                  :dtype :f64))))

(test vmap/composes-inside-with-tracing-and-reuses-trace
  "with-tracing の本体の中の vmap は現在のトレースへ展開される（eager と同じ結果）。"
  (let* ((h (nb:with-tracing (x y) (funcall (nb:vmap *vmap-add*) x (+ y y))))
         (x (make-random-array (make-array-spec '(3 2) :f64) :seed 5))
         (y (make-random-array (make-array-spec '(3 2) :f64) :seed 6)))
    (is (allclose (funcall h x y) (funcall *vmap-add* x (funcall *vmap-add* y y)) :dtype :f64))))

;;; --- コンディション ---

(test vmap/out-of-range-in-axes-signals-vmap-error
  "範囲外の in-axes（正にも負にも）、rank 0 の引数への 0 は vmap-error。"
  (let ((x (%mat '((1d0 2d0) (3d0 4d0)))))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :in-axes 2) x x))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :in-axes -3) x x))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :in-axes '(0 0)) x 1d0))))

(test vmap/malformed-axes-signal-vmap-error
  "in-axes のリストの個数が引数の個数と違う・整数でない要素、out-axes の型や個数の不正、
バッチされた引数が無い、バッチされた出力に out-axes nil は vmap-error。"
  (let ((x (%vec 1d0 2d0)))
    (signals nb:vmap-error (nb:vmap *vmap-add* :in-axes '(0)))
    (signals nb:vmap-error (nb:vmap *vmap-add* :in-axes '(0 :a)))
    (signals nb:vmap-error (nb:vmap *vmap-add* :out-axes 1.5))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :out-axes '(0 0)) x x))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :in-axes nil) x x))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :out-axes nil) x x))
    (signals nb:vmap-error (funcall (nb:vmap *vmap-add* :out-axes 2) x x))
    (signals nb:vmap-error (nb:vmap 42))))

(test vmap/axis-size-mismatch-signals-vmap-error
  "バッチ軸の長さが引数の間で違うと vmap-error。バッチされていない引数の長さは関係ない。"
  (signals nb:vmap-error (funcall (nb:vmap *vmap-add*) (%vec 1d0 2d0) (%vec 1d0 2d0 3d0))))

(nb:defprimitive %vmap-test-without-rule ()
  :abstract-eval (lambda (in-avals) (first in-avals))
  :eager (lambda (arrays in-avals) (declare (ignore in-avals)) (first arrays)))

(test vmap/missing-batch-rule-signals-no-batch-rule
  "バッチ軸を持つ値がバッチ化ルールの無いプリミティブに渡ると no-batch-rule（名前つき、vmap-error の子）。
バッチされていない値だけに適用されるときは、ルールが無くても通る。"
  (let ((f (nb:with-tracing (x y) (+ x (nb::%trace-eqn :%vmap-test-without-rule (list y)))))
        (x (%vec 1d0 2d0))
        (y 3d0))
    (handler-case (funcall (nb:vmap f :in-axes '(0 0)) x (%vec 3d0 4d0))
      (nb:no-batch-rule (c)
        (is (eq :%vmap-test-without-rule (nb:no-batch-rule-name c)))
        (is (typep c 'nb:vmap-error))))
    (is (equalp (%vec 4d0 5d0) (funcall (nb:vmap f :in-axes '(0 nil)) x y)) "y はバッチされないのでルール不要")))

(test vmap/batch-rule-survives-defprimitive-reevaluation
  "DEF-BATCH-RULE で設定したルールは、:BATCH を明示しない defprimitive の再評価で引き継がれる。"
  (let ((rule (lambda (args dims) (values args dims))))
    (nb::set-batch-rule :%vmap-test-without-rule rule)
    (unwind-protect
         (progn
           (nb:defprimitive %vmap-test-without-rule ()
             :abstract-eval (lambda (in-avals) (first in-avals)))
           (is (eq rule (nb::primitive-batch (nb::find-primitive :%vmap-test-without-rule)))))
      (nb:defprimitive %vmap-test-without-rule ()
        :abstract-eval (lambda (in-avals) (first in-avals))
        :eager (lambda (arrays in-avals) (declare (ignore in-avals)) (first arrays)))
      (setf (nb::primitive-batch (nb::find-primitive :%vmap-test-without-rule)) nil))))

(test vmap/accepts-jit-function-but-not-one-with-static-args
  "jit した関数は中の traceable-function を使って vmap できる。静的引数のある jit は vmap-error。"
  (let ((x (%vec 1d0 2d0)))
    (is (equalp (%vec 2d0 4d0) (funcall (nb:vmap (nb:jit *vmap-add*)) x x)))
    (signals nb:vmap-error
      (nb:vmap (nb:jit (nb:with-tracing (x n) (progn n x)) :static-args '(1))))))

(test vmap/wrong-call-arity-signals-vmap-error
  "vmap した関数を引数の個数が違う形で呼ぶと vmap-error。"
  (signals nb:vmap-error (funcall (nb:vmap *vmap-add*) (%vec 1d0 2d0)))
  (signals nb:vmap-error (funcall (nb:vmap *vmap-add*) (%vec 1d0 2d0) (%vec 1d0 2d0) (%vec 1d0 2d0))))

(test vmap/graph-constants-are-unbatched
  "f が定数（リテラルの 1d0 を持ち上げたもの）を使っても、バッチされない値として通り、結果は参照実装と一致する。"
  (let* ((f (nb:with-tracing (x) (+ x 1d0)))
         (x (make-random-array (make-array-spec '(3 2) :f64) :seed 3)))
    (is (allclose (funcall (nb:vmap f :in-axes 1 :out-axes 1) x)
                  (first (reference-vmap f (list x) :in-axes 1 :out-axes 1))
                  :dtype :f64))))

(defun %vmap-with-bad-rule (rule)
  "%vmap-test-without-rule のバッチ化ルールを RULE にして、(x) の1回だけ適用する関数の vmap を
長さ 3 のベクトルに適用する。後始末でルールを消す。"
  (let ((f (nb:with-tracing (x) (nb::%trace-eqn :%vmap-test-without-rule (list x)))))
    (nb::set-batch-rule :%vmap-test-without-rule rule)
    (unwind-protect (funcall (nb:vmap f) (%vec 1d0 2d0 3d0))
      (setf (nb::primitive-batch (nb::find-primitive :%vmap-test-without-rule)) nil))))

(test vmap/inconsistent-rule-results-signal-vmap-error
  "バッチ化ルールが、出力の個数・軸・形が元の eqn と整合しない結果を返したら vmap-error
（軸が rank 以上・負・バッチ軸の長さ違い、バッチされないのに形が違う、リストでない）。"
  (flet ((bad (outs-fn dims)
           (lambda (args batch-dims)
             (declare (ignore batch-dims))
             (values (funcall outs-fn (first args)) dims))))
    ;; 個数が違う
    (signals nb:vmap-error (%vmap-with-bad-rule (bad (lambda (x) (list x x)) '(0 0))))
    (signals nb:vmap-error (%vmap-with-bad-rule (bad #'list '(0 0))))
    ;; リストでない
    (signals nb:vmap-error (%vmap-with-bad-rule (bad #'identity 0)))
    ;; 軸が rank（= 1）以上・負
    (signals nb:vmap-error (%vmap-with-bad-rule (bad #'list '(1))))
    (signals nb:vmap-error (%vmap-with-bad-rule (bad #'list '(-1))))
    ;; バッチされない（nil）と言いながら形にバッチ軸が残っている（長さ 3 のまま）
    (signals nb:vmap-error (%vmap-with-bad-rule (bad #'list '(nil))))
    ;; 軸 0 を取り除くと元の形（rank 0）に一致しない形（rank 2）を返す
    (signals nb:vmap-error
      (%vmap-with-bad-rule
       (lambda (args batch-dims)
         (declare (ignore batch-dims))
         (values (list (nb::%trace-eqn :broadcast-in-dim (list (first args)) :shape '(3 2) :dims '(0)))
                 '(0)))))
    ;; dtype が違う（f32 に変換した値を返す）
    (signals nb:vmap-error
      (%vmap-with-bad-rule
       (lambda (args batch-dims)
         (declare (ignore batch-dims))
         (values (list (nb::%trace-eqn :convert (list (first args)) :dtype :f32)) '(0)))))
    ;; トレーサでない
    (signals nb:vmap-error (%vmap-with-bad-rule (bad (lambda (x) (declare (ignore x)) 1) '(0))))
    ;; 正しい結果は通る
    (is (equalp (%vec 1d0 2d0 3d0)
                (%vmap-with-bad-rule (bad #'list '(0)))))))

(test vmap/rule-may-return-unbatched-output-of-rank-above-zero
  "ルールがバッチされない出力（軸 nil）を、rank 1 の元の形のまま返すとき vmap は成功し、
out-axes 0 でバッチ軸に複製される。"
  (let ((f (nb:with-tracing (x) (nb::%trace-eqn :%vmap-test-without-rule (list x))))
        (rule (lambda (args batch-dims)
                (declare (ignore args batch-dims))
                (values (list (nb::%lift-constant (%vec 5d0 6d0) (nb:make-aval '(2) :f64) nb::*current-trace*))
                        '(nil)))))
    (nb::set-batch-rule :%vmap-test-without-rule rule)
    (unwind-protect
         (is (equalp (%mat '((5d0 6d0) (5d0 6d0) (5d0 6d0)))
                     (funcall (nb:vmap f) (%mat '((1d0 2d0) (3d0 4d0) (5d0 6d0))))))
      (setf (nb::primitive-batch (nb::find-primitive :%vmap-test-without-rule)) nil))))
