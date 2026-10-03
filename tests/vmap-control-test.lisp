;;;; cond* / while-loop のバッチ化ルールの性質（issue #140）。
;;;;
;;;; 守らせる性質: 「vmap f の結果は、バッチ軸で切り出した各要素に f を eager で適用して
;;;; 積み直したもの（REFERENCE-VMAP）と一致する」。引数ごとの in-axes（バッチしない NIL・
;;;; 軸 0・軸 1）をランダムにして、条件がバッチされる場合・されない場合、
;;;; carry の一部だけがバッチされる場合（最初はバッチされない carry が本体でバッチされる
;;;; 回帰を含む）を生成する。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(defun %vc-axis (code)
  "CODE から in-axes の1要素（NIL・0・1）を選ぶ。"
  (nth (mod code 3) '(nil 0 1)))

(defun %vc-in-axes (code-x code-y)
  "(x y) の in-axes。少なくとも1つはバッチされる（両方 NIL なら x を軸 0 にする）。"
  (let ((ax (%vc-axis code-x)) (ay (%vc-axis code-y)))
    (if (or ax ay) (list ax ay) (list 0 ay))))

(defun %vc-clamp-axis (axis inner-shape)
  "AXIS を INNER-SHAPE にバッチ軸を足せる範囲（0..rank）に収める（NIL はそのまま）。"
  (and axis (min axis (length inner-shape))))

(defun %vc-array (inner-shape size axis seed)
  (make-random-array
   (make-array-spec (if axis (axis-at inner-shape size axis) inner-shape) :f64)
   :seed seed))

(defun %vc-limit (inner-shape size axis seed)
  "0〜4 の整数値の f64 の配列（反復回数。要素ごとに違う）。"
  (let ((a (%vc-array inner-shape size axis seed)))
    (dotimes (i (array-total-size a) a)
      (setf (row-major-aref a i) (float (mod (+ seed (* 7 i)) 5) 1d0)))))

(defun %vc-case-generator ()
  (generator (tuple (uniform-integer :lo 1 :hi 4)       ; 0 バッチの長さ
                    (uniform-integer :lo 0 :hi 100000)  ; 1 seed
                    (uniform-integer :lo 0 :hi 99)      ; 2 x の in-axes
                    (uniform-integer :lo 0 :hi 99))))   ; 3 第2引数の in-axes

(defun %vc-property (f inner-x inner-y &key limit-y)
  "F（2引数。x は INNER-X、第2引数は INNER-Y の形）の vmap が参照実装と一致する性質。
LIMIT-Y が真なら第2引数は反復回数（整数値）の配列にする。"
  (lambda (case)
    (destructuring-bind (size seed code-x code-y) case
      (destructuring-bind (ax ay) (%vc-in-axes code-x code-y)
        (let* ((ay (%vc-clamp-axis ay inner-y))
               (ax (%vc-clamp-axis ax inner-x))
               (x (%vc-array inner-x size ax seed))
               (y (if limit-y
                      (%vc-limit inner-y size ay (+ seed 1))
                      (%vc-array inner-y size ay (+ seed 1))))
               (expected (reference-vmap f (list x y) :in-axes (list ax ay)))
               (actual (multiple-value-list
                        (funcall (nb:vmap f :in-axes (list ax ay) :out-axes 0) x y))))
          (and (= (length actual) (length expected))
               (every (lambda (a e) (allclose a e :dtype :f64)) actual expected)))))))

(defmacro def-vc-test (name doc f inner-x inner-y &key limit-y)
  `(test ,name
     ,doc
     (is (check-it (%vc-case-generator)
                   (%vc-property ,f ,inner-x ,inner-y :limit-y ,limit-y)
                   :regression-id ,name
                   :regression-file (regression-path "vmap-control")))))

;;; --- cond ---

(defparameter *vc-then*
  ;; 出力2は u だけに依存する（u がバッチされなければバッチされない出力）。
  (nb:with-tracing (u v) (values (+ u v) (* u 2.0))))

(defparameter *vc-else*
  ;; 出力1は v だけに依存する。
  (nb:with-tracing (u v) (values (- u v) v)))

(defparameter *vc-cond*
  (nb:with-tracing (x y)
    (multiple-value-bind (a b)
        (nb:cond* (< (nb:reduce-sum x :axes '(0)) 0.0) *vc-then* *vc-else* x y)
      (values a b))))

(def-vc-test vmap-control/cond-matches-per-slice-reference
  "cond*: 条件（x の総和の符号）・operand のバッチされ方をランダムにしても参照実装と一致する
（条件がバッチされる場合は select に落ちる。片方の枝でだけバッチされる出力を含む）。"
  *vc-cond* '(3) '(3))

(def-vc-test vmap-control/cond-with-unbatched-pred-matches-reference
  "cond*: 条件がバッチされない（条件は y だけから作る）とき、両枝をバッチ化した cond のまま
参照実装と一致する。"
  (nb:with-tracing (x y)
    (multiple-value-bind (a b)
        (nb:cond* (< (nb:reduce-sum y :axes '(0)) 0.0) *vc-then* *vc-else* x y)
      (values a b)))
  '(3) '(3))

(test vmap-control/cond-with-unbatched-pred-stays-a-cond
  "条件がバッチされなければ、vmap した graph は :cond を保つ（select に落とさない）。
条件がバッチされれば :cond は無く、select になる。"
  (flet ((prims (f in-axes)
           (mapcar (lambda (e) (nb::primitive-name (nb:eqn-prim e)))
                   (nb:graph-eqns (nb:trace-to-graph
                                   (nb:vmap f :in-axes in-axes)
                                   (list (nb:make-aval '(2 3) :f64) (nb:make-aval '(3) :f64)))))))
    (let ((f (nb:with-tracing (x y)
               (multiple-value-bind (a b)
                   (nb:cond* (< (nb:reduce-sum y :axes '(0)) 0.0) *vc-then* *vc-else* x y)
                 (values a b)))))
      (is (member :cond (prims f '(0 nil))))
      (is (not (member :select (prims f '(0 nil))))))
    (is (not (member :cond (prims *vc-cond* '(0 nil)))))
    (is (member :select (prims *vc-cond* '(0 nil))))))

;;; --- while-loop ---

(defparameter *vc-while-count*
  ;; 反復回数 limit（第2引数。スカラー）。条件は carry の i と limit の比較。
  (nb:with-tracing (x limit)
    (let ((r (nb:while-loop (nb:with-tracing (c) (< (first c) limit))
                            (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (* (second c) 1.5) 1.0)))
                            (list (nb::%scalar-array 0d0 :f64) x))))
      (values (first r) (second r)))))

(def-vc-test vmap-control/while-with-batched-pred-matches-reference
  "while-loop: 条件（反復回数 limit）が要素ごとに違う（バッチされる）とき、どれかが続く間回し、
終わった要素は据え置く。limit だけ・x だけ・両方がバッチされる組を含めて参照実装と一致する。"
  *vc-while-count* '(3) '() :limit-y t)

(defparameter *vc-while-captured*
  ;; 条件は定数回（i < 3）。a の init は y（バッチされないことがある）で、本体が x（バッチされる
  ;; ことがある。閉包で捕まえた loop 不変の値）を足すので、最初はバッチされない carry が
  ;; 本体を通るとバッチされる（回帰）。
  (nb:with-tracing (x y)
    (let ((r (nb:while-loop (nb:with-tracing (c) (< (first c) 3.0))
                            (nb:with-tracing (c) (list (+ (first c) 1.0) (+ (second c) x)))
                            (list (nb::%scalar-array 0d0 :f64) y))))
      (values (first r) (second r)))))

(def-vc-test vmap-control/while-carry-becoming-batched-matches-reference
  "while-loop: 条件がバッチされず、最初はバッチされない carry が本体（閉包で捕まえたバッチされる
x を足す）でバッチされる場合の回帰。不動点で carry を広げて参照実装と一致する。"
  *vc-while-captured* '(3) '(3))

(test vmap-control/while-carry-becoming-batched-keeps-unbatched-pred
  "回帰の graph: 条件がバッチされないので :while-loop のまま（cond は i < 3 だけで select が無い）。
x だけをバッチしても carry a がバッチされる（出力の形が [B, 3]）。"
  (let* ((g (nb:vmap *vc-while-captured* :in-axes '(0 nil)))
         (graph (nb:trace-to-graph g (list (nb:make-aval '(4 3) :f64) (nb:make-aval '(3) :f64)))))
    (is (member :while-loop (mapcar (lambda (e) (nb::primitive-name (nb:eqn-prim e)))
                                    (nb:graph-eqns graph))))
    (is (equal '(4 3) (nb:aval-shape (nb::var-aval (second (nb:graph-outvars graph))))))
    (is (equal '(4) (nb:aval-shape (nb::var-aval (first (nb:graph-outvars graph))))))))

(defparameter *vc-while-cond-body*
  ;; 本体の中で cond* を使う while-loop（cond と while-loop の両方を含む関数）。
  (nb:with-tracing (x limit)
    (let ((r (nb:while-loop
              (nb:with-tracing (c) (< (first c) limit))
              (nb:with-tracing (c)
                (list (+ (first c) 1.0)
                      (nb:cond* (< (nb:reduce-sum (second c) :axes '(0)) 0.0)
                                       (nb:with-tracing (u) (+ u u))
                                       (nb:with-tracing (u) (- u 0.5))
                                       (second c))))
              (list (nb::%scalar-array 0d0 :f64) x))))
      (values (first r) (second r)))))

(def-vc-test vmap-control/while-with-cond-in-body-matches-reference
  "本体に cond* を持つ while-loop（条件のバッチあり・なし）も参照実装と一致する。"
  *vc-while-cond-body* '(3) '() :limit-y t)

;;; --- 入れ子の vmap ---

(defun %vc-nested-reference (f x limit)
  "外側・内側とも軸 0 で切り出した参照実装（出力は軸 0 に積み直す）。"
  (reference-vmap (lambda (xs ls)
                    (values-list (reference-vmap f (list xs ls))))
                  (list x limit)))

(test vmap-control/nested-vmap-of-while-with-cond-matches-reference
  "(vmap (vmap f)) の f が while-loop と cond* の両方を使っても、2重に切り出した参照実装と
一致する（x は [B1, B2, 3]、limit は [B1, B2]）。"
  (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 3)
                                  (uniform-integer :lo 0 :hi 100000)))
                (lambda (case)
                  (destructuring-bind (b1 b2 seed) case
                    (let* ((x (make-random-array (make-array-spec (list b1 b2 3) :f64) :seed seed))
                           (limit (%vc-limit (list b1) b2 1 (+ seed 1)))
                           (expected (%vc-nested-reference *vc-while-cond-body* x limit))
                           (actual (multiple-value-list
                                    (funcall (nb:vmap (nb:vmap *vc-while-cond-body*)) x limit))))
                      (every (lambda (a e) (allclose a e :dtype :f64)) actual expected))))
                :regression-id vmap-control/nested-vmap-of-while-with-cond-matches-reference
                :regression-file (regression-path "vmap-control"))))

;;; --- scan ---

(defun %vc-scan-fn (reverse)
  "(w h0 xs) → (最終 h, 最終 c, ys の (h*x) と c)。w は本体が閉包で捕まえる const、
h は [3] の carry、c は定数 0 から始まる（バッチされない）スカラーの carry、xs は [L, 3]。"
  (nb:with-tracing (w h0 xs)
    (multiple-value-bind (carry ys)
        (nb:scan (nb:with-tracing (carry x)
                   (let ((h (first carry)) (c (second carry)) (u (first x)))
                     (values (list (+ (* h 0.5) (+ u w)) (+ c 1.0))
                             (list (* h u) c))))
                 (list h0 (nb::%scalar-array 0d0 :f64)) (list xs) :reverse reverse)
      (values (first carry) (second carry) (first ys) (second ys)))))

(defun %vc-scan-property (reverse)
  "(w h0 xs) の in-axes（NIL・0・1、xs は 2 も）をランダムにした vmap が参照実装と一致する。"
  (let ((f (%vc-scan-fn reverse)))
    (lambda (case)
      (destructuring-bind (size seed length code-w code-h code-x) case
        (let* ((aw (%vc-axis code-w))
               (ah (%vc-axis code-h))
               (ax (nth (mod code-x 4) '(nil 0 1 2)))
               (axes (if (or aw ah ax) (list aw ah ax) (list nil 0 ax)))
               (w (%vc-array '(3) size (first axes) seed))
               (h (%vc-array '(3) size (second axes) (+ seed 1)))
               (xs (%vc-array (list length 3) size (third axes) (+ seed 2)))
               (expected (reference-vmap f (list w h xs) :in-axes axes))
               (actual (multiple-value-list (funcall (nb:vmap f :in-axes axes :out-axes 0) w h xs))))
          (and (= (length actual) (length expected))
               (every (lambda (a e) (allclose a e :dtype :f64)) actual expected)))))))

(defun %vc-scan-generator ()
  (generator (tuple (uniform-integer :lo 1 :hi 3)       ; バッチの長さ
                    (uniform-integer :lo 0 :hi 100000)  ; seed
                    (uniform-integer :lo 0 :hi 3)       ; scan の長さ（0 を含む）
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99)
                    (uniform-integer :lo 0 :hi 99))))

(test vmap-control/scan-forward-matches-reference
  "scan: const・carry・xs のバッチされ方（xs のバッチ軸は走査の軸の前後どちらでも）と長さ
（0 と 1 を含む）をランダムにしても、参照実装と一致する。h0 がバッチされず const か xs だけが
バッチされる場合は、最初はバッチされない carry が本体でバッチされる回帰になる。"
  (is (check-it (%vc-scan-generator) (%vc-scan-property nil)
                :regression-id vmap-control/scan-forward-matches-reference
                :regression-file (regression-path "vmap-control"))))

(test vmap-control/scan-reverse-matches-reference
  "reverse の scan も参照実装と一致する（ys[t] には添字 t のステップの y が入る）。"
  (is (check-it (%vc-scan-generator) (%vc-scan-property t)
                :regression-id vmap-control/scan-reverse-matches-reference
                :regression-file (regression-path "vmap-control"))))

(test vmap-control/scan-unbatched-carry-becomes-batched
  "回帰: h0 も w もバッチされず xs だけがバッチされるとき、carry h は本体で x を足されてバッチされる。
定数から始まる carry c はバッチされないまま（出力の形はスカラー、ys の c は [L]）。ys の h*x は
バッチ軸が 1 の [L, B, 3] で、out-axes で軸 0 に移る。"
  (let* ((g (nb:vmap (%vc-scan-fn nil) :in-axes '(nil nil 1) :out-axes '(0 nil 0 nil)))
         (graph (nb:trace-to-graph g (list (nb:make-aval '(3) :f64) (nb:make-aval '(3) :f64)
                                           (nb:make-aval '(2 4 3) :f64))))
         (shapes (mapcar (lambda (v) (nb:aval-shape (nb::var-aval v))) (nb:graph-outvars graph))))
    (is (equal '((4 3) () (4 2 3) (2)) shapes))))

(test vmap-control/nested-vmap-of-scan-matches-reference
  "(vmap (vmap f)) の f が scan を使っても、2重に切り出した参照実装と一致する。"
  (let ((f (%vc-scan-fn nil)))
    (is (check-it (generator (tuple (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 3)
                                    (uniform-integer :lo 0 :hi 100000)))
                  (lambda (case)
                    (destructuring-bind (b1 b2 seed) case
                      (let* ((w (make-random-array (make-array-spec (list b1 3) :f64) :seed seed))
                             (h (make-random-array (make-array-spec (list b1 b2 3) :f64) :seed (+ seed 1)))
                             (xs (make-random-array (make-array-spec (list b1 2 b2 3) :f64) :seed (+ seed 2)))
                             (inner (lambda (w h xs) (values-list (reference-vmap f (list w h xs) :in-axes '(nil 0 1)))))
                             (expected (reference-vmap inner (list w h xs) :in-axes '(0 0 0)))
                             (actual (multiple-value-list
                                      (funcall (nb:vmap (nb:vmap f :in-axes '(nil 0 1)) :in-axes '(0 0 0))
                                               w h xs))))
                        (every (lambda (a e) (allclose a e :dtype :f64)) actual expected))))
                  :regression-id vmap-control/nested-vmap-of-scan-matches-reference
                  :regression-file (regression-path "vmap-control")))))

(test vmap-control/select-by-batched-pred-skips-identity-broadcast
  "バッチされた条件で select に落ちる cond* は、出力の形が [B] のときは条件をそのまま使って
broadcast-in-dim を作らず、形が [B, 3] の出力が複数あっても、その形のための broadcast は1つだけ。"
  (let* ((then (nb:with-tracing (u) (values (nb:reduce-sum u :axes '(0)) (* u 2.0) (* u 3.0))))
         (else (nb:with-tracing (u) (values (nb:reduce-sum u :axes '(0)) (+ u 1.0) (+ u 2.0))))
         (f (nb:with-tracing (x)
              (multiple-value-bind (a b c)
                  (nb:cond* (< (nb:reduce-sum x :axes '(0)) 0.0) then else x)
                (values a b c))))
         (graph (nb:trace-to-graph (nb:vmap f) (list (nb:make-aval '(2 3) :f64))))
         (eqns (nb:graph-eqns graph))
         (prims (mapcar (lambda (e) (nb::primitive-name (nb:eqn-prim e))) eqns))
         ;; 条件 [2] を [2, 3] へ広げる broadcast（他の broadcast は定数のリテラルなど）
         (pred-broadcasts (remove-if-not
                           (lambda (e)
                             (and (eq :broadcast-in-dim (nb::primitive-name (nb:eqn-prim e)))
                                  (equal '(2) (nb:aval-shape (nb::var-aval (first (nb:eqn-invars e)))))))
                           eqns)))
    (is (= 3 (count :select prims)))
    (is (= 1 (length pred-broadcasts)))))
