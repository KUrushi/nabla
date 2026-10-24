;;;; jit の入力の連結（issue #242、small）。
;;;;
;;;; 葉の個数が実行系の入力の上限（BACKEND-MAX-INPUTS）を超えるとき、jit は同じ dtype の葉を
;;;; 1 本の 1 次元バッファに連結して渡し、graph の先頭の slice と reshape で葉に戻す。
;;;; 守らせる性質（フェイク backend。コンパイルを断って USE-EAGER で graph を eval-graph する）:
;;;; - 連結→戻すは恒等: dtype の混ざった（bf16 / f16 / i1 / 整数 / rank 0 を含む）葉と device array の
;;;;   葉の木を、そのまま返す関数を jit すると、上限を超えたときも同じ値・同じ要素型の木が返る。
;;;;   そのときコンパイルされる main の入力は dtype の種類の数。
;;;; - 上限を超えなければ連結しない: StableHLO は上限なしの backend と同じテキスト。
;;;; - 戻した葉は後続の演算（x + x）で eager と一致する。要素数 0・rank 0・実数の葉も戻る。
;;;; - dtype の種類の数も上限を超えれば jit-error。
;;;; IREE の local-task で、上限を超える Transformer を学習するのは tests/nn/transformer-test.lisp（medium）。

(in-package #:nabla.tests)

(in-suite :nabla.small)

(define-condition %pack-test-refusal (nb:backend-error) ()
  (:documentation "%PACK-RECORDING-BACKEND がコンパイルを断るときに signal する。"))

(defclass %pack-recording-backend (fake-backend)
  ((max-inputs :initarg :max-inputs :reader %pack-max-inputs)
   (texts :initform nil :accessor %pack-texts))
  (:documentation "入力の上限 MAX-INPUTS を申告し、コンパイルした StableHLO を記録してから断るフェイク。"))

(defmethod nb:backend-max-inputs ((backend %pack-recording-backend))
  (%pack-max-inputs backend))

(defmethod nb:backend-compile ((backend %pack-recording-backend) text)
  (push text (%pack-texts backend))
  (error '%pack-test-refusal))

(defun %pack-call-eagerly (backend fn &rest args)
  "FN を BACKEND で jit して ARGS で呼ぶ。コンパイルは断られるので、USE-EAGER で（連結した入力の）
graph を評価した結果を返す。"
  (let ((nb:*compile-cache-directory* nil))
    (handler-bind ((nb:jit-compile-error (lambda (c) (declare (ignore c)) (invoke-restart 'nb:use-eager))))
      (apply (nb:jit fn :backend backend) args))))

(defun %main-input-count (text)
  "StableHLO の TEXT の main の入力の個数。"
  (let* ((start (search "func.func @main(" text))
         (end (search ") ->" text :start2 start)))
    (%count-subsequence "tensor<" text :start start :end end)))

(defun %count-subsequence (needle haystack &key (start 0) end)
  (loop with position = start
        for found = (search needle haystack :start2 position :end2 end)
        while found
        count t
        do (setf position (1+ found))))

(defun %pack-leaves (specs seed backend)
  "SPECS の配列の葉。bf16 / f16（ホストの配列では dtype が分からない）と、半分ほどの葉は BACKEND の
device array にする。"
  (loop for spec in specs
        for i from 0
        collect (let ((array (make-random-array spec :seed (+ seed i)))
                      (dtype (array-spec-dtype spec)))
                  (if (or (member dtype '(:bf16 :f16)) (oddp (+ seed i)))
                      (nb:to-device array backend :dtype dtype)
                      array))))

(defun %host-leaf (leaf)
  (if (arrayp leaf) leaf (nb:to-host leaf)))

(defun %pack-dtype-count (specs)
  (length (remove-duplicates (mapcar #'array-spec-dtype specs))))

(defparameter *pack-dtypes* '(:f32 :f64 :bf16 :f16 :i1 :i32 :u32))

(test jit-pack/concatenate-then-split-is-identity
  "上限を超える葉の木をそのまま返す関数を jit すると、同じ木（値・要素型）が返り、main の入力は
dtype の種類の数になる。"
  (is (check-it (generator (tuple (list (array-spec :dtypes *pack-dtypes* :max-rank 3 :max-dim 4)
                                        :min-length 2 :max-length 12)
                                  (uniform-integer :lo 0 :hi 1000)
                                  (uniform-integer :lo 0 :hi 11)))
                (lambda (value)
                  (destructuring-bind (specs seed extra) value
                    ;; 上限は dtype の種類の数以上、葉の個数未満（連結が起き、連結すれば収まる）
                    (let* ((low (%pack-dtype-count specs))
                           (high (1- (length specs))))
                      (or (> low high)
                          (let* ((limit (+ low (mod extra (1+ (- high low)))))
                                 (backend (make-instance '%pack-recording-backend :max-inputs limit))
                                 (leaves (%pack-leaves specs seed backend))
                                 (result (%pack-call-eagerly backend (nb:with-tracing (xs) xs)
                                                             (list (first leaves) (coerce (rest leaves) 'vector)))))
                            (and (= low (%main-input-count (first (%pack-texts backend))))
                                 (every (lambda (r leaf)
                                          (let ((expected (%host-leaf leaf)))
                                            (and (equal (array-element-type r) (array-element-type expected))
                                                 (equal (array-dimensions r) (array-dimensions expected))
                                                 (equalp r expected))))
                                        (cons (first result) (coerce (second result) 'list))
                                        leaves)))))))
                :regression-id jit-pack/concatenate-then-split-is-identity
                :regression-file (regression-path "jit-pack-identity"))))

(test jit-pack/split-leaves-feed-later-operations
  "上限を超える浮動小数点の葉の木に葉ごとの x + x を jit すると、連結して戻した葉が後続の演算に正しい形で
渡り、eager と一致する。"
  (is (check-it (generator (tuple (list (array-spec :dtypes '(:f32 :f64) :max-rank 3 :max-dim 4)
                                        :min-length 3 :max-length 10)
                                  (uniform-integer :lo 0 :hi 1000)))
                (lambda (value)
                  (destructuring-bind (specs seed) value
                    (let* ((leaves (loop for spec in specs for i from 0
                                         collect (make-random-array spec :seed (+ seed i))))
                           (backend (make-instance '%pack-recording-backend :max-inputs 2))
                           (f (nb:with-tracing (xs) (nb:tree-map (lambda (x) (+ x x)) xs)))
                           (result (%pack-call-eagerly backend f leaves)))
                      (every (lambda (r leaf spec)
                               (allclose r (funcall f leaf) :dtype (array-spec-dtype spec)))
                             result leaves specs))))
                :regression-id jit-pack/split-leaves-feed-later-operations
                :regression-file (regression-path "jit-pack-later-operations"))))

(test jit-pack/zero-size-rank-0-and-non-simple-leaves-round-trip
  "要素数 0 の葉（形 (0 3)）、rank 0 の葉・Lisp の実数、simple でない配列（adjustable）の葉が混ざった木も、
連結して戻すと同じ木になる（PBT の生成器は次元が 1 以上で simple な配列だけなので、ここで見る）。"
  (let* ((backend (make-instance '%pack-recording-backend :max-inputs 1))
         (empty (make-array '(0 3) :element-type 'single-float))
         (scalar (make-array '() :element-type 'single-float :initial-element 2.5))
         (vector (make-array '(2) :element-type 'single-float :initial-contents '(1.0 -1.0)))
         (adjustable (make-array '(2 2) :element-type 'single-float :adjustable t
                                        :initial-contents '((4.0 5.0) (6.0 7.0))))
         (result (%pack-call-eagerly backend (nb:with-tracing (xs) xs)
                                     (list empty scalar 3.0 vector adjustable))))
    (is (= 1 (%main-input-count (first (%pack-texts backend)))))
    (is (equal '(0 3) (array-dimensions (first result))))
    (is (equalp (list scalar (make-array '() :element-type 'single-float :initial-element 3.0) vector adjustable)
                (rest result)))))

(test jit-pack/within-the-limit-the-stablehlo-is-unchanged
  "葉の個数が上限以下なら連結せず、StableHLO は上限なしの backend と同じテキストになる。"
  (is (check-it (generator (tuple (list (array-spec :dtypes '(:f32 :f64 :i32 :i1) :max-rank 3 :max-dim 4)
                                        :min-length 1 :max-length 8)
                                  (uniform-integer :lo 0 :hi 4)))
                (lambda (value)
                  (destructuring-bind (specs extra) value
                    (let* ((leaves (loop for spec in specs for i from 0
                                         collect (make-random-array spec :seed i)))
                           (limited (make-instance '%pack-recording-backend
                                                   :max-inputs (+ (length specs) extra)))
                           (unlimited (make-instance '%pack-recording-backend :max-inputs nil))
                           (f (nb:with-tracing (xs) xs)))
                      (%pack-call-eagerly limited f leaves)
                      (%pack-call-eagerly unlimited f leaves)
                      (and (equal (%pack-texts limited) (%pack-texts unlimited))
                           (= (length specs) (%main-input-count (first (%pack-texts limited))))))))
                :regression-id jit-pack/within-the-limit-the-stablehlo-is-unchanged
                :regression-file (regression-path "jit-pack-within-limit"))))

(test jit-pack/more-dtypes-than-the-limit-signals-jit-error
  "dtype の種類の数が上限を超えると、連結しても呼べないので jit-error になる。"
  (let ((backend (make-instance '%pack-recording-backend :max-inputs 1)))
    ;; コンパイルの失敗（jit-compile-error）は USE-EAGER で流すので、ここで出る jit-error は連結の前の検査のもの
    (signals nb:jit-error
      (%pack-call-eagerly backend (nb:with-tracing (xs) xs)
                          (list (make-random-array (make-array-spec '(2 3) :f32))
                                (make-random-array (make-array-spec '(2) :f64)))))
    (is (null (%pack-texts backend)) "連結できないのにコンパイルしようとした")))

(test jit-pack/backend-max-inputs-defaults-to-no-limit
  "backend-max-inputs の既定は NIL（上限なし）。"
  (is (null (nb:backend-max-inputs (nb:make-backend :fake)))))

;;; --- 内部プリミティブ :slice（連結を戻す部品）---

(defun %slice-abstract-eval (in-avals &rest params)
  (apply (nb::primitive-abstract-eval (nb::find-primitive :slice)) in-avals params))

(defun %row-major-subscripts (index dims)
  (let ((subs nil))
    (dolist (d (reverse dims) subs)
      (multiple-value-bind (q r) (floor index d)
        (push r subs)
        (setf index q)))))

(defun %slice-eager (array start limit)
  (let ((aval (nb:array-aval array)))
    (funcall (nb::primitive-eager (nb::find-primitive :slice)) (list array) (list aval)
             :start start :limit limit)))

(test jit-pack/slice-cuts-the-subarray
  "slice は start（含む）から limit（含まない）までの部分配列を返し、その形は limit - start。範囲の外
（limit > 次元、start > limit、rank の不一致）は primitive-error。"
  (is (check-it (generator (tuple (array-spec :dtypes '(:f32 :f64 :i32 :i1) :max-rank 3 :max-dim 5)
                                  (list (uniform-integer :lo 0 :hi 100) :min-length 6 :max-length 6)))
                (lambda (value)
                  (destructuring-bind (spec cuts) value
                    (let* ((array (make-random-array spec :seed (first cuts)))
                           (shape (array-spec-shape spec))
                           (start (loop for d in shape for c in cuts collect (mod c (1+ d))))
                           (limit (loop for d in shape for s in start for c in (nthcdr 3 cuts)
                                        collect (+ s (mod c (1+ (- d s))))))
                           (result (%slice-eager array start limit)))
                      (and (equal (array-dimensions result) (mapcar #'- limit start))
                           (equal (nb:aval-shape (%slice-abstract-eval (list (nb:array-aval array))
                                                                              :start start :limit limit))
                                  (mapcar #'- limit start))
                           (let ((ok t))
                             (dotimes (i (array-total-size result) ok)
                               (let* ((sub (%row-major-subscripts i (array-dimensions result)))
                                      (src (mapcar #'+ sub start)))
                                 (unless (eql (row-major-aref result i) (apply #'aref array src))
                                   (setf ok nil)))))))))
                :regression-id jit-pack/slice-cuts-the-subarray
                :regression-file (regression-path "jit-pack-slice")))
  (let ((aval (nb:make-aval '(4 3) :f32)))
    (dolist (params '((:start (0 0) :limit (5 3)) (:start (2 0) :limit (1 3)) (:start (0) :limit (4))
                      (:start (a 0) :limit (1 3)) (:start (0 0) :limit (1 b))))
      (signals nb:primitive-error
        (apply #'%slice-abstract-eval (list aval) params)))
    (signals nb:primitive-error (%slice-abstract-eval (list aval aval) :start '(0 0) :limit '(1 1)))))
