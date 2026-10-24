;;;; gather / scatter-add（issue #240）の medium テスト。
;;;;
;;;; jit した関数を IREE (local) で実行し、jit しない（eager の）結果と比べる。添字は範囲外（負・N 以上）と
;;;; 重複を含む。StableHLO の gather は範囲外をクランプし scatter は実装定義なので、emit が自分で書いた範囲の
;;;; 判定（範囲外の行は 0、範囲外の更新は捨てる）が IREE で eager と同じ値になることと、重複する添字の更新が
;;;; scatter で足し合わさる（unique_indices = false）ことを確かめる。bf16 の scatter-add は f32 で累積する
;;;; convert → scatter → convert が IREE で畳まれずに動くことも見る。

(in-package #:nabla.iree.tests)

(defun %coerce-index-rows (contents dtype)
  "CONTENTS（整数の2次元配列）を、DTYPE（unsigned なら負の値を 2^32 / 2^64 で折り返す）の行のリストにする。"
  (loop for r below (array-dimension contents 0)
        collect (loop for c below (array-dimension contents 1)
                      collect (let ((v (aref contents r c)))
                                (cond ((>= v 0) v)
                                      ((eq dtype :i32) v)
                                      ((eq dtype :u32) (+ v (expt 2 32)))
                                      (t (+ v (expt 2 64))))))))

(defun %gather-test-indices (contents dtype)
  (make-array (array-dimensions contents) :element-type (nb:dtype-element-type dtype)
                                          :initial-contents (%coerce-index-rows contents dtype)))

(defparameter *gather-test-ids*
  #2a((0 4 4 -1) (5 2 0 4))
  "operand の先頭の軸が 5 のときの添字: 重複（4、0）・負（-1）・N 以上（5）を含む。")

(defun %gather-jit-matches-p (f arg expected dtype)
  "F を jit して IREE (local) で ARG に適用した結果が EXPECTED と DTYPE の許容誤差で一致するか。"
  (let ((jf (nb:jit f :backend (nabla:find-backend :iree))))
    (unwind-protect
         (multiple-value-bind (rtol atol) (dtype-tolerance dtype)
           (allclose (funcall jf arg) expected :rtol rtol :atol atol))
      (nb::%jit-cache-forget (nb::%jitted-function-fn jf)))))

(defun %round-to-bf16 (array)
  "f32 の ARRAY を bf16 に丸めて f32 に戻した配列（bf16 で正確に表せる値だけになる）。"
  (nb::decode-float16-array (nb::encode-float16-array array :bf16) :bf16))

(define-iree-test gather/jit-matches-eager
    "jit した gather（f32 / bf16 / :i32 の operand、:i32 / :u32 / :u64 の添字）と、jit した grad（scatter-add。
重複する添字の勾配は足し合わさり、範囲外の添字は寄与しない）を IREE (local) で実行した結果は eager と一致する。
bf16 は host の配列では dtype を区別できないので、関数の中で f32 と convert し、期待値は bf16 に丸めた入力の
f32 の eager にする（bf16 の scatter-add は f32 で累積するので、差は最後の丸めだけ）。"
  (skip-unless-iree :library :both)
  (let ((nb:*compile-cache-directory* nil))
    (dolist (index-dtype '(:i32 :u32 :u64))
      (let* ((ids (%gather-test-indices *gather-test-ids* index-dtype))
             (table (make-random-array (make-array-spec '(5 3) :f32) :seed 1))
             (integer-table (make-random-array (make-array-spec '(5 3) :i32) :seed 1))
             (weights (make-random-array (make-array-spec '(2 4 3) :f32) :seed 3))
             (gather (nb:with-tracing (x) (nb:gather x ids)))
             (grad (nb:grad (nb:with-tracing (x) (nb:reduce-sum (* (nb:gather x ids) weights)))))
             (rounded-weights (%round-to-bf16 weights))
             (rounded-grad (nb:grad (nb:with-tracing (x) (nb:reduce-sum (* (nb:gather x ids) rounded-weights))))))
        (is (%gather-jit-matches-p gather table (funcall gather table) :f32)
            "gather: f32、添字 ~S で jit が eager と一致しない" index-dtype)
        (is (%gather-jit-matches-p gather integer-table (funcall gather integer-table) :i32)
            "gather: :i32、添字 ~S で jit が eager と一致しない" index-dtype)
        (is (%gather-jit-matches-p
             (nb:with-tracing (x) (nb:convert (nb:gather (nb:convert x :bf16) ids) :f32))
             table (funcall gather (%round-to-bf16 table)) :bf16)
            "gather: bf16、添字 ~S で jit が eager と一致しない" index-dtype)
        ;; rank 0 の添字（index_vector_dim = 0）と rank 1 の operand（offset_dims が空）。範囲内と範囲外。
        (dolist (id '(3 7))
          (let* ((scalar-id (make-array '() :element-type (nb:dtype-element-type index-dtype) :initial-element id))
                 (pick (nb:with-tracing (x) (nb:gather x scalar-id)))
                 (vector (make-random-array (make-array-spec '(5) :f32) :seed 7)))
            (is (%gather-jit-matches-p pick table (funcall pick table) :f32)
                "gather: rank 0 の添字 ~S（~S）で jit が eager と一致しない" id index-dtype)
            (is (%gather-jit-matches-p pick vector (funcall pick vector) :f32)
                "gather: rank 1 の operand、rank 0 の添字 ~S（~S）で jit が eager と一致しない" id index-dtype)))
        (is (%gather-jit-matches-p grad table (funcall grad table) :f32)
            "grad（scatter-add）: f32、添字 ~S で jit が eager と一致しない" index-dtype)
        (is (%gather-jit-matches-p
             (nb:grad (nb:with-tracing (x)
                        (nb:reduce-sum (nb:convert (* (nb:gather (nb:convert x :bf16) ids)
                                                      (nb:convert weights :bf16))
                                                   :f32))))
             table (funcall rounded-grad table) :bf16)
            "grad（scatter-add）: bf16、添字 ~S で jit が eager と一致しない" index-dtype)))
    (gc-and-run-finalizers)))

(define-iree-test gather/jit-vmap-matches-eager
    "operand（updates）と添字の両方をバッチした vmap（平らにした添字の gather / scatter-add と select の判定）を
jit して IREE (local) で実行した結果は、jit しない vmap と一致する。添字だけをバッチした rank 1 の operand も通す。"
  (skip-unless-iree :library :both)
  (let ((nb:*compile-cache-directory* nil)
        (ids (%gather-test-indices *gather-test-ids* :i32)))
    (let ((gather (nb:vmap (primitive-function :gather '() 2) :in-axes '(0 0)))
          (scatter (nb:vmap (primitive-function :scatter-add '(:size 5) 2) :in-axes '(0 0)))
          (scalar (nb:vmap (primitive-function :gather '() 2) :in-axes '(nil 0)))
          (table (make-random-array (make-array-spec '(2 5 3) :f32) :seed 4))
          (updates (make-random-array (make-array-spec '(2 4 3) :f32) :seed 5))
          (vector (make-random-array (make-array-spec '(5) :f32) :seed 6))
          (flat-ids (make-array 4 :element-type '(signed-byte 32) :initial-contents '(4 -1 0 5))))
      (dolist (case (list (list gather (list table ids) "gather")
                          (list scatter (list ids updates) "scatter-add")
                          (list scalar (list vector flat-ids) "スカラーの添字")))
        (destructuring-bind (f args label) case
          (let ((jf (nb:jit f :backend (nabla:find-backend :iree))))
            (unwind-protect
                 (is (allclose (apply jf args) (apply f args) :dtype :f32) "~A の jit した vmap が eager と一致しない" label)
              (nb::%jit-cache-forget (nb::%jitted-function-fn jf)))))))
    (gc-and-run-finalizers)))
