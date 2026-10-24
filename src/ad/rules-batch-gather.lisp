;;;; ad/rules-batch-gather: gather と scatter-add のバッチ化ルール（issue #240）。
;;;;
;;;; 片側だけがバッチされるときは、バッチ軸を gather / scatter-add がそのまま運べる位置へ動かす
;;;; （operand / updates のバッチ軸は「rest」の先頭、添字のバッチ軸は添字の軸の1つ）。両方がバッチされる
;;;; ときは、operand を [B*N, rest...] に平らにし、添字に b*N を足して1回の gather / scatter-add にする。
;;;; 平らにした添字は常に [0, B*N) に入る（範囲外の添字は 0 にしてから b*N を足す）ので、プリミティブ自身の
;;;; 範囲外の扱いは働かない。範囲の判定は b*N を足す前に元の dtype で行い、gather は結果を、scatter-add は
;;;; 更新を select で 0 にする（負の添字が隣のバッチの行に落ちない）。平らにした添字の最大 B*N - 1 は :i32 に
;;;; 収まらなければならない（B*N <= 2^31）。

(in-package #:nabla)

(defun %gather-in-range (indices n)
  "INDICES（整数のトレーサ）の各要素が 0 以上 N 未満かの :i1 のトレーサ。長さ N の 1 の配列を gather で
引いて 0 でないかを見る（範囲の判定を gather の emit の1か所にまとめる。unsigned の比較もそこに任せる）。"
  (let* ((shape (aval-shape (tracer-aval indices)))
         (hits (%trace-eqn :gather (list (%lift-number-to 1 :i32 (list n)) indices))))
    (%trace-eqn :compare (list hits (%lift-number-to 0 :i32 shape)) :direction :ne)))

(defun %gather-batch-offsets (size n)
  "0, N, 2N, ..., (SIZE-1)N の :i32 の1次元配列。"
  (let ((array (make-array size :element-type '(signed-byte 32))))
    (dotimes (b size array) (setf (aref array b) (* b n)))))

(defun %gather-flat-indices (indices index-axis n)
  "INDICES のバッチ軸 INDEX-AXIS を先頭へ動かし、(values 平らにした添字 範囲内の判定) を返す。平らにした
添字は :i32 で、範囲内なら b*N + 添字、範囲外なら b*N（その行は呼ぶ側が判定で 0 にする）。"
  (let* ((moved (%vmap-move-axis indices index-axis 0))
         (aval (tracer-aval moved))
         (dtype (aval-dtype aval))
         (shape (aval-shape aval))
         (size (first shape)))
    (unless (<= (* size n) (expt 2 31))
      (error 'vmap-error :format-control "gather / scatter-add のバッチ化: バッチの長さ × N（~D × ~D）が :i32 に収まらない"
                         :format-arguments (list size n)))
    (let* ((ok (%gather-in-range moved n))
           (safe (%trace-eqn :select (list ok moved (%lift-number-to 0 dtype shape))))
           (safe32 (if (eq dtype :i32) safe (%trace-eqn :convert (list safe) :dtype :i32)))
           (offsets (%trace-eqn :broadcast-in-dim (list (%lift-array-to (%gather-batch-offsets size n) :i32))
                                :shape shape :dims '(0))))
      (values (%trace-eqn :add (list safe32 offsets)) ok))))

(defun %gather-mask (ok value)
  "判定 OK（形は添字の形 [B, idx...]）を VALUE の形へ広げ、偽の位置を 0 にした VALUE。"
  (let* ((aval (tracer-aval value))
         (shape (aval-shape aval))
         (wide (%trace-eqn :broadcast-in-dim (list ok) :shape shape
                                                       :dims (loop for d below (aval-rank (tracer-aval ok)) collect d))))
    (%trace-eqn :select (list wide value (%lift-number-to 0 (aval-dtype aval) shape)))))

(defun %gather-flatten-leading (x)
  "X [B, N, rest...] を [B*N, rest...] にする。"
  (let ((shape (aval-shape (tracer-aval x))))
    (%trace-eqn :reshape (list x) :shape (cons (* (first shape) (second shape)) (nthcdr 2 shape)))))

(def-batch-rule gather (args batch-dims)
  (destructuring-bind (operand indices) args
    (destructuring-bind (operand-axis index-axis) batch-dims
      (cond
        ((and operand-axis index-axis)
         (let* ((moved (%vmap-move-axis operand operand-axis 0))
                (n (second (aval-shape (tracer-aval moved))))
                (flat-operand (%gather-flatten-leading moved)))
           (multiple-value-bind (flat ok) (%gather-flat-indices indices index-axis n)
             (values (list (%gather-mask ok (%trace-eqn :gather (list flat-operand flat))))
                     (list 0)))))
        (operand-axis
         ;; operand [N, B, rest...] の B は rest の先頭なので、出力では添字の軸の直後に来る。
         (values (list (%trace-eqn :gather (list (%vmap-move-axis operand operand-axis 1) indices)))
                 (list (aval-rank (tracer-aval indices)))))
        (t
         ;; 添字のバッチ軸は、出力の先頭の添字の軸の中で同じ位置に残る。
         (values (list (%trace-eqn :gather (list operand indices)))
                 (list index-axis)))))))

(def-batch-rule scatter-add (args batch-dims &key size)
  (destructuring-bind (indices updates) args
    (destructuring-bind (index-axis updates-axis) batch-dims
      (if (null index-axis)
          ;; updates [idx..., B, rest...] の B は rest の先頭なので、出力 [N, B, rest...] の軸 1 になる。
          (let ((rank (aval-rank (tracer-aval indices))))
            (values (list (%trace-eqn :scatter-add (list indices (%vmap-move-axis updates updates-axis rank))
                                      :size size))
                    (list 1)))
          (let* ((batch (nth index-axis (aval-shape (tracer-aval indices))))
                 (moved (if updates-axis
                            (%vmap-move-axis updates updates-axis 0)
                            (%vmap-broadcast-batch updates 0 batch))))
            (multiple-value-bind (flat ok) (%gather-flat-indices indices index-axis size)
              (let* ((rest (nthcdr (aval-rank (tracer-aval ok)) (aval-shape (tracer-aval moved))))
                     (sum (%trace-eqn :scatter-add (list flat (%gather-mask ok moved)) :size (* batch size))))
                (values (list (%trace-eqn :reshape (list sum) :shape (list* batch size rest)))
                        (list 0)))))))))
