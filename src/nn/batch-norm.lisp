;;;; batch-norm と「状態を更新して返す call」の規約（issue #260）。
;;;;
;;;; 決めたこと:
;;;;
;;;; - 規約: 状態（:state のスロット）を更新するモジュールの call は (values 出力 新しいモジュール) を返す。
;;;;   新しいモジュールは :param と :static をそのまま共有し、:state だけを差し替えたもの。jit / vmap の戻り値の
;;;;   多値は「値のリスト」の木として flatten されるので（%flatten-function）そのまま通り、:state の aval は
;;;;   変わらないので treedef とチェックポイントのパス名も更新の前後で同じ。学習ステップは
;;;;   (value-and-grad loss :has-aux t) で新しいモジュールを aux として取り出し（aux は微分しない）、
;;;;   (apply-updates new-model updates) で :param だけを更新する（apply-updates は :state を素通しする）。
;;;;   Flax の apply(..., mutable=["batch_stats"]) が返す更新後の変数に当たる。別案の「状態の更新だけを別の
;;;;   総称関数に出す」は順伝播を2回トレースすることになり、「mutable collection に寄せた仕組み」は defmodule の
;;;;   外に新しい仕組みが要るので採らなかった。
;;;; - 学習と推論の切り替え use-running-average は、dropout の deterministic と同じく call の Lisp の真偽値の
;;;;   引数（jit では静的引数）。推論（真）でも常に2値を返し、新しいモジュールは元のモジュールそのもの（EQ）。
;;;;   呼び出し側はモードで分岐しなくてよい。2値を返すので sequential には入れられない（1値の層だけ）。
;;;; - batch-norm は Flax の nn.BatchNorm: スロットは scale / bias（:param）、mean / var（:state。Flax の
;;;;   batch_stats の mean / var）。学習モードは入力の統計で正規化し、移動平均を
;;;;   ra' = momentum * ra + (1 - momentum) * 統計 で更新する（Flax と同じ向き。momentum の既定 0.99 は
;;;;   古い値の重み）。分散は Flax と同じ「速い分散」 max(E[x^2] - E[x]^2, 0)（バイアスあり、n で割る。
;;;;   移動平均にもこの偏った分散を入れる。Flax と同じで、PyTorch の不偏分散とは違う）。推論モードは mean / var で
;;;;   正規化する。y = (x - mean) * rsqrt(var + epsilon) * scale + bias（layer-norm と同じ順の式）。
;;;; - axis は特徴の軸（整数かリスト、負の値は後ろから。既定 -1）で、統計は残りの軸すべて（バッチと空間）で取る。
;;;;   scale / bias / mean / var の形は特徴の軸の大きさ（軸の昇順）。統計を取る軸が残らない入力は error。
;;;; - 統計の勾配は x に流れる（Flax と同じ。統計を stop-gradient しない）。移動平均は aux なので学習ステップの
;;;;   grad からは外れる。推論モードでは mean / var が順伝播に使われるので、勾配から外すには
;;;;   (map-state #'nabla:stop-gradient model) を grad に渡す。
;;;; - mean / var の dtype はモデルの dtype（Flax は force_float32_reductions で常に f32 だが、nabla は f32 / f64 だけを
;;;;   扱うので、f64 のモデルは f64 の統計を持つ）。

(in-package #:nabla.nn)

;; defmodule を先に置く（後の defun がアクセサを使うので、構造体の定義を先に見せてインライン化させる）。
;; :init / :call の本体は後の defun を呼ぶだけ。
(defmodule batch-norm (scale bias (mean :state) (var :state) (momentum :static) (epsilon :static) (axis :static)
                             (use-scale :static) (use-bias :static))
  (:init (key features &key (dtype :f32) (momentum 0.99d0) (epsilon (if (eq dtype :f64) 1d-5 1f-5)) (axis -1)
                         (use-scale t) (use-bias t))
   (declare (ignore key))
   (%init-batch-norm features dtype momentum epsilon axis use-scale use-bias))
  (:call (self x use-running-average)
   (%batch-norm-apply self x use-running-average)))

(defun %batch-norm-axes (axis rank)
  "AXIS（整数かリスト）から (values 特徴の軸 統計を取る軸) を、それぞれ 0 以上の昇順のリストで返す。"
  (let* ((axes (%normalize-axes "batch-norm" "axis" axis))
         (feature-axes (%canonical-axes "batch-norm" "axis" axes rank))
         (reduction-axes (loop for a below rank unless (member a feature-axes) collect a)))
    ;; (1 -1) のように同じ軸を別の書き方で2回書くと、init の features の個数と食い違う
    (unless (= (length feature-axes) (length axes))
      (error "batch-norm: axis ~S に同じ軸が重複している（入力の rank ~D で ~S）" axis rank feature-axes))
    (unless reduction-axes
      (error "batch-norm: 統計を取る軸が残らない（入力の rank ~D、特徴の軸 ~S）" rank feature-axes))
    (values feature-axes reduction-axes)))

(defun %batch-norm-statistics (x reduction-axes count)
  "X を REDUCTION-AXES で縮約した平均と速い分散 max(E[x^2] - E[x]^2, 0) のリスト（形は特徴の軸の大きさ）。"
  (funcall (load-time-value
            (nabla:with-tracing (x reduction-axes count)
              (let* ((mean (/ (nabla:reduce-sum x :axes reduction-axes) count))
                     (mean-of-squares (/ (nabla:reduce-sum (* x x) :axes reduction-axes) count)))
                (list mean (max (- mean-of-squares (* mean mean)) 0)))))
           x reduction-axes count))

(defun %batch-norm-multiplier (var scale epsilon)
  "rsqrt(VAR + EPSILON) に SCALE（NIL ならかけない）をかけた倍率（形は特徴の軸の大きさ）。"
  (if scale
      (funcall (load-time-value (nabla:with-tracing (var scale epsilon) (* (nabla:rsqrt (+ var epsilon)) scale)))
               var scale epsilon)
      (funcall (load-time-value (nabla:with-tracing (var epsilon) (nabla:rsqrt (+ var epsilon)))) var epsilon)))

(defun %batch-norm-normalize (x mean multiplier bias feature-axes)
  "(X - MEAN) * MULTIPLIER + BIAS（BIAS が NIL なら足さない）。MEAN などは特徴の軸の大きさで、X の形に広げる。"
  (let ((centered (funcall (load-time-value
                            (nabla:with-tracing (x mean multiplier feature-axes)
                              (let ((shape (%shape x)))
                                (* (- x (%broadcast-features mean shape feature-axes))
                                   (%broadcast-features multiplier shape feature-axes)))))
                           x mean multiplier feature-axes)))
    (if bias
        (funcall (load-time-value
                  (nabla:with-tracing (centered bias feature-axes)
                    (+ centered (%broadcast-features bias (%shape centered) feature-axes))))
                 centered bias feature-axes)
        centered)))

(defun %running-average (old new momentum)
  "MOMENTUM * OLD + (1 - MOMENTUM) * NEW（Flax の移動平均の式）。"
  (funcall (load-time-value
            (nabla:with-tracing (old new momentum) (+ (* momentum old) (* (- 1 momentum) new))))
           old new momentum))

(defun %batch-norm-check (module x use-running-average)
  "引数と、スロットの形が入力の特徴の軸の大きさと合うことを確かめ、(values 特徴の軸 統計を取る軸) を返す。"
  (unless (member use-running-average '(t nil))
    (error "batch-norm: use-running-average は T か NIL でなければならない（jit では静的引数にする）: ~S"
           use-running-average))
  (%check-optional-slot "batch-norm" "use-scale" (batch-norm-use-scale module) "scale" (batch-norm-scale module))
  (%check-optional-slot "batch-norm" "use-bias" (batch-norm-use-bias module) "bias" (batch-norm-bias module))
  (let ((shape (%shape x))
        (dtype (nabla:aval-dtype (nabla:value-aval x))))
    (unless (eq dtype (nabla:aval-dtype (nabla:value-aval (batch-norm-mean module))))
      (error "batch-norm: 入力の dtype ~S がモデルの dtype ~S と違う" dtype
             (nabla:aval-dtype (nabla:value-aval (batch-norm-mean module)))))
    (multiple-value-bind (feature-axes reduction-axes) (%batch-norm-axes (batch-norm-axis module) (length shape))
      (let ((sizes (mapcar (lambda (a) (nth a shape)) feature-axes)))
        (dolist (slot (list (cons "scale" (batch-norm-scale module)) (cons "bias" (batch-norm-bias module))
                            (cons "mean" (batch-norm-mean module)) (cons "var" (batch-norm-var module))))
          (when (and (cdr slot) (not (equal sizes (%shape (cdr slot)))))
            (error "batch-norm: ~A の形は入力の特徴の軸 ~S の大きさ ~S でなければならない（~A の形 ~S、入力の形 ~S）"
                   (car slot) feature-axes sizes (car slot) (%shape (cdr slot)) shape))))
      (values feature-axes reduction-axes))))

(defun %batch-norm-apply (module x use-running-average)
  "batch-norm の call の本体。(values 出力 新しいモジュール)。"
  (multiple-value-bind (feature-axes reduction-axes) (%batch-norm-check module x use-running-average)
    (flet ((normalize (mean var)
             (%batch-norm-normalize x mean (%batch-norm-multiplier var (batch-norm-scale module)
                                                                   (batch-norm-epsilon module))
                                    (batch-norm-bias module) feature-axes)))
      (if use-running-average
          (values (normalize (batch-norm-mean module) (batch-norm-var module)) module)
          (destructuring-bind (mean var)
              (%batch-norm-statistics x reduction-axes
                                      (reduce #'* (mapcar (lambda (a) (nth a (%shape x))) reduction-axes)))
            (let ((momentum (batch-norm-momentum module)))
              (values (normalize mean var)
                      ;; :state の mean / var だけを差し替え、ほかのスロットは共有する
                      (%module-rebuild 'batch-norm
                                       (lambda (name kind)
                                         (declare (ignore kind))
                                         (let ((value (slot-value module name)))
                                           (case name
                                             (mean (%running-average value mean momentum))
                                             (var (%running-average value var momentum))
                                             (t value))))))))))))

(defun %init-batch-norm (features dtype momentum epsilon axis use-scale use-bias)
  (unless (and (realp epsilon) (plusp epsilon) (not (and (floatp epsilon) (sb-ext:float-infinity-p epsilon))))
    (error "batch-norm の epsilon は正の有限の実数でなければならない: ~S" epsilon))
  (unless (and (realp momentum) (<= 0 momentum 1))
    (error "batch-norm の momentum は 0 以上 1 以下の実数でなければならない: ~S" momentum))
  (%check-use-flag "batch-norm" "use-scale" use-scale)
  (%check-use-flag "batch-norm" "use-bias" use-bias)
  (let* ((axes (%normalize-axes "batch-norm" "axis" axis))
         (sizes (if (listp features) features (list features))))
    (unless (and (eql (%list-length sizes) (length axes))
                 (every (lambda (size) (and (integerp size) (plusp size))) sizes))
      (error "batch-norm: features は axis (~S) の個数 (~D) だけの正の整数（1 つなら整数でもよい）でなければならない: ~S"
             axis (length axes) features))
    (make-batch-norm :scale (and use-scale (%filled sizes dtype 1))
                     :bias (and use-bias (%filled sizes dtype 0))
                     :mean (%filled sizes dtype 0) :var (%filled sizes dtype 1)
                     :momentum momentum :epsilon epsilon :axis axis :use-scale use-scale :use-bias use-bias)))
