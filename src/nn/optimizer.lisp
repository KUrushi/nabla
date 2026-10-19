;;;; オプティマイザ（Optax 相当、issue #205）。
;;;;
;;;; 決めたこと:
;;;;
;;;; - オプティマイザは init（パラメータの木 → 状態の木）と update（更新量の木・状態・パラメータ
;;;;   → 更新量の木と新しい状態）の組（OPTIMIZER 構造体）。Optax の GradientTransformation と同じで、
;;;;   sgd / adam / adamw はもちろん clip-by-global-norm も scale も同じ型なので CHAIN で合成できる。
;;;;   update は純粋関数（入力を書き換えない）で、ホストの状態を持たない。ステップ数 count は
;;;;   状態の中の rank 0 の :i32 の配列（Lisp の整数ではない）なので、jit の入力として渡り、
;;;;   ステップごとに再コンパイルされない。
;;;; - 状態は DEFMODULE の構造体（adam-state など）で、スロット名がチェックポイントのパス名
;;;;   （opt.mu.dense.kernel のようなもの）になる。状態を持たない変換の状態は NIL（空の木）、
;;;;   CHAIN の状態は各変換の状態のリスト。mu / nu / trace は params と同じ構造の木。
;;;; - 内蔵のオプティマイザは DEFUN で書き、算術が要る小さな式だけを %TRACED（with-tracing の関数を
;;;;   その場で作って呼ぶ）で書く。with-tracing の本体では setq が使えないので、dtype ごとの
;;;;   スカラーの表（%PER-DTYPE）のような Lisp の処理は外に出す。利用者向けの DEFOPTIMIZER は
;;;;   init / update の本体を with-tracing の本体として書く（DEFMODULE の :call と同じ）。
;;;; - スカラー（学習率・バイアス補正）は更新する葉の dtype に合わせて convert する（f32 と f64 の
;;;;   混在でも dtype の不一致にならない）。bf16 / f16 の配列は eager の配列だけでは dtype を
;;;;   区別できないので、このファイルは f32 / f64 だけを対象にする。
;;;; - cos と pow のプリミティブが無いので、cosine decay は区間 [-π/2, π/2] の sin の Taylor 展開
;;;;   （15 次、打ち切り誤差 (π/2)^17/17! ≈ 6e-12。f32 の丸めの 1e-7 程度が残る）、バイアス補正の 1 - b^t は x = -t log b の -expm1(-x) の
;;;;   Taylor 展開（x < 1/4。b2 = 0.999、t = 1 でも相対誤差が f32 の丸め程度に収まる）で書く。

(in-package #:nabla.nn)

;;; --- 小さな部品 ---

(defmacro %traced ((&rest variables) &body body)
  "BODY を（VARIABLES を引数とする with-tracing の関数として）その場で作り、VARIABLES の値で呼ぶ。
VARIABLES は配列でもトレーサでもよい。"
  `(funcall (nabla:with-tracing ,variables ,@body) ,@variables))

(defun %hyper (who name value valid-p)
  "ハイパーパラメータ VALUE が有限の実数で (VALID-P VALUE) なら double-float にして返す。違えばエラー。"
  (unless (and (realp value)
               (not (and (floatp value) (or (sb-ext:float-nan-p value) (sb-ext:float-infinity-p value))))
               (funcall valid-p value))
    (error "~A: ~A が不正（有限の実数が要る）: ~S" who name value))
  (float value 1d0))

(defun %positive-integer (who name value)
  (unless (typep value '(integer 1))
    (error "~A: ~A は正の整数: ~S" who name value))
  value)

(defun %dtype-of (x)
  "X（配列・トレーサ・Lisp の実数）の dtype。配列とトレーサは NABLA:VALUE-AVAL に任せる。"
  (typecase x
    (double-float :f64)
    (real :f32)
    (t (nabla:aval-dtype (nabla:value-aval x)))))

(defun %to-dtype (x dtype)
  (if (eq (%dtype-of x) dtype) x (nabla:convert x dtype)))

(defun %per-dtype (function leaves)
  "LEAVES に現れる dtype ごとに (FUNCTION dtype) を1回ずつ呼んだ ((dtype . 値) ...) を返す。"
  (let ((table '()))
    (dolist (leaf leaves table)
      (let ((dtype (%dtype-of leaf)))
        (unless (assoc dtype table)
          (push (cons dtype (funcall function dtype)) table))))))

(defun %for-leaf (table leaf)
  (cdr (assoc (%dtype-of leaf) table)))

(defun %zero-count ()
  (make-array nil :element-type '(signed-byte 32) :initial-element 0))

(defun %increment (count)
  "COUNT（rank 0 の :i32）+ 1。Optax の safe_increment と同じく :i32 の最大値で止まる。"
  (%traced (count) (+ (min count 2147483646) 1)))

(defun %zeros-like (tree)
  "TREE と同じ構造・形・dtype のゼロの木。(< (* p p) 0) は p が NaN / 無限大でも偽なので、常にゼロ
\(p * 0 は p が無限大のとき NaN になる)。"
  (nabla:tree-map (lambda (p) (%traced (p) (nabla:where (< (* p p) 0) p 0))) tree))

;;; --- オプティマイザ型と DEFOPTIMIZER ---

(defstruct (optimizer (:constructor %make-optimizer (init-function update-function))
                      (:copier nil))
  "init（params → 状態）と update（updates 状態 params → 更新量と新しい状態）の組。
OPTIMIZER-INIT / OPTIMIZER-UPDATE で呼ぶ。"
  (init-function nil :read-only t)
  (update-function nil :read-only t))

(defun optimizer-init (optimizer params)
  "OPTIMIZER の状態を PARAMS（パラメータの木。モデルでもよい）から作る。状態は PyTree で、
mu / nu / trace のようにパラメータと同じ構造の木と、ステップ数 count（rank 0 の :i32 の配列）を含む。"
  (funcall (optimizer-init-function optimizer) params))

(defun optimizer-update (optimizer updates state &optional params)
  "OPTIMIZER で UPDATES（勾配の木。chain の途中では前の変換の出力）から (values 更新量の木 新しい状態) を
作る。PARAMS は adamw の weight decay のようにパラメータの値を使う変換が読む（使わなければ省略できる）。
純粋関数で、引数を書き換えない。UPDATES・PARAMS・状態は jit の引数にできる。木の構造が違えば
pytree-structure-mismatch。更新量は (apply-updates params updates) でパラメータに足す。"
  (funcall (optimizer-update-function optimizer) updates state params))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %ignorable-body (clause)
    "(:init (vars...) body...) の本体を、使わない引数で警告が出ないよう再束縛で包んだフォームのリスト。"
    (destructuring-bind (head variables &rest forms) clause
      (declare (ignore head))
      (list `(let ,(mapcar (lambda (v) (list v v)) variables)
               (declare (ignorable ,@variables))
               ,@forms)))))

(defmacro defoptimizer (name lambda-list &body body)
  "オプティマイザを返す関数 NAME を定義する。LAMBDA-LIST は NAME のハイパーパラメータの引数で、
BODY は省略可の docstring と次の2つの節:
  (:init (params) body...)                  → 状態（PyTree。状態を持たなければ NIL）を返す
  (:update (updates state params) body...)  → (values 新しい updates 新しい state) を返す
どちらの本体も nabla:with-tracing の本体として書く（defmodule の :call と同じ。CL の + * や nabla の関数が
配列にもトレーサにも使え、setq は使えない）ので、jit / grad の中で呼べる。ハイパーパラメータは
閉包で捕まえた Lisp の値になる（jit のたびに定数としてコンパイルされる）。ステップ数のように
ステップごとに変わる値は Lisp の整数でなく、状態の中の配列に持つ。

  (defoptimizer scale-by (factor)
    (:init (params) nil)
    (:update (updates state params)
      (values (nabla:tree-map (lambda (g) (* g factor)) updates) state)))
  (chain (scale-by 0.5) (adam 1e-3))

節は :init と :update が1つずつ。それ以外の形は展開時にエラー。"
  (let ((doc (when (stringp (first body)) (pop body)))
        (init (find :init (remove-if-not #'consp body) :key #'first))
        (update (find :update (remove-if-not #'consp body) :key #'first)))
    (unless (and (= (length body) 2) (every #'consp body) init update
                 (destructuring-bind (head variables &rest forms) init
                   (declare (ignore head forms))
                   (and (listp variables) (= (length variables) 1) (symbolp (first variables))))
                 (destructuring-bind (head variables &rest forms) update
                   (declare (ignore head forms))
                   (and (listp variables) (= (length variables) 3) (every #'symbolp variables))))
      (error "defoptimizer ~S: 節は (:init (params) body...) と (:update (updates state params) body...) の2つ: ~S"
             name body))
    `(defun ,name ,lambda-list
       ,@(when doc (list doc))
       (%make-optimizer (nabla:with-tracing ,(second init) ,@(%ignorable-body init))
                        (nabla:with-tracing ,(second update) ,@(%ignorable-body update))))))

(defun apply-updates (params updates)
  "PARAMS（パラメータの木。モデルでもよい）の各葉に UPDATES の同じ位置の葉を足した、同じ構造の木を返す
\(optimizer-update が返す更新量は、学習率の符号を含むので足すだけでよい)。"
  (nabla:tree-map (lambda (p u) (%traced (p u) (+ p u))) params updates))

;;; --- 状態を持たない変換 ---

(defun scale (factor)
  "更新量を FACTOR 倍する（状態なし）。sgd の学習率は (scale (- lr))。"
  (let ((factor (%hyper 'scale "factor" factor (constantly t))))
    (%make-optimizer (lambda (params) (declare (ignore params)) nil)
                     (lambda (updates state params)
                       (declare (ignore params))
                       (values (nabla:tree-map (lambda (g) (%traced (g) (* g factor))) updates)
                               state)))))

(defun add-decayed-weights (weight-decay)
  "更新量に WEIGHT-DECAY * パラメータ を足す（状態なし。パラメータが要るので optimizer-update に
params を渡す）。adamw は adam の後にこれを置く（decoupled weight decay）。"
  (let ((weight-decay (%hyper 'add-decayed-weights "weight-decay" weight-decay (constantly t))))
    (%make-optimizer (lambda (params) (declare (ignore params)) nil)
                     (lambda (updates state params)
                       (when (and (null params) (nabla:tree-leaves updates))
                         (error "add-decayed-weights: weight decay にはパラメータの値が要る。optimizer-update の引数 params を渡すこと"))
                       (values (nabla:tree-map (lambda (g p) (%traced (g p) (+ g (* weight-decay p))))
                                               updates params)
                               state)))))

(defun %global-norm (leaves dtype)
  "LEAVES の全要素の二乗和の平方根（DTYPE のスカラー）。"
  (let ((total (reduce (lambda (a b) (%traced (a b) (+ a b)))
                       (mapcar (lambda (g) (%to-dtype (%traced (g) (nabla:reduce-sum (* g g))) dtype))
                               leaves))))
    ;; sqrt の微分は 0 で無限大になる。二重 where（sqrt の引数を 0 の外へ逃がす）で、
    ;; 全勾配が 0 のときも、この更新を微分した結果が NaN にならないようにする
    (%traced (total)
      (nabla:where (> total 0) (sqrt (nabla:where (> total 0) total 1)) 0))))

(defun clip-by-global-norm (max-norm)
  "更新量の全葉をまとめた大域ノルム（二乗和の平方根）が MAX-NORM を超えるとき、向きを変えずに
ノルムを MAX-NORM に縮める（状態なし）。超えなければそのまま。ノルムは最初の葉の dtype で数える。
縮める側の割り算の分母は (max ノルム MAX-NORM) にし、ノルムの sqrt も二重 where（%GLOBAL-NORM）にして、選ばれない側でもゼロ割り・sqrt(0) の微分が出ない
\(勾配がすべて 0 でも、この更新を微分したときに NaN が出ない)。"
  (let ((max-norm (%hyper 'clip-by-global-norm "max-norm" max-norm #'plusp)))
    (%make-optimizer
     (lambda (params) (declare (ignore params)) nil)
     (lambda (updates state params)
       (declare (ignore params))
       (let ((leaves (nabla:tree-leaves updates)))
         (if (null leaves)
             (values updates state)
             (let* ((norm (%global-norm leaves (%dtype-of (first leaves))))
                    (table (%per-dtype (lambda (dtype) (%to-dtype norm dtype)) leaves)))
               (values (nabla:tree-map
                        (lambda (g)
                          (let ((n (%for-leaf table g)))
                            (%traced (g n)
                              (nabla:where (< n max-norm) g (* (/ g (max n max-norm)) max-norm)))))
                        updates)
                       state))))))))

;;; --- momentum / adam / スケジュール ---

;; スロットはすべて :state（学習するパラメータではない）。map-state で一括して扱え、チェックポイントの
;; パス名はスロット名（opt.mu.0 など）で :param と変わらない
(defmodule momentum-state ((trace :state)))
(defmodule adam-state ((count :state) (mu :state) (nu :state)))
(defmodule schedule-state ((count :state)))

(defun momentum-trace (decay &key nesterov)
  "heavy-ball の運動量（Optax の trace）: trace' = g + DECAY * trace、更新量は trace'
\(NESTEROV なら g + DECAY * trace')。状態は trace（params と同じ構造の木）。"
  (let ((decay (%hyper 'momentum-trace "decay" decay (lambda (x) (and (<= 0 x) (< x 1))))))
    (%make-optimizer
     (lambda (params) (make-momentum-state :trace (%zeros-like params)))
     (lambda (updates state params)
       (declare (ignore params))
       (let ((trace (nabla:tree-map (lambda (g tr) (%traced (g tr) (+ g (* decay tr))))
                                    updates (momentum-state-trace state))))
         (values (if nesterov
                     (nabla:tree-map (lambda (g tr) (%traced (g tr) (+ g (* decay tr)))) updates trace)
                     trace)
                 (make-momentum-state :trace trace)))))))

(defun %one-minus-pow (base count dtype)
  "1 - BASE^COUNT（COUNT は rank 0 の :i32、結果は DTYPE のスカラー）。pow のプリミティブが無いので
x = -COUNT log BASE として 1 - exp(-x) を計算する。ただし BASE が 1 に近いと x が小さく、1 - exp(-x) は
桁落ちするので、x < 1/4 では -expm1(-x) の Taylor 展開（13 次）を使う。BASE = 0 は常に 1。"
  (if (zerop base)
      1
      (let ((rate (- (log base)))
            (orders (loop for k from 13 downto 2 collect k)))
        (%traced (count)
          (let ((x (* (nabla:convert count dtype) rate)))
            (nabla:where (< x 0.25)
                         (* x (reduce (lambda (s k) (- 1 (/ (* x s) k))) orders :initial-value 1))
                         (- 1 (exp (- x)))))))))

(defun scale-by-adam (&key (b1 0.9d0) (b2 0.999d0) (eps 1d-8) (eps-root 0d0))
  "Adam の更新量 mu_hat / (sqrt(nu_hat + EPS-ROOT) + EPS)（学習率は掛けない。Optax の scale_by_adam）。
mu = B1 mu + (1 - B1) g、nu = B2 nu + (1 - B2) g^2 で、mu_hat / nu_hat は 1 - b^count による
バイアス補正。状態は adam-state（count: rank 0 の :i32、mu・nu: params と同じ構造の木）。"
  (let ((b1 (%hyper 'scale-by-adam "b1" b1 (lambda (x) (and (<= 0 x) (< x 1)))))
        (b2 (%hyper 'scale-by-adam "b2" b2 (lambda (x) (and (<= 0 x) (< x 1)))))
        (eps (%hyper 'scale-by-adam "eps" eps (lambda (x) (>= x 0))))
        (eps-root (%hyper 'scale-by-adam "eps-root" eps-root (lambda (x) (>= x 0)))))
    (%make-optimizer
     (lambda (params)
       (make-adam-state :count (%zero-count) :mu (%zeros-like params) :nu (%zeros-like params)))
     (lambda (updates state params)
       (declare (ignore params))
       (let* ((count (%increment (adam-state-count state)))
              (mu (nabla:tree-map (lambda (g m) (%traced (g m) (+ (* (- 1 b1) g) (* b1 m))))
                                  updates (adam-state-mu state)))
              (nu (nabla:tree-map (lambda (g n) (%traced (g n) (+ (* (- 1 b2) (* g g)) (* b2 n))))
                                  updates (adam-state-nu state)))
              (leaves (nabla:tree-leaves updates))
              (c1 (%per-dtype (lambda (dtype) (%one-minus-pow b1 count dtype)) leaves))
              (c2 (%per-dtype (lambda (dtype) (%one-minus-pow b2 count dtype)) leaves)))
         (values (nabla:tree-map
                  (lambda (m n)
                    (let ((c1 (%for-leaf c1 m)) (c2 (%for-leaf c2 m)))
                      (%traced (m n c1 c2)
                        (/ (/ m c1) (+ (sqrt (+ (/ n c2) eps-root)) eps)))))
                  mu nu)
                 (make-adam-state :count count :mu mu :nu nu)))))))

(defun scale-by-schedule (schedule)
  "更新量に (SCHEDULE count) を掛ける。SCHEDULE は count（rank 0 の :i32）を受け取って rank 0 の
浮動小数点のスカラーを返す関数（このファイルの *-schedule、または with-tracing で作った関数）。
状態は schedule-state（count）で、update ごとに 1 増える。"
  (unless (functionp schedule)
    (error "scale-by-schedule: スケジュールは関数: ~S" schedule))
  (%make-optimizer
   (lambda (params) (declare (ignore params)) (make-schedule-state :count (%zero-count)))
   (lambda (updates state params)
     (declare (ignore params))
     (let* ((count (schedule-state-count state))
            (step (funcall schedule count))
            (table (%per-dtype (lambda (dtype) (%to-dtype step dtype)) (nabla:tree-leaves updates))))
       (values (nabla:tree-map (lambda (g)
                                 (let ((s (%for-leaf table g)))
                                   (%traced (g s) (* g s))))
                               updates)
               (make-schedule-state :count (%increment count)))))))

(defun %scale-by-learning-rate (learning-rate)
  "-LEARNING-RATE 倍（LEARNING-RATE がスケジュールなら -(schedule count) 倍）。"
  (if (functionp learning-rate)
      (scale-by-schedule (lambda (count) (%traced (count) (- (funcall learning-rate count)))))
      (scale (- (%hyper 'optimizer "learning-rate" learning-rate (constantly t))))))

(defun %check-learning-rate (who learning-rate)
  (unless (or (realp learning-rate) (functionp learning-rate))
    (error "~A: 学習率は実数かスケジュール（関数）: ~S" who learning-rate)))

;;; --- chain と sgd / adam / adamw ---

(defun chain (&rest optimizers)
  "OPTIMIZERS を左から順に適用する1つのオプティマイザにする（前の update の出力が次の入力になる）。
状態は各オプティマイザの状態のリスト。引数なしは恒等（更新量をそのまま返し、状態は NIL）。
結合的: (chain a (chain b c)) と (chain a b c) は同じ更新量を返す（状態の葉の並びも同じ）。"
  (dolist (optimizer optimizers)
    (unless (optimizer-p optimizer)
      (error "chain: オプティマイザでないものが渡された: ~S" optimizer)))
  (%make-optimizer
   (lambda (params)
     (mapcar (lambda (optimizer) (optimizer-init optimizer params)) optimizers))
   (lambda (updates states params)
     (unless (and (listp states) (= (length states) (length optimizers)))
       (error "chain: 状態の個数がオプティマイザの個数と違う: ~S" states))
     (let ((new-states '()))
       (loop for optimizer in optimizers
             for state in states
             do (multiple-value-bind (new-updates new-state)
                    (optimizer-update optimizer updates state params)
                  (setf updates new-updates)
                  (push new-state new-states)))
       (values updates (nreverse new-states))))))

(defun sgd (learning-rate &key momentum nesterov)
  "確率的勾配降下法。LEARNING-RATE は実数かスケジュール（count → スカラーの関数）で、更新量は
-lr * g。MOMENTUM（[0, 1)）があれば先に momentum-trace を通す。NESTEROV は MOMENTUM が必要。"
  (%check-learning-rate 'sgd learning-rate)
  (when (and nesterov (null momentum))
    (error "sgd: nesterov には momentum が要る"))
  (apply #'chain
         (append (when momentum (list (momentum-trace momentum :nesterov nesterov)))
                 (list (%scale-by-learning-rate learning-rate)))))

(defun adam (learning-rate &key (b1 0.9d0) (b2 0.999d0) (eps 1d-8) (eps-root 0d0))
  "Adam（Optax の adam と同じ既定値）。LEARNING-RATE は実数かスケジュール。
\(chain (scale-by-adam ...) (scale (- lr)))。"
  (%check-learning-rate 'adam learning-rate)
  (chain (scale-by-adam :b1 b1 :b2 b2 :eps eps :eps-root eps-root)
         (%scale-by-learning-rate learning-rate)))

(defun adamw (learning-rate &key (b1 0.9d0) (b2 0.999d0) (eps 1d-8) (eps-root 0d0) (weight-decay 1d-4))
  "AdamW: adam の更新量に WEIGHT-DECAY * パラメータ を足してから学習率を掛ける（decoupled weight decay。
Optax の adamw と同じ既定値。mask は未対応で、すべてのパラメータに減衰をかける）。
update には params を渡すこと。"
  (%check-learning-rate 'adamw learning-rate)
  (chain (scale-by-adam :b1 b1 :b2 b2 :eps eps :eps-root eps-root)
         (add-decayed-weights weight-decay)
         (%scale-by-learning-rate learning-rate)))

;;; --- 学習率スケジュール ---

(defun %linear-value (count init end steps begin)
  "COUNT（rank 0 の :i32）での線形補間: BEGIN まで INIT、BEGIN + STEPS 以降は END。"
  (%traced (count)
    (let ((progress (min (max (- count begin) 0) steps)))
      (+ (* (- init end) (- 1 (/ (nabla:convert progress :f32) steps))) end))))

(defun %cosine-value (count init end steps)
  "COUNT での cosine decay: 0.5 (1 + cos(pi t)) = 0.5 (1 - sin(pi (t - 1/2))) で INIT から END へ、
t = min(COUNT, STEPS) / STEPS。sin は [-pi/2, pi/2] の Taylor 展開 u (1 - u^2/(2 3) (1 - u^2/(4 5) ...))。"
  (let ((orders (loop for n from 7 downto 1 collect n)))
    (%traced (count)
      (let* ((fraction (/ (nabla:convert (min count steps) :f32) steps))
             (u (* pi (- fraction 0.5)))
             (u2 (* u u))
             (sine (* u (reduce (lambda (s n) (- 1 (/ (* u2 s) (* (* 2 n) (+ (* 2 n) 1)))))
                                orders :initial-value 1)))
             (cosine-weight (* 0.5 (+ 1 (- sine)))))
        (+ (* (- init end) cosine-weight) end)))))

(defun constant-schedule (value)
  "どの count でも VALUE を返すスケジュール（count → rank 0 の :f32 のスカラー）。"
  (let ((value (%hyper 'constant-schedule "value" value (constantly t))))
    (nabla:with-tracing (count)
      (+ (* (nabla:convert count :f32) 0.0) value))))

(defun linear-schedule (init-value end-value transition-steps &key (transition-begin 0))
  "TRANSITION-BEGIN まで INIT-VALUE、そこから TRANSITION-STEPS かけて END-VALUE へ直線で動き、
以降は END-VALUE のスケジュール（Optax の linear_schedule。count → rank 0 の :f32 のスカラー）。"
  (let ((init (%hyper 'linear-schedule "init-value" init-value (constantly t)))
        (end (%hyper 'linear-schedule "end-value" end-value (constantly t)))
        (steps (%positive-integer 'linear-schedule "transition-steps" transition-steps))
        (begin (if (typep transition-begin '(integer 0))
                   transition-begin
                   (error "linear-schedule: transition-begin は 0 以上の整数: ~S" transition-begin))))
    (nabla:with-tracing (count) (%linear-value count init end steps begin))))

(defun cosine-decay-schedule (init-value decay-steps &key (alpha 0d0))
  "INIT-VALUE から DECAY-STEPS かけて半周期の cosine で INIT-VALUE * ALPHA まで下がり、以降はその値
\(Optax の cosine_decay_schedule。count → rank 0 の :f32 のスカラー）。cos のプリミティブが無いので
多項式近似（打ち切り誤差は 1e-11 未満で、f32 の丸め 1e-7 程度が残る）。"
  (let* ((init (%hyper 'cosine-decay-schedule "init-value" init-value (constantly t)))
         (alpha (%hyper 'cosine-decay-schedule "alpha" alpha (constantly t)))
         (steps (%positive-integer 'cosine-decay-schedule "decay-steps" decay-steps))
         (end (* init alpha)))
    (nabla:with-tracing (count) (%cosine-value count init end steps))))

(defun warmup-cosine-decay-schedule (init-value peak-value warmup-steps decay-steps &key (end-value 0d0))
  "INIT-VALUE から WARMUP-STEPS かけて直線で PEAK-VALUE へ上がり、そこから DECAY-STEPS（warmup を
含む総ステップ数）まで cosine で END-VALUE へ下がり、以降は END-VALUE（Optax の
warmup_cosine_decay_schedule。count → rank 0 の :f32 のスカラー）。DECAY-STEPS は WARMUP-STEPS より大きい。"
  (let ((init (%hyper 'warmup-cosine-decay-schedule "init-value" init-value (constantly t)))
        (peak (%hyper 'warmup-cosine-decay-schedule "peak-value" peak-value (constantly t)))
        (end (%hyper 'warmup-cosine-decay-schedule "end-value" end-value (constantly t)))
        (warmup (if (typep warmup-steps '(integer 0))
                    warmup-steps
                    (error "warmup-cosine-decay-schedule: warmup-steps は 0 以上の整数: ~S" warmup-steps)))
        (decay (%positive-integer 'warmup-cosine-decay-schedule "decay-steps" decay-steps)))
    (unless (> decay warmup)
      (error "warmup-cosine-decay-schedule: decay-steps ~D は warmup-steps ~D より大きい" decay warmup))
    (if (zerop warmup)
        (nabla:with-tracing (count) (%cosine-value count peak end decay))
        (nabla:with-tracing (count)
          (nabla:where (< count warmup)
                       (%linear-value count init peak warmup 0)
                       (%cosine-value (- count warmup) peak end (- decay warmup)))))))
