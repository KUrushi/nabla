;;;; prng: JAX の jax.random に相当する、明示的なキー渡しの PRNG の公開 API（issue #136）。
;;;;
;;;; キーは :u32 の (2) の配列（計画どおり）。乱数は rng-bit-generator（THREE_FRY。
;;;; src/primitives/rng.lisp）のビットから作る。決めたこと:
;;;;
;;;; - キー → 状態: キー [k0, k1] を ui64[2] の状態 [k0 | k1 << 32, カウンタ] にする
;;;;   （rng-bit-generator は鍵を s0 の (下位32, 上位32) として読むので、キーの2語を
;;;;   bitcast-convert で u64 にまとめれば、そのまま鍵になる）。
;;;; - 乱数を引く（uniform / normal / split）はカウンタ 0 から始め、ビットは常に1次元で作って
;;;;   reshape する（多次元の u32 の配置の癖に依らない。同じ個数なら同じ列）。fold-in は
;;;;   カウンタ 2^32 + data から2語（= 新しいキー）を取る。引く量が 2^32 要素未満なら
;;;;   fold-in の出力が uniform / split の列と重なることは無い。
;;;; - 同じキーで uniform と split の両方を引くと、同じビット列を共有する（JAX と同じく、
;;;;   キーは「使うか、split するか」のどちらか一方にだけ使う）。
;;;; - JAX の既定（threefry_2x32 を直接呼ぶ実装）とはビット単位では一致しない
;;;;   （nabla は rng_bit_generator を使う。分布としては同じ）。
;;;; - uniform: ビット列の仮数部だけを取り出して [1, 2) の浮動小数点数にし、1 を引く
;;;;   （JAX と同じ手法。f32 は上位 23 ビット、f64 は上位 52 ビット）。
;;;; - normal: 一様乱数 u ∈ (-1, 1) の erf の逆関数 × √2（JAX と同じ方式）。Box–Muller は
;;;;   sin / cos のプリミティブが無いため採らなかった。erf の逆関数は Giles の多項式近似で、
;;;;   :f32 は単精度用（JAX の f32 と同じ係数）、:f64 は倍精度用（XLA の f64 と同じ係数）。
;;;;   eager の相対誤差の実測（|x| < 1 の 650 点、Lisp の多倍長で求めた参照値と比較。
;;;;   tests/prng-test.lisp）は f32 で最大 2.0e-7、f64 で最大 4.9e-16。f64 で単精度用を
;;;;   使うと（#166 以前）裾で崩れ、u = 1 - 2^-53 で 12% 小さかった（√2 倍で 7.32、真は 8.29）。
;;;;
;;;; 実装は内部のトレース用の関数（トレーサを受けてトレーサを返す）で書き、公開関数が
;;;; 引数の種類（配列だけなら eager、トレーサがあれば現在のトレースに足す）で振り分ける
;;;; ので、eager・jit・vmap・grad の中のどこでも同じ式が動く。

(in-package #:nabla)

(define-condition prng-error (error)
  ((format-control :initarg :format-control :initform "" :reader prng-error-format-control)
   (format-arguments :initarg :format-arguments :initform nil
                     :reader prng-error-format-arguments))
  (:report
   (lambda (condition stream)
     (format stream "PRNG エラー: ~?"
             (prng-error-format-control condition)
             (prng-error-format-arguments condition))))
  (:documentation
   "PRNG の公開 API（PRNG-KEY / SPLIT / FOLD-IN / UNIFORM / NORMAL）に不正な引数
（キーが :u32 の (2) の配列でない、shape が正の整数のリストでない、dtype が浮動小数点
（:f32 / :f64）でない、MINVAL >= MAXVAL など）を渡したときに signal する。"))

(defun %prng-error (control &rest arguments)
  (error 'prng-error :format-control control :format-arguments arguments))

;;; --- 引数の検査 ---

(defun %prng-value-aval (value)
  "VALUE（配列かトレーサ）の aval。それ以外は NIL。"
  (typecase value
    (tracer (tracer-aval value))
    (array (ignore-errors (array-aval value)))
    (t nil)))

(defun %prng-check-key (name key)
  (let ((aval (%prng-value-aval key)))
    (unless (and aval (eq (aval-dtype aval) :u32) (equal (aval-shape aval) '(2)))
      (%prng-error "~A のキーは :u32 の (2) の配列でなければならない（PRNG-KEY / SPLIT の結果）: ~S"
                   name key))))

(defun %prng-check-shape (shape)
  (unless (and (listp shape) (every (lambda (d) (and (integerp d) (plusp d))) shape))
    (%prng-error "shape は正の整数のリストでなければならない: ~S" shape)))

(defun %prng-check-float-dtype (dtype)
  (unless (member dtype '(:f32 :f64))
    (%prng-error "dtype は :f32 か :f64 でなければならない: ~S" dtype)))

;;; --- トレース用の部品（引数・戻り値はトレーサ） ---

(defun %prng-constant (array)
  (%lift-constant array (array-aval array) *current-trace*))

(defun %prng-u64-array (&rest values)
  (make-array (length values) :element-type '(unsigned-byte 64) :initial-contents values))

(defun %prng-state-from (key counter-tracer)
  "キー（u32[2] のトレーサ）とカウンタ（u64 の rank 0 のトレーサ。NIL なら 0）から
状態 ui64[2] のトレーサを作る: [k0 | k1 << 32, counter]。"
  (let* ((s0 (%trace-eqn :bitcast-convert (list key) :dtype :u64))
         (s0-vector (%trace-eqn :broadcast-in-dim (list s0) :shape '(2) :dims '()))
         (s0-lane (%trace-eqn :mul (list s0-vector (%prng-constant (%prng-u64-array 1 0))))))
    (if counter-tracer
        (let* ((c-vector (%trace-eqn :broadcast-in-dim (list counter-tracer) :shape '(2) :dims '()))
               (c-lane (%trace-eqn :mul (list c-vector (%prng-constant (%prng-u64-array 0 1))))))
          (%trace-eqn :add (list s0-lane c-lane)))
        s0-lane)))

(defun %prng-bits (key shape dtype)
  "KEY から、SHAPE・DTYPE（:u32 / :u64）の乱数ビットのトレーサ（カウンタ 0 から）。
rng-bit-generator の :u32 の出力は多次元だと配置が形に依存する（先頭の偶数の次元を半分に
して並べる）ので、常に1次元で作ってから SHAPE に reshape する。これで値が形の
配置の癖に依らず、同じ個数なら同じ列になる。"
  (let* ((count (reduce #'* shape))
         (flat (second (%trace-eqn* :rng-bit-generator (list (%prng-state-from key nil))
                                    :shape (list count) :dtype dtype))))
    (if (equal shape (list count))
        flat
        (%trace-eqn :reshape (list flat) :shape shape))))

(defun %prng-uniform-01 (key shape dtype)
  "[0, 1) の一様乱数（DTYPE は :f32 / :f64）のトレーサ。ビット列の仮数部だけを取り出して
指数を 1.0 のものにして [1, 2) の数にし、1 を引く。"
  (multiple-value-bind (bit-dtype shift one-bits)
      (ecase dtype
        (:f32 (values :u32 9 #x3F800000))
        (:f64 (values :u64 12 #x3FF0000000000000)))
    (let* ((bits (%prng-bits key shape bit-dtype))
           (mantissa (%trace-eqn :shift-right-logical
                                 (list bits (%lift-number-to shift bit-dtype shape))))
           (exponent-one (%trace-eqn :bitwise-or
                                     (list mantissa (%lift-number-to one-bits bit-dtype shape))))
           (floats (%trace-eqn :bitcast-convert (list exponent-one) :dtype dtype)))
      (%trace-eqn :sub (list floats (%lift-number-to 1 dtype shape))))))

(defun %prng-uniform (key shape dtype minval maxval)
  (let* ((unit (%prng-uniform-01 key shape dtype))
         (scale (- maxval minval))
         (scaled (if (= scale 1) unit (%t-mul unit scale))))
    (if (zerop minval) scaled (%t-add scaled minval))))

(defparameter *erf-inv-f32-coefficients*
  '((5 2.5d0
     (2.81022636d-08 3.43273939d-07 -3.5233877d-06 -4.39150654d-06 0.00021858087d0
      -0.00125372503d0 -0.00417768164d0 0.246640727d0 1.50140941d0))
    (nil 3
     (-0.000200214257d0 0.000100950558d0 0.00134934322d0 -0.00367342844d0 0.00573950773d0
      -0.0076224613d0 0.00943887047d0 1.00167406d0 2.83297682d0)))
  "erf の逆関数の Giles の単精度近似（JAX / XLA の f32 と同じ係数）。区間ごとの
(上限 中心 係数) のリストで、w = -log(1 - x^2) が上限未満の最初の区間を使う（上限 NIL は
残り全部）。最初の区間は w - 中心、それ以外は sqrt(w) - 中心 の多項式で、係数は最高次から並べる。")

;; 係数は XLA の xla/hlo/builder/lib/math.cc の ErfInv64 と突き合わせて確かめた（#175 のレビュー）。
;; f64 の係数 59 個（23 + 19 + 17）、区間の上限 6.25 / 16、中心 3.125 / 3.25 / 5 がすべて一致する。
;; *ERF-INV-F32-COEFFICIENTS* も ErfInv32 と一致する。係数を直すときは、この2つと突き合わせ直す。
(defparameter *erf-inv-f64-coefficients*
  '((6.25d0 3.125d0
     (-3.6444120640178196996d-21 -1.685059138182016589d-19 1.2858480715256400167d-18
      1.115787767802518096d-17 -1.333171662854620906d-16 2.0972767875968561637d-17
      6.6376381343583238325d-15 -4.0545662729752068639d-14 -8.1519341976054721522d-14
      2.6335093153082322977d-12 -1.2975133253453532498d-11 -5.4154120542946279317d-11
      1.051212273321532285d-09 -4.1126339803469836976d-09 -2.9070369957882005086d-08
      4.2347877827932403518d-07 -1.3654692000834678645d-06 -1.3882523362786468719d-05
      0.0001867342080340571352d0 -0.00074070253416626697512d0 -0.0060336708714301490533d0
      0.24015818242558961693d0 1.6536545626831027356d0))
    (16 3.25d0
     (2.2137376921775787049d-09 9.0756561938885390979d-08 -2.7517406297064545428d-07
      1.8239629214389227755d-08 1.5027403968909827627d-06 -4.013867526981545969d-06
      2.9234449089955446044d-06 1.2475304481671778723d-05 -4.7318229009055733981d-05
      6.8284851459573175448d-05 2.4031110387097893999d-05 -0.0003550375203628474796d0
      0.00095328937973738049703d0 -0.0016882755560235047313d0 0.0024914420961078508066d0
      -0.0037512085075692412107d0 0.005370914553590063617d0 1.0052589676941592334d0
      3.0838856104922207635d0))
    (nil 5
     (-2.7109920616438573243d-11 -2.5556418169965252055d-10 1.5076572693500548083d-09
      -3.7894654401267369937d-09 7.6157012080783393804d-09 -1.4960026627149240478d-08
      2.9147953450901080826d-08 -6.7711997758452339498d-08 2.2900482228026654717d-07
      -9.9298272942317002539d-07 4.5260625972231537039d-06 -1.9681778105531670567d-05
      7.5995277030017761139d-05 -0.00021503011930044477347d0 -0.00013871931833623122026d0
      1.0103004648645343977d0 4.8499064014085844221d0)))
  "erf の逆関数の Giles の倍精度近似（XLA の f64 の ErfInv と同じ係数）。形は
*ERF-INV-F32-COEFFICIENTS* と同じ。")

(defun %prng-horner (coefficients w)
  "COEFFICIENTS（最高次から）の多項式を W（トレーサ）で Horner 法で評価する。"
  (let ((p (first coefficients)))
    (dolist (c (rest coefficients) p)
      (setf p (%t-add (%t-mul p w) c)))))

(defun %prng-erf-inv (x)
  "erf の逆関数（X は (-1, 1) の :f32 / :f64 のトレーサ）。Giles の近似を X の dtype で選ぶ
（:f64 は倍精度用の係数）。全区間の多項式を計算し、w の区間で select する。"
  (let* ((w (%t-neg (%t-log (%t-mul (%t-sub 1 x) (%t-add 1 x)))))
         (sqrt-w (%t-exp (%t-mul (%t-log w) 0.5d0)))
         (pieces (if (eq (aval-dtype (tracer-aval x)) :f64)
                     *erf-inv-f64-coefficients*
                     *erf-inv-f32-coefficients*)))
    (labels ((piece (pieces first)
               (destructuring-bind ((bound center coefficients) &rest rest) pieces
                 (let ((p (%prng-horner coefficients (%t-sub (if first w sqrt-w) center))))
                   (if bound
                       (%t-select (%t-compare w bound :lt) p (piece rest nil))
                       p)))))
      (%t-mul (piece pieces t) x))))

(defun %prng-normal (key shape dtype)
  (let* ((lowest (if (eq dtype :f64)
                     (- (- 1d0 (scale-float 1d0 -53)))
                     (coerce (- (- 1 (scale-float 1f0 -24))) 'double-float)))
         (u (%prng-uniform key shape dtype lowest 1)))
    (%t-mul (%prng-erf-inv u) (sqrt 2d0))))

(defun %prng-split (key n)
  (%prng-bits key (list n 2) :u32))

(defun %prng-fold-in (key data)
  "DATA（u32 / i32 の rank 0 のトレーサ）を KEY に混ぜた新しいキー。"
  (let* ((as-u32 (if (eq (aval-dtype (tracer-aval data)) :u32)
                     data
                     ;; i32 → u32 はビット列の再解釈（負の値は 2 の補数。convert の飽和・折り返しに
                     ;; 依らず、実行系によらず定義される）
                     (%trace-eqn :bitcast-convert (list data) :dtype :u32)))
         (wide (%trace-eqn :convert (list as-u32) :dtype :u64))
         (counter (%trace-eqn :add (list wide (%lift-number-to (expt 2 32) :u64 '())))))
    (second (%trace-eqn* :rng-bit-generator
                         (list (%prng-state-from key counter))
                         :shape '(2) :dtype :u32))))

;;; --- 引数の種類による振り分け ---

(defun %prng-dispatch (arguments function)
  "ARGUMENTS（配列かトレーサのリスト）を FUNCTION（トレーサを受けてトレーサを返す）に渡す。
トレーサが1つでもあるか、トレース中（*CURRENT-TRACE*）なら、配列を定数として現在のトレースに
持ち上げて FUNCTION を呼ぶ（結果はトレーサ）。全部配列でトレース中でなければ、新しい
トレースで graph にして eager に評価する（結果は配列）。"
  (if (or *current-trace* (some (lambda (a) (typep a 'tracer)) arguments))
      (apply function
             (mapcar (lambda (a) (if (typep a 'tracer) a (%lift-constant a (array-aval a) *current-trace*)))
                     arguments))
      (apply #'eval-graph
             (%call-with-fresh-trace (mapcar #'array-aval arguments)
                                     (lambda (&rest tracers) (apply function tracers)))
             arguments)))

;;; --- 公開 API ---

(defun prng-key (seed)
  "整数 SEED（符号付き・符号なしの64ビットに収まる整数）から PRNG のキー（:u32 の (2) の配列）を
作る。キーは [SEED の上位32ビット, 下位32ビット]（JAX の PRNGKey と同じ並び）。
SEED は Lisp の整数でなければならない（トレースしない定数。トレース中に呼んでもキーは配列）。"
  (unless (and (integerp seed) (<= (- (expt 2 63)) seed (1- (expt 2 64))))
    (%prng-error "SEED は64ビットに収まる整数でなければならない: ~S" seed))
  (let ((bits (ldb (byte 64 0) seed)))
    (make-array 2 :element-type '(unsigned-byte 32)
                  :initial-contents (list (ldb (byte 32 32) bits) (ldb (byte 32 0) bits)))))

(defun split (key &optional (n 2))
  "KEY から独立な N 個の新しいキーを作り、shape (N 2)・:u32 の配列（またはトレーサ）で返す
（行 i が i 番目のキー）。N は正の整数（既定 2）。同じ KEY・N からは常に同じ結果になる。
KEY を split したら、元の KEY では乱数を引かない（引くと split の結果と同じビット列を共有する）。
KEY は :u32 の (2) の配列かトレーサ。eager・JIT・VMAP の中のどこでも使え、VMAP でキーを
バッチすると各要素のキーを単独に split した結果と一致する。"
  (%prng-check-key "split" key)
  (unless (and (integerp n) (plusp n))
    (%prng-error "split の個数は正の整数でなければならない: ~S" n))
  (%prng-dispatch (list key) (lambda (k) (%prng-split k n))))

(defun fold-in (key data)
  "KEY に整数 DATA を混ぜた新しいキー（:u32 の (2)）を返す。ループの反復番号のように、
同じ KEY から別々のキーを順に作るときに使う。DATA は Lisp の整数（0 以上 2^32 未満）か、
:u32 / :i32 の rank 0 の配列・トレーサ（トレースされた反復番号でもよい）。同じ引数からは
常に同じキーになる。i32 の負の値はビット列を u32 として読む（-1 は 2^32-1。配列でもトレーサでも同じ）。"
  (%prng-check-key "fold-in" key)
  (let ((data-aval (%prng-value-aval data)))
    (cond
      ((integerp data)
       (unless (< -1 data (expt 2 32))
         (%prng-error "fold-in の DATA（整数）は 0 以上 2^32 未満でなければならない: ~S" data))
       (setf data (make-array '() :element-type '(unsigned-byte 32) :initial-element data)))
      ((and data-aval (member (aval-dtype data-aval) '(:u32 :i32)) (null (aval-shape data-aval))))
      (t (%prng-error "fold-in の DATA は整数か、:u32 / :i32 の rank 0 の配列・トレーサでなければならない: ~S"
                      data))))
  (%prng-dispatch (list key data) #'%prng-fold-in))

(defun uniform (key shape &key (dtype :f32) (minval 0) (maxval 1))
  "KEY から、[MINVAL, MAXVAL) の一様乱数の配列（またはトレーサ）を返す。SHAPE は正の整数の
リスト（空リストは rank 0）、DTYPE は :f32 か :f64、MINVAL / MAXVAL は MINVAL < MAXVAL の
Lisp の実数（既定 0 と 1）。同じ KEY・SHAPE・DTYPE からは常に同じ値になり、別のキーからは
別の値になる。浮動小数点の丸めのため、MAXVAL にちょうど等しい値が出うる（JAX と同じ）。
乱数ビット列の仮数部から作る（f32 は 23 ビット、f64 は 52 ビットの粒度）。
eager・JIT・VMAP の中のどこでも使え、VMAP でキーをバッチすると各要素のキーで単独に呼んだ
結果と一致する。JAX とはビット単位では一致しない（README 参照）。"
  (%prng-check-key "uniform" key)
  (%prng-check-shape shape)
  (%prng-check-float-dtype dtype)
  (unless (and (realp minval) (realp maxval) (< minval maxval))
    (%prng-error "MINVAL < MAXVAL の実数でなければならない: ~S ~S" minval maxval))
  (let ((limit (rational (if (eq dtype :f64) most-positive-double-float most-positive-single-float))))
    (unless (and (<= (abs (rational minval)) limit) (<= (abs (rational maxval)) limit)
                 (<= (- (rational maxval) (rational minval)) limit))
      (%prng-error "MINVAL / MAXVAL と幅 (MAXVAL - MINVAL) は ~S に収まらなければならない: ~S ~S"
                   dtype minval maxval)))
  (%prng-dispatch (list key) (lambda (k) (%prng-uniform k shape dtype minval maxval))))

(defun normal (key shape &key (dtype :f32))
  "KEY から、標準正規分布 N(0, 1) に従う乱数の配列（またはトレーサ）を返す。SHAPE は正の整数の
リスト、DTYPE は :f32 か :f64。方式は JAX と同じ、(-1, 1) の一様乱数に erf の逆関数
（Giles の多項式近似。:f32 は単精度用、:f64 は倍精度用で、相対誤差はそれぞれ約 2e-7・5e-16）を
かけて √2 倍する。値は有限（絶対値の上限は f32 で約 5.42、f64 で約 8.29）。同じ KEY・SHAPE・DTYPE からは常に同じ値になる。
eager・JIT・VMAP の中のどこでも使える。"
  (%prng-check-key "normal" key)
  (%prng-check-shape shape)
  (%prng-check-float-dtype dtype)
  (%prng-dispatch (list key) (lambda (k) (%prng-normal k shape dtype))))
