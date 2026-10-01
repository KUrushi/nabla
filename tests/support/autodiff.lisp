;;;; autodiff: 自動微分（jvp / transpose / vjp / grad）のテストが共通で使う
;;;; ヘルパー（issue #76）。
;;;;
;;;; 守らせる性質は .claude/skills/nabla-testing/references/properties.md
;;;; の「自動微分」の節にある。ここはそのための道具だけを持つ:
;;;;
;;;;   - f64 の中心差分による方向微分（CENTRAL-DIFFERENCE-JVP）と
;;;;     勾配・vjp（CENTRAL-DIFFERENCE-GRADIENT）
;;;;   - 任意 rank の配列の内積（INNER-PRODUCT）
;;;;   - ランダムな接線・余接線（RANDOM-TANGENT / RANDOM-COTANGENT）
;;;;
;;;; 評価は常に eager の NB:EVAL-GRAPH（または呼び出し側が渡す関数）で
;;;; 行う。IREE を通さないので f64 のまま計算でき、CPU だけで動く。

(in-package #:nabla.tests.support)

(defparameter *central-difference-step* 1d-6
  "中心差分の既定の刻み幅 h。

f64 の機械イプシロンは約 2.2e-16。中心差分 (f(x+h) - f(x-h)) / 2h の
誤差は、打ち切り誤差 O(h^2 f''') と丸め誤差 O(eps |f| / h) の和で、
h = 1e-6 のとき前者は約 1e-12 · f'''、後者は約 1e-10 · |f| になり、
どちらも rtol 1e-4 より十分小さい。h を 1e-8 まで小さくすると丸め誤差が
1e-8 付近まで膨らみ、1e-3 まで大きくすると打ち切り誤差が 1e-6 付近まで
膨らむ。1e-6 はその中間の安全な値。")

(defparameter *autodiff-rtol* 1d-4
  "自動微分の結果と中心差分を比べるときの、推奨する相対許容誤差。

中心差分そのものが近似（打ち切り誤差と丸め誤差、上の
*CENTRAL-DIFFERENCE-STEP* 参照）なので、f64 の厳密な一致（1e-12 程度）
は求めず、1e-4 に緩める。これより厳しくしても、差分側の誤差で偶発的に
落ちるだけで、ルールの間違いを見つける力は増えない。")

(defparameter *autodiff-atol* 1d-6
  "自動微分の比較で、真の値が 0 に近いときに効く推奨の絶対許容誤差。")

(defun %require-f64-array (array what)
  (unless (typep array '(array double-float))
    (error "~A は f64（double-float）の配列でなければならない: ~S" what array)))

(defun %call-autodiff-function (fn arrays)
  "FN（GRAPH か関数）を ARRAYS に適用し、出力の配列のリストを返す。"
  (multiple-value-list
   (if (nb::graph-p fn)
       (apply #'nb:eval-graph fn arrays)
       (apply fn arrays))))

(defun %shift-arrays (primals tangents step)
  "各 PRIMALS[i] + STEP * TANGENTS[i] を f64 の新しい配列のリストにして返す。"
  (mapcar (lambda (p v)
            (let ((out (make-array (array-dimensions p) :element-type 'double-float)))
              (dotimes (i (array-total-size p) out)
                (setf (row-major-aref out i)
                      (+ (row-major-aref p i) (* step (row-major-aref v i)))))))
          primals tangents))

(defun central-difference-jvp (fn primals tangents &key (h *central-difference-step*))
  "FN の PRIMALS での、TANGENTS 方向の方向微分を中心差分で求める。

FN は NB::GRAPH（NB:EVAL-GRAPH で評価する）か、f64 配列を受け取って
配列を多値で返す関数。PRIMALS と TANGENTS は同じ形の f64 配列のリスト。
戻り値は出力ごとの方向微分（f64 配列）のリスト:
  (FN(x + h v) - FN(x - h v)) / 2h
H の既定値と許容誤差の根拠は *CENTRAL-DIFFERENCE-STEP* /
*AUTODIFF-RTOL* を参照。"
  (mapc (lambda (a) (%require-f64-array a "primal")) primals)
  (mapc (lambda (a) (%require-f64-array a "tangent")) tangents)
  (let ((plus (%call-autodiff-function fn (%shift-arrays primals tangents h)))
        (minus (%call-autodiff-function fn (%shift-arrays primals tangents (- h)))))
    (mapcar (lambda (fp fm)
              (let ((out (make-array (array-dimensions fp) :element-type 'double-float)))
                (dotimes (i (array-total-size fp) out)
                  (setf (row-major-aref out i)
                        (/ (- (row-major-aref fp i) (row-major-aref fm i)) (* 2 h))))))
            plus minus)))

(defun central-difference-gradient (fn primals &key cotangents (h *central-difference-step*))
  "FN の vjp（余接線 COTANGENTS との縮約）を、入力の要素を1つずつ動かす
中心差分で求める。戻り値は PRIMALS と同じ形の f64 配列のリスト。

COTANGENTS は FN の出力ごとの f64 配列のリスト。省略すると全て 1
（FN がスカラーを返すなら、これが通常の勾配になる）。入力 i の要素 j の
結果は <COTANGENTS, d FN / d x_ij>。FN の呼び出しは入力の総要素数の2倍。"
  (mapc (lambda (a) (%require-f64-array a "primal")) primals)
  (let ((cotangents (or cotangents
                        (mapcar (lambda (out)
                                  (make-array (array-dimensions out) :element-type 'double-float
                                                                     :initial-element 1d0))
                                (%call-autodiff-function fn primals))))
        (tangents (mapcar (lambda (p) (make-array (array-dimensions p) :element-type 'double-float
                                                                       :initial-element 0d0))
                          primals)))
    (loop for tangent in tangents
          collect (let ((grad (make-array (array-dimensions tangent) :element-type 'double-float)))
                    (dotimes (j (array-total-size tangent) grad)
                      (setf (row-major-aref tangent j) 1d0)
                      (setf (row-major-aref grad j)
                            (reduce #'+ (mapcar #'inner-product cotangents
                                                (central-difference-jvp fn primals tangents :h h))))
                      (setf (row-major-aref tangent j) 0d0))))))

(defun inner-product (a b)
  "同じ形の配列 A と B（任意 rank、single / double-float）の内積 <A, B> を
DOUBLE-FLOAT で返す。f32 の入力でも各要素を f64 に直してから積和を f64 で
累積する（丸め誤差で性質のテストが揺れないように）。形が違えば ERROR。
rank 0 の配列は要素1つの積。"
  (unless (equal (array-dimensions a) (array-dimensions b))
    (error "内積の2つの配列の形が違う: ~S と ~S" (array-dimensions a) (array-dimensions b)))
  (let ((sum 0d0))
    (declare (type double-float sum))
    (dotimes (i (array-total-size a) sum)
      (incf sum (* (coerce (row-major-aref a i) 'double-float)
                   (coerce (row-major-aref b i) 'double-float))))))

(defun random-tangent (aval &key (seed 0))
  "AVAL（NB:AVAL。shape だけを使う）と同じ形の、決定的な f64 の乱数配列
（各要素は概ね [-1, 1)）を返す。SEED が同じなら同じ配列になる。
jvp に渡す接線 v に使う。AVAL の dtype に関わらず f64 を返すのは、
中心差分（f64）との比較に使うため。"
  (make-random-array (make-array-spec (nb:aval-shape aval) :f64) :seed seed))

(defun random-cotangent (aval &key (seed 0))
  "vjp に渡す余接線 u を作る。出力の AVAL と同じ形の f64 乱数配列で、
RANDOM-TANGENT と同じ分布（接線と余接線を区別して読めるようにした別名）。"
  (random-tangent aval :seed seed))
