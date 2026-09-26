;;;; allclose / approx=: 許容誤差つきの数値比較。
;;;;
;;;; 比較の規則は |actual - expected| <= atol + rtol * |expected|。
;;;; NaN はどちらか一方でも含まれていれば、決して一致しない。

(in-package #:nabla.tests.support)

(defun %nan-p (x)
  (sb-ext:float-nan-p x))

(defun %default-tolerance (dtype rtol atol)
  "DTYPE があれば dtype-tolerance を、なければ緩めの既定値を使う。
RTOL / ATOL が明示されていれば、そちらを優先する。"
  (multiple-value-bind (default-rtol default-atol)
      (if dtype (dtype-tolerance dtype) (values 1d-5 1d-8))
    (values (or rtol default-rtol) (or atol default-atol))))

(defun approx= (a b &key dtype rtol atol)
  "スカラー A と B が許容誤差つきで一致するか。

|A - B| <= ATOL + RTOL * |B| なら真。A・B のどちらかが NaN なら常に偽。
比較は DOUBLE-FLOAT で行う。"
  (multiple-value-bind (rtol atol) (%default-tolerance dtype rtol atol)
    (let ((a (coerce a 'double-float))
          (b (coerce b 'double-float)))
      (and (not (%nan-p a))
           (not (%nan-p b))
           (<= (abs (- a b)) (+ atol (* rtol (abs b))))))))

(defun allclose (actual expected &key dtype rtol atol)
  "配列 ACTUAL と EXPECTED の全要素が許容誤差つきで一致するか。

一致しない要素があれば、最大誤差とその添字、使った許容誤差を
*standard-output* に表示し、NIL を返す。一致すれば T を返す。
DTYPE を渡すと、その dtype の要素の格納形式（bf16 / f16 のビット列など）
を数値に戻してから比較する。"
  (unless (equal (array-dimensions actual) (array-dimensions expected))
    (error "allclose: 形状が違う: ~A と ~A"
           (array-dimensions actual) (array-dimensions expected)))
  (multiple-value-bind (rtol atol) (%default-tolerance dtype rtol atol)
    (let ((max-diff 0d0)
          (max-index nil)
          (nan-seen nil)
          (out-of-tolerance nil))
      (dotimes (i (array-total-size actual))
        (let* ((a (if dtype
                      (decode-element dtype (row-major-aref actual i))
                      (coerce (row-major-aref actual i) 'double-float)))
               (b (if dtype
                      (decode-element dtype (row-major-aref expected i))
                      (coerce (row-major-aref expected i) 'double-float))))
          (cond
            ((or (%nan-p a) (%nan-p b))
             (setf nan-seen t
                   out-of-tolerance t))
            (t
             (let ((diff (abs (- a b))))
               (when (> diff max-diff)
                 (setf max-diff diff
                       max-index i))
               (when (> diff (+ atol (* rtol (abs b))))
                 (setf out-of-tolerance t)))))))
      (if out-of-tolerance
          (progn
            (format t "~&allclose: 不一致。最大誤差 ~A (添字 ~A)、rtol=~A atol=~A、NaN を含む: ~A~%"
                    max-diff max-index rtol atol nan-seen)
            nil)
          t))))
