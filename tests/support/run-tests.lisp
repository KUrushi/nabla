;;;; run-tests / sizes-from-env: テストスイートの実行を制御する。

(in-package #:nabla.tests.support)

(defun %split-comma (string)
  (loop for start = 0 then (1+ pos)
        for pos = (position #\, string :start start)
        collect (string-trim '(#\Space #\Tab) (subseq string start pos))
        while pos))

(defun %size-keyword (name)
  (cond
    ((string-equal name "small") :small)
    ((string-equal name "medium") :medium)
    ((string-equal name "large") :large)
    (t (error "NABLA_TEST_SIZES: 知らないテストサイズ ~S（small / medium / large のどれか）" name))))

(defun sizes-from-env (&optional (env-value (sb-ext:posix-getenv "NABLA_TEST_SIZES")))
  "NABLA_TEST_SIZES（カンマ区切り。既定は \"small,medium\"）を
(:small :medium :large) のリストにして返す。空文字列や未設定は既定値に
フォールバックし、区切りの前後の空要素（先頭・末尾・連続するカンマ）は
無視する。"
  (let* ((trimmed (and env-value (string-trim '(#\Space #\Tab) env-value)))
         (value (if (or (null trimmed) (zerop (length trimmed)))
                    "small,medium"
                    env-value)))
    (mapcar #'%size-keyword
            (remove "" (%split-comma value) :test #'string=))))

(defun %size-suite (size)
  (ecase size
    (:small :nabla.small)
    (:medium :nabla.medium)
    (:large :nabla.large)))

(defun run-tests (&key (sizes (sizes-from-env)))
  "SIZES（既定は sizes-from-env の結果）に対応する FiveAM スイートを
すべて実行し、全部通れば T、1つでも落ちれば NIL を返す。
スイートごとに一行の要約を標準出力に表示する。"
  (let ((all-ok t))
    (dolist (size sizes)
      (let* ((suite (%size-suite size))
             (ok (fiveam:run! suite)))
        (format t "~&[nabla] suite ~A: ~:[FAILED~;ok~]~%" suite ok)
        (unless ok (setf all-ok nil))))
    all-ok))
