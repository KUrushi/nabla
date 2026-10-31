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

(defun shard-from-env (&optional (env-value (sb-ext:posix-getenv "NABLA_TEST_SHARD")))
  "NABLA_TEST_SHARD（\"I/N\"。1 始まりの番号 I と個数 N）を (I N) のリストに
して返す。未設定や空文字列は NIL（シャードしない）。I が 1〜N の外の値や
形の崩れた値はエラーにする（CI で一部のテストが黙って抜けないように）。
CI はテストを N 個のジョブに分けて並列に走らせるのに使う。"
  (let ((value (and env-value (string-trim '(#\Space #\Tab) env-value))))
    (unless (or (null value) (zerop (length value)))
      (flet ((parse (string)
               (and (plusp (length string))
                    (every #'digit-char-p string)
                    (parse-integer string))))
        (let* ((slash (position #\/ value))
               (index (and slash (parse (subseq value 0 slash))))
               (count (and slash (parse (subseq value (1+ slash))))))
          (unless (and index count (<= 1 index count))
            (error "NABLA_TEST_SHARD: ~S は \"I/N\"（1 <= I <= N）の形ではない" env-value))
          (list index count))))))

(defun %select-shard (names shard)
  "NAMES（テストやスイートの名前）のうち、SHARD = (I N) が受け持つものを返す。
名前を印字した文字列で並べ、その順で I 番目から N 個おきに取る（並べてから
配るので、どのジョブでも同じ分け方になり、同じファイルのテストは別々の
シャードに散る）。"
  (destructuring-bind (index count) shard
    (loop for name in (sort (copy-list names) #'string< :key #'prin1-to-string)
          for position from 0
          when (= (mod position count) (1- index))
            collect name)))

(defun %run-suite-shard (suite shard)
  "SUITE の直下のテスト（と子スイート）のうち SHARD の受け持ちだけを実行し、
fiveam:run! と同じく結果を説明して、全部通れば T を返す。"
  ;; FiveAM はスイートの直下の要素を列挙する関数を export していないので、
  ;; スイートの中身の表（fiveam::tests）を直接読む。
  (let ((names (loop for name being the hash-keys of (fiveam::tests (get-test suite))
                     collect name)))
    (explain! (loop for name in (%select-shard names shard)
                    append (run name)))))

(defun run-tests (&key (sizes (sizes-from-env)) (shard (shard-from-env)))
  "SIZES（既定は sizes-from-env の結果）に対応する FiveAM スイートを
すべて実行し、全部通れば T、1つでも落ちれば NIL を返す。
SHARD（既定は shard-from-env の結果）が (I N) なら、各スイートの直下の
テストを N 個に分けたうちの I 番目だけを実行する。
スイートごとに一行の要約を標準出力に表示する。"
  (let ((all-ok t))
    (dolist (size sizes)
      (let* ((suite (%size-suite size))
             (ok (if shard
                     (%run-suite-shard suite shard)
                     (fiveam:run! suite))))
        (format t "~&[nabla] suite ~A~@[ (shard ~{~D/~D~})~]: ~:[FAILED~;ok~]~%"
                suite shard ok)
        (unless ok (setf all-ok nil))))
    all-ok))
