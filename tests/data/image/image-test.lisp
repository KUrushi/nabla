;;;; 画像のデコードと image-source（issue #283）。
;;;;
;;;; 守らせる性質（ファイルを読むので medium）:
;;;; - Pillow との一致: tests/fixtures/image/ の PNG は Pillow がデコードした画素とちょうど同じ、
;;;;   JPEG は成分ごとの差が JPEG-TOLERANCE 以下。形は (H W C)、element-type は (unsigned-byte 8)。
;;;; - image-source のレコード: index 番目のレコードは index 番目のファイルの画像（ラベルがあれば
;;;;   (画像 ラベルの index 番目の行)）。パスはパスネームでも OS の書き方の文字列でもよい。
;;;; - PNG の IEND の後ろのバイトは読まない（足しても同じ画素）。
;;;; - 壊れたファイル: 途中で切れたファイル（シグネチャより後ろで切ったもの）と、シグネチャより
;;;;   後ろの1バイトを変えた PNG は :corrupt。1バイトを変えた JPEG は画像になるか
;;;;   image-decode-error になり、範囲外のメモリに触れない（scripts/setup-lisp-deps.sh が clone 時に
;;;;   cl-jpeg の safety を 2 に書き換える約束。safety 0 のままだと Memory fault でプロセスが落ちる）。
;;;; - cl-jpeg の safety: ロードした cl-jpeg の jpeg::*optimize* は safety 2 以上。safety が 2 未満か
;;;;   書かれていなければ、JPEG はデコードせずに setup-lisp-deps.sh を挙げる error（PNG は読める）。
;;;; - 読まない形式は :unsupported、PNG でも JPEG でもないファイルは :unknown-format、
;;;;   開けない・読めないファイル（無い・ディレクトリ）は :unreadable。
;;;; - image-source の範囲外の添字・ラベルの個数の食い違いは data-source-error。
;;;; - 依存の方針: nabla/data/image が直接依存する nabla のシステムは nabla/data だけ。

(in-package #:nabla.data.image.tests)

(defparameter +jpeg-tolerance+ 2
  "JPEG の画素の許容誤差。cl-jpeg と Pillow（libjpeg）は逆 DCT の丸めと色差の補間が違う
（フィクスチャでの実測は最大 1）。")

(defun fixture-path (name)
  (asdf:system-relative-pathname "nabla" (format nil "tests/fixtures/image/~A" name)))

(defun image-fixtures ()
  "tests/fixtures/image/expected.lisp の (名前 (H W C) (画素値 ...)) のリスト。"
  (with-open-file (in (fixture-path "expected.lisp"))
    (with-standard-io-syntax (read in))))

(defun fixture-tolerance (name)
  (if (string= "jpg" (pathname-type name)) +jpeg-tolerance+ 0))

(defun image-matches-p (image shape values tolerance)
  (and (equal '(unsigned-byte 8) (array-element-type image))
       (equal shape (array-dimensions image))
       (loop for value in values
             for i from 0
             always (<= (abs (- value (row-major-aref image i))) tolerance))))

(defun file-octets (pathname)
  (with-open-file (in pathname :element-type '(unsigned-byte 8))
    (let ((octets (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (read-sequence octets in)
      octets)))

(defun write-octets (pathname octets)
  (with-open-file (out pathname :direction :output :if-exists :supersede
                                :element-type '(unsigned-byte 8))
    (write-sequence octets out))
  pathname)

(defun decode-reason (pathname)
  "PATHNAME を READ-IMAGE したときの IMAGE-DECODE-ERROR の reason。読めたら :no-error。"
  (handler-case (progn (read-image pathname) :no-error)
    (image-decode-error (c) (image-decode-error-reason c))))

(defun source-reason (thunk)
  (handler-case (progn (funcall thunk) :no-error)
    (data-source-error (c) (data-source-error-reason c))))

(defmacro def-image-property (name docstring (&rest vars) generator &body body)
  "GENERATOR が作る値を VARS に束縛して BODY（真偽）を medium の PBT で確かめる。"
  `(test (,name :suite :nabla.medium)
     ,docstring
     (is (check-it ,generator
                   (lambda (case) (destructuring-bind ,vars case ,@body))
                   :regression-id ,name
                   :regression-file (regression-path ,(substitute #\- #\/ (string-downcase (string name)))
                                                     :package "NABLA.DATA.IMAGE.TESTS"))
         ,(format nil "~A の性質が成り立たなかった" name))))

;;; --- Pillow との一致 ---

(test (image/decodes-like-pillow :suite :nabla.medium)
  "PNG（RGB・RGBA・グレースケール・グレースケール + アルファ・パレット・1 ビット・インタレース・
256 バイト以上のチャンク）と
JPEG（ベースライン・4:2:0・プログレッシブ・グレースケール・Adobe の APP14 つき）の画素が Pillow と一致する。"
  (dolist (fixture (image-fixtures))
    (destructuring-bind (name shape values) fixture
      (is (image-matches-p (read-image (fixture-path name)) shape values (fixture-tolerance name))
          "~A が Pillow の ~A の画素と一致しない" name shape))))

;;; --- image-source ---

(def-image-property image-source/record-is-the-image-and-the-label-row
  "index 番目のレコードは index 番目のファイルの画像で、ラベルがあればラベルの index 番目の行と組になる。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (fixtures (image-fixtures))
         (chosen (loop repeat (random 5 state) collect (nth (random (length fixtures) state) fixtures)))
         (paths (mapcar (lambda (fixture)
                          (let ((path (fixture-path (first fixture))))
                            (if (zerop (random 2 state)) path (namestring path))))
                        chosen))
         (labels (and (zerop (random 3 state))
                      (make-array (length chosen) :element-type '(signed-byte 32)
                                                  :initial-contents (loop repeat (length chosen)
                                                                          collect (- (random 2000 state) 1000)))))
         (source (make-image-source paths :labels labels)))
    (and (= (length chosen) (source-length source))
         (loop for (name shape values) in chosen
               for i from 0
               for record = (source-item source i)
               for image = (if labels (first record) record)
               always (and (image-matches-p image shape values (fixture-tolerance name))
                           (or (null labels)
                               (and (= 2 (length record))
                                    (= (aref labels i) (aref (second record)))
                                    (equal '() (array-dimensions (second record))))))))))

(def-image-property image-source/index-out-of-range-is-reported
  "ファイルが n 個のとき、0 未満・n 以上・整数でない添字は :index-out-of-range（ファイルは開かない）。"
  (n offset)
  (generator (tuple (uniform-integer :lo 0 :hi 4) (uniform-integer :lo 0 :hi 3)))
  (let ((source (make-image-source (loop repeat n collect "missing.png"))))
    (every (lambda (index) (eq :index-out-of-range (source-reason (lambda () (source-item source index)))))
           (list (+ n offset) (- -1 offset) (+ offset 0.5) nil))))

(def-image-property image-source/rejects-labels-of-another-length
  "ファイルが n 個でラベルが n 個でなければ :length-mismatch、n 個なら作れる。"
  (n m)
  (generator (tuple (uniform-integer :lo 0 :hi 4) (uniform-integer :lo 0 :hi 4)))
  (eq (if (= n m) :no-error :length-mismatch)
      (source-reason (lambda ()
                       (make-image-source (loop repeat n collect "a.png")
                                          :labels (make-array m :element-type '(signed-byte 32)
                                                                :initial-element 0))))))

;;; --- 壊れたファイル ---

(def-image-property image/bytes-after-iend-are-ignored
  "PNG の IEND の後ろに足したバイトは読まない（png-read と Pillow と同じ）。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (pngs (remove "png" (image-fixtures) :test-not #'string= :key (lambda (f) (pathname-type (first f)))))
         (fixture (nth (random (length pngs) state) pngs))
         (extra (make-array (1+ (random 20 state)) :element-type '(unsigned-byte 8))))
    (dotimes (i (length extra))
      (setf (aref extra i) (random 256 state)))
    (with-temporary-directory (dir)
      (destructuring-bind (name shape values) fixture
        (image-matches-p (read-image (write-octets (merge-pathnames "padded.png" dir)
                                                   (concatenate '(vector (unsigned-byte 8))
                                                                (file-octets (fixture-path name)) extra)))
                         shape values 0)))))

(def-image-property image/truncated-file-is-corrupt
  "シグネチャ（8 バイト）より後ろで切ったファイルは :corrupt。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (fixtures (image-fixtures))
         (octets (file-octets (fixture-path (first (nth (random (length fixtures) state) fixtures)))))
         (end (+ 8 (random (- (length octets) 8) state))))
    (with-temporary-directory (dir)
      (eq :corrupt (decode-reason (write-octets (merge-pathnames "cut" dir) (subseq octets 0 end)))))))

(def-image-property image/png-with-a-changed-byte-is-corrupt
  "シグネチャより後ろの1バイトを変えた PNG は :corrupt（長さ以外は CRC で守られ、壊れた長さは
ファイルの大きさと比べて見つける）。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (names '("rgb.png" "rgba.png" "gray.png" "gray-alpha.png" "palette.png" "bilevel.png"
                  "interlaced.png" "noise.png"))
         (octets (file-octets (fixture-path (nth (random (length names) state) names))))
         (position (+ 8 (random (- (length octets) 8) state))))
    (setf (aref octets position) (logxor (aref octets position) (1+ (random 255 state))))
    (with-temporary-directory (dir)
      (eq :corrupt (decode-reason (write-octets (merge-pathnames "changed" dir) octets))))))

(def-image-property image/huge-png-chunk-length-is-corrupt-without-allocating-it
  "IHDR の長さを 256 MiB 以上に書き換えた PNG は :corrupt で、その大きさの配列を確保しない
（png-read に渡す前に、長さがファイルの中に収まるかを確かめる）。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (names '("rgb.png" "palette.png" "noise.png"))
         (octets (file-octets (fixture-path (nth (random (length names) state) names))))
         (length (+ (ash 1 28) (random (ash 1 30) state))))
    (dotimes (i 4)
      (setf (aref octets (+ 8 i)) (ldb (byte 8 (* 8 (- 3 i))) length)))
    (with-temporary-directory (dir)
      (let* ((path (write-octets (merge-pathnames "huge.png" dir) octets))
             (before (sb-ext:get-bytes-consed))
             (reason (decode-reason path)))
        (and (eq :corrupt reason)
             (< (- (sb-ext:get-bytes-consed) before) (ash 1 24)))))))

(def-image-property image/jpeg-with-changed-bytes-stays-in-bounds
  "1バイトを変えた JPEG は、(H W C) の u8 の画像になるか IMAGE-DECODE-ERROR になり、範囲外の
メモリに触れない（Memory fault にならない）。1 回の試行で 20 個のファイルを試す。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let ((state (sb-ext:seed-random-state seed))
        (names '("baseline.jpg" "subsampled.jpg" "progressive.jpg" "gray.jpg")))
    (with-temporary-directory (dir)
      (loop repeat 20
            always (let* ((octets (file-octets (fixture-path (nth (random (length names) state) names))))
                          (position (random (length octets) state)))
                     (setf (aref octets position) (logxor (aref octets position) (1+ (random 255 state))))
                     (handler-case
                         (let ((image (read-image (write-octets (merge-pathnames "changed" dir) octets))))
                           (and (equal '(unsigned-byte 8) (array-element-type image))
                                (= 3 (array-rank image))))
                       (image-decode-error (c)
                         (not (search "memory fault"
                                      (string-downcase (image-decode-error-detail c)))))))))))

;;; --- cl-jpeg の safety ---

(test (image/loaded-cl-jpeg-optimizes-with-safety-2 :suite :nabla.small)
  "テストのイメージにロードされた cl-jpeg の jpeg::*optimize* の safety は 2
（scripts/setup-lisp-deps.sh が clone 時に書き換える。上流の safety 0 のままなら落ちる）。"
  (is (equal '(safety 2)
             (find 'safety (rest jpeg::*optimize*) :key (lambda (quality)
                                                          (if (consp quality) (first quality) quality))))
      "jpeg::*optimize* が ~S" jpeg::*optimize*))

(def-image-property image/jpeg-needs-cl-jpeg-with-safety-2
  "jpeg::*optimize* の safety が 2 未満か書かれていないと、READ-IMAGE は JPEG をデコードせずに
（IMAGE-DECODE-ERROR ではない）error を signal し、その文言は scripts/setup-lisp-deps.sh を挙げる。
safety が 2 以上（値の無い safety は 3）なら JPEG を読める。PNG はどちらでも読める。"
  (choice seed)
  (generator (tuple (uniform-integer :lo 0 :hi 5) (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (safety (nth choice '((safety 0) (safety 1) (safety 2) (safety 3) safety nil)))
         ;; safety の位置で結果が変わらないよう、宣言の中の順を混ぜる。
         (qualities (mapcar #'cdr (sort (mapcar (lambda (quality) (cons (random 1000 state) quality))
                                                (remove nil (list safety '(space 0) '(debug 0) '(speed 3))))
                                        #'< :key #'car)))
         (outcome (let ((jpeg::*optimize* (cons 'optimize qualities)))
                    (list (handler-case (progn (read-image (fixture-path "baseline.jpg")) :decoded)
                            (image-decode-error () :decode-error)
                            (error (c)
                              (if (search "scripts/setup-lisp-deps.sh" (princ-to-string c))
                                  :refused
                                  :other-error)))
                          (decode-reason (fixture-path "rgb.png"))))))
    (equal outcome (list (if (member choice '(0 1 5)) :refused :decoded) :no-error))))

;;; --- 読まない形式・読めないファイル ---

(test (image/unsupported-formats-are-reported :suite :nabla.medium)
  "16 ビットの PNG、CMYK（4 成分）の JPEG、Adobe の APP14 が RGB（色変換 0）と言う JPEG は :unsupported。"
  (is (eq :unsupported (decode-reason (fixture-path "gray16.png"))))
  (is (eq :unsupported (decode-reason (fixture-path "cmyk.jpg"))))
  (is (eq :unsupported (decode-reason (fixture-path "adobe-rgb.jpg")))))

(test (image/other-files-are-unknown-format :suite :nabla.medium)
  (is (eq :unknown-format (decode-reason (fixture-path "generate.py"))))
  (with-temporary-directory (dir)
    (is (eq :unknown-format
            (decode-reason (write-octets (merge-pathnames "empty.png" dir)
                                         (make-array 0 :element-type '(unsigned-byte 8))))))))

(test (image/missing-file-is-unreadable :suite :nabla.medium)
  (with-temporary-directory (dir)
    (let ((condition (handler-case (read-image (merge-pathnames "missing.png" dir))
                       (image-decode-error (c) c))))
      (is (typep condition 'data-error))
      (is (eq :unreadable (image-decode-error-reason condition)))
      (is (equal (merge-pathnames "missing.png" dir) (image-decode-error-pathname condition))))
    (is (eq :unreadable (decode-reason (ensure-directories-exist (merge-pathnames "dir.png/" dir))))
        "ディレクトリが :unreadable にならない")))

;;; --- 依存の方針 ---

(test (data/image/depends-only-on-data :suite :nabla.small)
  "nabla/data/image は nabla のシステムのうち nabla/data にだけ依存し、nabla/nn・nabla/iree・
nabla/pjrt を知らない（issue #283。画像のライブラリには依存してよい）。"
  (is (equal '("nabla/data")
             (remove-if-not (lambda (dependency)
                              (and (stringp dependency) (eql 0 (search "nabla" dependency))))
                            (asdf:system-depends-on (asdf:find-system "nabla/data/image"))))))
