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
;;;;   image-decode-error になり、範囲外のメモリに触れない（scripts/setup-lisp-deps.sh が safety 2 に
;;;;   直した cl-jpeg の fork を入れる約束。上流の safety 0 だと Memory fault でプロセスが落ちる）。
;;;; - cl-jpeg の safety: ロードした cl-jpeg の jpeg::*optimize* は safety 2 以上。safety が 2 未満か
;;;;   書かれていなければ、JPEG はデコードせずに setup-lisp-deps.sh を挙げる error（PNG は読める）。
;;;; - 読まない形式は :unsupported、PNG でも JPEG でもないファイルは :unknown-format、
;;;;   開けない・読めないファイル（無い・ディレクトリ）は :unreadable。
;;;; - image-source の範囲外の添字・ラベルの個数の食い違いは data-source-error。
;;;; - 寸法の上限（#324）: ヘッダの幅 × 高さが上限（:max-pixels、既定 *max-image-pixels*）以下なら
;;;;   今までどおり読め、超えれば :too-large（nil なら上限なし）。寸法だけを巨大に書き換えた PNG（IHDR の
;;;;   CRC も直したもの）と JPEG（SOF の寸法）は、その大きさの配列を確保せずに :too-large。image-source は
;;;;   作ったときの上限を持つ（ワーカーのスレッドは呼び出し側の動的束縛を見ないので）。
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

;;; --- 寸法の上限（#324） ---

(defun image-test-crc32 (octets &key (start 0) (end (length octets)))
  "PNG のチャンクの CRC-32（ISO-HDLC。多項式 #xEDB88320 の反転形）。"
  (let ((crc #xffffffff))
    (loop for i from start below end
          do (setf crc (logxor crc (aref octets i)))
             (dotimes (bit 8)
               (setf crc (if (logbitp 0 crc) (logxor #xedb88320 (ash crc -1)) (ash crc -1)))))
    (logxor crc #xffffffff)))

(defun put-big-endian (octets position value count)
  "OCTETS の POSITION から COUNT バイトに VALUE をビッグエンディアンで書く。"
  (dotimes (i count octets)
    (setf (aref octets (+ position i)) (ldb (byte 8 (* 8 (- count 1 i))) value))))

(defun png-with-dimensions (octets width height)
  "PNG の OCTETS の IHDR の幅と高さを書き換え、IHDR の CRC も直したもの。"
  (put-big-endian octets 16 width 4)
  (put-big-endian octets 20 height 4)
  (put-big-endian octets 29 (image-test-crc32 octets :start 12 :end 29) 4))

(defun jpeg-sof-position (octets)
  "JPEG の OCTETS の最初の SOF マーカー（#xFF の次の #xC0〜#xCF のうち DHT・JPG・DAC 以外）の位置。"
  (loop for i from 2 below (1- (length octets))
        when (and (= #xff (aref octets i))
                  (<= #xc0 (aref octets (1+ i)) #xcf)
                  (not (member (aref octets (1+ i)) '(#xc4 #xc8 #xcc))))
          return i))

(defun jpeg-with-dimensions (octets width height)
  "JPEG の OCTETS の SOF の高さと幅を書き換えたもの（SOF の後ろはマーカー 2・長さ 2・精度 1・高さ 2・幅 2）。"
  (let ((sof (jpeg-sof-position octets)))
    (put-big-endian octets (+ sof 5) height 2)
    (put-big-endian octets (+ sof 7) width 2)))

(def-image-property image/images-up-to-the-pixel-limit-decode-and-larger-ones-are-too-large
  "幅 × 高さが :max-pixels 以下の画像は Pillow と同じ画素で読め、超えれば :too-large。nil なら上限なし。
image-source は作ったときの上限を持つ。"
  (seed delta)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo -3 :hi 3)))
  (let* ((state (sb-ext:seed-random-state seed))
         (fixtures (image-fixtures))
         (fixture (nth (random (length fixtures) state) fixtures)))
    (destructuring-bind (name (height width channels) values) fixture
      (declare (ignore channels))
      (let* ((limit (max 0 (+ (* height width) delta)))
             (path (fixture-path name))
             (expected (if (<= (* height width) limit) :decoded :too-large)))
        (flet ((outcome (thunk)
                 (handler-case (if (image-matches-p (funcall thunk) (second fixture) values
                                                    (fixture-tolerance name))
                                   :decoded
                                   :wrong-pixels)
                   (image-decode-error (c) (image-decode-error-reason c)))))
          (and (eq expected (outcome (lambda () (read-image path :max-pixels limit))))
               (eq :decoded (outcome (lambda () (read-image path :max-pixels nil))))
               (eq expected (outcome (let ((source (make-image-source (list path) :max-pixels limit)))
                                       (lambda () (let ((*max-image-pixels* nil)) (source-item source 0))))))
               (eq expected (outcome (let ((source (let ((*max-image-pixels* limit))
                                                     (make-image-source (list path)))))
                                       (lambda () (source-item source 0)))))))))))

(def-image-property image/huge-dimensions-are-too-large-without-allocating
  "ヘッダの寸法だけを、幅 × 高さが既定の上限を超えるように書き換えた PNG（IHDR の CRC も直す）と
JPEG（SOF）は :too-large で、その大きさの配列を確保しない。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (pngs '("rgb.png" "rgba.png" "gray.png" "palette.png" "interlaced.png" "noise.png"))
         (jpegs '("baseline.jpg" "subsampled.jpg" "progressive.jpg" "gray.jpg"))
         (png-width (+ 9500 (random (- (ash 1 31) 9500) state)))
         (png-height (+ (ceiling (1+ *max-image-pixels*) png-width) (random 1000 state)))
         (jpeg-width (+ 1366 (random (- 65536 1366) state)))
         (jpeg-height (min 65535 (+ (ceiling (1+ *max-image-pixels*) jpeg-width) (random 1000 state))))
         (png (png-with-dimensions (file-octets (fixture-path (nth (random (length pngs) state) pngs)))
                                   png-width png-height))
         (jpeg (jpeg-with-dimensions (file-octets (fixture-path (nth (random (length jpegs) state) jpegs)))
                                     jpeg-width jpeg-height)))
    (with-temporary-directory (dir)
      (flet ((too-large-without-allocating-p (name octets)
               (let* ((path (write-octets (merge-pathnames name dir) octets))
                      (before (sb-ext:get-bytes-consed))
                      (reason (decode-reason path)))
                 (and (eq :too-large reason)
                      (< (- (sb-ext:get-bytes-consed) before) (ash 1 24))))))
        (and (> (* png-width png-height) *max-image-pixels*)
             (> (* jpeg-width jpeg-height) *max-image-pixels*)
             (too-large-without-allocating-p "huge.png" png)
             (too-large-without-allocating-p "huge.jpg" jpeg))))))

;; JPEG のマーカーの並び（ITU-T T.81 の B.1.1）。SOF でないセグメントの中身は 0xFF で埋め、寸法として
;; 読み違えると巨大な画素数になるようにする。
(defparameter +sof-markers+ '(#xc0 #xc1 #xc2 #xc3 #xc5 #xc6 #xc7 #xc9 #xca #xcb #xcd #xce #xcf)
  "フレームの見出し（SOF0〜SOF15 のうち DHT・JPG・DAC 以外）のマーカー。")

(defparameter +non-sof-segment-markers+ '(#xc4 #xc8 #xcc #xdb #xdd #xe0 #xe1 #xef #xfe #x02 #xbf)
  "長さを持つ、SOF でないセグメントのマーカー（DHT・JPG・DAC・DQT・DRI・APPn・COM・予約）。")

(defparameter +standalone-markers+ '(#x01 #xd0 #xd1 #xd2 #xd3 #xd4 #xd5 #xd6 #xd7)
  "長さを持たない単独のマーカー（TEM・RSTn）。")

(defun synthetic-jpeg (state width height ending)
  "SOI、ランダムな単独のマーカーと SOF でないセグメント（どのマーカーの前にも 0〜2 個の詰め物の 0xFF）、
幅 WIDTH・高さ HEIGHT の SOF（種類はランダム）、EOI の並び。画像のデータは無い。ENDING が :eoi / :sos
なら SOF の前に EOI / SOS を置く（その後ろの SOF は見出しとして読まれない）。:truncated なら SOF の
幅の直後でファイルを切る。"
  (let ((out (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
    (labels ((emit (&rest octets) (dolist (octet octets) (vector-push-extend octet out)))
             (pick (list) (nth (random (length list) state) list))
             (marker (code)
               (loop repeat (random 3 state) do (emit #xff))
               (emit #xff code)))
      (emit #xff #xd8)
      (loop repeat (random 5 state)
            do (if (zerop (random 3 state))
                   (marker (pick +standalone-markers+))
                   (let ((payload (+ 5 (random 6 state))))
                     (marker (pick +non-sof-segment-markers+))
                     (emit (ldb (byte 8 8) (+ 2 payload)) (ldb (byte 8 0) (+ 2 payload)))
                     (loop repeat payload do (emit #xff)))))
      (ecase ending
        (:eoi (marker #xd9))
        (:sos (marker #xda) (emit 0 8 1 1 0 0 63 0))
        ((:sof :truncated)))
      (marker (pick +sof-markers+))
      (emit 0 17 8 (ldb (byte 8 8) height) (ldb (byte 8 0) height) (ldb (byte 8 8) width) (ldb (byte 8 0) width))
      (unless (eq ending :truncated)
        (emit 3 1 #x11 0 2 #x11 1 3 #x11 1)
        (marker #xd9)))
    (coerce out '(simple-array (unsigned-byte 8) (*)))))

(def-image-property image/jpeg-pixel-limit-reads-the-first-frame-header
  "JPEG の上限は、SOI の後の単独のマーカー・詰め物の 0xFF・SOF でないセグメントを飛ばして最初の SOF の
寸法で見る: 幅 × 高さ − 1 を上限にすると :too-large、幅 × 高さなら :too-large でない（SOF の幅の直後で
切れていても同じ）。SOF より前に EOI か SOS があれば、上限 0 でも :too-large にならない。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (width (1+ (random 300 state)))
         (height (1+ (random 300 state)))
         (ending (nth (random 4 state) '(:sof :truncated :eoi :sos)))
         (octets (synthetic-jpeg state width height ending)))
    (with-temporary-directory (dir)
      (let ((path (write-octets (merge-pathnames "synthetic.jpg" dir) octets)))
        (flet ((reason (limit)
                 (handler-case (progn (read-image path :max-pixels limit) :no-error)
                   (image-decode-error (c) (image-decode-error-reason c)))))
          (if (member ending '(:eoi :sos))
              (not (eq :too-large (reason 0)))
              (and (eq :too-large (reason (1- (* width height))))
                   (not (eq :too-large (reason (* width height)))))))))))

(test (image/default-pixel-limit-is-pillows :suite :nabla.small)
  "既定の上限は Pillow の Image.MAX_IMAGE_PIXELS（1024 × 1024 × 1024 / 4 / 3 画素）と同じ。"
  (is (= (floor (* 1024 1024 1024) (* 4 3)) *max-image-pixels*)))

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
