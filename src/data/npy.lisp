;;;; .npy の読み書きと npy-source（issue #282）。
;;;;
;;;; 形式（numpy の NEP 1、numpy/lib/format.py）:
;;;;   6 バイト: マジック #x93 "NUMPY"
;;;;   2 バイト: 版（major minor）。1.0 / 2.0 / 3.0
;;;;   ヘッダ長: 版 1.0 は u16、2.0 / 3.0 は u32（リトルエンディアン）
;;;;   ヘッダ: Python の dict リテラル {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
;;;;     を空白で詰めて改行で終えたもの（1.0 / 2.0 は latin-1、3.0 は UTF-8。読めるのは ASCII だけ）。
;;;;   残り: データ（row-major の生バイト列）
;;;;
;;;; ファイル全体を配列に読む（大きなデータセットの遅延読み込みは測ってから。設計レビュー D3）。
;;;; ヘッダは 3 つのキーだけを読む自前の最小のパーサで読む（nabla/nn の JSON は形式が違い、
;;;; data は nn に依存しない）。対応は C 順・リトルエンディアンだけで、配列のメモリ表現は
;;;; リトルエンディアンのホストを前提にビット列をそのままコピーする（NaN のペイロードも保つ）。

(in-package #:nabla.data)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (member :little-endian *features*)
    (error "nabla/data の .npy はリトルエンディアンのホストだけに対応している")))

(define-condition npy-error (data-error)
  ((path :initarg :path :initform nil :reader npy-error-path
         :documentation "読み書きしていたファイル。")
   (reason :initarg :reason :reader npy-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader npy-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "npy~@[ (~A)~]: ~A (~S)"
                     (npy-error-path c) (npy-error-detail c) (npy-error-reason c))))
  (:documentation
   ".npy を読めない・書けないときに signal する。REASON は次のどれか。
  :invalid-header      マジックが違う・ヘッダが途中で終わる・dict として読めない・キーの過不足や
                       重複・shape が非負の整数のタプルでない
  :unsupported-version 版が 1.0 / 2.0 / 3.0 でない
  :fortran-order       fortran_order が True（C 順だけに対応する）
  :big-endian          ビッグエンディアンの dtype（'>f4' など）
  :unsupported-dtype   対応しない dtype（object・complex・構造化など）。書き込みでは、対応する
                       dtype の無い element-type
  :size-mismatch       データの長さが shape × 要素のバイト数と合わない
  :invalid-bool        bool のバイトが 0 / 1 でない"))

(defun %npy-error (path reason control &rest arguments)
  (error 'npy-error :path path :reason reason :detail (apply #'format nil control arguments)))

(defparameter +npy-dtypes+
  '(("<f4" single-float 4) ("<f8" double-float 8) ("|i1" (signed-byte 8) 1)
    ("<i4" (signed-byte 32) 4) ("<i8" (signed-byte 64) 8) ("|u1" (unsigned-byte 8) 1)
    ("<u4" (unsigned-byte 32) 4) ("<u8" (unsigned-byte 64) 8) ("|b1" bit 1))
  "対応する dtype（descr 要素型 要素のバイト数）。bool（|b1）は (unsigned-byte 8) の 0 / 1 で読み、
bit の配列を bool として書く。")

(defparameter +npy-magic+ #(#x93 78 85 77 80 89)
  "マジック #x93 \"NUMPY\"。")

;;; --- ヘッダ ---

(defun %parse-npy-header (text path)
  "ヘッダの文字列 TEXT を読み、(values descr fortran-order shape) を返す。"
  (let ((position 0) (entries '()))
    (labels ((fail (control &rest arguments)
               (apply #'%npy-error path :invalid-header
                      (concatenate 'string "ヘッダ ~S の ~D 文字目: " control) text position arguments))
             (peek ()
               (loop while (and (< position (length text))
                                (member (char text position) '(#\Space #\Tab #\Newline #\Return)))
                     do (incf position))
               (and (< position (length text)) (char text position)))
             (accept (char) (when (eql (peek) char) (incf position)))
             (expect (char) (unless (accept char) (fail "~S が無い" char)))
             (parse-token (predicate what)
               (peek)
               (let ((end (or (position-if-not predicate text :start position) (length text))))
                 (when (= end position) (fail "~A が無い" what))
                 (prog1 (subseq text position end) (setf position end))))
             (parse-string ()
               (let* ((delimiter (peek))
                      (end (and (member delimiter '(#\' #\"))
                                (position delimiter text :start (1+ position)))))
                 (unless end (fail "文字列が無い、または閉じていない"))
                 (prog1 (subseq text (1+ position) end) (setf position (1+ end)))))
             (parse-dimension ()
               (prog1 (parse-integer (parse-token (lambda (c) (char<= #\0 c #\9)) "非負の整数"))
                 (accept #\L)))         ; Python 2 の numpy が書いた長整数 (3L, 4L)
             (parse-shape ()
               (expect #\()
               (if (accept #\))
                   '()
                   (let ((dimensions (list (parse-dimension))))
                     (expect #\,)       ; 1 要素のタプルは (3,)。(3) は整数
                     (loop until (accept #\))
                           do (push (parse-dimension) dimensions)
                              (unless (accept #\,) (expect #\)) (loop-finish)))
                     (nreverse dimensions))))
             (parse-value (key)
               (cond ((string= key "descr")
                      (when (eql (peek) #\[)
                        (%npy-error path :unsupported-dtype "構造化 dtype には対応していない: ~S" text))
                      (parse-string))
                     ((string= key "fortran_order")
                      (let ((word (parse-token #'alpha-char-p "True / False")))
                        (cond ((string= word "True") t)
                              ((string= word "False") nil)
                              (t (fail "fortran_order ~S は True / False でない" word)))))
                     ((string= key "shape") (parse-shape))
                     (t (fail "知らないキー ~S" key)))))
      (expect #\{)
      (loop until (accept #\})
            do (let ((key (parse-string)))
                 (when (assoc key entries :test #'string=) (fail "キー ~S が重複している" key))
                 (expect #\:)
                 (push (cons key (parse-value key)) entries)
                 (unless (accept #\,) (expect #\}) (loop-finish))))
      (when (peek) (fail "dict の後に余分な文字がある"))
      (unless (= 3 (length entries)) (fail "descr / fortran_order / shape のどれかが無い"))
      (flet ((value (key) (cdr (assoc key entries :test #'string=))))
        (values (value "descr") (value "fortran_order") (value "shape"))))))

(defun %npy-dtype (descr fortran-order shape path)
  "ヘッダの値を検査して、+NPY-DTYPES+ の項目を返す。"
  (let* ((one-octet (and (= 3 (length descr)) (find (char descr 0) "<>=|")
                         (member (subseq descr 1) '("i1" "u1" "b1") :test #'string=)))
         ;; 1 バイトの dtype にバイト順は無い（numpy 以外は '<u1' と書くことがある）
         (dtype (assoc (if one-octet (format nil "|~A" (subseq descr 1)) descr)
                       +npy-dtypes+ :test #'string=)))
    (cond (dtype)
          ((and (plusp (length descr)) (char= #\> (char descr 0))
                (assoc (substitute #\< #\> descr :count 1) +npy-dtypes+ :test #'string=))
           (%npy-error path :big-endian "ビッグエンディアンの dtype ~S には対応していない" descr))
          (t (%npy-error path :unsupported-dtype "dtype ~S には対応していない" descr)))
    ;; 1 を超える軸が 1 つ以下なら、Fortran 順と C 順のバイト列は同じ
    (when (and fortran-order (> (count-if (lambda (d) (> d 1)) shape) 1))
      (%npy-error path :fortran-order "Fortran 順の配列には対応していない"))
    (unless (and (< (length shape) array-rank-limit)
                 (every (lambda (d) (< d array-dimension-limit)) shape)
                 (< (reduce #'* shape) array-total-size-limit))
      (%npy-error path :invalid-header "shape ~S は SBCL の配列の大きさの上限を超える" shape))
    dtype))

(defun %npy-header-octets (descr shape)
  "版 1.0 のマジックからヘッダの改行までのバイト列。numpy（_write_array_header / _wrap_header）と
同じバイト列にする: dict の後に、先頭の軸を追記で伸ばせるよう (21 - 先頭の次元の桁数) 個の空白を置き、
データの開始位置が 64 の倍数になるよう 1〜64 個の空白と改行で詰める（rank は ARRAY-RANK-LIMIT 未満
なので、ヘッダ長は u16 に必ず収まる）。"
  (let* ((dict (format nil "{'descr': '~A', 'fortran_order': False, 'shape': (~{~D~^, ~}~:[~;,~]), }~vA"
                       descr shape (= 1 (length shape))
                       (if shape (- 21 (length (princ-to-string (first shape)))) 0) ""))
         (padding (- 64 (mod (+ 10 (length dict) 1) 64)))
         (text (format nil "~A~vA~%" dict padding ""))
         (length (length text)))
    (concatenate '(vector (unsigned-byte 8))
                 +npy-magic+ (list 1 0 (ldb (byte 8 0) length) (ldb (byte 8 8) length))
                 (map 'list #'char-code text))))

;;; --- データ ---

(defun %octet-length (vector)
  "特殊化した 1 次元のベクタ VECTOR の記憶域のバイト数。"
  (* (length vector)
     (etypecase vector
       ((or (simple-array (unsigned-byte 8) (*)) (simple-array (signed-byte 8) (*))) 1)
       ((or (simple-array single-float (*)) (simple-array (signed-byte 32) (*))
            (simple-array (unsigned-byte 32) (*)))
        4)
       ((or (simple-array double-float (*)) (simple-array (signed-byte 64) (*))
            (simple-array (unsigned-byte 64) (*)))
        8))))

(defun %copy-octets (from from-start to to-start count)
  "特殊化したベクタ FROM の FROM-START バイト目から COUNT バイトを、TO の TO-START バイト目へ
コピーする。コピー自体は範囲を検査しないので、範囲外ならコピーする前に error にする（記憶域を壊さない）。"
  (unless (and (<= 0 from-start) (<= 0 to-start) (<= 0 count)
               (<= (+ from-start count) (%octet-length from))
               (<= (+ to-start count) (%octet-length to)))
    (error "%copy-octets: 範囲外のコピー（~D バイト目から ~D バイトを ~D バイトの配列の ~D バイト目へ）"
           from-start count (%octet-length to) to-start))
  (sb-sys:with-pinned-objects (from to)
    (sb-kernel:system-area-ub8-copy (sb-sys:vector-sap from) from-start
                                    (sb-sys:vector-sap to) to-start count)))

(defparameter +chunk-octets+ 65536
  "データを読み書きするときの作業用バッファの大きさ（バイト）。")

(defun %read-npy-data (stream array itemsize path)
  "STREAM の今の位置から ARRAY の記憶域へ、作業用バッファ 1 本を通してデータを読む。"
  (let ((storage (sb-ext:array-storage-vector array))
        (buffer (make-array +chunk-octets+ :element-type '(unsigned-byte 8)))
        (total (* (array-total-size array) itemsize)))
    (loop for done from 0 below total by +chunk-octets+
          for count = (min +chunk-octets+ (- total done))
          do (unless (= count (read-sequence buffer stream :end count))
               (%npy-error path :size-mismatch "データが途中で終わった（読んでいる間に短くなった）"))
             (%copy-octets buffer 0 storage done count))))

(defun %write-npy-data (stream array itemsize)
  "ARRAY の要素のバイト列を、作業用バッファ 1 本を通して STREAM に書く。"
  (let* ((bitp (eq 'bit (array-element-type array)))
         (storage (unless bitp
                    (sb-ext:array-storage-vector
                     (if (typep array 'simple-array)
                         array
                         (let ((copy (make-array (array-dimensions array)
                                                 :element-type (array-element-type array))))
                           (dotimes (i (array-total-size array) copy)
                             (setf (row-major-aref copy i) (row-major-aref array i))))))))
         (buffer (make-array +chunk-octets+ :element-type '(unsigned-byte 8)))
         (total (* (array-total-size array) itemsize)))
    (loop for done from 0 below total by +chunk-octets+
          for count = (min +chunk-octets+ (- total done))
          do (if bitp
                 (dotimes (i count) (setf (aref buffer i) (row-major-aref array (+ done i))))
                 (%copy-octets storage done buffer 0 count))
             (write-sequence buffer stream :end count))))

;;; --- 公開する関数 ---

(defun read-npy (path)
  "PATH の .npy ファイルを読み、配列を返す。ヘッダの版 1.0 / 2.0 / 3.0、C 順、リトルエンディアンの
f32 / f64 / i8 / i32 / i64 / u8 / u32 / u64 / bool（descr '<f4' '<f8' '|i1' '<i4' '<i8' '|u1'
'<u4' '<u8' '|b1'）に対応し、要素型はそれぞれ single-float / double-float / (signed-byte 8 / 32 / 64) /
\(unsigned-byte 8 / 32 / 64)。bool は bit 配列でなく (unsigned-byte 8) の 0 / 1 で返す。
u8・i8・bool・i64 の配列は nabla の dtype に無い（NABLA:ARRAY-DTYPE が DTYPE-MISMATCH）ので、
jit や to-device に渡す前に変換（ローダの map など）で f32 / i32 などに直す。
rank 0 の配列も読める。ファイル全体を配列に読む。1 バイトの dtype はバイト順の記号（'<u1' など）を問わず、
fortran_order が True でも 1 を超える軸が 1 つ以下なら（バイト列が C 順と同じなので）読む。
形式が読めないファイルは NPY-ERROR（理由はその docstring）。ファイルが無いなどは CL の FILE-ERROR。
相対パスは WRITE-NPY と同じく現在のディレクトリ（UIOP:GETCWD）を基準にする。"
  (with-open-file (stream (merge-pathnames path (uiop:getcwd)) :element-type '(unsigned-byte 8))
    (let ((preamble (make-array 12 :element-type '(unsigned-byte 8))))
      (unless (and (= 10 (read-sequence preamble stream :end 10))
                   (equalp +npy-magic+ (subseq preamble 0 6)))
        (%npy-error path :invalid-header "先頭が .npy のマジックで始まらない"))
      (let ((major (aref preamble 6)))
        (unless (and (<= 1 major 3) (zerop (aref preamble 7)))
          (%npy-error path :unsupported-version "版 ~D.~D には対応していない" major (aref preamble 7)))
        (unless (or (= major 1) (= 12 (read-sequence preamble stream :start 10)))
          (%npy-error path :invalid-header "ヘッダ長の途中でファイルが終わった"))
        (let* ((header-start (if (= major 1) 10 12))
               (header-length (loop for i from 8 below header-start
                                    sum (ash (aref preamble i) (* 8 (- i 8)))))
               (header (make-array (min header-length (file-length stream))
                                   :element-type '(unsigned-byte 8))))
          (unless (= header-length (read-sequence header stream))
            (%npy-error path :invalid-header "ヘッダ長 ~D がファイルの大きさを超える" header-length))
          (multiple-value-bind (descr fortran-order shape)
              ;; 版 1.0 / 2.0 のヘッダは latin-1、3.0 は UTF-8 だが、読めるヘッダは ASCII だけなので、
              ;; どの版も UTF-8 で読む（ASCII でないバイトはパーサが弾く）
              (%parse-npy-header (sb-ext:octets-to-string header :external-format '(:utf-8 :replacement #\?))
                                 path)
            (destructuring-bind (descr element-type itemsize) (%npy-dtype descr fortran-order shape path)
              ;; 配列を確保する前にファイルの大きさと照らす（壊れた shape で巨大な確保をしない）
              (let ((data-length (- (file-length stream) header-start header-length)))
                (unless (= data-length (* (reduce #'* shape) itemsize))
                  (%npy-error path :size-mismatch "データの長さ ~D バイトが shape ~S × ~D バイトと合わない"
                              data-length shape itemsize)))
              (let ((array (make-array shape :element-type (if (eq element-type 'bit)
                                                               '(unsigned-byte 8)
                                                               element-type))))
                (%read-npy-data stream array itemsize path)
                (when (and (string= descr "|b1")
                           (find-if (lambda (x) (> x 1)) (sb-ext:array-storage-vector array)))
                  (%npy-error path :invalid-bool "bool の配列に 0 / 1 以外のバイトがある"))
                array))))))))

(defun write-npy (path array)
  "ARRAY を PATH に .npy（版 1.0、C 順、リトルエンディアン）で書き、書いたファイルのパス名を返す。
要素型は READ-NPY が返すもの（single-float / double-float / (signed-byte 8 / 32 / 64) /
\(unsigned-byte 8 / 32 / 64)）と、bool として書く bit。それ以外（(unsigned-byte 16) や t など）は
:unsupported-dtype の NPY-ERROR で、ファイルを作らない。ヘッダは numpy と同じく、データの開始位置が
64 バイトの倍数になるよう空白で詰める。NABLA:WITH-ATOMIC-OUTPUT-FILE で書くので、読む側は書きかけの
ファイルを見ず、失敗すれば PATH の既存のファイルは変わらない。"
  (let ((dtype (find (array-element-type array) +npy-dtypes+ :key #'second :test #'equal)))
    (unless dtype
      (%npy-error path :unsupported-dtype "要素型 ~S の配列は .npy に書けない" (array-element-type array)))
    (destructuring-bind (descr element-type itemsize) dtype
      (declare (ignore element-type))
      (nabla:with-atomic-output-file (stream path)
        (write-sequence (%npy-header-octets descr (array-dimensions array)) stream)
        (%write-npy-data stream array itemsize)))))

(defun npy-source (paths)
  "PATHS（葉が .npy のパス名（文字列か pathname）の PyTree）の各ファイルを READ-NPY で読み、その配列の
木から MAKE-ARRAY-SOURCE で作った ARRAY-SOURCE を返す。INDEX 番目のレコードは、各葉を
そのファイルの配列の先頭の軸の INDEX 番目の行に置き換えた同じ形の木。
例: (npy-source (list \"x.npy\" \"y.npy\")) のレコードは (x の行 y の行)。
各ファイルの配列は rank 1 以上で、先頭の軸の長さが揃っている必要がある（rank 0・長さの食い違いは
MAKE-ARRAY-SOURCE と同じ DATA-SOURCE-ERROR）。パス名でない葉は :bad-leaf の DATA-SOURCE-ERROR。
u8・bool のレコードは nabla の dtype に無いので、jit や to-device に渡す前にローダの map で
f32 / i32 などに直す。"
  (multiple-value-bind (leaves treedef) (nabla:tree-flatten paths)
    (make-array-source
     (nabla:tree-unflatten treedef
                           (mapcar (lambda (path)
                                     (unless (typep path '(or string pathname))
                                       (%data-source-error :bad-leaf "葉 ~S が .npy のパス名でない" path))
                                     (read-npy path))
                                   leaves)))))
