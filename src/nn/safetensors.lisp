;;;; safetensors の読み書き（平坦な名前 → 配列。issue #198）。
;;;;
;;;; 形式（https://github.com/huggingface/safetensors）:
;;;;   先頭 8 バイト: JSON ヘッダのバイト長 N（u64 リトルエンディアン）
;;;;   次の N バイト: UTF-8 の JSON ヘッダ
;;;;     {"名前": {"dtype": "F32", "shape": [2, 3], "data_offsets": [開始, 終了]}, ...,
;;;;      "__metadata__": {"キー": "値", ...}}
;;;;     data_offsets はデータ領域の先頭からのバイト位置。
;;;;   残り: データ領域（各テンソルの生バイト列をリトルエンディアンで隙間なく並べたもの）
;;;;
;;;; JSON ライブラリ（apt の cl-yason）を使わず、ヘッダに必要な部分だけを自前で読み書きする
;;;; （src/nn/json.lisp）。理由: (1) ヘッダは「文字列・整数・配列・オブジェクト」だけで
;;;; 足りるので 100 行ほどで済む。(2) 依存を増やすと scripts/setup-lisp-deps.sh の APT_PACKAGES が
;;;; 変わり CI の依存キャッシュが無効になる。(3) 壊れたファイル（重複キー、整数以外の数値、
;;;; 対の無いサロゲートなど）の扱いを safetensors の仕様に合わせて自分で決められる。
;;;; 一方、自前のパーサは JSON の全仕様を読めない（小数・指数は読まずに
;;;; :invalid-header にする）。safetensors のヘッダには現れない。
;;;;
;;;; 配列のメモリ表現はリトルエンディアンのホストを前提にし（x86-64 / arm64）、
;;;; ビット列をそのままコピーする（NaN のペイロードも保つ）。:i1（BOOL）は 1 要素 1 バイト。

(in-package #:nabla.nn)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (member :little-endian *features*)
    (error "nabla/nn の safetensors はリトルエンディアンのホストだけに対応している")))

(define-condition safetensors-error (error)
  ((path :initarg :path :initform nil :reader safetensors-error-path
         :documentation "読んでいたファイル。バイト列から読んだときは NIL。")
   (reason :initarg :reason :reader safetensors-error-reason
           :documentation "原因を表すキーワード。")
   (detail :initarg :detail :reader safetensors-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "safetensors~@[ (~A)~]: ~A (~S)"
                     (safetensors-error-path c) (safetensors-error-detail c)
                     (safetensors-error-reason c))))
  (:documentation
   "safetensors の読み書きで、入力が仕様に合わないときに signal する。
REASON は次のどれか。読み込み:
  :truncated       8 バイトに満たない
  :header-length   ヘッダ長がファイルの大きさを超える、または 100 MB を超える
  :invalid-header  ヘッダが UTF-8 でない・JSON として不正・形式に合わない
                   （項目の欠落、負の shape、メタデータが文字列でない、キーの重複など）
  :unknown-dtype   対応していない dtype 文字列（F32 F64 BF16 F16 I32 U32 U64 BOOL 以外）
  :size-mismatch   data_offsets の幅が shape × 要素のバイト数と合わない
  :bad-offsets     テンソルのデータ範囲が重なる・隙間がある・先頭が 0 でない・
                   終了が開始より小さい・データ領域をはみ出す、または余る
  :invalid-bool    BOOL のバイトが 0 / 1 でない
書き込み:
  :invalid-entries 名前の重複、予約名 __metadata__、UTF-8 にできない名前、
                   文字列でないメタデータ
書き込みで配列の要素型と dtype が合わないときは NABLA:DTYPE-MISMATCH になる。
PATH は READ-SAFETENSORS で読んでいたファイル、DETAIL は説明の文字列。"))

(defun %fail (path reason control &rest args)
  (error 'safetensors-error :path path :reason reason
                            :detail (apply #'format nil control args)))

(defparameter +dtype-names+
  '((:f32 . "F32") (:f64 . "F64") (:bf16 . "BF16") (:f16 . "F16")
    (:i32 . "I32") (:u32 . "U32") (:u64 . "U64") (:i1 . "BOOL"))
  "nabla の dtype と safetensors の dtype 文字列の対応。")

(defparameter +max-header-length+ 100000000
  "ヘッダ長の上限（参照実装と同じ 100 MB）。壊れた長さで巨大な確保をしないため。")

(deftype octet-vector () '(simple-array (unsigned-byte 8) (*)))

;;; --- バイト列と配列の変換 ---

(defun %octet-length (vector)
  "特殊化した 1 次元のベクタ VECTOR の記憶域のバイト数。"
  (* (length vector)
     (etypecase vector
       ((simple-array (unsigned-byte 8) (*)) 1)
       ((simple-array (unsigned-byte 16) (*)) 2)
       ((or (simple-array single-float (*)) (simple-array (signed-byte 32) (*))
            (simple-array (unsigned-byte 32) (*)))
        4)
       ((or (simple-array double-float (*)) (simple-array (unsigned-byte 64) (*))) 8))))

(defun %copy-octets (from from-start to to-start count)
  "FROM（特殊化したベクタ）の FROM-START 番目のバイトから COUNT バイトを、
TO の TO-START 番目のバイトへコピーする（両方とも要素のバイト位置ではなくバイト単位）。
コピー自体は範囲を検査しないので、範囲外ならコピーする前に error にする（記憶域を壊さない）。"
  (unless (and (<= 0 from-start) (<= 0 to-start) (<= 0 count)
               (<= (+ from-start count) (%octet-length from))
               (<= (+ to-start count) (%octet-length to)))
    (error "%copy-octets: 範囲外のコピー（~D バイト目から ~D バイトを ~D バイトの配列へ ~D バイト目から）"
           from-start count (%octet-length to) to-start))
  (when (plusp count)
    (sb-sys:with-pinned-objects (from to)
      (sb-kernel:system-area-ub8-copy (sb-sys:vector-sap from) from-start
                                      (sb-sys:vector-sap to) to-start count))))

(defun %storage-vector (array)
  "ARRAY の要素を row-major の順に持つ、特殊化した 1 次元のベクタ（コピーかもしれない）。"
  (if (typep array 'simple-array)
      (sb-ext:array-storage-vector array)
      (let ((copy (make-array (array-dimensions array) :element-type (array-element-type array))))
        (dotimes (i (array-total-size array))
          (setf (row-major-aref copy i) (row-major-aref array i)))
        (sb-ext:array-storage-vector copy))))

(defparameter +chunk-octets+ 65536
  "葉のバイト列を読み書きするときの作業用バッファの大きさ（バイト）。ファイル全体や葉全体の
バイト列を作らず、この大きさずつ配列の記憶域とやり取りする（issue #262）。")

(defun %chunk-buffer ()
  (make-array +chunk-octets+ :element-type '(unsigned-byte 8)))

(defun %emit-array-octets (array dtype buffer emit)
  "ARRAY の要素のバイト列（DTYPE のリトルエンディアン表現）を BUFFER に区切って詰め、
\(funcall EMIT BUFFER 詰めたバイト数) で先頭から順に渡す。"
  (let ((total (* (array-total-size array) (nabla:dtype-byte-width dtype)))
        (storage (unless (eq dtype :i1) (%storage-vector array))))
    (loop for done from 0 below total by (length buffer)
          for count = (min (length buffer) (- total done))
          do (if storage
                 (%copy-octets storage done buffer 0 count)
                 (dotimes (i count) (setf (aref buffer i) (row-major-aref array (+ done i)))))
             (funcall emit buffer count))))

(defun %array-from-chunks (dtype shape buffer fill path)
  "DTYPE・SHAPE の配列を作り、その要素のバイト列を BUFFER に区切って
\(funcall FILL BUFFER バイト数) に先頭から順に埋めさせる。"
  (let* ((array (make-array shape :element-type (nabla:dtype-element-type dtype)))
         (total (* (array-total-size array) (nabla:dtype-byte-width dtype)))
         (storage (unless (eq dtype :i1) (sb-ext:array-storage-vector array))))
    (loop for done from 0 below total by (length buffer)
          for count = (min (length buffer) (- total done))
          do (funcall fill buffer count)
             (if storage
                 (%copy-octets buffer 0 storage done count)
                 (dotimes (i count)
                   (let ((byte (aref buffer i)))
                     (when (> byte 1)
                       (%fail path :invalid-bool "BOOL のテンソルに 0 / 1 以外のバイト ~D がある" byte))
                     (setf (row-major-aref array (+ done i)) byte)))))
    array))

;;; --- 書き込み ---

(defun %proper-list-p (object)
  (and (listp object) (handler-case (integerp (list-length object)) (error () nil))))

(defun %normalize-entry (entry)
  "ENTRY の (name array [dtype]) を (name array dtype) にする。"
  (unless (and (%proper-list-p entry) (<= 2 (length entry) 3) (arrayp (second entry)))
    (%fail nil :invalid-entries "エントリは (名前 配列 [dtype]) でなければならない: ~S" entry))
  (destructuring-bind (name array &optional dtype) entry
    (list name array (nabla:array-dtype array dtype))))

(defun %check-names (names metadata)
  (let ((seen (make-hash-table :test 'equal)))
    (dolist (name names)
      (unless (stringp name)
        (%fail nil :invalid-entries "名前が文字列でない: ~S" name))
      (when (string= name "__metadata__")
        (%fail nil :invalid-entries "名前 __metadata__ は予約されている"))
      (when (gethash name seen)
        (%fail nil :invalid-entries "名前 ~S が重複している" name))
      (setf (gethash name seen) t)))
  (unless (and (%proper-list-p metadata) (every #'consp metadata))
    (%fail nil :invalid-entries "メタデータは (キー . 値) のリストでなければならない: ~S" metadata))
  (dolist (string (append names (loop for (key . value) in metadata collect key collect value)))
    (unless (stringp string)
      (%fail nil :invalid-entries "メタデータのキーと値は文字列でなければならない: ~S" string))
    (when (find-if (lambda (c) (<= #xD800 (char-code c) #xDFFF)) string)
      (%fail nil :invalid-entries "UTF-8 にできない文字（サロゲート）を含む文字列: ~S" string))))

(defun encode-safetensors (entries &key metadata)
  "ENTRIES を safetensors のバイト列（(simple-array (unsigned-byte 8) (*))）にする。

ENTRIES は (名前 配列 [dtype]) のリスト。dtype は :f32 :f64 :bf16 :f16 :i32 :u32 :u64 :i1
のどれか。省略すると配列の要素型から決める。(unsigned-byte 16) の配列は bf16 と
f16 のどちらか決められないので dtype が必須（省略すると NABLA:DTYPE-MISMATCH）。
:i1 は BOOL。METADATA は (キー . 値)（どちらも文字列）のリストで、__metadata__ に書く。

テンソルは要素のバイト数の大きい順、同じなら名前の順に並べて書く（入力の順序に
依らず同じバイト列になり、各テンソルはその要素の大きさに整列する。この順序は Python の
safetensors の書く順序とは限らず、保証するのは読み書きの意味が同じことだけでバイト列の一致ではない）。ヘッダは空白で
8 バイト境界まで埋める。名前の重複・予約名 __metadata__・サロゲートを含む文字列・
文字列でないメタデータは SAFETENSORS-ERROR（reason :invalid-entries）。"
  (let* ((plan (%encoding-plan entries metadata))
         (result (make-array (+ 8 (third plan) (fourth plan)) :element-type '(unsigned-byte 8)))
         (position 0))
    (%emit-safetensors plan (lambda (octets count)
                              (replace result octets :start1 position :end2 count)
                              (incf position count)))
    result))

(defun %encoding-plan (entries metadata)
  "ENTRIES と METADATA（ENCODE-SAFETENSORS と同じ）を検査し、書く内容
\(正規化したエントリ ヘッダのバイト列 空白で埋めた後のヘッダ長 データ領域の長さ) を返す。
葉のバイト列はまだ作らない。"
  (unless (%proper-list-p entries)
    (%fail nil :invalid-entries "ENTRIES はリストでなければならない: ~S" entries))
  (let ((normalized (mapcar #'%normalize-entry entries)))
    (%check-names (mapcar #'first normalized) metadata)
    (setf normalized
          (sort normalized
                (lambda (a b)
                  (let ((wa (nabla:dtype-byte-width (third a)))
                        (wb (nabla:dtype-byte-width (third b))))
                    (or (> wa wb) (and (= wa wb) (string< (first a) (first b))))))))
    (let* ((offset 0)
           (header (with-output-to-string (out)
                     (write-char #\{ out)
                     (let ((first-member t))
                       (flet ((comma () (if first-member (setf first-member nil) (write-char #\, out))))
                         (when metadata
                           (comma)
                           (write-string "\"__metadata__\":{" out)
                           (loop for ((key . value) . more) on metadata
                                 do (%json-write-string key out)
                                    (write-char #\: out)
                                    (%json-write-string value out)
                                    (when more (write-char #\, out)))
                           (write-char #\} out))
                         (loop for (name array dtype) in normalized
                               for size = (* (array-total-size array) (nabla:dtype-byte-width dtype))
                               do (comma)
                                  (%json-write-string name out)
                                  (format out ":{\"dtype\":\"~A\",\"shape\":[~{~D~^,~}],\"data_offsets\":[~D,~D]}"
                                          (cdr (assoc dtype +dtype-names+))
                                          (array-dimensions array) offset (+ offset size))
                                  (incf offset size))))
                     (write-char #\} out)))
           (header-octets (sb-ext:string-to-octets header :external-format :utf-8)))
      (list normalized header-octets (* 8 (ceiling (length header-octets) 8)) offset))))

(defun %emit-safetensors (plan emit)
  "%ENCODING-PLAN の PLAN のファイルのバイト列を、先頭から順に (funcall EMIT バイト列 バイト数) で
渡す。ヘッダは空白で 8 バイト境界まで埋め、葉は作業用バッファ 1 本を通して 1 つずつ渡す
\(全体や葉全体のバイト列を作らない。issue #262)。"
  (destructuring-bind (normalized header-octets padded-length data-length) plan
    (declare (ignore data-length))
    (let ((prefix (make-array 8 :element-type '(unsigned-byte 8))))
      (dotimes (i 8) (setf (aref prefix i) (ldb (byte 8 (* 8 i)) padded-length)))
      (funcall emit prefix 8))
    (funcall emit header-octets (length header-octets))
    (let ((padding (- padded-length (length header-octets))))
      (funcall emit (make-array padding :element-type '(unsigned-byte 8) :initial-element 32) padding))
    (let ((buffer (%chunk-buffer)))
      (loop for (nil array dtype) in normalized
            do (%emit-array-octets array dtype buffer emit)))))

(defun write-safetensors (path entries &key metadata)
  "ENTRIES（ENCODE-SAFETENSORS と同じ）を PATH に safetensors として書く。
NABLA:WITH-ATOMIC-OUTPUT-FILE で一時ファイルに書いてから fsync して rename し、
親ディレクトリも fsync する（issue #263）。読む側は書きかけのファイルを見ず、
rename までに失敗すれば PATH の既存のファイルは変わらない。正常に戻った後は、
プロセスの失敗に加えて OS のクラッシュ・電源断の後も PATH は完全な新しい中身に
なる（macOS では NABLA:*FULL-FSYNC* を真にしない限り、ドライブの書き込み
キャッシュは対象外）。書き込んだファイルのパス名（PATH を現在のディレクトリを
基準に絶対化したもの）を返す。"
  (let ((plan (%encoding-plan entries metadata)))
    (nabla:with-atomic-output-file (stream path)
      (%emit-safetensors plan (lambda (octets count) (write-sequence octets stream :end count))))))

;;; --- 読み込み ---

(defun %header-field (info key path name)
  (let ((member (assoc key (rest info) :test #'string=)))
    (unless member
      (%fail path :invalid-header "テンソル ~S に ~S が無い" name key))
    (cdr member)))

(defun %parse-tensor-info (name info path)
  "名前 NAME のヘッダの項目 INFO を検査して (dtype shape begin end) にする。"
  (unless (%json-object-p info)
    (%fail path :invalid-header "テンソル ~S の項目がオブジェクトでない" name))
  (let* ((dtype-name (%header-field info "dtype" path name))
         (shape (%header-field info "shape" path name))
         (offsets (%header-field info "data_offsets" path name))
         (dtype (car (rassoc dtype-name +dtype-names+ :test #'equal))))
    (unless (stringp dtype-name)
      (%fail path :invalid-header "テンソル ~S の dtype が文字列でない" name))
    (unless dtype
      (%fail path :unknown-dtype "テンソル ~S の dtype ~S には対応していない" name dtype-name))
    (unless (and (listp shape) (every (lambda (d) (and (integerp d) (>= d 0))) shape))
      (%fail path :invalid-header "テンソル ~S の shape は非負の整数のリストでなければならない" name))
    (unless (and (< (length shape) array-rank-limit)
                 (every (lambda (d) (< d array-dimension-limit)) shape)
                 (< (reduce #'* shape) array-total-size-limit))
      (%fail path :invalid-header "テンソル ~S の shape ~S は SBCL の配列の大きさの上限を超える" name shape))
    (unless (and (listp offsets) (= (length offsets) 2)
                 (every (lambda (o) (and (integerp o) (>= o 0))) offsets))
      (%fail path :invalid-header "テンソル ~S の data_offsets は非負の整数 2 つでなければならない" name))
    (destructuring-bind (begin end) offsets
      (when (< end begin)
        (%fail path :bad-offsets "テンソル ~S の data_offsets は終了 ~D が開始 ~D より小さい" name end begin))
      (unless (= (- end begin) (* (reduce #'* shape) (nabla:dtype-byte-width dtype)))
        (%fail path :size-mismatch "テンソル ~S のデータの長さ ~D バイトが shape ~S × ~A と合わない"
               name (- end begin) shape dtype-name))
      (list dtype shape begin end))))

(defun %parse-metadata (metadata path)
  (unless (and (%json-object-p metadata)
               (every (lambda (m) (stringp (cdr m))) (rest metadata)))
    (%fail path :invalid-header "__metadata__ は文字列から文字列へのオブジェクトでなければならない"))
  (rest metadata))

(defun decode-safetensors (octets &key path)
  "safetensors のバイト列 OCTETS を (values entries metadata) にする。

ENTRIES は (名前 配列 dtype) のリストで、ファイル内のデータの順。配列の要素型は
NABLA:DTYPE-ELEMENT-TYPE のとおり（bf16 / f16 は (unsigned-byte 16) のビット列、
BOOL は :i1 の bit 配列）。METADATA は (キー . 値) のリスト（無ければ NIL）。
入力が壊れていれば SAFETENSORS-ERROR（reason は同条件のドキュメントを参照）。
PATH はエラーに入れるファイル名（読み込みには使わない）。"
  (let ((octets (coerce octets 'octet-vector)))
    (%read-safetensors-from (length octets)
                            (lambda (position buffer count)
                              (replace buffer octets :start2 position :end1 count))
                            path)))

(defun %read-safetensors-from (length read-at path)
  "長さ LENGTH バイトの safetensors を読み、(values entries metadata) を返す。
\(funcall READ-AT 位置 バッファ バイト数) は、入力の「位置」からの「バイト数」バイトを
バッファの先頭に読む。ヘッダを読んで LENGTH と照らして検査してから、葉ごとに作業用バッファ
1 本を通して配列へ読むので、入力全体のバイト列を作らず、LENGTH の外を読まない（issue #262）。"
  (when (< length 8)
    (%fail path :truncated "~D バイトしかなく、ヘッダ長の 8 バイトにも満たない" length))
  (let ((header-length (let ((prefix (make-array 8 :element-type '(unsigned-byte 8))))
                         (funcall read-at 0 prefix 8)
                         (loop for i below 8 sum (ash (aref prefix i) (* 8 i))))))
    (when (or (> header-length (- length 8)) (> header-length +max-header-length+))
      (%fail path :header-length "ヘッダ長 ~D が範囲外（ファイルは ~D バイト）" header-length length))
    (let* ((data-start (+ 8 header-length))
           (header-octets (let ((octets (make-array header-length :element-type '(unsigned-byte 8))))
                            (funcall read-at 8 octets header-length)
                            octets))
           (header (handler-case
                       (%json-parse (sb-ext:octets-to-string
                                     header-octets :external-format '(:utf-8 :replacement nil)))
                     (%json-syntax-error (c)
                       (%fail path :invalid-header "ヘッダが JSON として不正: ~A" c))
                     (error (c)
                       (%fail path :invalid-header "ヘッダが UTF-8 の JSON として読めない: ~A" c))))
           (metadata '())
           (tensors '()))
      (unless (%json-object-p header)
        (%fail path :invalid-header "ヘッダが JSON のオブジェクトでない"))
      (dolist (member (rest header))
        (if (string= (car member) "__metadata__")
            (setf metadata (%parse-metadata (cdr member) path))
            (push (cons (car member) (%parse-tensor-info (car member) (cdr member) path))
                  tensors)))
      ;; データ領域を先頭から隙間なく、重なりなく覆っていること。
      ;; 大きさ 0 のテンソルが同じ位置に並ぶので、開始 → 終了の順に並べる（参照実装と同じ）。
      ;; この検査で、各テンソルの範囲が LENGTH の内側にあることも決まる（葉を読む前に確かめる）。
      (setf tensors (stable-sort (nreverse tensors)
                                 (lambda (a b) (or (< (fourth a) (fourth b))
                                                   (and (= (fourth a) (fourth b))
                                                        (< (fifth a) (fifth b)))))))
      (let ((expected 0))
        (loop for (name nil nil begin end) in tensors
              do (unless (= begin expected)
                   (%fail path :bad-offsets "テンソル ~S のデータが ~D から始まるが、直前の終わりは ~D"
                          name begin expected))
                 (setf expected end))
        (unless (= expected (- length data-start))
          (%fail path :bad-offsets "テンソルのデータの終わり ~D がデータ領域の長さ ~D と合わない"
                 expected (- length data-start))))
      (let ((buffer (%chunk-buffer)))
        (values
         (loop for (name dtype shape begin) in tensors
               collect (let ((position (+ data-start begin)))
                         (list name
                               (%array-from-chunks dtype shape buffer
                                                   (lambda (buffer count)
                                                     (funcall read-at position buffer count)
                                                     (incf position count))
                                                   path)
                               dtype)))
         metadata)))))

(defun read-safetensors (path)
  "PATH の safetensors ファイルを読み、(values entries metadata) を返す
（DECODE-SAFETENSORS と同じ。エラーの PATH にこのファイルが入る）。
ファイル全体を読み込まず、ヘッダを読んでからテンソルごとに配列へ読む（ピークは結果の配列 +
作業用バッファ。issue #262）。"
  (with-open-file (stream path :element-type '(unsigned-byte 8))
    (%read-safetensors-from (file-length stream)
                            (lambda (position buffer count)
                              ;; 今はテンソルを開始位置の順に続けて読むので位置は常に合っているが、
                              ;; READ-AT は任意の位置から読む約束（必要な葉だけ読む形を足すときに使う）。
                              (file-position stream position)
                              (unless (= (read-sequence buffer stream :end count) count)
                                (%fail path :truncated "ファイルが途中で終わった（読んでいる間に短くなった）")))
                            path)))
