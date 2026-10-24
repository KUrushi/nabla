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

(defun %copy-octets (from from-start to to-start count)
  "FROM（特殊化したベクタ）の FROM-START 番目のバイトから COUNT バイトを、
TO の TO-START 番目のバイトへコピーする（両方とも要素のバイト位置ではなくバイト単位）。"
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

(defun %array-octets (array dtype)
  "ARRAY の要素のバイト列（DTYPE のリトルエンディアン表現）を返す。"
  (let* ((count (array-total-size array))
         (octets (make-array (* count (nabla:dtype-byte-width dtype)) :element-type '(unsigned-byte 8))))
    (if (eq dtype :i1)
        (dotimes (i count) (setf (aref octets i) (row-major-aref array i)))
        (let ((storage (%storage-vector array)))
          (%copy-octets storage 0 octets 0 (length octets))))
    octets))

(defun %octets-array (octets start dtype shape path)
  "OCTETS の START から始まる、DTYPE・SHAPE の配列を作る。"
  (let ((array (make-array shape :element-type (nabla:dtype-element-type dtype))))
    (if (eq dtype :i1)
        (dotimes (i (array-total-size array))
          (let ((byte (aref octets (+ start i))))
            (when (> byte 1)
              (%fail path :invalid-bool "BOOL のテンソルに 0 / 1 以外のバイト ~D がある" byte))
            (setf (row-major-aref array i) byte)))
        (%copy-octets octets start (sb-ext:array-storage-vector array) 0
                      (* (array-total-size array) (nabla:dtype-byte-width dtype))))
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
    (let* ((blobs (mapcar (lambda (e) (%array-octets (second e) (third e))) normalized))
           (header (with-output-to-string (out)
                     (write-char #\{ out)
                     (let ((first-member t) (offset 0))
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
                               for blob in blobs
                               do (comma)
                                  (%json-write-string name out)
                                  (format out ":{\"dtype\":\"~A\",\"shape\":[~{~D~^,~}],\"data_offsets\":[~D,~D]}"
                                          (cdr (assoc dtype +dtype-names+))
                                          (array-dimensions array) offset (+ offset (length blob)))
                                  (incf offset (length blob)))))
                     (write-char #\} out)))
           (header-octets (sb-ext:string-to-octets header :external-format :utf-8))
           (padded-length (* 8 (ceiling (length header-octets) 8)))
           (data-length (reduce #'+ blobs :key #'length))
           (result (make-array (+ 8 padded-length data-length) :element-type '(unsigned-byte 8)
                                                                  :initial-element 32)))
      (dotimes (i 8) (setf (aref result i) (ldb (byte 8 (* 8 i)) padded-length)))
      (replace result header-octets :start1 8)
      (let ((position (+ 8 padded-length)))
        (dolist (blob blobs)
          (replace result blob :start1 position)
          (incf position (length blob))))
      result)))

(defun write-safetensors (path entries &key metadata)
  "ENTRIES（ENCODE-SAFETENSORS と同じ）を PATH に safetensors として書く。
NABLA:WITH-ATOMIC-OUTPUT-FILE で一時ファイルに書いてから fsync して rename し、
親ディレクトリも fsync する（issue #263）。読む側は書きかけのファイルを見ず、
rename までに失敗すれば PATH の既存のファイルは変わらない。正常に戻った後は、
プロセスの失敗に加えて OS のクラッシュ・電源断の後も PATH は完全な新しい中身に
なる（macOS では NABLA:*FULL-FSYNC* を真にしない限り、ドライブの書き込み
キャッシュは対象外）。書き込んだファイルのパス名（PATH を現在のディレクトリを
基準に絶対化したもの）を返す。"
  (let ((octets (encode-safetensors entries :metadata metadata)))
    (nabla:with-atomic-output-file (stream path)
      (write-sequence octets stream))))

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
  (let* ((octets (coerce octets 'octet-vector))
         (length (length octets)))
    (when (< length 8)
      (%fail path :truncated "~D バイトしかなく、ヘッダ長の 8 バイトにも満たない" length))
    (let ((header-length (loop for i below 8 sum (ash (aref octets i) (* 8 i)))))
      (when (or (> header-length (- length 8)) (> header-length +max-header-length+))
        (%fail path :header-length "ヘッダ長 ~D が範囲外（ファイルは ~D バイト）" header-length length))
      (let* ((data-start (+ 8 header-length))
             (header (handler-case
                         (%json-parse (sb-ext:octets-to-string
                                       octets :start 8 :end data-start
                                              :external-format '(:utf-8 :replacement nil)))
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
        (values
         (loop for (name dtype shape begin) in tensors
               collect (list name (%octets-array octets (+ data-start begin) dtype shape path) dtype))
         metadata)))))

(defun read-safetensors (path)
  "PATH の safetensors ファイルを読み、(values entries metadata) を返す
（DECODE-SAFETENSORS と同じ。エラーの PATH にこのファイルが入る）。"
  (let ((octets (with-open-file (stream path :element-type '(unsigned-byte 8))
                  (let ((octets (make-array (file-length stream) :element-type '(unsigned-byte 8))))
                    (unless (= (read-sequence octets stream) (length octets))
                      (%fail path :truncated "ファイルを最後まで読めなかった"))
                    octets))))
    (decode-safetensors octets :path path)))
