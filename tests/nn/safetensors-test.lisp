;;;; safetensors の読み書き（issue #198）。
;;;;
;;;; 守らせる性質:
;;;; - 往復: encode → decode は、名前・dtype・形・値（ビット列まで）・メタデータを保つ
;;;;   （全 dtype、形に 0 と rank 0 を含む、名前は引用符・バックスラッシュ・制御文字・
;;;;   非 ASCII・BMP 外の文字を含む）。
;;;; - 入力の順序に依らず同じバイト列になる。データ領域の先頭は 8 バイト境界。
;;;; - 壊れた入力: 真の接頭辞や、1バイトを変えた入力は、SAFETENSORS-ERROR になるか
;;;;   正しく読めるかのどちらか（他のエラーにならず、落ちない）。
;;;; - 文書化した理由ごとの SAFETENSORS-ERROR（例ベース）。
;;;; - ファイル: 書いて読むと同じ。一時ファイルを残さず、失敗しても既存のファイルを壊さない。
;;;; - Python の safetensors が書いたファイルを読め、nabla が書いたファイルは
;;;;   tests/fixtures/safetensors/generate.py で Python が読めることを確かめてある（例ベース）。

(in-package #:nabla.nn.tests)

;;; --- 補助 ---

(defun element-bits (element)
  "ELEMENT のビット列を表す値（NaN を含めて比べるため）。"
  (etypecase element
    (single-float (sb-kernel:single-float-bits element))
    (double-float (list (sb-kernel:double-float-high-bits element)
                        (sb-kernel:double-float-low-bits element)))
    (integer element)))

(defun array-bits (array)
  (loop for i below (array-total-size array)
        collect (element-bits (row-major-aref array i))))

(defun same-array-p (a b)
  "A と B が同じ element-type・形・ビット列なら真。"
  (and (equal (array-element-type a) (array-element-type b))
       (equal (array-dimensions a) (array-dimensions b))
       (equal (array-bits a) (array-bits b))))

(defun same-entries-p (expected actual)
  "EXPECTED の各 (name array dtype) が ACTUAL に同じものとして含まれ、個数も等しいなら真。"
  (and (= (length expected) (length actual))
       (every (lambda (e)
                (let ((a (find (first e) actual :key #'first :test #'string=)))
                  (and a
                       (eq (third e) (third a))
                       (same-array-p (second e) (second a)))))
              expected)))

(defun random-bits-element (dtype state)
  "DTYPE の配列要素を、ビット列を一様に選んで作る（NaN・無限大・非正規化数も出る）。"
  (ecase dtype
    (:f32 (sb-kernel:make-single-float (- (random (ash 1 32) state) (ash 1 31))))
    (:f64 (sb-kernel:make-double-float (- (random (ash 1 32) state) (ash 1 31))
                                       (random (ash 1 32) state)))
    ((:bf16 :f16) (random (ash 1 16) state))
    (:i1 (random 2 state))
    (:i32 (- (random (ash 1 32) state) (ash 1 31)))
    (:u32 (random (ash 1 32) state))
    (:u64 (random (ash 1 64) state))))

(defun random-array-of (dtype shape state)
  (let ((array (make-array shape :element-type (nabla:dtype-element-type dtype))))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i) (random-bits-element dtype state)))))

(defparameter *name-alphabet*
  (coerce (list #\a #\b #\. #\_ #\0 #\Space #\" #\\ #\/ #\Newline (code-char 1) (code-char 31)
                (code-char #xE9) (code-char #x91CD) (code-char #x1F600) (code-char #x7F))
          'string))

(defun random-name (state)
  (let ((name (make-string (random 9 state))))
    (dotimes (i (length name) name)
      (setf (char name i) (char *name-alphabet* (random (length *name-alphabet*) state))))))

(defun random-entries (seed count)
  "SEED から決まる、名前が重なり合わない COUNT 個以下の (name array dtype)。
bf16 / f16 は dtype を明示する。形は rank 0..3、各次元 0..4。"
  (let ((state (sb-ext:seed-random-state seed))
        (names '()))
    (loop repeat count
          for name = (random-name state)
          for dtype = (nth (random (length *dtypes*) state) *dtypes*)
          for shape = (loop repeat (random 4 state) collect (random 5 state))
          unless (or (member name names :test #'string=) (string= name "__metadata__"))
            do (push name names)
            and collect (list name (random-array-of dtype shape state) dtype))))

(defun random-metadata (seed)
  (let ((state (sb-ext:seed-random-state seed)))
    (loop repeat (random 3 state)
          for i from 0
          collect (cons (format nil "k~D~A" i (random-name state)) (random-name state)))))

(defun entry (name dtype shape &rest elements)
  (list name
        (make-array shape :element-type (nabla:dtype-element-type dtype)
                          :initial-contents (if shape
                                                (reshape-list elements shape)
                                                (first elements)))
        dtype))

(defun reshape-list (flat shape)
  "平坦なリスト FLAT を SHAPE の入れ子のリストにする（initial-contents 用）。"
  (if (null (cdr shape))
      flat
      (let ((stride (reduce #'* (cdr shape))))
        (loop for i below (first shape)
              collect (reshape-list (subseq flat (* i stride) (* (1+ i) stride)) (cdr shape))))))

(defun octets (&rest bytes)
  (coerce bytes '(simple-array (unsigned-byte 8) (*))))

(defun u64le (n)
  (loop for i below 8 collect (ldb (byte 8 (* 8 i)) n)))

(defun raw-file (header-text &optional (data '()))
  "HEADER-TEXT（JSON のヘッダ）と DATA（バイトのリスト）から、長さ接頭辞つきのバイト列を作る。"
  (let ((header (sb-ext:string-to-octets header-text :external-format :utf-8)))
    (coerce (append (u64le (length header)) (coerce header 'list) data)
            '(simple-array (unsigned-byte 8) (*)))))

(defun failure-reason (octets)
  "OCTETS を decode して SAFETENSORS-ERROR の reason を返す（出なければ :no-error）。"
  (handler-case (progn (decode-safetensors octets) :no-error)
    (safetensors-error (c) (safetensors-error-reason c))))

;;; --- 性質（small） ---

(test (safetensors/roundtrip :suite :nabla.small)
  "encode → decode は名前・dtype・形・ビット列・メタデータを保つ。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000000)
                                  (uniform-integer :lo 0 :hi 6)))
                (lambda (case)
                  (destructuring-bind (seed count) case
                    (let ((entries (random-entries seed count))
                          (metadata (random-metadata seed)))
                      (multiple-value-bind (decoded decoded-metadata)
                          (decode-safetensors (encode-safetensors entries :metadata metadata))
                        (and (same-entries-p entries decoded)
                             (equal metadata decoded-metadata))))))
                :regression-id safetensors/roundtrip
                :regression-file (regression-path "safetensors-roundtrip"
                                                  :package "NABLA.NN.TESTS"))
      "往復で値が変わった"))

(test (safetensors/encoding-is-independent-of-entry-order :suite :nabla.small)
  "入力の順序を逆にしても同じバイト列になり、データ領域は 8 バイト境界から始まる。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000000)
                                  (uniform-integer :lo 0 :hi 6)))
                (lambda (case)
                  (destructuring-bind (seed count) case
                    (let* ((entries (random-entries seed count))
                           (bytes (encode-safetensors entries))
                           (header-length (loop for i below 8
                                                sum (ash (aref bytes i) (* 8 i)))))
                      (and (equalp bytes (encode-safetensors (reverse entries)))
                           (zerop (mod (+ 8 header-length) 8))))))
                :regression-id safetensors/order
                :regression-file (regression-path "safetensors-order"
                                                  :package "NABLA.NN.TESTS"))
      "順序で結果が変わった、またはデータ領域が整列していない"))

(test (safetensors/corrupted-input-fails-cleanly :suite :nabla.small)
  "真の接頭辞と、1バイトを書き換えた入力は、SAFETENSORS-ERROR になるか読めるかのどちらか。"
  (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000000)
                                  (uniform-integer :lo 1 :hi 4)
                                  (uniform-integer :lo 0 :hi 1000000)
                                  (uniform-integer :lo 0 :hi 255)))
                (lambda (case)
                  (destructuring-bind (seed count position byte) case
                    (let* ((bytes (encode-safetensors (random-entries seed count)))
                           (cut (mod position (length bytes)))
                           (flipped (copy-seq bytes)))
                      (setf (aref flipped cut) byte)
                      (and (eq :safetensors-error
                               (handler-case (progn (decode-safetensors (subseq bytes 0 cut)) nil)
                                 (safetensors-error () :safetensors-error)))
                           (handler-case (progn (decode-safetensors flipped) t)
                             (safetensors-error () t))))))
                :regression-id safetensors/corrupted
                :regression-file (regression-path "safetensors-corrupted"
                                                  :package "NABLA.NN.TESTS"))
      "壊れた入力が SAFETENSORS-ERROR 以外になった、または接頭辞が読めてしまった"))

;;; --- 例（small） ---

(test (safetensors/special-float-bits-survive :suite :nabla.small)
  "NaN（ペイロード付き）・±無限大・-0.0・非正規化数がビット列のまま往復する。"
  (let* ((f32 (make-array 5 :element-type 'single-float
                            :initial-contents
                            (mapcar #'sb-kernel:make-single-float
                                    '(#x7FC00001 #x7F800000 #x-800000 #x-80000000 1))))
         (f64 (make-array 2 :element-type 'double-float
                            :initial-contents
                            (list (sb-kernel:make-double-float #x7FF80000 1)
                                  (sb-kernel:make-double-float #x-80000000 0))))
         (entries (list (list "f32" f32 :f32) (list "f64" f64 :f64))))
    (is (same-entries-p entries (decode-safetensors (encode-safetensors entries))))))

(test (safetensors/sixteen-bit-dtype-is-kept :suite :nabla.small)
  "(unsigned-byte 16) の配列は dtype を明示すれば bf16 / f16 として書ける。省略すると
dtype-mismatch（bf16 か f16 か決められない）。"
  (let ((bits (make-array 2 :element-type '(unsigned-byte 16) :initial-contents '(#x3F80 #xC000))))
    (dolist (dtype '(:bf16 :f16))
      (is (eq dtype (third (first (decode-safetensors
                                   (encode-safetensors (list (list "x" bits dtype)))))))))
    (signals nabla:dtype-mismatch (encode-safetensors (list (list "x" bits))))))

(test (safetensors/dtype-is-inferred-from-the-array :suite :nabla.small)
  "dtype を省略した (name array) は、配列の要素型から決める。"
  (let ((x (make-array 1 :element-type 'single-float :initial-element 1.0)))
    (is (eq :f32 (third (first (decode-safetensors (encode-safetensors (list (list "x" x))))))))))

(test (safetensors/invalid-arguments-are-rejected :suite :nabla.small)
  "名前の重複・予約名 __metadata__・サロゲートを含む名前・文字列でないメタデータは
reason :invalid-entries。"
  (let ((x (make-array 1 :element-type 'single-float :initial-element 1.0)))
    (flet ((reason (&rest args)
             (handler-case (progn (apply #'encode-safetensors args) :no-error)
               (safetensors-error (c) (safetensors-error-reason c)))))
      (is (eq :invalid-entries (reason (list (list "x" x) (list "x" x)))))
      (is (eq :invalid-entries (reason (list (list "__metadata__" x)))))
      (is (eq :invalid-entries (reason (list (list (string (code-char #xD800)) x)))))
      (is (eq :invalid-entries (reason (list (list "x" x)) :metadata '(("k" . 1))))))))

(test (safetensors/malformed-entries-and-unicode-digits :suite :nabla.small)
  "不正な形のエントリ・リスト・メタデータは :invalid-entries、Unicode の数字は JSON の数に使えない。"
  (flet ((reason (&rest args)
           (handler-case (progn (apply #'encode-safetensors args) :no-error)
             (safetensors-error (c) (safetensors-error-reason c)))))
    (is (eq :invalid-entries (reason (list (list "x")))))
    (is (eq :invalid-entries (reason "x")))
    (is (eq :invalid-entries (reason '() :metadata '(("a" . "b") . 3))))
    (is (eq :invalid-entries (reason (list (list "x" 5))))))
  (is (eq :invalid-header
          (failure-reason (raw-file (format nil "{\"x\":{\"dtype\":\"F32\",\"shape\":[~C],\"data_offsets\":[0,0]}}"
                                            (code-char #x663))))))
  (is (eq :invalid-header
          (failure-reason (raw-file (format nil "{\"x\\u~C~C~C~C\":{}}" #\0 #\0 (code-char #x663) #\1))))))

(test (safetensors/accepts-python-style-headers :suite :nabla.small)
  "空白や \\u エスケープ（サロゲート対を含む）、順序の違う項目を含む標準的なヘッダを読める。"
  (let ((entries (decode-safetensors
                  (raw-file "{ \"a\\u00e9\\ud83d\\ude00\" : { \"data_offsets\" : [0, 4], \"shape\" : [1], \"dtype\" : \"F32\" } }   "
                            '(0 0 128 63)))))
    (is (string= (format nil "a~A~A" (code-char #xE9) (code-char #x1F600)) (first (first entries))))
    (is (= 1.0 (row-major-aref (second (first entries)) 0)))))

(test (safetensors/empty-file :suite :nabla.small)
  "テンソルの無いファイルも読み書きできる。"
  (multiple-value-bind (entries metadata) (decode-safetensors (encode-safetensors '()))
    (is (null entries))
    (is (null metadata))))

(test (safetensors/documented-failure-reasons :suite :nabla.small)
  "壊れたファイルは、文書化した reason の SAFETENSORS-ERROR になる。"
  (let ((f32-info "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[~D,~D]}}"))
    (flet ((file (offsets &optional (data '(0 0 0 0)))
             (raw-file (format nil f32-info (first offsets) (second offsets)) data)))
      (is (eq :truncated (failure-reason (octets 1 0 0))))
      (is (eq :header-length (failure-reason (coerce (u64le 100) '(simple-array (unsigned-byte 8) (*))))))
      (is (eq :header-length (failure-reason (coerce (append (u64le (ash 1 63)) '(0))
                                                     '(simple-array (unsigned-byte 8) (*))))))
      (is (eq :invalid-header (failure-reason (raw-file "{not json"))))
      (is (eq :invalid-header (failure-reason (raw-file "[]"))))
      (is (eq :invalid-header (failure-reason (raw-file "{} x"))))
      (is (eq :invalid-header (failure-reason (octets 1 0 0 0 0 0 0 0 #xFF))))
      (is (eq :invalid-header (failure-reason (raw-file "{\"x\":{\"dtype\":\"F32\",\"shape\":[1]}}"))))
      (is (eq :invalid-header (failure-reason (raw-file "{\"x\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}" '(0 0 0 0)))))
      (is (eq :invalid-header (failure-reason (raw-file "{\"__metadata__\":{\"k\":1}}"))))
      (is (eq :invalid-header (failure-reason (raw-file "{\"x\":1,\"x\":2}"))))
      (is (eq :unknown-dtype (failure-reason (raw-file "{\"x\":{\"dtype\":\"F8\",\"shape\":[1],\"data_offsets\":[0,1]}}" '(0)))))
      (is (eq :unknown-dtype (failure-reason (raw-file "{\"x\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[0,8]}}" '(0 0 0 0 0 0 0 0)))))
      (is (eq :size-mismatch (failure-reason (raw-file "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}" '(0 0 0 0)))))
      ;; データ領域をはみ出す・余る・先頭が 0 でない
      (is (eq :bad-offsets (failure-reason (file '(0 4) '(0 0)))))
      (is (eq :bad-offsets (failure-reason (file '(0 4) '(0 0 0 0 0)))))
      (is (eq :bad-offsets (failure-reason (file '(4 8) '(0 0 0 0 0 0 0 0)))))
      (is (eq :bad-offsets (failure-reason (file '(4 0)))))
      ;; 重なり・隙間
      (is (eq :bad-offsets
              (failure-reason (raw-file "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[2,6]}}" '(0 0 0 0 0 0)))))
      (is (eq :bad-offsets
              (failure-reason (raw-file "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}" '(0 0 0 0 0 0 0 0 0 0 0 0)))))
      (is (eq :invalid-bool
              (failure-reason (raw-file "{\"x\":{\"dtype\":\"BOOL\",\"shape\":[1],\"data_offsets\":[0,1]}}" '(2)))))
      (is (eq :no-error (failure-reason (file '(0 4))))))))

(test (safetensors/error-carries-reason-and-detail :suite :nabla.small)
  "SAFETENSORS-ERROR は reason・detail を読め、メッセージに detail が入る。"
  (handler-case (decode-safetensors (octets 1 0 0) :path #p"/tmp/x.safetensors")
    (safetensors-error (c)
      (is (eq :truncated (safetensors-error-reason c)))
      (is (equal #p"/tmp/x.safetensors" (safetensors-error-path c)))
      (is (stringp (safetensors-error-detail c)))
      (is (search (safetensors-error-detail c) (princ-to-string c))))))

;;; 追加の例（レビューと mutation testing で足した）

(test (safetensors/zero-sized-tensor-may-share-an-offset :suite :nabla.small)
  "大きさ 0 のテンソルが、ほかのテンソルの先頭と同じ位置を指していても読める。"
  (let ((entries (decode-safetensors
                  (raw-file "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"z\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}"
                            '(0 0 128 63)))))
    (is (equal '("z" "a") (mapcar #'first entries)))))

(test (safetensors/header-fields-are-validated :suite :nabla.small)
  "ヘッダの項目の欠落・型違い・数値の書式違い・大きすぎる shape は :invalid-header。"
  (flet ((reason (info &optional (data '()))
           (failure-reason (raw-file (format nil "{\"x\":~A}" info) data))))
    (is (eq :invalid-header (reason "{\"shape\":[1],\"data_offsets\":[0,4]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"data_offsets\":[0,4]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "7")))
    (is (eq :invalid-header (reason "{\"dtype\":1,\"shape\":[1],\"data_offsets\":[0,4]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4,8]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-4,0]}")))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[00,4]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-,4]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[1.0],\"data_offsets\":[0,4]}" '(0 0 0 0))))
    (is (eq :invalid-header (reason "{\"dtype\":\"F32\",\"shape\":[0,1000000000000000000000],\"data_offsets\":[0,0]}")))
    (is (eq :bad-offsets (reason "{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[4,0]}")))
    (is (eq :no-error (reason "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"extra\":true}" '(0 0 0 0))))))

(test (safetensors/header-length-boundaries :suite :nabla.small)
  "ヘッダがちょうどファイルの終わりまでなら読め、1バイトでも超えれば :header-length。7バイトは :truncated。"
  (is (eq :no-error (failure-reason (raw-file "{}"))))
  (is (eq :header-length (failure-reason (octets 3 0 0 0 0 0 0 0 123 125))))
  (is (eq :truncated (failure-reason (octets 0 0 0 0 0 0 0))))
  (is (eq :invalid-header (failure-reason (octets 0 0 0 0 0 0 0 0)))))

(test (safetensors/json-nesting-and-escapes :suite :nabla.small)
  "深すぎる入れ子はスタックを使い切らず :invalid-header。壊れたエスケープも同じ。"
  (let ((deep (format nil "{\"__metadata__\":~A~A}"
                      (make-string 100000 :initial-element #\[)
                      (make-string 100000 :initial-element #\]))))
    (is (eq :invalid-header (failure-reason (raw-file deep)))))
  (is (eq :invalid-header (failure-reason (raw-file "{\"\\ud83d\":{}}"))))
  (is (eq :invalid-header (failure-reason (raw-file "{\"\\ude00\":{}}"))))
  (is (eq :invalid-header (failure-reason (raw-file "{\"\\q\":{}}"))))
  (is (eq :invalid-header (failure-reason (raw-file "{\"\\u12g4\":{}}")))))

(test (safetensors/array-kinds-and-names :suite :nabla.small)
  "調整可能・displaced の配列も書ける。文字列でない名前と (キー . 値) でないメタデータは
:invalid-entries。サロゲートの直前・直後の文字は名前に使える。"
  (let* ((base (make-array 4 :element-type 'single-float :initial-contents '(1.0 2.0 3.0 4.0)))
         (adjustable (make-array '(2 2) :element-type 'single-float :adjustable t
                                        :initial-contents '((1.0 2.0) (3.0 4.0))))
         (displaced (make-array '(2) :element-type 'single-float :displaced-to base
                                     :displaced-index-offset 2)))
    (is (same-entries-p (list (list "a" (make-array '(2 2) :element-type 'single-float
                                                           :initial-contents '((1.0 2.0) (3.0 4.0)))
                                    :f32))
                        (decode-safetensors (encode-safetensors (list (list "a" adjustable))))))
    (is (equalp #(3.0 4.0)
                (second (first (decode-safetensors (encode-safetensors (list (list "d" displaced)))))))))
  (let ((x (make-array 1 :element-type 'single-float :initial-element 1.0)))
    (flet ((reason (&rest args)
             (handler-case (progn (apply #'encode-safetensors args) :no-error)
               (safetensors-error (c) (safetensors-error-reason c)))))
      (is (eq :invalid-entries (reason (list (list 1 x)))))
      (is (eq :invalid-entries (reason (list (list "x" x)) :metadata '("k"))))
      (is (eq :no-error (reason (list (list (string (code-char #xD7FF)) x)
                                      (list (string (code-char #xE000)) x))))))))

;;; JSON ヘッダのリーダの境界（mutation testing で足した）

(defparameter *sample-header*
  "{\"__metadata__\":{\"k\":\"v\\u00e9\\ud83d\\ude00\\n\"},\"a\\\"b\":{\"dtype\":\"F32\",\"shape\":[1,0],\"data_offsets\":[0,0],\"extra\":[1,-20,true,false,null,\"s\",{}]}}"
  "全部の構文（エスケープ・サロゲート対・整数・真偽・null・空のオブジェクト）を含む有効なヘッダ。")

(test (safetensors/json-sample-header-is-valid :suite :nabla.small)
  "サンプルのヘッダは読める（次の接頭辞のテストの前提）。"
  (multiple-value-bind (entries metadata) (decode-safetensors (raw-file *sample-header*))
    (is (equal '("a\"b") (mapcar #'first entries)))
    (is (equal (list (cons "k" (format nil "v~A~A~A" (code-char #xE9) (code-char #x1F600) #\Newline)))
               metadata))))

(test (safetensors/every-proper-prefix-of-the-header-is-rejected :suite :nabla.small)
  "ヘッダの JSON の真の接頭辞（長さは接頭辞に合わせる）は、どこで切っても :invalid-header。"
  (is (check-it (generator (uniform-integer :lo 0 :hi (1- (length *sample-header*))))
                (lambda (cut) (eq :invalid-header (failure-reason (raw-file (subseq *sample-header* 0 cut)))))
                :regression-id safetensors/header-prefix
                :regression-file (regression-path "safetensors-header-prefix"
                                                  :package "NABLA.NN.TESTS"))
      "接頭辞が :invalid-header にならなかった"))

(test (safetensors/json-surrogate-pairs :suite :nabla.small)
  "\\u のサロゲート対は範囲の端まで正しく復号し、対にならないものは :invalid-header。"
  (flet ((name (escape)
           (first (first (decode-safetensors
                          (raw-file (format nil "{\"~A\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}" escape)))))))
    (is (string= (string (code-char #x10000)) (name "\\ud800\\udc00")))
    (is (string= (string (code-char #x10FFFF)) (name "\\udbff\\udfff")))
    (is (string= "A" (name "\\u0041")))
    (is (string= (string (code-char #xFFFF)) (name "\\uFFFF")))
    (is (string= (string (code-char #xD7FF)) (name "\\ud7ff")))
    (is (string= (string (code-char #xE000)) (name "\\ue000"))))
  (flet ((reason (escape)
           (failure-reason (raw-file (format nil "{\"~A\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}" escape)))))
    (is (eq :invalid-header (reason "\\ud800\\ud800")))
    (is (eq :invalid-header (reason "\\ud800\\u0041")))
    (is (eq :invalid-header (reason "\\ud800\\udbff")))
    (is (eq :invalid-header (reason "\\ud800")))
    (is (eq :invalid-header (reason "\\udbff")))
    (is (eq :invalid-header (reason "\\udc00")))
    (is (eq :invalid-header (reason "\\udfff")))
    (is (eq :invalid-header (reason "\\ud800\\u00")))
    (is (eq :invalid-header (reason "\\udc00\\udc00")))
    (is (eq :invalid-header (reason "\\ud800\\ue000")))
    (is (eq :invalid-header (reason "\\u+041")))
    (is (eq :invalid-header (reason "\\ud800\\")))))

(test (safetensors/json-whitespace-and-duplicate-keys :suite :nabla.small)
  "トークンの間の空白（スペース・タブ・改行・CR）は読み飛ばし、重複したキーは :invalid-header。"
  (is (eq :no-error
          (failure-reason (raw-file (format nil "{~C\"x\"~C:~C{\"dtype\"~C:\"F32\",~C\"shape\":[0~C],\"data_offsets\":[0,0]}~C}"
                                            #\Tab #\Newline #\Return #\Tab #\Newline #\Return #\Tab)))))
  (let ((info "{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}"))
    (is (eq :invalid-header (failure-reason (raw-file (format nil "{\"x\":~A,\"x\":~A}" info info)))))))

(test (safetensors/json-nesting-limit-boundary :suite :nabla.small)
  "入れ子の深さの上限ちょうどは読め、1つ超えれば :invalid-header。"
  (flet ((reason (arrays)
           (failure-reason
            (raw-file (format nil "{\"x\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0],\"extra\":~A0~A}}"
                              (make-string arrays :initial-element #\[)
                              (make-string arrays :initial-element #\]))))))
    (is (eq :no-error (reason 62)))
    (is (eq :invalid-header (reason 63)))))

(test (safetensors/shape-limits :suite :nabla.small)
  "SBCL の配列の上限ちょうどの次元・階数は :invalid-header（make-array の型エラーにならない）。"
  (flet ((reason (shape-text)
           (failure-reason
            (raw-file (format nil "{\"x\":{\"dtype\":\"F32\",\"shape\":~A,\"data_offsets\":[0,0]}}" shape-text)))))
    (is (eq :invalid-header (reason (format nil "[0,~D]" array-dimension-limit))))
    (is (eq :invalid-header
            (reason (format nil "[~{~D~^,~}]" (make-list array-rank-limit :initial-element 0)))))))

(test (safetensors/invalid-names-in-entries :suite :nabla.small)
  "名前にサロゲート領域の最後の文字（U+DFFF）があれば :invalid-entries。"
  (is (eq :invalid-entries
          (handler-case (encode-safetensors
                         (list (list (string (code-char #xDFFF))
                                     (make-array 1 :element-type 'single-float :initial-element 1.0))))
            (safetensors-error (c) (safetensors-error-reason c))))))

;;; --- ファイル（medium） ---

(test (safetensors/file-roundtrip :suite :nabla.medium)
  "write-safetensors → read-safetensors で同じ名前・dtype・形・値・メタデータが戻り、
一時ファイルは残らない。上書きもできる。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "model.safetensors" dir))
          (first (random-entries 7 5))
          (second (random-entries 8 5)))
      (write-safetensors path first :metadata '(("format" . "nabla")))
      (multiple-value-bind (entries metadata) (read-safetensors path)
        (is (same-entries-p first entries))
        (is (equal '(("format" . "nabla")) metadata)))
      (write-safetensors path second)
      (is (same-entries-p second (read-safetensors path)))
      (is (equal (list (namestring path))
                 (mapcar #'namestring (directory (merge-pathnames "*.*" dir))))))))

(test (safetensors/failed-write-keeps-the-existing-file :suite :nabla.medium)
  "書き込みに失敗しても、既存のファイルは壊れず、一時ファイルも残らない。"
  (with-temporary-directory (dir)
    (let* ((path (merge-pathnames "model.safetensors" dir))
           (good (random-entries 3 4))
           (bits (make-array 1 :element-type '(unsigned-byte 16) :initial-element 0)))
      (write-safetensors path good)
      (signals error (write-safetensors path (list (list "x" bits))))
      (is (same-entries-p good (read-safetensors path)))
      (is (= 1 (length (directory (merge-pathnames "*.*" dir))))))))

(test (safetensors/write-creates-missing-directories :suite :nabla.medium)
  "存在しないディレクトリの下にも書け、相対パスは現在のディレクトリを基準にする。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "a/b/model.safetensors" dir)))
      (write-safetensors path (random-entries 1 2))
      (is (probe-file path)))
    (uiop:with-current-directory (dir)
      (write-safetensors "rel.safetensors" '())
      (is (probe-file (merge-pathnames "rel.safetensors" dir))))))

(test (safetensors/failed-rename-leaves-no-temporary-file :suite :nabla.medium)
  "最後の rename に失敗しても（書き込み先がディレクトリ）、一時ファイルを残さずエラーにする。"
  (with-temporary-directory (dir)
    (let ((target (merge-pathnames "model.safetensors/" dir)))
      (ensure-directories-exist target)
      (signals error (write-safetensors (string-right-trim "/" (namestring target)) (random-entries 2 2)))
      (is (equal (list (namestring target))
                 (mapcar #'namestring (directory (merge-pathnames "*.*" dir))))))))

(test (safetensors/write-to-paths-without-a-type :suite :nabla.medium)
  "拡張子の無いパスにも、ある パスにも、指定した名前のファイルがちょうど1つ書かれ、*.tmp は残らない。"
  (with-temporary-directory (dir)
    (dolist (name '("model" "model.ckpt" "model.safetensors"))
      (let ((path (merge-pathnames name dir))
            (entries (random-entries 5 3)))
        (write-safetensors path entries)
        (is (probe-file path) "~A" name)
        (is (same-entries-p entries (read-safetensors path)))))
    (is (equal '("model" "model.ckpt" "model.safetensors")
               (sort (mapcar #'file-namestring (directory (merge-pathnames "*.*" dir))) #'string<)))))

(test (safetensors/read-reports-the-path :suite :nabla.medium)
  "壊れたファイルを read-safetensors で読むと、エラーの path にそのファイルが入る。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "broken.safetensors" dir)))
      (with-open-file (out path :direction :output :element-type '(unsigned-byte 8))
        (write-sequence (octets 1 2 3) out))
      (handler-case (read-safetensors path)
        (safetensors-error (c)
          (is (eq :truncated (safetensors-error-reason c)))
          (is (equal (namestring path) (namestring (safetensors-error-path c)))))))))

;;; --- ファイルを葉ごとに読み書きする（medium、issue #262） ---
;;;
;;; read-safetensors / write-safetensors はファイル全体を 1 本のバイト列にせず、葉ごとに
;;; 小さな作業用バッファを通して読み書きする。バイト列版（encode / decode）と同じ結果になる
;;; ことを、作業用バッファより大きな葉（境界をまたぐ）と壊したファイルで確かめる。

(defparameter *all-safetensors-dtypes* '(:f32 :f64 :bf16 :f16 :i32 :u32 :u64 :i1))

(defun random-entries-with-a-large-leaf (seed count)
  "RANDOM-ENTRIES に、0〜約 300 KB のランダムな dtype の 1 次元の葉 \"large\" を 1 つ足す
（読み書きの作業用バッファ 64 KiB より大きくなり、境界をまたぐ）。半分の場合は、バイト数が
64 KiB・128 KiB ちょうどとその前後になる長さを選ぶ。"
  (let* ((state (sb-ext:seed-random-state (+ seed 1)))
         (dtype (nth (random (length *all-safetensors-dtypes*) state) *all-safetensors-dtypes*))
         (width (nabla:dtype-byte-width dtype))
         (length (if (zerop (random 2 state))
                     (+ (floor (nth (random 2 state) '(65536 131072)) width) (1- (random 3 state)))
                     (random (floor 300000 width) state))))
    (cons (list "large" (random-array-of dtype (list length) state) dtype)
          (remove "large" (random-entries seed count) :key #'first :test #'string=))))

(defun write-octets-file (path octets)
  (with-open-file (out path :direction :output :element-type '(unsigned-byte 8) :if-exists :supersede)
    (write-sequence octets out))
  path)

(defun file-octets (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((octets (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (read-sequence octets in)
      octets)))

(defun read-outcome (thunk)
  "THUNK の (values entries metadata) を (:ok entries metadata) に、SAFETENSORS-ERROR を
\(:error reason) にする。"
  (handler-case (multiple-value-bind (entries metadata) (funcall thunk)
                  (list :ok entries metadata))
    (safetensors-error (c) (list :error (safetensors-error-reason c)))))

(defun same-outcome-p (a b)
  (if (eq (first a) :ok)
      (and (eq (first b) :ok)
           (same-entries-p (second a) (second b))
           (equal (third a) (third b)))
      (equal a b)))

(test (safetensors/written-file-is-the-encoding :suite :nabla.medium)
  "write-safetensors が書いたファイルは、encode-safetensors のバイト列とビット単位で同じ
（全 dtype、作業用バッファより大きな葉、メタデータを含む）。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "model.safetensors" dir)))
      (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000000)
                                      (uniform-integer :lo 0 :hi 4)))
                    (lambda (case)
                      (destructuring-bind (seed count) case
                        (let ((entries (random-entries-with-a-large-leaf seed count))
                              (metadata (random-metadata seed)))
                          (write-safetensors path entries :metadata metadata)
                          (equalp (encode-safetensors entries :metadata metadata)
                                  (file-octets path)))))
                    :regression-id safetensors/written-file
                    :regression-file (regression-path "safetensors-written-file"
                                                      :package "NABLA.NN.TESTS"))
          "ファイルのバイト列が encode-safetensors と違った"))))

(test (safetensors/read-agrees-with-decode-on-corrupted-files :suite :nabla.medium)
  "そのままの・切り詰めた・1バイト変えた・末尾にバイトを足したファイルを read-safetensors で
読んだ結果は、同じバイト列を decode-safetensors した結果と同じ（同じ値か、同じ reason の
SAFETENSORS-ERROR）。切り詰めたファイルを読み過ぎたり、他のエラーになったりしない。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "model.safetensors" dir)))
      (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000000)
                                      (uniform-integer :lo 0 :hi 4)
                                      (uniform-integer :lo 0 :hi 3)
                                      (uniform-integer :lo 0 :hi 1000000)
                                      (uniform-integer :lo 0 :hi 255)))
                    (lambda (case)
                      (destructuring-bind (seed count kind position byte) case
                        (let* ((bytes (encode-safetensors (random-entries-with-a-large-leaf seed count)
                                                          :metadata (random-metadata seed)))
                               (cut (mod position (length bytes)))
                               (corrupted (ecase kind
                                            (0 bytes)
                                            (1 (subseq bytes 0 cut))
                                            (2 (let ((copy (copy-seq bytes)))
                                                 (setf (aref copy cut) byte)
                                                 copy))
                                            (3 (concatenate '(simple-array (unsigned-byte 8) (*)) bytes (octets byte))))))
                          (write-octets-file path corrupted)
                          (same-outcome-p (read-outcome (lambda () (decode-safetensors corrupted)))
                                          (read-outcome (lambda () (read-safetensors path)))))))
                    :regression-id safetensors/read-agrees
                    :regression-file (regression-path "safetensors-read-agrees"
                                                      :package "NABLA.NN.TESTS"))
          "read-safetensors と decode-safetensors の結果が違った"))))

(defun bytes-consed-by (thunk)
  (let ((before (sb-ext:get-bytes-consed)))
    (funcall thunk)
    (- (sb-ext:get-bytes-consed) before)))

(test (safetensors/file-io-allocates-about-one-file :suite :nabla.medium)
  "8 MiB のファイルで、write-safetensors が確保するのはファイルの 1/4 未満（全体をまとめた
バイト列を作らない）、read-safetensors はファイルの 1.25 倍未満（結果の配列の 1 倍 +
作業用バッファ。ファイル全体のバイト列を作らない）。issue #262。"
  (with-temporary-directory (dir)
    (let* ((path (merge-pathnames "big.safetensors" dir))
           (entries (loop for i below 4
                          collect (list (format nil "w~D" i)
                                        (make-array (floor (* 2 1048576) 4) :element-type 'single-float
                                                                            :initial-element 1.5)
                                        :f32)))
           (written (bytes-consed-by (lambda () (write-safetensors path entries))))
           (size (with-open-file (in path :element-type '(unsigned-byte 8)) (file-length in)))
           (read (bytes-consed-by (lambda () (read-safetensors path)))))
      (is (< written (/ size 4)) "write-safetensors が ~D バイト確保した（ファイルは ~D バイト）" written size)
      (is (< read (* 5/4 size)) "read-safetensors が ~D バイト確保した（ファイルは ~D バイト）" read size)
      (is (same-entries-p entries (read-safetensors path))))))

;;; --- Python の safetensors との相互運用（medium、例ベース） ---

(defun fixture-path (name)
  (asdf:system-relative-pathname "nabla" (format nil "tests/fixtures/safetensors/~A" name)))

(defun fixture-entries ()
  "tests/fixtures/safetensors/generate.py が作る PYTHON_TENSORS と同じ内容。"
  (list (entry "f32" :f32 '(2 2) 1.5 -2.0 0.25 3.0)
        (entry "f64" :f64 '(3) 0.1d0 -2.5d0 1d100)
        (entry "bf16" :bf16 '(3) #x3F80 #xBF80 #x4000)
        (entry "f16" :f16 '(2) #x3C00 #xC000)
        (entry "i32" :i32 '(3) -2147483648 0 2147483647)
        (entry "u32" :u32 '(2) 0 4294967295)
        (entry "u64" :u64 '(2) 0 18446744073709551615)
        (entry "bool" :i1 '(2 2) 1 0 1 1)
        (entry "scalar" :f32 '() 7.0)
        (entry "empty" :f32 '(0 3))
        (entry (format nil "layer.0/~A" (code-char #x91CD)) :f32 '(1) 1.0)))

(test (safetensors/reads-python-written-file :suite :nabla.medium)
  "Python の safetensors が書いたファイルを、全 dtype・rank 0・サイズ 0・非 ASCII の名前・
メタデータ込みで読める。"
  (multiple-value-bind (entries metadata)
      (read-safetensors (fixture-path "python-written.safetensors"))
    (is (same-entries-p (fixture-entries) entries))
    (is (equal '(("format" . "pt") ("note" . "fixture")) (sort (copy-list metadata) #'string< :key #'car)))))

(test (safetensors/written-bytes-match-the-committed-file :suite :nabla.medium)
  "nabla が書くバイト列は tests/fixtures/safetensors/nabla-written.safetensors と一致する
（形式の変化を検出する。generate.py がこのファイルを Python で読めることを確かめている）。"
  (with-open-file (in (fixture-path "nabla-written.safetensors") :element-type '(unsigned-byte 8))
    (let ((committed (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (read-sequence committed in)
      (is (equalp committed
                  (encode-safetensors (fixture-entries)
                                      :metadata '(("format" . "pt") ("note" . "fixture"))))))))

(test (safetensors/write-fsyncs-the-file-and-the-directory :suite :nabla.medium)
  "write-safetensors は rename の前に一時ファイルを、後に親ディレクトリを fsync する（#263）。
fsync の呼び出しは内部のフック nabla::*fsync-function* で数える。"
  (with-temporary-directory (dir)
    (let ((kinds '()))
      (let ((nabla::*fsync-function*
              (lambda (fd)
                (push (if (sb-posix:s-isdir (sb-posix:stat-mode (sb-posix:fstat fd))) :directory :file)
                      kinds))))
        (write-safetensors (merge-pathnames "m.safetensors" dir) (random-entries 2 2)))
      (is (equal '(:file :directory) (reverse kinds))))))
