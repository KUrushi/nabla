;;;; 画像のデコード（PNG / JPEG）と image-source（issue #283）。
;;;;
;;;; 画像は (H W C) の (unsigned-byte 8) の配列で返す。C はファイルのまま（グレースケール 1、
;;;; グレースケール + アルファ 2、RGB 3、RGBA 4）で、パレットは RGB に、1・2・4 ビットの
;;;; グレースケールは 0..255 に直す（Pillow の convert("RGB") / convert("L") と同じ値）。
;;;; 形式はファイルの先頭のバイト（シグネチャ）で見分け、拡張子は見ない。

(in-package #:nabla.data.image)

;;; --- コンディション ---

(define-condition image-decode-error (nabla.data:data-error)
  ((reason :initarg :reason :reader image-decode-error-reason
           :documentation "原因を表すキーワード。")
   (pathname :initarg :pathname :reader image-decode-error-pathname
             :documentation "読もうとしたファイル。")
   (detail :initarg :detail :reader image-decode-error-detail
           :documentation "人間向けの詳しい説明（文字列）。"))
  (:report (lambda (c s)
             (format s "画像 ~A: ~A (~S)" (image-decode-error-pathname c)
                     (image-decode-error-detail c) (image-decode-error-reason c))))
  (:documentation
   "画像ファイルを読めないときに signal する（NABLA.DATA:DATA-ERROR の子）。REASON は次のどれか。
  :unreadable      ファイルを開けない・読めない（無い・権限が無い・ディレクトリなど）
  :unknown-format  先頭のバイトが PNG でも JPEG でもない
  :corrupt         デコードに失敗した（途中で切れている・CRC の不一致・壊れたデータなど）
  :too-large       ヘッダの幅 × 高さが上限（READ-IMAGE の MAX-PIXELS）を超える（配列を確保する前に止める）
  :unsupported     正しいファイルだが読まない形式（16 ビットの PNG、成分が 1 でも 3 でもない
                   CMYK などの JPEG、Adobe の APP14 が YCbCr でないと言う3成分の JPEG、
                   ベースラインでもプログレッシブでもない JPEG）"))

(defun %image-error (reason pathname control &rest arguments)
  (error 'image-decode-error :reason reason :pathname pathname
                             :detail (apply #'format nil control arguments)))

(defmacro %with-library-errors ((pathname) &body body)
  "BODY（ライブラリの呼び出し）が signal したエラーを :corrupt の IMAGE-DECODE-ERROR にする。
png-read の CRC の不一致は WARNING の子なので、別に捕まえる。"
  `(handler-case (progn ,@body)
     (jpeg:unsupported-jpeg-format (e) (%image-error :unsupported ,pathname "~A" e))
     ((or error png-read:crc-failure) (e) (%image-error :corrupt ,pathname "~A" e))))

;;; --- 寸法の上限（issue #324） ---

(defparameter *max-image-pixels* (floor (* 1024 1024 1024) (* 4 3))
  "READ-IMAGE と MAKE-IMAGE-SOURCE の MAX-PIXELS の既定値（幅 × 高さの上限。nil なら上限なし）。
既定は Pillow の Image.MAX_IMAGE_PIXELS と同じ 89478485 画素（RGB の u8 で約 256 MiB）。
ヘッダに巨大な寸法を書いた壊れたファイル・悪意のあるファイル 1 つで、ヒープを使い切らないための上限。
ローダのワーカーのスレッドは呼び出し側の動的束縛を見ないので、IMAGE-SOURCE には作るときに渡す
（MAKE-IMAGE-SOURCE は作ったときの値を持つ）。")

(defun %check-pixel-limit (width height max-pixels pathname)
  (let ((pixels (* width height)))
    (when (and max-pixels (> pixels max-pixels))
      (%image-error :too-large pathname "幅 ~D × 高さ ~D = ~D 画素が上限 ~D を超える"
                    width height pixels max-pixels))))

(defun %big-endian (octets start count)
  (let ((value 0))
    (dotimes (i count value)
      (setf value (+ (* value 256) (aref octets (+ start i)))))))

(defun %crc32 (octets start end)
  "OCTETS の [START, END) の CRC-32（PNG のチャンクの CRC。多項式 #xEDB88320 の反転形）。"
  (let ((crc #xffffffff))
    (declare (type (unsigned-byte 32) crc))
    (loop for i from start below end
          do (setf crc (logxor crc (aref octets i)))
             (dotimes (bit 8)
               (setf crc (if (logbitp 0 crc) (logxor #xedb88320 (ash crc -1)) (ash crc -1)))))
    (logxor crc #xffffffff)))

;;; --- PNG ---

(defun %check-png-dimensions (stream pathname max-pixels)
  "先頭の IHDR の幅と高さが MAX-PIXELS の上限に収まることを、png-read が配列を確保する前に確かめる。
IHDR の CRC が合わないとき（寸法のバイトが壊れている）は :corrupt にする（壊れた寸法を :too-large と
取り違えない）。先頭が長さ 13 の IHDR でなければ、ここでは何もしない（png-read が :corrupt にする）。
STREAM の位置は先頭に戻す。"
  (let ((header (make-array 33 :element-type '(unsigned-byte 8))))
    (file-position stream 0)
    (when (and (= 33 (read-sequence header stream))
               (= 13 (%big-endian header 8 4))
               (equalp (subseq header 12 16) #(73 72 68 82))) ; "IHDR"
      (unless (= (%crc32 header 12 29) (%big-endian header 29 4))
        (%image-error :corrupt pathname "IHDR の CRC が合わない"))
      (%check-pixel-limit (%big-endian header 16 4) (%big-endian header 20 4) max-pixels pathname))
    (file-position stream 0)))

(defun %check-png-chunk-lengths (stream pathname)
  "PNG のチャンクの見出し（長さと種類）を IEND までたどり、どの長さもファイルの中に収まることを
確かめる（png-read は長さのとおりに配列を確保するので、壊れた長さで巨大な確保をさせない）。
IEND より後ろのバイトは png-read と同じく見ない。STREAM の位置は先頭に戻す。"
  (let ((size (file-length stream))
        (header (make-array 8 :element-type '(unsigned-byte 8)))
        (position 8))
    (loop
      (when (> (+ position 8) size)
        (return))
      (file-position stream position)
      (read-sequence header stream)
      (let ((length (reduce (lambda (a b) (+ (* a 256) b)) header :end 4)))
        (when (> (+ position 12 length) size)
          (%image-error :corrupt pathname "チャンクの長さ ~D がファイルの大きさを超える" length))
        (when (equalp (subseq header 4) #(73 69 78 68)) ; "IEND"
          (return))
        (incf position (+ 12 length))))
    (file-position stream 0)))

(defun %png-to-hwc (data scale)
  "png-read の image-data（(W H) か (W H C)、x の軸が先）を (H W C) の u8 にし、各値を SCALE 倍する。"
  (let* ((width (array-dimension data 0))
         (height (array-dimension data 1))
         (channels (if (= 3 (array-rank data)) (array-dimension data 2) 1))
         (image (make-array (list height width channels) :element-type '(unsigned-byte 8))))
    (dotimes (y height image)
      (dotimes (x width)
        (dotimes (k channels)
          (setf (aref image y x k)
                (* scale (if (= 3 (array-rank data)) (aref data x y k) (aref data x y)))))))))

(defun %read-png (stream pathname max-pixels)
  (%check-png-chunk-lengths stream pathname)
  (%check-png-dimensions stream pathname max-pixels)
  (let* ((state (%with-library-errors (pathname)
                  (let ((png-read:*crc-fail-behaviour* :error))
                    (png-read:read-png-datastream stream))))
         (depth (png-read:bit-depth state)))
    ;; ビット深度は IHDR にあるが、CRC を確かめた後で見る（壊れた IHDR を :unsupported と
    ;; 取り違えない）ので、デコードの後で判定する。
    (when (= depth 16)
      (%image-error :unsupported pathname "16 ビットの PNG は読まない"))
    ;; 1・2・4 ビットのグレースケールは 0..255 に引き伸ばす（255 / (2^depth - 1) 倍）。
    ;; パレットの depth は添字の幅で、image-data は既に 8 ビットの RGB。
    (%png-to-hwc (png-read:image-data state)
                 (if (eq (png-read:colour-type state) :greyscale)
                     (floor 255 (1- (ash 1 depth)))
                     1))))

;;; --- JPEG ---

;;; cl-jpeg は jpeg.lisp の jpeg::*optimize* を全関数の declare に #.*optimize* で配るので、上流の
;;; (safety 0) のままだと壊れた JPEG で範囲外のメモリを読み書きし、コンディションではなく Memory
;;; fault でプロセスごと落ちる。scripts/setup-lisp-deps.sh は safety 2 でコンパイルするよう直した
;;; fork（KUrushi/cl-jpeg の bounds-checked、issue #315）を入れる（1 では cl-jpeg の (speed 3) より
;;; 低く、SBCL が型の検査を弱めるので足りない）。ほかの手段で入れた上流の cl-jpeg（Quicklisp など）を
;;; 使っていれば、JPEG をデコードする前にここで止める。

(defun %cl-jpeg-safety (optimize)
  "OPTIMIZE（(optimize (safety 0) (speed 3)) の形の宣言）の safety の値。safety だけを書いた
（値の無い）ものは 3、safety が無ければ nil。"
  (dolist (quality (rest optimize))
    (cond ((eq quality 'safety) (return 3))
          ((and (consp quality) (eq (first quality) 'safety)) (return (second quality))))))

(defun %check-cl-jpeg-safety ()
  "ロードされている cl-jpeg の jpeg::*optimize* の safety が 2 以上であることを確かめ、そうでなければ
error を signal する。見るのは jpeg::*optimize* の値で、関数のコンパイルに使われたものと同じ
（どちらも jpeg.lisp の中にあり、同じ fasl から来るので、上流の cl-jpeg からコンパイルした古い fasl も
safety 0 として見つかる）。ただし safety 0 でコンパイルした後で *optimize* だけを書き換えた・
束縛した場合は見分けられない。"
  (let* ((symbol (find-symbol "*OPTIMIZE*" "JPEG"))
         (optimize (and symbol (boundp symbol) (symbol-value symbol)))
         (safety (and (consp optimize) (%cl-jpeg-safety optimize))))
    (unless (and (integerp safety) (<= 2 safety))
      (error "cl-jpeg が safety ~:[不明~;~:*~D~] でコンパイルされている（jpeg::*optimize* = ~A）。~
壊れた JPEG で範囲外のメモリに触れてプロセスが落ちるので、nabla/data/image は safety 2 以上の ~
cl-jpeg だけを使う。scripts/setup-lisp-deps.sh で入れた cl-jpeg（safety 2 に直した fork）を ~
CL_SOURCE_REGISTRY に置き、上流の cl-jpeg からコンパイルした fasl（~~/.cache/common-lisp/ 以下の ~
cl-jpeg のもの）が残っていれば消して、コンパイルし直すこと。"
             safety (let ((*print-pretty* nil)) (prin1-to-string optimize))))))

(defun %jpeg-frame-dimensions (stream)
  "JPEG の最初のフレームの見出し（SOF0〜SOF15 のうち DHT・JPG・DAC 以外）の (values 幅 高さ)。
SOS か EOI までに見つからない、または途中でファイルが切れていれば nil（cl-jpeg が :corrupt にする）。
STREAM の位置は先頭に戻す。"
  (let ((size (file-length stream))
        (buffer (make-array 7 :element-type '(unsigned-byte 8))))
    (flet ((octets-at (position count)
             (when (<= (+ position count) size)
               (file-position stream position)
               (read-sequence buffer stream :end count)
               buffer)))
      (unwind-protect
           (loop with position = 2
                 for head = (octets-at position 2)
                 while (and head (= #xff (aref head 0)))
                 do (let ((marker (aref head 1)))
                      (cond ((= marker #xff) (incf position)) ; 詰め物の #xFF
                            ((or (<= #xd0 marker #xd7) (= marker #x01)) (incf position 2))
                            ((member marker '(#xd9 #xda)) (return nil)) ; EOI・SOS
                            (t
                             (let ((segment (octets-at (+ position 2) 7)))
                               (unless segment (return nil))
                               (when (and (<= #xc0 marker #xcf) (not (member marker '(#xc4 #xc8 #xcc))))
                                 ;; 長さ 2・精度 1・高さ 2・幅 2
                                 (return (values (%big-endian segment 5 2) (%big-endian segment 3 2))))
                               (incf position (+ 2 (%big-endian segment 0 2))))))))
        (file-position stream 0)))))

(defun %read-jpeg (stream pathname max-pixels)
  ;; ロード時には検査しない（PNG だけを読む利用者まで止めない）。JPEG を読む前に毎回見る
  ;; （*optimize* の短いリストを1つたどるだけで、デコードに比べて無視できる）。
  (%check-cl-jpeg-safety)
  ;; cl-jpeg はフレームの見出しの寸法のとおりに配列を確保するので、その前に上限を見る。
  (multiple-value-bind (width height) (%jpeg-frame-dimensions stream)
    (when width
      (%check-pixel-limit width height max-pixels pathname)))
  (multiple-value-bind (buffer height width components transform)
      (%with-library-errors (pathname) (jpeg:decode-stream stream))
    ;; 読むのはグレースケール（1 成分）と YCbCr の3成分だけ。cl-jpeg は3成分を常に YCbCr と
    ;; みなして RGB に直すので、Adobe の APP14 が YCbCr でない（RGB のまま、など）と言う画像は
    ;; 読まない（色を取り違えたまま返さない）。CMYK などの4成分も読まない。
    (unless (or (= components 1)
                (and (= components 3) (member transform '(nil :ycbcr-rgb))))
      (%image-error :unsupported pathname "成分が ~D 個で Adobe の色変換が ~S の JPEG は読まない"
                    components transform))
    ;; cl-jpeg の3成分は B, G, R の順なので、成分の順を逆にする。
    (let ((image (make-array (list height width components) :element-type '(unsigned-byte 8))))
      (dotimes (pixel (* height width) image)
        (dotimes (k components)
          (setf (row-major-aref image (+ (* pixel components) k))
                (aref buffer (+ (* pixel components) (- components 1 k)))))))))

;;; --- 読み込み ---

(defun read-image (pathname &key (max-pixels *max-image-pixels*))
  "PNG か JPEG のファイル PATHNAME をデコードして、(H W C) の (unsigned-byte 8) の配列を返す。
C はファイルのまま（グレースケール 1、グレースケール + アルファ 2、RGB 3、RGBA 4）。パレットの
PNG は RGB（C = 3。tRNS の透過は読まない）、1・2・4 ビットのグレースケールは 0..255 に直す。
形式は先頭のバイトで見分ける。読めなければ IMAGE-DECODE-ERROR を signal する。
ヘッダの幅 × 高さが MAX-PIXELS（0 以上の整数か nil。既定 *MAX-IMAGE-PIXELS*、nil なら上限なし）を
超えれば、配列を確保する前に :too-large の IMAGE-DECODE-ERROR にする。
u8 は nabla の dtype に無いので、jit に渡す前に f32 などに直すこと。"
  (check-type max-pixels (or null (integer 0)))
  (handler-case
      (with-open-file (stream pathname :element-type '(unsigned-byte 8))
        (let ((head (make-array 8 :element-type '(unsigned-byte 8) :initial-element 0)))
          (read-sequence head stream)
          (file-position stream 0)
          (cond ((equalp head #(137 80 78 71 13 10 26 10)) (%read-png stream pathname max-pixels))
                ((and (= #xff (aref head 0)) (= #xd8 (aref head 1))) (%read-jpeg stream pathname max-pixels))
                (t (%image-error :unknown-format pathname "PNG でも JPEG でもない")))))
    ;; ディレクトリなどは開けても読めず、STREAM-ERROR になる。ライブラリのエラーは
    ;; %WITH-LIBRARY-ERRORS が先に :corrupt にするので、ここに来るのはファイル自体の問題。
    ((or file-error stream-error) (e) (%image-error :unreadable pathname "~A" e))))

;;; --- image-source ---

(defclass image-source (nabla.data:data-source)
  ((pathnames :initarg :pathnames :reader %image-source-pathnames)
   (labels :initarg :labels :reader %image-source-labels)
   (max-pixels :initarg :max-pixels :reader %image-source-max-pixels))
  (:documentation
   "画像ファイルのリスト（とラベル）から作るデータソース。MAKE-IMAGE-SOURCE で作る。
画像はレコードを取るとき（SOURCE-ITEM）にデコードする。"))

(defmethod nabla.data:source-length ((source image-source))
  (length (%image-source-pathnames source)))

(defun make-image-source (pathnames &key labels (max-pixels *max-image-pixels*))
  "画像ファイルの列 PATHNAMES（パスネームか、OS の書き方のパスの文字列）から IMAGE-SOURCE を作る。
INDEX 番目のレコードは、LABELS が nil なら (READ-IMAGE 番目のファイル) の画像だけ、そうでなければ
(画像 ラベル) のリスト。LABELS は NABLA.DATA:MAKE-ARRAY-SOURCE と同じ木（葉の先頭の軸がレコードの
軸）で、ラベルはその INDEX 番目の行（例: (signed-byte 32) のベクタなら rank 0 の配列）。
LABELS のレコードの個数がファイルの数と違えば :length-mismatch の NABLA.DATA:DATA-SOURCE-ERROR。
画像は READ-IMAGE に MAX-PIXELS（既定は作るときの *MAX-IMAGE-PIXELS*）を渡して読む（ワーカーの
スレッドは呼び出し側の動的束縛を見ないので、上限は作るときに決める）。
ファイルは作るときには開かない。画像の (unsigned-byte 8) は nabla の dtype に無いので、jit に
渡す前にローダの変換（map）で f32 などに直すこと。"
  (let ((pathnames (map 'simple-vector
                        (lambda (p) (if (stringp p) (uiop:parse-native-namestring p) (pathname p)))
                        pathnames))
        (labels (and labels (nabla.data:make-array-source labels))))
    (when (and labels (/= (length pathnames) (nabla.data:source-length labels)))
      (error 'nabla.data:data-source-error
             :reason :length-mismatch
             :detail (format nil "ファイルが ~D 個でラベルが ~D 個" (length pathnames)
                             (nabla.data:source-length labels))))
    (check-type max-pixels (or null (integer 0)))
    (make-instance 'image-source :pathnames pathnames :labels labels :max-pixels max-pixels)))

(defmethod nabla.data:source-item ((source image-source) index)
  (unless (and (integerp index) (< -1 index (nabla.data:source-length source)))
    (error 'nabla.data:data-source-error
           :reason :index-out-of-range
           :detail (format nil "添字 ~S は 0 以上 ~D 未満の整数でない"
                           index (nabla.data:source-length source))))
  (let ((image (read-image (svref (%image-source-pathnames source) index)
                           :max-pixels (%image-source-max-pixels source)))
        (labels (%image-source-labels source)))
    (if labels
        (list image (nabla.data:source-item labels index))
        image)))
