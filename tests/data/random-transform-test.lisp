;;;; random-map-transform（issue #284）。
;;;;
;;;; 守らせる性質（small）:
;;;; - 導出: キーは、テストに書いた SplitMix64 の参照実装で (シード, 何番目の random-map-transform か,
;;;;   位置) から作った値と一致する（算法の版 1。固定値は版と一緒に変える）。シードの省略は 0。
;;;; - 決定性: 同じ (シード, 位置) からは同じキー（:u32 の (2) の新しい配列）と同じ出力。
;;;; - 別の乱数: 位置・シード・リストの中の順番のどれかが違えばキーが違い、NABLA:UNIFORM の値も違う
;;;;   （位置については 2^64 未満で全単射なので、確率でなく必ず成り立つ）。
;;;; - 位置なし・範囲外の位置で random-map-transform をかけると :bad-position、形の違うシードは
;;;;   :bad-seed の transform-error（map / filter だけなら位置は要らない）。
;;;; - ローダ: バッチの列は、オラクルが通し位置 k とローダのシードで APPLY-TRANSFORMS したものと
;;;;   一致する（filter・shuffle・エポックをまたぐ場合を含む）。
;;;; - 再開: 状態から再開した残りの列は、中断しなかった列と一致する。
;;;; - 拡張の例（左右反転、パディング付きのランダムな切り抜き）: 出力はパディングした画像
;;;;   （反転したかどうか）の窓のどれか。位置ごとに反転と切り抜きの位置が変わる。

(in-package #:nabla.data.tests)

;;; --- キー ---

(defun random-keys-of (seed position &optional (count 1))
  "SEED（nil なら省略）と通し位置 POSITION で、COUNT 個並べた random-map-transform が順に受けるキーの
リスト。"
  (let ((capture (random-map-transform (lambda (keys key) (append keys (list key))))))
    (if seed
        (apply-transforms (make-list count :initial-element capture) '() :seed seed :position position)
        (apply-transforms (make-list count :initial-element capture) '() :position position))))

(defun random-key-of (seed position)
  (first (random-keys-of seed position)))

(defun reference-mix64 (z)
  "SplitMix64 の出力関数（Steele・Lea・Flood 2014 の定数）の参照実装。"
  (let* ((z (ldb (byte 64 0) (* (logxor z (ash z -30)) #xBF58476D1CE4E5B9)))
         (z (ldb (byte 64 0) (* (logxor z (ash z -27)) #x94D049BB133111EB))))
    (logxor z (ash z -31))))

(defun reference-record-key (hi lo ordinal position)
  "版 1 のキー: 状態 mix64(mix64(シード) xor mix64(順番)) に位置 × γ を足して攪拌した 64 ビットを
[上位 32 ビット, 下位 32 ビット] に割る。"
  (let* ((state (reference-mix64 (logxor (reference-mix64 (+ (ash hi 32) lo)) (reference-mix64 ordinal))))
         (bits (reference-mix64 (ldb (byte 64 0) (+ state (* position #x9E3779B97F4A7C15))))))
    (sampler-test-seed (ash bits -32) (ldb (byte 32 0) bits))))

(test (random-map-transform/key-derivation-of-version-1 :suite :nabla.small)
  "参照実装の SplitMix64 は既知の出力列と一致し、版 1 のキーの固定値。シードを省略すると 0 のシード。"
  (is (equal '(1) (list nabla.data::+index-sampler-algorithm-version+)))
  ;; 状態 0 の SplitMix64 の出力列 #xE220A8397B1DCDAF, #x6E789E6AA1B965F4（参照実装の既知の値）
  (is (= #xE220A8397B1DCDAF (reference-mix64 #x9E3779B97F4A7C15)))
  (is (= #x6E789E6AA1B965F4 (reference-mix64 (ldb (byte 64 0) (* 2 #x9E3779B97F4A7C15)))))
  (is (equalp (reference-record-key 0 0 0 0) (random-key-of nil 0)))
  (is (equalp (random-key-of (sampler-test-seed 0 0) 5) (random-key-of nil 5)))
  (is (equalp (reference-record-key 1 2 1 7) (second (random-keys-of (sampler-test-seed 1 2) 7 2))))
  (is (equalp (sampler-test-seed #xCCA9DD8D #x7C0CF5CB) (random-key-of (sampler-test-seed 1 2) 7))))

(def-source-property random-map-transform/keys-follow-the-reference-derivation
  "キーは、シード・リストの中の順番・位置から参照実装で作ったキーと一致する（位置は 2^64 未満の全域）。"
  (hi lo k count)
  (generator (tuple (uniform-integer :lo 0 :hi 4294967295) (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 18446744073709551615) (uniform-integer :lo 1 :hi 3)))
  (equalp (loop for ordinal below count collect (reference-record-key hi lo ordinal k))
          (random-keys-of (sampler-test-seed hi lo) k count)))

(def-source-property random-map-transform/same-seed-and-position-give-the-same-key
  "同じ (シード, 位置) からは同じキーと同じ一様乱数。キーは呼ぶたびに新しい :u32 の (2) の配列。"
  (hi lo k)
  (generator (tuple (uniform-integer :lo 0 :hi 4294967295) (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 1099511627776)))
  (let ((a (random-key-of (sampler-test-seed hi lo) k))
        (b (random-key-of (sampler-test-seed hi lo) k)))
    (and (typep a '(simple-array (unsigned-byte 32) (2)))
         (equalp a b)
         (not (eq a b))
         (equalp (nabla:uniform a '(2) :dtype :f64) (nabla:uniform b '(2) :dtype :f64)))))

(def-source-property random-map-transform/different-positions-seeds-or-ordinals-give-different-keys
  "位置・シード・リストの中の順番のどれかが違えば、キーと一様乱数が違う。"
  (hi lo k d)
  (generator (tuple (uniform-integer :lo 0 :hi 4294967295) (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 1099511627776) (uniform-integer :lo 1 :hi 1000)))
  (destructuring-bind (key second-transform) (random-keys-of (sampler-test-seed hi lo) k 2)
    (let ((next-position (random-key-of (sampler-test-seed hi lo) (+ k d)))
          (other-seed (random-key-of (sampler-test-seed hi (mod (+ lo d) (ash 1 32))) k)))
      (and (not (equalp key next-position))
           (not (equalp key other-seed))
           (not (equalp key second-transform))
           (not (equalp (nabla:uniform key '(2) :dtype :f64)
                        (nabla:uniform next-position '(2) :dtype :f64)))))))

(defparameter *bad-positions* (list nil -1 1.5 "3" (ash 1 64))
  "通し位置にならない値。定数を直接渡すとコンパイル時に型の WARNING になるので、変数から渡す。")

(defparameter *bad-seeds* (list 0 "seed" (make-array 2 :initial-element 0)
                                (make-array 3 :element-type '(unsigned-byte 32) :initial-element 0))
  "シードにならない値（:u32 の (2) の配列でない）。")

(defun random-transform-failure-reason (thunk)
  "THUNK が signal した transform-error の reason。signal しなければ :no-error。"
  (handler-case (progn (funcall thunk) :no-error)
    (transform-error (c) (transform-error-reason c))))

(test (random-map-transform/rejects-bad-positions-and-seeds :suite :nabla.small)
  "位置なし・範囲外の位置で random-map-transform をかけると :bad-position、形の違うシードは（変換に
依らず）:bad-seed。どちらも data-error の子の transform-error。map / filter だけなら位置は要らない。
filter が落としたレコードには random-map-transform を呼ばない。"
  (dolist (position *bad-positions*)
    (is (eq :bad-position (random-transform-failure-reason (lambda () (random-key-of nil position))))))
  (is (eq :no-error (random-transform-failure-reason (lambda () (random-key-of nil (1- (ash 1 64)))))))
  (dolist (seed *bad-seeds*)
    (is (eq :bad-seed (random-transform-failure-reason
                       (lambda () (apply-transforms (list (map-transform #'1+)) 1 :seed seed))))))
  (is (typep (handler-case (random-key-of nil nil) (error (c) c)) 'data-error))
  (is (equal '(2 t) (multiple-value-list (apply-transforms (list (map-transform #'1+)) 1))))
  (is (equal '(nil nil)
             (multiple-value-list
              (apply-transforms (list (filter-transform #'evenp)
                                      (random-map-transform (lambda (r k) (error "呼ばない ~S ~S" r k))))
                                1))))
  (dolist (not-a-function *non-functions*)
    (signals type-error (random-map-transform not-a-function))))

;;; --- ローダ ---

(defun random-key-transform ()
  "レコードの末尾にキーを足す random-map-transform（ローダが渡したキーをバッチから読むため）。"
  (random-map-transform (lambda (record key) (append record (list key)))))

(def-source-property random-map-transform/loader-keys-each-record-by-its-position
  "ローダのバッチの列は、オラクルが通し位置 k とローダのシード（無ければ 0）で変換したものと一致する。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (let* ((arguments (list* :transforms (append (getf arguments :transforms) (list (random-key-transform)))
                             arguments))
           (expected (apply #'loader-oracle n arguments))
           (actual (loader-batches-or-reason (apply #'data-loader (loader-index-source n) arguments))))
      (if (eq expected :empty-epoch)
          (eq actual :empty-epoch)
          (and (listp actual)
               (= (length expected) (length actual))
               (every (lambda (records batch) (same-tree-of-arrays-p (batch-records records) batch))
                      expected actual))))))

(def-source-property random-map-transform/restoring-continues-the-uninterrupted-sequence
  "m 個のバッチを取った後の状態から再開した残りの列は、中断しなかった列の m 番目以降と一致する。"
  (seed m)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 0 :hi 30)))
  (multiple-value-bind (arguments n) (loader-random-case seed)
    (flet ((make-loader ()
             (apply #'data-loader (loader-index-source n)
                    :transforms (append (getf arguments :transforms) (list (random-key-transform)))
                    arguments)))
      (let ((all (loader-batches-or-reason (make-loader))))
        (or (eq all :empty-epoch)
            (let ((it (make-data-iterator (make-loader))))
              (loader-take it m)
              (let ((rest (loader-all-batches (restore-iterator (make-loader)
                                                                (loader-copy-state (iterator-state it))))))
                (and (= (length all) (+ (min m (length all)) (length rest)))
                     (every #'same-tree-of-arrays-p rest (nthcdr (min m (length all)) all))))))))))

;;; --- 拡張の例（画像分類の受け入れテスト #287 で使う形。README にも同じ例がある） ---

(defun example-flip-left-right (image)
  "(H W C) の IMAGE を左右（W の軸）に反転した新しい配列。"
  (destructuring-bind (h w c) (array-dimensions image)
    (let ((out (make-array (list h w c) :element-type (array-element-type image))))
      (dotimes (i h out)
        (dotimes (j w)
          (dotimes (k c)
            (setf (aref out i j k) (aref image i (- w 1 j) k))))))))

(defun example-crop-padded (image padding dy dx)
  "(H W C) の IMAGE の周りに 0 を PADDING 個ずつ足した画像から、(DY, DX) を左上とする (H W C) の窓。"
  (destructuring-bind (h w c) (array-dimensions image)
    (let ((out (make-array (list h w c) :element-type (array-element-type image)
                                        :initial-element (coerce 0 (array-element-type image)))))
      (dotimes (i h out)
        (dotimes (j w)
          (let ((si (+ i dy (- padding))) (sj (+ j dx (- padding))))
            (when (and (< -1 si h) (< -1 sj w))
              (dotimes (k c)
                (setf (aref out i j k) (aref image si sj k))))))))))

(defun example-augment (padding)
  "レコード (画像 ラベル) の画像を、確率 1/2 で左右に反転し、PADDING 個の 0 で囲んでランダムな位置で
元の大きさに切り抜く random-map-transform。乱数はキーから NABLA:UNIFORM で 3 つ引く。"
  (random-map-transform
   (lambda (record key)
     (destructuring-bind (image label) record
       (let* ((u (nabla:uniform key '(3) :dtype :f64))
              (span (1+ (* 2 padding)))
              ;; uniform は丸めで 1 ちょうどを返しうるので、オフセットは 2 * PADDING で止める
              (dy (min (* 2 padding) (floor (* (aref u 1) span))))
              (dx (min (* 2 padding) (floor (* (aref u 2) span)))))
         (list (example-crop-padded (if (< (aref u 0) 0.5d0) (example-flip-left-right image) image)
                                    padding dy dx)
               label))))))

(defun example-random-image (state h w c)
  (let ((image (make-array (list h w c) :element-type 'single-float)))
    (dotimes (i (array-total-size image) image)
      (setf (row-major-aref image i) (float (1+ (random 100 state)) 1f0)))))

(defun example-candidates (image padding)
  "IMAGE を反転するか・しないかと、[0, 2 PADDING] の (DY, DX) の窓をすべて並べたリスト。"
  (loop for flipped in (list image (example-flip-left-right image))
        append (loop for dy to (* 2 padding)
                     append (loop for dx to (* 2 padding)
                                  collect (example-crop-padded flipped padding dy dx)))))

(def-source-property random-map-transform/augmentation-is-a-window-of-the-padded-image
  "拡張の例の出力はパディングした画像（反転したかどうか）の窓のどれかで、ラベルはそのまま。
同じ (シード, 位置) なら同じ出力。"
  (seed k)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000) (uniform-integer :lo 0 :hi 100000)))
  (let* ((state (sb-ext:seed-random-state seed))
         (image (example-random-image state (1+ (random 4 state)) (1+ (random 4 state)) (1+ (random 3 state))))
         (padding (random 3 state))
         (transforms (list (example-augment padding)))
         (key (sampler-test-seed (random (ash 1 32) state) (random (ash 1 32) state)))
         (out (apply-transforms transforms (list image :label) :seed key :position k)))
    (and (eq :label (second out))
         (member (first out) (example-candidates image padding) :test #'equalp)
         (equalp out (apply-transforms transforms (list image :label) :seed key :position k)))))

(test (random-map-transform/augmentation-varies-with-the-position :suite :nabla.small)
  "同じレコードでも位置（エポック）ごとに拡張が変わり、64 の位置で反転する・しない両方と
複数の切り抜きの位置が出る。"
  (let* ((image (example-random-image (sb-ext:seed-random-state 7) 3 4 2))
         (transforms (list (example-augment 1)))
         (outs (loop for k below 64
                     collect (first (apply-transforms transforms (list image 0)
                                                      :seed (sampler-test-seed 1 2) :position k)))))
    ;; 候補の先頭の 9 個は反転しない窓、残りの 9 個は反転した窓
    (is (some (lambda (out) (find out (subseq (example-candidates image 1) 0 9) :test #'equalp)) outs))
    (is (some (lambda (out) (find out (subseq (example-candidates image 1) 9) :test #'equalp)) outs))
    (is (< 4 (length (remove-duplicates outs :test #'equalp))))))
