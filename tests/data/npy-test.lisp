;;;; .npy の読み書きと npy-source（issue #282）。ファイル I/O なのですべて medium。
;;;;
;;;; 守らせる性質:
;;;; - numpy との一致（例ベース）: numpy が書いた各 dtype・形（rank 0、長さ 0 の軸、ヘッダの版
;;;;   1.0 / 2.0 / 3.0）のファイルを読むと、tests/fixtures/npy/expected.lisp の element-type・形・
;;;;   値（浮動小数点はビット列）と一致する。bool は (unsigned-byte 8) の 0 / 1。
;;;; - 読めないファイル: Fortran 順・ビッグエンディアン・object / complex / 構造化 dtype・壊れた
;;;;   ヘッダ・データの長さの食い違い・0 / 1 でない bool は、理由つきの npy-error。
;;;; - 往復: write-npy → read-npy で同じ element-type（bit は (unsigned-byte 8)）・形・値
;;;;   （浮動小数点は eql。-0.0 と NaN を含む）が戻り、ヘッダは numpy と同じく 64 バイト境界で終わる。
;;;; - numpy と同じバイト列: numpy が書いたファイルを読んで write-npy で書き直すと、同じバイト列になる。
;;;; - 1 バイトの dtype のバイト順の記号、1 を超える軸が 1 つ以下の Fortran 順、Python 2 の (3L,) は読む。
;;;; - npy-source: パス名の木の各ファイルを読んだ配列の木の array-source と同じレコードを返す。

(in-package #:nabla.data.tests)

;;; --- 補助 ---

(defun npy-fixture (name)
  (asdf:system-relative-pathname "nabla" (format nil "tests/fixtures/npy/~A" name)))

(defparameter *npy-kind-element-types*
  '((:f32 . single-float) (:f64 . double-float) (:i32 . (signed-byte 32)) (:i64 . (signed-byte 64))
    (:u32 . (unsigned-byte 32)) (:u64 . (unsigned-byte 64)) (:u8 . (unsigned-byte 8))
    (:i8 . (signed-byte 8)) (:bool . (unsigned-byte 8))))

(defun npy-element-value (x)
  "浮動小数点はビット列（u32 / u64）、整数はそのまま。"
  (etypecase x
    (single-float (ldb (byte 32 0) (sb-kernel:single-float-bits x)))
    (double-float (logior (ash (ldb (byte 32 0) (sb-kernel:double-float-high-bits x)) 32)
                          (sb-kernel:double-float-low-bits x)))
    (integer x)))

(defun npy-array-values (array)
  (loop for i below (array-total-size array) collect (npy-element-value (row-major-aref array i))))

(defun npy-reason (thunk)
  (handler-case (progn (funcall thunk) :no-error)
    (npy-error (c) (npy-error-reason c))))

(defun npy-octets (header &key (version '(1 0)) (data '()) (magic '(#x93 78 85 77 80 89)))
  "ヘッダの文字列 HEADER（改行と詰め物は付けない）と DATA（バイトのリスト）から .npy のバイト列を作る。"
  (let* ((text (format nil "~A~%" header))
         (length-octets (if (= 1 (first version)) 2 4)))
    (coerce (append magic version
                    (loop for i below length-octets collect (ldb (byte 8 (* 8 i)) (length text)))
                    (map 'list #'char-code text)
                    data)
            '(vector (unsigned-byte 8)))))

(defun read-npy-octets (dir octets)
  "OCTETS を DIR のファイルに書いて READ-NPY で読む。"
  (let ((path (merge-pathnames "raw.npy" dir)))
    (with-open-file (out path :direction :output :element-type '(unsigned-byte 8) :if-exists :supersede)
      (write-sequence octets out))
    (read-npy path)))

(defun raw-npy-reason (dir header &rest keys)
  (npy-reason (lambda () (read-npy-octets dir (apply #'npy-octets header keys)))))

;;; --- numpy が書いたファイル ---

(test (npy/reads-files-written-by-numpy :suite :nabla.medium)
  "numpy の各 dtype・形・ヘッダの版のファイルを読むと、element-type・形・値が期待値と一致する。"
  (let ((expected (with-open-file (in (npy-fixture "expected.lisp"))
                    (let ((*read-eval* nil)) (read in)))))
    (is (= 13 (length expected)))
    (loop for (name kind shape values) in expected
          for array = (read-npy (npy-fixture name))
          do (is (equal (cdr (assoc kind *npy-kind-element-types*)) (array-element-type array))
                 "~A: element-type ~S" name (array-element-type array))
             (is (equal shape (array-dimensions array)) "~A: 形 ~S" name (array-dimensions array))
             (is (equal values (npy-array-values array)) "~A: 値が違う" name))))

(test (npy/rejects-files-numpy-can-write-but-nabla-cannot-read :suite :nabla.medium)
  (is (eq :fortran-order (npy-reason (lambda () (read-npy (npy-fixture "error-fortran.npy"))))))
  (is (eq :big-endian (npy-reason (lambda () (read-npy (npy-fixture "error-big-endian.npy"))))))
  (dolist (name '("error-object.npy" "error-complex.npy" "error-structured.npy"))
    (is (eq :unsupported-dtype (npy-reason (lambda () (read-npy (npy-fixture name))))) "~A" name)))

;;; --- 手で作ったヘッダ ---

(test (npy/reads-header-variants :suite :nabla.medium)
  "キーの順・空白・二重引用符・最後のカンマの有無に依らず読める。"
  (with-temporary-directory (dir)
    (let ((array (read-npy-octets dir (npy-octets "{\"shape\":(2,) ,'fortran_order' :False,'descr':\"<i4\"}"
                                                  :data '(7 0 0 0 255 255 255 255)))))
      (is (equalp #(7 -1) array))
      (is (equal '(signed-byte 32) (array-element-type array))))
    (is (equal '(2 1) (array-dimensions
                       (read-npy-octets dir (npy-octets "{'descr': '|u1', 'fortran_order': False, 'shape': (2,1)}"
                                                        :data '(1 2))))))
    ;; 1 バイトの dtype のバイト順の記号は問わない。1 を超える軸が 1 つなら Fortran 順も C 順と同じ。
    ;; Python 2 の numpy は形を (3L,) と書く
    (let ((array (read-npy-octets dir (npy-octets "{'descr': '<u1', 'fortran_order': True, 'shape': (1, 3L)}"
                                                  :data '(1 2 3)))))
      (is (equal '((1 3) (1 2 3)) (list (array-dimensions array) (npy-array-values array)))))
    (is (equalp #(-1) (read-npy-octets dir (npy-octets "{'descr': '=i1', 'fortran_order': False, 'shape': (1,)}"
                                                       :data '(255)))))
    (is (equalp #(1 0) (read-npy-octets dir (npy-octets "{'descr': '<b1', 'fortran_order': False, 'shape': (2,)}"
                                                        :data '(1 0)))))
    ;; タブと復帰も空白
    (is (equalp #(9) (read-npy-octets dir (npy-octets (format nil "{'descr':~C'|u1',~C'fortran_order': False, 'shape': (1,)}"
                                                              #\Tab #\Return)
                                                      :data '(9)))))))

(test (npy/rejects-broken-files :suite :nabla.medium)
  (with-temporary-directory (dir)
    (flet ((reason (header &rest keys) (apply #'raw-npy-reason dir header keys)))
      (let ((ok "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }")
            (four '(0 0 128 63)))
        (is (eq :no-error (reason ok :data four)))
        (is (eq :invalid-header (reason ok :data four :magic '(#x93 78 85 77 80 90))))
        (is (eq :unsupported-version (reason ok :data four :version '(4 0))))
        (is (eq :unsupported-version (reason ok :data four :version '(0 0))))
        (is (eq :unsupported-version (reason ok :data four :version '(1 1))))
        (is (eq :size-mismatch (reason ok :data '(0 0 128))))
        (is (eq :size-mismatch (reason ok :data '(0 0 128 63 0))))
        ;; 壊れた巨大な shape は、配列を確保する前にファイルの大きさで弾く
        (is (eq :size-mismatch (reason "{'descr': '<f4', 'fortran_order': False, 'shape': (100000, 100000), }"
                                       :data four)))
        (is (eq :fortran-order (reason "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 1, 2), }"
                                       :data (append four four four four))))
        (is (eq :big-endian (reason "{'descr': '>i4', 'fortran_order': False, 'shape': (1,), }" :data four)))
        ;; SBCL の配列の上限（rank・各次元・要素数）を超える shape
        (dolist (shape (list (make-list array-rank-limit :initial-element 0)
                             (list 0 array-dimension-limit)
                             (list (expt 2 31) (expt 2 31) (expt 2 31))))
          (is (eq :invalid-header
                  (reason (format nil "{'descr': '|u1', 'fortran_order': False, 'shape': (~{~D,~})}" shape)))
              "shape ~S" shape))
        (is (eq :invalid-bool (reason "{'descr': '|b1', 'fortran_order': False, 'shape': (2,), }" :data '(1 2))))
        (is (eq :unsupported-dtype (reason "{'descr': '<c8', 'fortran_order': False, 'shape': (), }")))
        (dolist (header '("" "[]" "{'descr': '<f4', 'fortran_order': False}"
                          "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'x': 1}"
                          "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (1,)}"
                          "{'descr': '<f4', 'descr': '<f4', 'shape': (1,)}"
                          "{'descr': '<f4', 'fortran_order': False, 'shape': (1)}"
                          "{'descr': '<f4', 'fortran_order': False, 'shape': (-1,)}"
                          "{'descr': '<f4', 'fortran_order': False, 'shape': (1,,)}"
                          "{'descr': '<f4', 'fortran_order': 0, 'shape': (1,)}"
                          "{'descr': 4, 'fortran_order': False, 'shape': (1,)}"
                          "{'descr': '<f4, 'fortran_order': False, 'shape': (1,)}"
                          "{'descr': '<f4', 'fortran_order': False, 'shape': (1,)} x"))
          (is (eq :invalid-header (reason header :data four)) "~S" header))
        (is (eq :invalid-header
                (npy-reason (lambda () (read-npy-octets dir (subseq (npy-octets ok :data four) 0 9))))))
        (is (eq :invalid-header
                (npy-reason (lambda () (read-npy-octets dir (subseq (npy-octets ok) 0 30))))))
        (is (eq :invalid-header
                (npy-reason (lambda () (read-npy-octets dir (subseq (npy-octets ok :version '(2 0)) 0 11))))))))))

;;; --- 往復 ---

(defparameter *npy-writable-element-types*
  '(single-float double-float (signed-byte 8) (signed-byte 32) (signed-byte 64)
    (unsigned-byte 8) (unsigned-byte 32) (unsigned-byte 64) bit))

(defun npy-random-element (element-type state)
  (cond ((member element-type '(single-float double-float))
         (let ((one (coerce 1 element-type)))
           (case (random 8 state)
             (0 (- (coerce 0 element-type)))
             (1 (if (eq element-type 'single-float)
                    (sb-kernel:make-single-float #x7FC00001)
                    (sb-kernel:make-double-float #x7FF80000 1)))
             (t (- (random (* 2 one) state) one)))))
        ((eq element-type 'bit) (random 2 state))
        ((eq (first element-type) 'signed-byte)
         (- (random (ash 1 (second element-type)) state) (ash 1 (1- (second element-type)))))
        (t (random (ash 1 (second element-type)) state))))

(defun npy-random-array (seed)
  "SEED から決まる、書ける element-type・rank 0..3・各次元 0..3 の配列。たまに simple でない
（adjustable、または大きなベクタの途中に displaced した）配列にする。"
  (let* ((state (sb-ext:seed-random-state seed))
         (element-type (nth (random (length *npy-writable-element-types*) state)
                            *npy-writable-element-types*))
         (dimensions (loop repeat (random 4 state) collect (random 4 state)))
         (array (ecase (random 4 state)
                  ((0 1) (make-array dimensions :element-type element-type))
                  (2 (make-array dimensions :element-type element-type :adjustable t))
                  (3 (let ((base (make-array (+ 2 (reduce #'* dimensions)) :element-type element-type)))
                       (dotimes (i (length base))
                         (setf (aref base i) (npy-random-element element-type state)))
                       (make-array dimensions :element-type element-type
                                              :displaced-to base :displaced-index-offset 1))))))
    (dotimes (i (array-total-size array) array)
      (setf (row-major-aref array i) (npy-random-element element-type state)))))

(defun npy-roundtrip-p (dir array)
  (let ((path (merge-pathnames "a.npy" dir)))
    (write-npy path array)
    (let ((read (read-npy path))
          (preamble (make-array 10 :element-type '(unsigned-byte 8))))
      (with-open-file (in path :element-type '(unsigned-byte 8))
        (read-sequence preamble in))
      (and (equal (if (eq 'bit (array-element-type array)) '(unsigned-byte 8) (array-element-type array))
                  (array-element-type read))
           (equal (array-dimensions array) (array-dimensions read))
           (equal (npy-array-values array) (npy-array-values read))
           ;; 版 1.0 で、ヘッダ長（u16）を足したデータの開始位置が 64 の倍数
           (equalp #(1 0) (subseq preamble 6 8))
           (zerop (mod (+ 10 (aref preamble 8) (* 256 (aref preamble 9))) 64))))))

(test (npy/write-read-roundtrip :suite :nabla.medium)
  "write-npy → read-npy で同じ element-type・形・値が戻り、ヘッダは 64 バイト境界で終わる。"
  (with-temporary-directory (dir)
    (is (check-it (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
                  (lambda (case) (npy-roundtrip-p dir (npy-random-array (first case))))
                  :regression-id npy/write-read-roundtrip
                  :regression-file (regression-path "npy-write-read-roundtrip" :package "NABLA.DATA.TESTS"))
        "write-npy → read-npy の往復が成り立たなかった")))

(defun file-octets (path)
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((octets (make-array (file-length in) :element-type '(unsigned-byte 8))))
      (read-sequence octets in)
      octets)))

(test (npy/write-reproduces-the-bytes-numpy-wrote :suite :nabla.medium)
  "numpy が版 1.0 で書いたファイルを読んで write-npy で書き直すと、ヘッダの詰め物まで同じバイト列になる
（bool は |u1 として書くので除く）。"
  (with-temporary-directory (dir)
    (dolist (name '("f32.npy" "f64.npy" "i32.npy" "i64.npy" "u32.npy" "u64.npy" "u8.npy" "i8.npy"
                    "scalar.npy" "empty.npy"))
      (let ((path (merge-pathnames name dir)))
        (write-npy path (read-npy (npy-fixture name)))
        (is (equalp (file-octets (npy-fixture name)) (file-octets path)) "~A" name)))))

(test (npy/write-read-roundtrip-of-a-long-header :suite :nabla.medium)
  "rank が大きくヘッダが 256 バイトを超えても往復する（rank 110 でヘッダ長は #x1B6。u16 の上位バイトと
下位バイトの最上位ビットが立つ）。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "long.npy" dir))
          (array (make-array (make-list 110 :initial-element 1) :element-type 'single-float
                                                               :initial-element 2.5)))
      (write-npy path array)
      (is (equalp #(#xB6 #x01) (subseq (file-octets path) 8 10)))
      (is (equalp array (read-npy path))))))

(test (npy/write-read-roundtrip-across-chunks :suite :nabla.medium)
  "作業用バッファ（64 KiB）より大きなデータも往復する。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "big.npy" dir))
          (array (make-array '(3 10000) :element-type 'single-float)))
      (dotimes (i (array-total-size array))
        (setf (row-major-aref array i) (float i)))
      (write-npy path array)
      (is (equalp array (read-npy path))))))

(test (npy/write-rejects-unsupported-element-types :suite :nabla.medium)
  "nabla に読めない element-type は :unsupported-dtype で、ファイルを作らない。"
  (with-temporary-directory (dir)
    (let ((path (merge-pathnames "a.npy" dir)))
      (is (eq :unsupported-dtype
              (npy-reason (lambda () (write-npy path (make-array 2 :element-type '(unsigned-byte 16)))))))
      (is (eq :unsupported-dtype (npy-reason (lambda () (write-npy path (vector 1 2))))))
      (is (null (directory (merge-pathnames "*.*" dir)))))))

;;; --- npy-source ---

(test (npy-source/records-are-rows-of-the-files :suite :nabla.medium)
  "パス名の木の各ファイルを読み、array-source と同じレコード（同じ形の木）を返す。"
  (with-temporary-directory (dir)
    (let ((x (merge-pathnames "x.npy" dir))
          (y (merge-pathnames "y.npy" dir))
          (xs (make-array '(3 2) :element-type 'single-float
                                 :initial-contents '((1.0 2.0) (3.0 4.0) (5.0 6.0))))
          (ys (make-array 3 :element-type '(signed-byte 32) :initial-contents '(7 8 9))))
      (write-npy x xs)
      (write-npy y ys)
      (let ((source (npy-source (list (namestring x) (vector y)))))
        (is (= 3 (source-length source)))
        (dotimes (i 3)
          (is (equalp (source-item (make-array-source (list xs (vector ys))) i)
                      (source-item source i))))))))

(test (npy-source/rejects-bad-trees :suite :nabla.medium)
  "パス名でない葉は :bad-leaf、rank 0 のファイルは :rank-zero-leaf、長さの食い違いは :length-mismatch。"
  (with-temporary-directory (dir)
    (let ((scalar (merge-pathnames "s.npy" dir))
          (two (merge-pathnames "two.npy" dir))
          (three (merge-pathnames "three.npy" dir)))
      (write-npy scalar (make-array '() :element-type 'single-float :initial-element 1.0))
      (write-npy two (make-array 2 :element-type 'single-float :initial-element 1.0))
      (write-npy three (make-array 3 :element-type 'single-float :initial-element 1.0))
      (is (eq :bad-leaf (source-error-reason (lambda () (npy-source (list two 3))))))
      (is (eq :rank-zero-leaf (source-error-reason (lambda () (npy-source scalar)))))
      (is (eq :length-mismatch (source-error-reason (lambda () (npy-source (list two three)))))))))
