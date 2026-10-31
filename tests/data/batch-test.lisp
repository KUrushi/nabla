;;;; レコードをバッチに積む batch-records（issue #281）。
;;;;
;;;; 守らせる性質（small）:
;;;; - 往復: array-source の全レコードを順に積むと、元の木（同じ形・element-type・要素）に戻る。
;;;; - 順に積む: 任意の添字の列（重複・順不同、1 個以上）のレコードを積んだバッチの葉の j 番目の
;;;;   行は、j 番目のレコードの葉と一致する。
;;;; - 揃わないレコード: treedef・葉の形・element-type の食い違いは、理由つきの batch-error
;;;;   （食い違うレコードが列のどこにあっても）。レコードが無い・葉が無い・配列でない葉も batch-error。

(in-package #:nabla.data.tests)

;;; --- 生成とオラクル ---

(defun batch-random-source (seed)
  "SEED から決まる、長さ 1 以上の (values 木 array-source 長さ)。"
  (multiple-value-bind (tree length) (source-random-source-tree seed)
    (let ((tree (if (plusp length) tree (source-random-leaf 3 (sb-ext:seed-random-state seed)))))
      (values tree (make-array-source tree) (if (plusp length) length 3)))))

(defun same-array-p (a b)
  (and (equal (array-element-type a) (array-element-type b))
       (equal (array-dimensions a) (array-dimensions b))
       (loop for i below (array-total-size a)
             always (eql (row-major-aref a i) (row-major-aref b i)))))

(defun same-tree-of-arrays-p (a b)
  (and (nabla:treedef= (nabla:tree-structure a) (nabla:tree-structure b))
       (every #'same-array-p (nabla:tree-leaves a) (nabla:tree-leaves b))))

(defun batch-failure-reason (thunk)
  "THUNK が signal した batch-error の (values reason detail)。signal しなければ :no-error。"
  (handler-case (progn (funcall thunk) :no-error)
    (batch-error (c) (values (batch-error-reason c) (batch-error-detail c)))))

(defun replace-first-leaf (record function)
  "RECORD の最初の葉を (FUNCTION 葉) に置き換えた同じ形の木。"
  (multiple-value-bind (leaves treedef) (nabla:tree-flatten record)
    (nabla:tree-unflatten treedef (cons (funcall function (first leaves)) (rest leaves)))))

(defun mismatched-record (kind record)
  "RECORD と KIND（:treedef-mismatch / :shape-mismatch / :element-type-mismatch）だけが
食い違うレコード。"
  (ecase kind
    (:treedef-mismatch (list record))
    (:shape-mismatch
     (replace-first-leaf record (lambda (leaf)
                                  (make-array (cons 1 (array-dimensions leaf))
                                              :element-type (array-element-type leaf)))))
    (:element-type-mismatch
     (replace-first-leaf record (lambda (leaf)
                                  (make-array (array-dimensions leaf)
                                              :element-type (if (eq 'double-float (array-element-type leaf))
                                                                'single-float
                                                                'double-float)))))))

;;; --- 積む ---

(def-source-property batch-records/stacking-every-record-rebuilds-the-source-tree
  "array-source の全レコードを順に積むと、元の木に戻る。"
  (seed)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)))
  (multiple-value-bind (tree source length) (batch-random-source seed)
    (same-tree-of-arrays-p tree (batch-records (loop for i below length
                                                      collect (source-item source i))))))

(def-source-property batch-records/row-j-is-the-j-th-record
  "任意の添字の列のレコードを積むと、バッチの葉の j 番目の行は j 番目のレコードの葉。"
  (seed indices)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)
                    (list (uniform-integer :lo 0 :hi 100) :min-length 1 :max-length 6)))
  (multiple-value-bind (tree source length) (batch-random-source seed)
    (declare (ignore tree))
    (let* ((records (mapcar (lambda (i) (source-item source (mod i length))) indices))
           (batch (and records (batch-records records))))
      ;; 縮小（shrinking）は :min-length を下回る空の列を作りうるので、空なら性質の外。
      (or (null records)
          (and (nabla:treedef= (nabla:tree-structure batch) (nabla:tree-structure (first records)))
           (loop for record in records
                 for j from 0
                 always (every (lambda (batch-leaf record-leaf) (row-matches-p batch-leaf j record-leaf))
                               (nabla:tree-leaves batch)
                               (nabla:tree-leaves record))))))))

;;; --- 揃わないレコード ---

(def-source-property batch-records/reports-the-mismatched-record
  "treedef・形・element-type が食い違うレコードがどこにあっても、その理由の batch-error。
detail は最初のレコードと食い違うレコードの位置（0 始まり。先頭が食い違うなら 1）を示す。"
  (seed kind position count)
  (generator (tuple (uniform-integer :lo 0 :hi 1000000)
                    (uniform-integer :lo 0 :hi 2)
                    (uniform-integer :lo 0 :hi 3)
                    (uniform-integer :lo 1 :hi 3)))
  (multiple-value-bind (tree source) (batch-random-source seed)
    (declare (ignore tree))
    (let* ((reason (nth kind '(:treedef-mismatch :shape-mismatch :element-type-mismatch)))
           (records (loop repeat count collect (source-item source 0)))
           (position (min position count))
           (records (append (subseq records 0 position)
                            (list (mismatched-record reason (source-item source 0)))
                            (subseq records position))))
      (multiple-value-bind (actual detail) (batch-failure-reason (lambda () (batch-records records)))
        (and (eq reason actual)
             (search (format nil "~D 番目" (max position 1)) detail))))))

(test (batch-records/rejects-an-empty-list :suite :nabla.small)
  (is (eq :no-records (batch-failure-reason (lambda () (batch-records '()))))))

(test (batch-records/rejects-records-without-leaves :suite :nabla.small)
  "葉の無いレコードはバッチの軸を作れない（バッチの大きさが黙って失われないように）。"
  (is (eq :no-leaves (batch-failure-reason (lambda () (batch-records (list nil nil))))))
  (is (eq :no-leaves (batch-failure-reason (lambda () (batch-records (list (list nil) (list nil))))))))

(test (batch-records/rejects-leaves-that-are-not-arrays :suite :nabla.small)
  (is (eq :bad-leaf (batch-failure-reason (lambda () (batch-records (list (list 1)))))))
  (is (eq :bad-leaf (batch-failure-reason (lambda () (batch-records (list (list 1) (list 2)))))))
  (is (eq :bad-leaf (batch-failure-reason
                     (lambda () (batch-records (list (list :x) (list (f32-vector 1.0))))))))
  (is (eq :bad-leaf (batch-failure-reason
                     (lambda () (batch-records (list (list (f32-vector 1.0)) (list :x))))))))

(test (batch-records/stacks-bf16-leaves-by-element-type :suite :nabla.small)
  "(unsigned-byte 16) の葉（bf16 / f16）は element-type だけを見て積む。"
  (let ((a (make-array '(2) :element-type '(unsigned-byte 16) :initial-contents '(1 2)))
        (b (make-array '(2) :element-type '(unsigned-byte 16) :initial-contents '(3 4))))
    (is (same-array-p (make-array '(2 2) :element-type '(unsigned-byte 16)
                                         :initial-contents '((1 2) (3 4)))
                      (batch-records (list a b))))))

(test (batch-error/is-a-data-error :suite :nabla.small)
  (signals data-error (batch-records '())))
