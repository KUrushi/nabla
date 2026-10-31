;;;; index-sampler（issue #280）。
;;;;
;;;; 守らせる性質（small）:
;;;; - 決定性: 同じ引数で作ったサンプラーは同じ列を返す（シードの配列を後で書き換えても変わらない）。
;;;; - 置換: 位置 k の (epoch index) は epoch = floor(k / n) で、各エポックの index の列は 0..n-1 の置換。
;;;; - 順序の違い: シードかエポックが違えば違う順序になる（n ≥ 4 で、8 通りの別のシード・エポックの
;;;;   どれかと違えばよい。n ≤ 3 は順序の数が少なく、偶然一致してよい）。n ≥ 2 なら、シードを変えると
;;;;   偶置換と奇置換の両方が出る（均衡 Feistel は偶置換しか作らないので、領域の取り方を守る）。
;;;; - 再開: 位置 k から始めた列は、0 から始めた列の k 番目以降と一致する。
;;;; - shuffle nil: 各エポックが 0, 1, ..., n-1 の順。
;;;; - 範囲外: 負・整数でない・n × num-epochs 以上の位置は :position-out-of-range。不正な引数は理由つき。
;;;; - 固定値: 決まったシードの順序（版をまたいでチェックポイントから再開する列が変わらないこと）。

(in-package #:nabla.data.tests)

(defparameter *sampler-boundary-lengths*
  '(0 1 2 3 4 5 15 16 17 63 64 65 255 256 257)
  "cycle-walking の領域 2^(2h) の境界（4 の冪とその前後）を含む、レコードの個数の候補。")

(defun sampler-test-length (selector)
  "SELECTOR（0..59）を n に写す。0..39 はそのまま、残りは境界の候補。"
  (if (< selector 40)
      selector
      (nth (mod selector (length *sampler-boundary-lengths*)) *sampler-boundary-lengths*)))

(defun sampler-test-seed (hi lo)
  (make-array 2 :element-type '(unsigned-byte 32) :initial-contents (list hi lo)))

(defun epoch-order (sampler n epoch)
  "SAMPLER の EPOCH 番目のエポックの index の列。"
  (loop for i below n collect (nth-value 1 (sampler-index sampler (+ (* epoch n) i)))))

(defun permutation-p (indices n)
  (equal (sort (copy-list indices) #'<) (loop for i below n collect i)))

(defun permutation-odd-p (indices)
  "INDICES（0..n-1 の置換）が奇置換なら真。偶数長の巡回の個数の偶奇で決める。"
  (let* ((perm (coerce indices 'simple-vector))
         (seen (make-array (length perm) :element-type 'bit :initial-element 0))
         (odd nil))
    (dotimes (start (length perm) odd)
      (when (zerop (aref seen start))
        (let ((cycle-length 0))
          (loop for i = start then (svref perm i)
                until (= 1 (aref seen i))
                do (setf (aref seen i) 1) (incf cycle-length))
          (when (evenp cycle-length) (setf odd (not odd))))))))

(defun sampler-error-reason-of (thunk)
  (handler-case (progn (funcall thunk) :no-error)
    (sampler-error (c) (sampler-error-reason c))))

(defparameter *sampler-case-generator*
  (generator (tuple (uniform-integer :lo 0 :hi 59)
                    (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 4294967295)
                    (uniform-integer :lo 0 :hi 5)))
  "(長さの選択子 シードの上位語 シードの下位語 エポック) の生成器。")

;;; --- 決定性と置換 ---

(def-source-property index-sampler/same-arguments-give-the-same-sequence
  "同じ引数で別々に作ったサンプラーは、全位置で同じ (epoch index) を返す。シードの配列を作った
後で書き換えても列は変わらない。"
  (selector hi lo epochs)
  *sampler-case-generator*
  (let* ((n (sampler-test-length selector))
         (seed (sampler-test-seed hi lo))
         (a (make-index-sampler n :shuffle t :seed seed :num-epochs (1+ epochs)))
         (before (loop for k below (* n (1+ epochs)) collect (multiple-value-list (sampler-index a k)))))
    (fill seed (logxor hi 1))
    (let ((b (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo) :num-epochs (1+ epochs))))
      (and (equal before (loop for k below (* n (1+ epochs)) collect (multiple-value-list (sampler-index a k))))
           (equal before (loop for k below (* n (1+ epochs)) collect (multiple-value-list (sampler-index b k))))))))

(def-source-property index-sampler/each-epoch-is-a-permutation
  "位置 k の epoch は floor(k / n) で、各エポックの index の列は 0..n-1 の置換。"
  (selector hi lo epoch)
  *sampler-case-generator*
  (let* ((n (sampler-test-length selector))
         (sampler (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo))))
    (and (permutation-p (epoch-order sampler n epoch) n)
         (loop for i below n
               always (= epoch (sampler-index sampler (+ (* epoch n) i)))))))

(def-source-property index-sampler/a-far-epoch-is-a-permutation
  "num-epochs が nil なら、64 ビットを超えるエポックでも置換になる。"
  (selector hi lo extra)
  *sampler-case-generator*
  (let* ((n (sampler-test-length selector))
         (epoch (+ (expt 2 64) extra))
         (sampler (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo))))
    (permutation-p (epoch-order sampler n epoch) n)))

;;; --- 順序の違い ---

(def-source-property index-sampler/another-seed-gives-another-order
  "n ≥ 4 なら、シードの下位語の 1 ビットを変えた 8 通りのどれかで順序が変わり、上位語でも同じ。"
  (selector hi lo epoch)
  *sampler-case-generator*
  (let* ((n (max 4 (sampler-test-length selector)))
         (order (epoch-order (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo)) n epoch)))
    (and (some (lambda (bit)
                 (not (equal order (epoch-order (make-index-sampler n :shuffle t
                                                                      :seed (sampler-test-seed hi (logxor lo (ash 1 bit))))
                                                n epoch))))
               '(0 1 2 3 4 5 16 31))
         (some (lambda (bit)
                 (not (equal order (epoch-order (make-index-sampler n :shuffle t
                                                                      :seed (sampler-test-seed (logxor hi (ash 1 bit)) lo))
                                                n epoch))))
               '(0 1 2 3 4 5 16 31)))))

(def-source-property index-sampler/another-epoch-gives-another-order
  "n ≥ 4 なら、続く 8 個のエポックのどれかの順序は EPOCH の順序と違う。"
  (selector hi lo epoch)
  *sampler-case-generator*
  (let* ((n (max 4 (sampler-test-length selector)))
         (sampler (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo)))
         (order (epoch-order sampler n epoch)))
    (some (lambda (other) (not (equal order (epoch-order sampler n other))))
          (loop for other from (1+ epoch) repeat 8 collect other))))

(def-source-property index-sampler/orders-of-both-parities-occur
  "n ≥ 2 なら、シードの下位語を変えた 32 通りの順序に偶置換と奇置換の両方がある。"
  (selector hi lo epoch)
  *sampler-case-generator*
  (let* ((n (max 2 (sampler-test-length selector)))
         (parities (loop for i below 32
                         collect (permutation-odd-p
                                  (epoch-order (make-index-sampler n :shuffle t
                                                                     :seed (sampler-test-seed hi (logxor lo i)))
                                               n epoch)))))
    (and (member t parities) (member nil parities) t)))

;;; --- 再開 ---

(def-source-property index-sampler/resuming-at-k-continues-the-sequence
  "位置 k から作り直したサンプラーで引いた列は、0 から引いた列の k 番目以降と一致する。"
  (selector hi lo k)
  *sampler-case-generator*
  (let* ((n (sampler-test-length selector))
         (total (* n 3))
         (k (if (zerop total) 0 (mod (* k 7) total)))
         (whole (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo) :num-epochs 3))
         (resumed (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo) :num-epochs 3)))
    (equal (nthcdr k (loop for j below total collect (multiple-value-list (sampler-index whole j))))
           (loop for j from k below total collect (multiple-value-list (sampler-index resumed j))))))

;;; --- shuffle nil ---

(def-source-property index-sampler/without-shuffle-each-epoch-is-in-order
  "shuffle が nil なら、位置 k は (floor(k / n), k mod n)。シードは順序に効かない。"
  (selector hi lo epoch)
  *sampler-case-generator*
  (let* ((n (sampler-test-length selector))
         (plain (make-index-sampler n))
         (seeded (make-index-sampler n :seed (sampler-test-seed hi lo))))
    (and (equal (loop for i below n collect i) (epoch-order plain n epoch))
         (equal (loop for i below n collect i) (epoch-order seeded n epoch))
         (loop for i below n always (= epoch (sampler-index plain (+ (* epoch n) i)))))))

;;; --- 範囲外と不正な引数 ---

(def-source-property index-sampler/positions-out-of-range-are-reported
  "負の位置と n × num-epochs 以上の位置は :position-out-of-range。直前の位置は引ける。"
  (selector hi lo epochs)
  *sampler-case-generator*
  (let* ((n (sampler-test-length selector))
         (total (* n epochs))
         (sampler (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo) :num-epochs epochs))
         (endless (make-index-sampler n :shuffle t :seed (sampler-test-seed hi lo))))
    (and (eq :position-out-of-range (sampler-error-reason-of (lambda () (sampler-index sampler total))))
         (eq :position-out-of-range (sampler-error-reason-of (lambda () (sampler-index sampler -1))))
         (eq :position-out-of-range (sampler-error-reason-of (lambda () (sampler-index endless (- -1 lo)))))
         (or (zerop total) (eq :no-error (sampler-error-reason-of (lambda () (sampler-index sampler (1- total))))))
         (if (zerop n)
             (eq :position-out-of-range (sampler-error-reason-of (lambda () (sampler-index endless lo))))
             (eq :no-error (sampler-error-reason-of (lambda () (sampler-index endless (* lo n)))))))))

(test (index-sampler/non-integer-positions-are-reported :suite :nabla.small)
  (let ((sampler (make-index-sampler 3)))
    (is (eq :position-out-of-range (sampler-error-reason-of (lambda () (sampler-index sampler 1.0)))))
    (is (eq :position-out-of-range (sampler-error-reason-of (lambda () (sampler-index sampler nil)))))))

(test (index-sampler/rejects-bad-arguments :suite :nabla.small)
  (is (eq :bad-length (sampler-error-reason-of (lambda () (make-index-sampler -1)))))
  (is (eq :bad-length (sampler-error-reason-of (lambda () (make-index-sampler 2.0)))))
  (is (eq :bad-length (sampler-error-reason-of (lambda () (make-index-sampler (expt 2 64))))))
  (is (eq :no-error (sampler-error-reason-of (lambda () (make-index-sampler (1- (expt 2 64)))))))
  (is (eq :bad-num-epochs (sampler-error-reason-of (lambda () (make-index-sampler 2 :num-epochs -1)))))
  (is (eq :bad-num-epochs (sampler-error-reason-of (lambda () (make-index-sampler 2 :num-epochs 1.5)))))
  (is (eq :no-error (sampler-error-reason-of (lambda () (make-index-sampler 2 :num-epochs 0)))))
  (is (eq :bad-seed (sampler-error-reason-of (lambda () (make-index-sampler 2 :shuffle t)))))
  (is (eq :bad-seed (sampler-error-reason-of (lambda () (make-index-sampler 2 :shuffle t :seed 7)))))
  (is (eq :bad-seed (sampler-error-reason-of
                     (lambda () (make-index-sampler 2 :seed (make-array 3 :element-type '(unsigned-byte 32)
                                                                          :initial-element 0))))))
  (is (eq :bad-seed (sampler-error-reason-of
                     (lambda () (make-index-sampler 2 :shuffle t
                                                      :seed (make-array 2 :element-type '(signed-byte 32)
                                                                          :initial-element 0)))))))

(test (sampler-error/is-a-data-error :suite :nabla.small)
  (signals data-error (make-index-sampler -1)))

;;; --- 固定値 ---

(test (index-sampler/order-of-a-fixed-seed-is-stable :suite :nabla.small)
  "決まったシードの順序の固定値。版をまたいで同じ位置から再開した列が変わらないことを守る
（SplitMix64 の定数・ラウンドの数・領域の取り方を変えると落ちる。変えるなら破壊的変更として扱う）。"
  (let ((sampler (make-index-sampler 10 :shuffle t :seed (nabla:prng-key 0))))
    (is (equal '(8 1 6 7 2 3 0 9 4 5) (epoch-order sampler 10 0)))
    (is (equal '(3 0 9 6 1 4 8 5 2 7) (epoch-order sampler 10 1)))
    ;; エポックは下位 64 ビットまで順序に効く
    (is (equal '(2 0 5 8 1 6 3 9 7 4) (epoch-order sampler 10 (1+ (expt 2 63))))))
  (let ((sampler (make-index-sampler 5 :shuffle t :seed (nabla:prng-key 42))))
    (is (equal '(3 1 0 2 4) (epoch-order sampler 5 0))))
  ;; シードの上位語（最上位ビットを含む）も順序に効く
  (let ((sampler (make-index-sampler 1000 :shuffle t :seed (nabla:prng-key #x89ABCDEF01234567))))
    (is (equal '((2 491) (2 678) (2 628) (2 837) (2 993) (2 147) (2 508) (2 267))
               (loop for k from 2000 below 2008 collect (multiple-value-list (sampler-index sampler k)))))))
