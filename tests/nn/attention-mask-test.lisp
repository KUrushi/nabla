;;;; attention のマスクの部品 make-causal-mask / make-attention-mask / combine-masks（issue #208、small）。
;;;;
;;;; 守らせる性質: 因果マスクは下三角（k <= q が 1）、padding マスクは q と k がともに有効な所だけ 1、
;;;; combine-masks は論理積（大きさ 1 の軸を広げ、足りない先頭の軸を補う）。不正な引数は関数の名前入りの error。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)
(test attention/make-causal-mask-is-lower-triangular
  "make-causal-mask n は形 (1 n n) の bit 配列で、クエリ位置 q がキー位置 k 以上（k <= q）のとき 1。"
  (let ((mask (make-causal-mask 3)))
    (is (equal '(1 3 3) (array-dimensions mask)))
    (is (eq 'bit (array-element-type mask)))
    (is (equalp #3A(((1 0 0) (1 1 0) (1 1 1))) mask))))


(test attention/make-attention-mask-combines-valid-flags
  "make-attention-mask は [..., q] と [..., k] の bit 配列から [..., 1, q, k] を作る（q と k がともに 1 の所だけ 1）。"
  (let ((mask (make-attention-mask (make-array '(2 2) :element-type 'bit :initial-contents '((1 0) (1 1)))
                                   (make-array '(2 3) :element-type 'bit :initial-contents '((1 1 0) (0 1 1))))))
    (is (equal '(2 1 2 3) (array-dimensions mask)))
    (is (equalp #4A((((1 1 0) (0 0 0))) (((0 1 1) (0 1 1)))) mask))))


(test attention/combine-masks-ands-with-broadcasting
  "combine-masks は マスクの論理積を取る。ランクが違えば右に揃えてブロードキャストし、NIL は無視、
全部 NIL なら NIL。"
  (let ((combined (combine-masks (make-causal-mask 2)
                                 (make-array '(2 1 2 2) :element-type 'bit :initial-contents '((((1 1) (1 0))) (((1 1) (1 1))))))))
    (is (equal '(2 1 2 2) (array-dimensions combined)))
    (is (equalp #4A((((1 0) (1 0))) (((1 0) (1 1)))) combined)))
  (is (null (combine-masks nil nil)))
  (let ((mask (make-causal-mask 2)))
    (is (equalp mask (combine-masks nil mask)))))


(test attention/mask-helpers-reject-bad-arguments-naming-the-function
  "make-causal-mask / make-attention-mask / combine-masks は、不正な引数（正でない長さ・bit でない配列・
先頭の軸が合わない・ブロードキャストできない形）を、関数の名前を含む error で断る。"
  (flet ((names (name thunk) (search name (or (%error-message thunk) "")))
         (bits (&rest dims) (make-array dims :element-type 'bit :initial-element 1)))
    (is (names "make-causal-mask" (lambda () (make-causal-mask 0))))
    (is (names "make-causal-mask" (lambda () (make-causal-mask -2))))
    (is (names "make-causal-mask" (lambda () (make-causal-mask 2.5))))
    (is (names "make-attention-mask" (lambda () (make-attention-mask (%rand '(2) 1) (bits 2)))))
    (is (names "make-attention-mask" (lambda () (make-attention-mask (bits 2) (%rand '(2) 1)))))
    (is (names "make-attention-mask" (lambda () (make-attention-mask (bits 2 3) (bits 3 3)))))
    (is (names "make-attention-mask" (lambda () (make-attention-mask (bits) (bits 2)))))
    (is (names "combine-masks" (lambda () (combine-masks (%rand '(2) 1) (bits 2)))))
    (is (names "combine-masks" (lambda () (combine-masks (bits 2 3) (bits 3 3)))))
    (is (names "combine-masks" (lambda () (combine-masks (bits 2 3) (bits 4)))))))


(test attention/combine-masks-broadcasts-size-one-axes-and-extends-rank
  "combine-masks は大きさ 1 の軸を広げ、足りない先頭の軸を 1 として補う（バッチ軸が 3 で、因果マスクの先頭の 1 が
バッチ軸の大きさと食い違わない）。結果の各要素は、広げた各マスクの対応する要素の論理積。"
  (let* ((padding (make-array '(3 1 2 2) :element-type 'bit
                                         :initial-contents '((((1 1) (1 1))) (((1 0) (1 0))) (((0 0) (0 0))))))
         (combined (combine-masks (make-causal-mask 2) padding)))
    (is (equal '(3 1 2 2) (array-dimensions combined)))
    (is (equalp #4A((((1 0) (1 1))) (((1 0) (1 0))) (((0 0) (0 0)))) combined))))

