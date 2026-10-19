;;;; dense-general（issue #208、small）。
;;;;
;;;; 守らせる性質: kernel / bias を平らにした dense と同じ計算（縮約する軸が 1 つでも 2 つでも）、init の形・
;;;; 分布・スロット（contract-axes は :static でチェックポイントのパス名に現れない）、不正な形は層名入りの error。

(in-package #:nabla.nn.tests)

(in-suite :nabla.small)
(def-layer-property dense-general/equals-dense-on-flattened-axes
  "dense-general は、kernel と bias を平らにした dense と同じ。縮約する軸が1つ（入力 → ヘッド × 幅）でも
2つ（ヘッド × 幅 → 出力）でも、出力の形は 先頭のバッチ軸 ++ 特徴の軸。"
  (seed heads dim in) (tuple (uniform-integer :lo 0 :hi 100000) (uniform-integer :lo 1 :hi 3)
                             (uniform-integer :lo 1 :hi 3) (uniform-integer :lo 1 :hi 4))
  (let* ((split (make-dense-general :kernel (%rand (list in heads dim) seed :f64)
                                    :bias (%rand (list heads dim) (+ seed 1) :f64)
                                    :contract-axes 1))
         (flat (make-dense :kernel (nabla:reshape (dense-general-kernel split) (list in (* heads dim)))
                           :bias (nabla:reshape (dense-general-bias split) (list (* heads dim)))))
         (x (%rand (list 2 3 in) (+ seed 2) :f64))
         (merged (make-dense-general :kernel (%rand (list heads dim in) seed :f64)
                                     :bias (%rand (list in) (+ seed 1) :f64)
                                     :contract-axes 2))
         (merged-flat (make-dense :kernel (nabla:reshape (dense-general-kernel merged) (list (* heads dim) in))
                                  :bias (dense-general-bias merged)))
         (h (%rand (list 2 3 heads dim) (+ seed 3) :f64)))
    (and (equal (list 2 3 heads dim) (array-dimensions (call split x)))
         (allclose (nabla:reshape (call split x) (list 2 3 (* heads dim))) (call flat x)
                   :dtype :f64 :rtol 1d-12 :atol 1d-12)
         (equal (list 2 3 in) (array-dimensions (call merged h)))
         (allclose (call merged h) (call merged-flat (nabla:reshape h (list 2 3 (* heads dim))))
                   :dtype :f64 :rtol 1d-12 :atol 1d-12))))


(test dense-general/misuse-names-the-layer
  "dense-general の kernel と bias の形が合わない・contract-axes が範囲外・入力の最後の軸が合わない・入力の
rank が足りないときは、層の名前を含む error。"
  (flet ((names (thunk) (search "dense-general" (or (%error-message thunk) ""))))
    (let ((good (make-dense-general :kernel (%rand '(4 2 3) 1) :bias (%rand '(2 3) 2) :contract-axes 1))
          (zero-axes (make-dense-general :kernel (%rand '(4 3) 1) :bias (%rand '(4 3) 2) :contract-axes 0))
          (all-axes (make-dense-general :kernel (%rand '(4 3) 1) :bias (%rand '() 2) :contract-axes 2))
          (wrong-bias (make-dense-general :kernel (%rand '(4 2 3) 1) :bias (%rand '(3 2) 2) :contract-axes 1))
          (wrong-axes (make-dense-general :kernel (%rand '(4 2 3) 1) :bias (%rand '(3) 2) :contract-axes 1)))
      (is (names (lambda () (call good (%rand '(5 5) 1)))) "最後の軸が合わない")
      (is (names (lambda () (call good (%rand '(5 2 3) 1)))) "縮約する軸が in の大きさと違う")
      (is (names (lambda () (call zero-axes (%rand '(5 4) 1)))) "contract-axes が 0")
      (is (names (lambda () (call all-axes (%rand '(4 3) 1)))) "contract-axes が kernel の rank 以上")
      (is (names (lambda () (call wrong-bias (%rand '(5 4) 1)))) "bias が kernel の後ろの軸と違う")
      (is (names (lambda () (call wrong-axes (%rand '(5 4) 1)))) "contract-axes と bias が食い違う")
      (is (equal '(5 2 3) (array-dimensions (call good (%rand '(5 4) 1)))) "正しい入力は通る")
      (is (equal '(2 3) (array-dimensions (call good (%rand '(4) 1)))) "バッチ軸が無く、縮約する軸だけの入力も通る")
      (is (names (lambda () (init-dense-general (nabla:prng-key 1) 0 3))) "in が正でない")
      (is (names (lambda () (init-dense-general (nabla:prng-key 1) 4 '()))) "features が空"))))

(test dense-general/init-and-checkpoint-paths
  "init-dense-general は kernel (in... features...)・bias 0・contract-axes = in の軸の数を作る。スロットは
kernel, bias（:param）と contract-axes（:static）で、:static は葉でないのでチェックポイントのパス名は
Flax と同じ kernel と bias のまま。"
  (let ((one (init-dense-general (nabla:prng-key 1) 5 '(2 3)))
        (two (init-dense-general (nabla:prng-key 1) '(2 3) 4)))
    (is (equal '(5 2 3) (array-dimensions (dense-general-kernel one))))
    (is (equal '(2 3) (array-dimensions (dense-general-bias one))))
    (is (= 1 (dense-general-contract-axes one)))
    (is (equal '(2 3 4) (array-dimensions (dense-general-kernel two))))
    (is (equal '(4) (array-dimensions (dense-general-bias two))))
    (is (= 2 (dense-general-contract-axes two)))
    (is (every #'zerop (%elements (dense-general-bias two))))
    (is (equal '(("kernel" . :param) ("bias" . :param) ("contract-axes" . :static))
               (%slot-names 'dense-general)))
    (is (equal '("kernel" "bias") (nabla.nn:tree-paths two)))))
