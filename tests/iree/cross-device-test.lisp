;;;; local と cuda で同じ StableHLO をコンパイル・実行し、数値一致を
;;;; 確かめる large テスト（issue #12）。
;;;;
;;;; このマシンには GPU が無いので、SKIP-UNLESS-CUDA が常にこのテストを
;;;; スキップする（NABLA_REQUIRE_CUDA を立てれば失敗にできる。CI では
;;;; 立てない）。実際の GPU での実行結果は docs/iree-build.md の
;;;; 「GPU で確かめる手順」節に記録する（この PR の時点ではすべて未測定）。
;;;;
;;;; local backend は find-backend（プロセス寿命の共有インスタンス）、
;;;; cuda backend は各テストの中で make-backend :iree :target :cuda を
;;;; 1回だけ呼んで作る（契約 §4）。cuda-arch は指定せず IREE の既定に任せる。

(in-package #:nabla.iree.tests)

(defun %cross-device-to-device (array backend dtype)
  "ARRAY を BACKEND にコピーする。DTYPE が :BF16 なら :DTYPE :BF16 を明示する
（(UNSIGNED-BYTE 16) の配列は :DTYPE なしでは NABLA:DTYPE-MISMATCH になるため）。"
  (if (eq dtype :bf16)
      (to-device array backend :dtype :bf16)
      (to-device array backend)))

(defun %cross-device-run (backend text function-name arrays dtype)
  "TEXT を BACKEND で BACKEND-COMPILE → BACKEND-LOAD し、ARRAYS（Lisp の
多次元配列）を BACKEND にコピーして BACKEND-INVOKE した結果を、
DECODE-ARRAY で DOUBLE-FLOAT の配列に戻して返す。module・device array は
使い終わったらすべて解放する。"
  (let ((module (nabla:backend-load backend (nabla:backend-compile backend text))))
    (unwind-protect
         (let ((device-arrays (mapcar (lambda (array) (%cross-device-to-device array backend dtype))
                                       arrays)))
           (unwind-protect
                (multiple-value-bind (result)
                    (apply #'nabla:backend-invoke backend module function-name device-arrays)
                  (unwind-protect
                       (decode-array (to-host result) dtype)
                    (release-device-array result)))
             (dolist (device-array device-arrays)
               (release-device-array device-array))))
      (nabla:backend-unload backend module))))

(define-iree-test/large cross-device/add/f32-local-matches-cuda
    "add.mlir（shape 4x8）を local と cuda で、同じ seed から作った同じ
入力に対して実行した結果は、f32 の既定の許容誤差で一致する。

同じテキストに対する vmfb ディスクキャッシュ（issue #10）のファイルが
local と cuda で別々に（2つ）できることもあわせて確かめる
（issue #12 の完了条件『キャッシュのキーにターゲットが入っている』）。"
  (skip-unless-iree :library :both)
  (skip-unless-cuda)
  (with-temporary-directory (dir)
    (let ((nabla:*compile-cache-directory* dir)
          (local (nabla:find-backend :iree))
          (cuda (nabla:make-backend :iree :target :cuda))
          (text (stablehlo-fixture "add")))
      (multiple-value-bind (rtol atol) (dtype-tolerance :f32)
        (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                      (lambda (seed)
                        (let* ((a (make-random-array (make-array-spec '(4 8) :f32) :seed seed))
                               (b (make-random-array (make-array-spec '(4 8) :f32) :seed (1+ seed)))
                               (local-result (%cross-device-run local text "main" (list a b) :f32))
                               (cuda-result (%cross-device-run cuda text "main" (list a b) :f32)))
                          (allclose local-result cuda-result :rtol rtol :atol atol)))
                      :regression-id cross-device/add/f32-local-matches-cuda
                      :regression-file (regression-path "iree-cross-device-add-f32" :package "NABLA.IREE.TESTS"))))
      (is (= 2 (length (directory (make-pathname :name :wild :type "module" :defaults dir))))
          "local と cuda の vmfb は別々のキャッシュファイルになっているはず"))))

(define-iree-test/large cross-device/add/bf16-local-matches-cuda
    "add_bf16.mlir（shape 4x8）を local と cuda で、同じ seed から作った
同じ入力に対して実行した結果は、bf16 の既定の許容誤差（rtol 1e-2 /
atol 1e-3）で一致する。"
  (skip-unless-iree :library :both)
  (skip-unless-cuda)
  (let ((local (nabla:find-backend :iree))
        (cuda (nabla:make-backend :iree :target :cuda))
        (text (stablehlo-fixture "add_bf16")))
    (multiple-value-bind (rtol atol) (dtype-tolerance :bf16)
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((a (make-random-array (make-array-spec '(4 8) :bf16) :seed seed))
                             (b (make-random-array (make-array-spec '(4 8) :bf16) :seed (1+ seed)))
                             (local-result (%cross-device-run local text "main" (list a b) :bf16))
                             (cuda-result (%cross-device-run cuda text "main" (list a b) :bf16)))
                        (allclose local-result cuda-result :rtol rtol :atol atol)))
                    :regression-id cross-device/add/bf16-local-matches-cuda
                    :regression-file (regression-path "iree-cross-device-add-bf16" :package "NABLA.IREE.TESTS"))))))

(define-iree-test/large cross-device/matmul/f32-local-matches-cuda
    "matmul.mlir（2x3 · 3x2）を local と cuda で、同じ seed から作った同じ
入力に対して実行した結果は、f32 の既定の許容誤差で一致する。

CPU と GPU では総和（dot_general の内積）の順序が違いうるので、既定の
許容誤差で不安定に失敗するようなら、その理由をここに書いてから緩める
（issue の補足を参照。現時点では GPU が無く未検証なので、まずは既定値
から始める）。"
  (skip-unless-iree :library :both)
  (skip-unless-cuda)
  (let ((local (nabla:find-backend :iree))
        (cuda (nabla:make-backend :iree :target :cuda))
        (text (stablehlo-fixture "matmul")))
    (multiple-value-bind (rtol atol) (dtype-tolerance :f32)
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((a (make-random-array (make-array-spec '(2 3) :f32) :seed seed))
                             (b (make-random-array (make-array-spec '(3 2) :f32) :seed (1+ seed)))
                             (local-result (%cross-device-run local text "main" (list a b) :f32))
                             (cuda-result (%cross-device-run cuda text "main" (list a b) :f32)))
                        (allclose local-result cuda-result :rtol rtol :atol atol)))
                    :regression-id cross-device/matmul/f32-local-matches-cuda
                    :regression-file (regression-path "iree-cross-device-matmul-f32" :package "NABLA.IREE.TESTS"))))))

(define-iree-test/large cross-device/matmul/bf16-local-matches-cuda
    "matmul_bf16.mlir（2x3 · 3x2）を local と cuda で、同じ seed から作った
同じ入力に対して実行した結果は、bf16 の既定の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (skip-unless-cuda)
  (let ((local (nabla:find-backend :iree))
        (cuda (nabla:make-backend :iree :target :cuda))
        (text (stablehlo-fixture "matmul_bf16")))
    (multiple-value-bind (rtol atol) (dtype-tolerance :bf16)
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((a (make-random-array (make-array-spec '(2 3) :bf16) :seed seed))
                             (b (make-random-array (make-array-spec '(3 2) :bf16) :seed (1+ seed)))
                             (local-result (%cross-device-run local text "main" (list a b) :bf16))
                             (cuda-result (%cross-device-run cuda text "main" (list a b) :bf16)))
                        (allclose local-result cuda-result :rtol rtol :atol atol)))
                    :regression-id cross-device/matmul/bf16-local-matches-cuda
                    :regression-file (regression-path "iree-cross-device-matmul-bf16" :package "NABLA.IREE.TESTS"))))))

(define-iree-test/large cross-device/reduce-sum/f32-local-matches-cuda
    "reduce_sum.mlir（shape 4x8 を dimension 1 で総和）を local と cuda で、
同じ seed から作った同じ入力に対して実行した結果は、f32 の既定の許容誤差で
一致する。CPU と GPU で総和の順序が違いうる点は matmul と同じ注意が要る。"
  (skip-unless-iree :library :both)
  (skip-unless-cuda)
  (let ((local (nabla:find-backend :iree))
        (cuda (nabla:make-backend :iree :target :cuda))
        (text (stablehlo-fixture "reduce_sum")))
    (multiple-value-bind (rtol atol) (dtype-tolerance :f32)
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((a (make-random-array (make-array-spec '(4 8) :f32) :seed seed))
                             (local-result (%cross-device-run local text "main" (list a) :f32))
                             (cuda-result (%cross-device-run cuda text "main" (list a) :f32)))
                        (allclose local-result cuda-result :rtol rtol :atol atol)))
                    :regression-id cross-device/reduce-sum/f32-local-matches-cuda
                    :regression-file (regression-path "iree-cross-device-reduce-sum-f32" :package "NABLA.IREE.TESTS"))))))

(define-iree-test/large cross-device/reduce-sum/bf16-local-matches-cuda
    "reduce_sum_bf16.mlir（shape 4x8 を dimension 1 で総和）を local と
cuda で、同じ seed から作った同じ入力に対して実行した結果は、bf16 の
既定の許容誤差で一致する。"
  (skip-unless-iree :library :both)
  (skip-unless-cuda)
  (let ((local (nabla:find-backend :iree))
        (cuda (nabla:make-backend :iree :target :cuda))
        (text (stablehlo-fixture "reduce_sum_bf16")))
    (multiple-value-bind (rtol atol) (dtype-tolerance :bf16)
      (is (check-it (generator (uniform-integer :lo 0 :hi (1- (expt 2 31))))
                    (lambda (seed)
                      (let* ((a (make-random-array (make-array-spec '(4 8) :bf16) :seed seed))
                             (local-result (%cross-device-run local text "main" (list a) :bf16))
                             (cuda-result (%cross-device-run cuda text "main" (list a) :bf16)))
                        (allclose local-result cuda-result :rtol rtol :atol atol)))
                    :regression-id cross-device/reduce-sum/bf16-local-matches-cuda
                    :regression-file (regression-path "iree-cross-device-reduce-sum-bf16" :package "NABLA.IREE.TESTS"))))))
