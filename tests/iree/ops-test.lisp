;;;; issue #30（u2）: StableHLO op 対応表のフィクスチャが、IREE の
;;;; embedded compiler で実際に非空の vmfb へコンパイルできることを確かめる。
;;;;
;;;; src/ の変更を伴わないので mutation testing は対象外（CLAUDE.md /
;;;; 契約 §0 参照。PR 本文にも明記する）。数値の正しさはここでは確かめない
;;;; （wave 2 のプリミティブ実装の仕事。issue #30 の完了条件はコンパイルが
;;;; 通ることだけ）。

(in-package #:nabla.iree.tests)

(defparameter *stablehlo-op-fixtures*
  '("ops/add" "ops/add_bf16"
    "ops/subtract" "ops/subtract_bf16"
    "ops/multiply" "ops/multiply_bf16"
    "ops/divide" "ops/divide_bf16"
    "ops/maximum" "ops/maximum_bf16"
    "ops/minimum" "ops/minimum_bf16"
    "ops/negate" "ops/negate_bf16"
    "ops/exponential" "ops/exponential_bf16"
    "ops/log" "ops/log_bf16"
    "ops/tanh" "ops/tanh_bf16"
    "ops/compare" "ops/compare_bf16"
    "ops/select" "ops/select_bf16"
    "ops/convert" "ops/convert_bf16"
    "ops/constant" "ops/constant_bf16"
    "ops/broadcast_in_dim" "ops/broadcast_in_dim_bf16"
    "ops/reshape" "ops/reshape_bf16"
    "ops/transpose" "ops/transpose_bf16"
    "ops/dot_general" "ops/dot_general_bf16"
    "ops/reduce_add" "ops/reduce_add_bf16"
    "ops/reduce_max" "ops/reduce_max_bf16")
  "docs/stablehlo-ops.md の表に載る、対象19 op それぞれの f32 / bf16
フィクスチャ名（tests/fixtures/stablehlo/<名前>.mlir、拡張子なし）。この
リストと docs/stablehlo-ops.md の表、tests/fixtures/stablehlo/ops/ 配下の
ファイルの3つは、常に同じ38個（19 op × 2）を指す。")

(define-iree-test ops/all-fixtures-compile-to-non-empty-vmfb
    "docs/stablehlo-ops.md の表にある全フィクスチャ（f32 と bf16）について、
backend-compile が非空の (simple-array (unsigned-byte 8) (*)) を返し、
先頭4バイトが vmfb の ZIP local-file-header シグネチャと一致する。フィクス
チャごとに個別に is で報告するので、1つが壊れても他の結果が見える。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree)))
    (dolist (name *stablehlo-op-fixtures*)
      (let ((bytes (nabla:backend-compile backend (stablehlo-fixture name))))
        (is (typep bytes '(simple-array (unsigned-byte 8) (*))) "~A" name)
        (is (plusp (length bytes)) "~A" name)
        (is (equalp *vmfb-magic* (subseq bytes 0 4)) "~A" name)))))

(define-iree-test ops/fixture-count-matches-directory
    "*stablehlo-op-fixtures* に列挙した個数が、
tests/fixtures/stablehlo/ops/ に実際に置かれた .mlir ファイルの個数と
一致する。docs/stablehlo-ops.md の表・このリスト・フィクスチャの3つが
ずれたら検出する、簡単な整合テスト（tests/regressions.lisp と同じ
uiop:directory-files を使う）。"
  (let* ((dir (asdf:system-relative-pathname "nabla" "tests/fixtures/stablehlo/ops/"))
         (files (uiop:directory-files dir "*.mlir")))
    (is (= (length *stablehlo-op-fixtures*) (length files))
        "*stablehlo-op-fixtures* has ~D entries but ~A has ~D .mlir files"
        (length *stablehlo-op-fixtures*) dir (length files))))

(define-iree-test ops/add-dot-general-reduce-execute-end-to-end
    "表の中でも add / dot_general / reduce(add) の3つは、コンパイルだけで
なく to-device → backend-load → backend-invoke → to-host まで通し、実際に
IREE の local バックエンドで実行できることを確かめる（既存の
tests/iree/backend-test.lisp と同じプロトコルの使い方）。数値は
tests/support/reference.lisp の reference-* と allclose :dtype :f32 で
一致することを確かめる。数値の正しさの網羅的な検査は wave 2 の仕事。"
  (skip-unless-iree :library :both)
  (let ((backend (nabla:find-backend :iree)))
    ;; add: 4x8 + 4x8。
    (let* ((a (make-random-array (make-array-spec '(4 8) :f32) :seed 1))
           (b (make-random-array (make-array-spec '(4 8) :f32) :seed 2))
           (module (nabla:backend-load backend (nabla:backend-compile backend (stablehlo-fixture "ops/add")))))
      (unwind-protect
           (with-device-arrays ((da (to-device a backend)) (db (to-device b backend)))
             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
               (is (allclose (to-host result) (reference-add a b) :dtype :f32))))
        (nabla:backend-unload backend module)))
    ;; dot_general: 2x3 @ 3x2。
    (let* ((a (make-random-array (make-array-spec '(2 3) :f32) :seed 3))
           (b (make-random-array (make-array-spec '(3 2) :f32) :seed 4))
           (module (nabla:backend-load backend (nabla:backend-compile backend (stablehlo-fixture "ops/dot_general")))))
      (unwind-protect
           (with-device-arrays ((da (to-device a backend)) (db (to-device b backend)))
             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da db)))
               (is (allclose (to-host result) (reference-matmul a b) :dtype :f32))))
        (nabla:backend-unload backend module)))
    ;; reduce(add): 4x8 を dimension 1 に沿って総和。
    (let* ((a (make-random-array (make-array-spec '(4 8) :f32) :seed 5))
           (module (nabla:backend-load backend (nabla:backend-compile backend (stablehlo-fixture "ops/reduce_add")))))
      (unwind-protect
           (with-device-arrays ((da (to-device a backend)))
             (with-device-arrays ((result (nabla:backend-invoke backend module "main" da)))
               (is (allclose (to-host result) (reference-reduce-sum a 1) :dtype :f32))))
        (nabla:backend-unload backend module)))))
