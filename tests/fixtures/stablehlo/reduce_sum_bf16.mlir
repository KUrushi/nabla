// reduce_sum.mlir の bf16 版。issue #12（local/cuda 数値一致）で bf16 の
// 経路も確かめるために用意する。f32 を bf16 に置き換えただけ
// （dense<0.0> : tensor<f32> も tensor<bf16> に変えるだけでよい）。
//
// 入力（行優先、shape 4x8）:
//   a = 4x8xbf16 の任意の値
//
// 期待される出力（shape 4）: a の dimension 1（各行）に沿った総和。
// tests/support/reference.lisp の reference-reduce-sum（axis=1）がこの
// 期待値を計算する。

func.func @main(%a: tensor<4x8xbf16>) -> tensor<4xbf16> {
  %init = stablehlo.constant dense<0.0> : tensor<bf16>
  %0 = "stablehlo.reduce"(%a, %init) ({
  ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
    %1 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    stablehlo.return %1 : tensor<bf16>
  }) {dimensions = array<i64: 1>} : (tensor<4x8xbf16>, tensor<bf16>) -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
