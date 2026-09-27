// 手書きの reduce（総和）サンプル。matmul.mlir / add.mlir と同じ理由で用意する。
//
// third_party/stablehlo/stablehlo/tests/ops_stablehlo.mlir と同じ、
// "stablehlo.reduce"(...) ({ ^bb0(...): ... }) { dimensions = array<i64: ...> }
// という一般形の書き方をそのまま使う（バージョンに左右されにくい）。
//
// 入力（行優先、shape 4x8）:
//   a = 4x8xf32 の任意の値
//
// 期待される出力（shape 4）: a の dimension 1（各行）に沿った総和。
// tests/support/reference.lisp の reference-reduce-sum（axis=1）がこの
// 期待値を計算する。

func.func @main(%a: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = "stablehlo.reduce"(%a, %init) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %1 = stablehlo.add %arg0, %arg1 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {dimensions = array<i64: 1>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
