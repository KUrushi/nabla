// stablehlo.dot_general の f32 フィクスチャ（pretty form）。既存の
// tests/fixtures/stablehlo/matmul.mlir と同じ意味（2x3 @ 3x2 → 2x2）だが、
// generic form ではなく pretty form の綴りを記録する。
func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
