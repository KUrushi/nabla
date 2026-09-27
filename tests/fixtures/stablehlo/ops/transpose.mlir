// stablehlo.transpose の f32 フィクスチャ（pretty form）。
func.func @main(%a: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  %0 = stablehlo.transpose %a, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  func.return %0 : tensor<4x2x3xf32>
}
