// stablehlo.reshape の f32 フィクスチャ（pretty form）。
func.func @main(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = stablehlo.reshape %a : (tensor<2x3xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}
