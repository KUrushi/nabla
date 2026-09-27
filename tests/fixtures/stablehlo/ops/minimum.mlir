// stablehlo.minimum の f32 フィクスチャ（pretty form）。
func.func @main(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.minimum %a, %b : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
