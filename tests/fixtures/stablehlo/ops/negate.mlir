// stablehlo.negate の f32 フィクスチャ（pretty form、単項）。
func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.negate %a : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
