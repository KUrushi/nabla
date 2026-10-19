// stablehlo.reverse の f32 フィクスチャ（pretty form。issue #203）。
func.func @main(%a: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = stablehlo.reverse %a, dims = [0, 2] : tensor<2x3x4xf32>
  func.return %0 : tensor<2x3x4xf32>
}
