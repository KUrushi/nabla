// stablehlo.convert の f32 → bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<4xf32>) -> tensor<4xbf16> {
  %0 = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
