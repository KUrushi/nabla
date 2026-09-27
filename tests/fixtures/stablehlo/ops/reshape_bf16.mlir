// stablehlo.reshape の bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<2x3xbf16>) -> tensor<3x2xbf16> {
  %0 = stablehlo.reshape %a : (tensor<2x3xbf16>) -> tensor<3x2xbf16>
  func.return %0 : tensor<3x2xbf16>
}
