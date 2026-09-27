// stablehlo.transpose の bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<2x3x4xbf16>) -> tensor<4x2x3xbf16> {
  %0 = stablehlo.transpose %a, dims = [2, 0, 1] : (tensor<2x3x4xbf16>) -> tensor<4x2x3xbf16>
  func.return %0 : tensor<4x2x3xbf16>
}
