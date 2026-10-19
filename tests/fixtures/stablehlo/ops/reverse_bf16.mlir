// stablehlo.reverse の bf16 フィクスチャ（pretty form。issue #203）。
func.func @main(%a: tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16> {
  %0 = stablehlo.reverse %a, dims = [0, 2] : tensor<2x3x4xbf16>
  func.return %0 : tensor<2x3x4xbf16>
}
