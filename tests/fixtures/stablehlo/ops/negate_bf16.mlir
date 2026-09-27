// stablehlo.negate の bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<4xbf16>) -> tensor<4xbf16> {
  %0 = stablehlo.negate %a : tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
