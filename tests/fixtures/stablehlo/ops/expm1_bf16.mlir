// stablehlo.exponential_minus_one の bf16 フィクスチャ（pretty form、単項）。
func.func @main(%a: tensor<4xbf16>) -> tensor<4xbf16> {
  %0 = stablehlo.exponential_minus_one %a : tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
