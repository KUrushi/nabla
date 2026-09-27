// stablehlo.reduce（max）の bf16 フィクスチャ（pretty form）。init は bf16 の
// -inf（0xFF80）。
func.func @main(%a: tensor<4x8xbf16>) -> tensor<4xbf16> {
  %init = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<4x8xbf16>, tensor<bf16>) -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
