// stablehlo.reduce（add）の bf16 フィクスチャ（pretty form）。init は 0.0 の
// bf16 ビット列（0x0000）。
func.func @main(%a: tensor<4x8xbf16>) -> tensor<4xbf16> {
  %init = stablehlo.constant dense<0x0000> : tensor<bf16>
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xbf16>, tensor<bf16>) -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
