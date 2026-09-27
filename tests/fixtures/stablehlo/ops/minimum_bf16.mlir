// stablehlo.minimum の bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<4x8xbf16>, %b: tensor<4x8xbf16>) -> tensor<4x8xbf16> {
  %0 = stablehlo.minimum %a, %b : tensor<4x8xbf16>
  func.return %0 : tensor<4x8xbf16>
}
