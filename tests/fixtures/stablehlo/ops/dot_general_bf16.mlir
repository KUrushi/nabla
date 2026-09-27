// stablehlo.dot_general の bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<2x3xbf16>, %b: tensor<3x2xbf16>) -> tensor<2x2xbf16> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xbf16>, tensor<3x2xbf16>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>
}
