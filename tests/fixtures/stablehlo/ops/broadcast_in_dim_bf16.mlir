// stablehlo.broadcast_in_dim の bf16 フィクスチャ（pretty form）。
func.func @main(%a: tensor<3xbf16>) -> tensor<2x3xbf16> {
  %0 = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xbf16>) -> tensor<2x3xbf16>
  func.return %0 : tensor<2x3xbf16>
}
