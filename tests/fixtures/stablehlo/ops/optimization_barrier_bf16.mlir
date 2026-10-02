// stablehlo.optimization_barrier の bf16 フィクスチャ（pretty form、単項。nabla の stop-gradient の出力先）。
func.func @main(%a: tensor<4xbf16>) -> tensor<4xbf16> {
  %0 = stablehlo.optimization_barrier %a : tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
