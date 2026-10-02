// stablehlo.optimization_barrier の f32 フィクスチャ（pretty form、単項。nabla の stop-gradient の出力先）。
func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.optimization_barrier %a : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
