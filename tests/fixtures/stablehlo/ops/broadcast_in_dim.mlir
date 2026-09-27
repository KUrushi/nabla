// stablehlo.broadcast_in_dim の f32 フィクスチャ（pretty form）。
// shape (3) の入力を dims=[1] で shape (2,3) に broadcast する。
func.func @main(%a: tensor<3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}
