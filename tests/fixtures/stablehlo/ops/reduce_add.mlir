// stablehlo.reduce（add）の f32 フィクスチャ（pretty form）。dimension 1
// （各行）に沿った総和。既存の reduce_sum.mlir（generic form）と同じ意味。
func.func @main(%a: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
