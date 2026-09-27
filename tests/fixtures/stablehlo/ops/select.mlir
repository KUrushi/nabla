// stablehlo.select の f32 フィクスチャ（pretty form）。%pred は :i1。
func.func @main(%pred: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.select %pred, %a, %b : tensor<4xi1>, tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
