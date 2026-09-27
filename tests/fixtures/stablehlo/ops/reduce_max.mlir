// stablehlo.reduce（max）の f32 フィクスチャ（pretty form）。init は f32 の
// -inf を16進ビット列で書く（dense<-inf> は書かない。契約 §3 参照）。
func.func @main(%a: tensor<4x8xf32>) -> tensor<4xf32> {
  %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
