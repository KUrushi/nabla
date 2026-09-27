// stablehlo.add の f32 フィクスチャ（pretty form）。
// 入力: %a, %b はどちらも 4x8xf32。出力: 要素ごとの和。
func.func @main(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.add %a, %b : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
