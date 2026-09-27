// stablehlo.compare の f32 フィクスチャ（pretty form）。方向は LT（他に LE GT
// GE EQ NE も同じ書き方で通ることを advisor が確認済み）。出力 dtype は
// nabla の :i1（issue #37）。
func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}
