// stablehlo.constant の f32 フィクスチャ。dense 属性に直接リテラルを書く形。
func.func @main() -> tensor<2xf32> {
  %c = stablehlo.constant dense<[1.0, 2.5]> : tensor<2xf32>
  func.return %c : tensor<2xf32>
}
