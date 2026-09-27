// stablehlo.convert の bf16 → f32 フィクスチャ（入力側が bf16 のケース）。
func.func @main(%a: tensor<4xbf16>) -> tensor<4xf32> {
  %0 = stablehlo.convert %a : (tensor<4xbf16>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
