// stablehlo.select の bf16 フィクスチャ（pretty form）。
func.func @main(%pred: tensor<4xi1>, %a: tensor<4xbf16>, %b: tensor<4xbf16>) -> tensor<4xbf16> {
  %0 = stablehlo.select %pred, %a, %b : tensor<4xi1>, tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
