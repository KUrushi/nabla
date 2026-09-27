// stablehlo.compare の bf16 フィクスチャ（pretty form）。出力は :i1。
func.func @main(%a: tensor<4xbf16>, %b: tensor<4xbf16>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %a, %b : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}
