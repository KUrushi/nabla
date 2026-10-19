// stablehlo.sqrt の bf16 フィクスチャ（pretty form、単項）。
func.func @main(%a: tensor<4xbf16>) -> tensor<4xbf16> {
  %0 = stablehlo.sqrt %a : tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
