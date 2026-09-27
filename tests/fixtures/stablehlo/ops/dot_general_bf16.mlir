// stablehlo.dot_general の bf16 フィクスチャ（pretty form）。issue #54: IREE
// は bf16 の dot_general を f32 で累積しないので、f32 の結果型で
// dot_general を出してから convert で bf16 に戻す。
func.func @main(%a: tensor<2x3xbf16>, %b: tensor<3x2xbf16>) -> tensor<2x2xbf16> {
  %acc = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x3xbf16>, tensor<3x2xbf16>) -> tensor<2x2xf32>
  %0 = stablehlo.convert %acc : (tensor<2x2xf32>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>
}
