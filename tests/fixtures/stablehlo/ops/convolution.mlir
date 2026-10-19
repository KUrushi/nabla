// stablehlo.convolution の f32 フィクスチャ（pretty form。issue #197）。channels-last
// （NHWC / HWIO）、stride・padding・lhs / rhs の膨張・feature_group_count が全部効く設定。
func.func @main(%a: tensor<2x7x6x3xf32>, %b: tensor<3x2x1x6xf32>) -> tensor<2x4x4x6xf32> {
  %0 = stablehlo.convolution(%a, %b) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [2, 1], pad = [[1, 1], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 2]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<2x7x6x3xf32>, tensor<3x2x1x6xf32>) -> tensor<2x4x4x6xf32>
  func.return %0 : tensor<2x4x4x6xf32>
}
