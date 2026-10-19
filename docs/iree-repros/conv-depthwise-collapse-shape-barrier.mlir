// conv-depthwise-collapse-shape.mlir の回避策（対照）。
func.func @main(%a: tensor<1x5x6x3xbf16>, %w: tensor<3x2x1x3xbf16>) -> tensor<1x3x5x3xbf16> {
  %l0 = stablehlo.convert %a : (tensor<1x5x6x3xbf16>) -> tensor<1x5x6x3xf32>
  %r0 = stablehlo.convert %w : (tensor<3x2x1x3xbf16>) -> tensor<3x2x1x3xf32>
  %l, %r = stablehlo.optimization_barrier %l0, %r0 : tensor<1x5x6x3xf32>, tensor<3x2x1x3xf32>
  %acc = stablehlo.convolution(%l, %r) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<1x5x6x3xf32>, tensor<3x2x1x3xf32>) -> tensor<1x3x5x3xf32>
  %o = stablehlo.convert %acc : (tensor<1x3x5x3xf32>) -> tensor<1x3x5x3xbf16>
  func.return %o : tensor<1x3x5x3xbf16>
}
