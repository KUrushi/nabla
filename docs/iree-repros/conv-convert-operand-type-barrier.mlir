// conv-convert-operand-type.mlir の回避策（対照）。f32 に convert した2つの入力をまとめて
// optimization_barrier に通すと、同じ畳み込みがコンパイルできる。
func.func @main(%a: tensor<2x5x6x4xbf16>, %w: tensor<3x3x3x4xbf16>) -> tensor<2x5x6x3xbf16> {
  %9 = stablehlo.transpose %w, dims = [0, 1, 3, 2] : (tensor<3x3x3x4xbf16>) -> tensor<3x3x4x3xbf16>
  %l0 = stablehlo.convert %a : (tensor<2x5x6x4xbf16>) -> tensor<2x5x6x4xf32>
  %r0 = stablehlo.convert %9 : (tensor<3x3x4x3xbf16>) -> tensor<3x3x4x3xf32>
  %l, %r = stablehlo.optimization_barrier %l0, %r0 : tensor<2x5x6x4xf32>, tensor<3x3x4x3xf32>
  %acc = stablehlo.convolution(%l, %r) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x5x6x4xf32>, tensor<3x3x4x3xf32>) -> tensor<2x5x6x3xf32>
  %o = stablehlo.convert %acc : (tensor<2x5x6x3xf32>) -> tensor<2x5x6x3xbf16>
  func.return %o : tensor<2x5x6x3xbf16>
}
