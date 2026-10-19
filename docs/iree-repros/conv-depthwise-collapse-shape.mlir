// IREE 3.11.0 で O/G = 1 の depthwise 畳み込み（rhs [k.., 1, C]、feature_group_count = C）を
// bf16 → f32 に convert して畳み込むと "tensor.collapse_shape ... expected collapsed type to be
// 'tensor<3x2x3xbf16>', but got 'tensor<3x2x3xf32>'" で落ちる最小の再現（nabla issue #203）。f32 のままなら通る。
// 回避策: conv-depthwise-collapse-shape-barrier.mlir（2つの f32 を optimization_barrier に通す）。
func.func @main(%a: tensor<1x5x6x3xbf16>, %w: tensor<3x2x1x3xbf16>) -> tensor<1x3x5x3xbf16> {
  %l = stablehlo.convert %a : (tensor<1x5x6x3xbf16>) -> tensor<1x5x6x3xf32>
  %r = stablehlo.convert %w : (tensor<3x2x1x3xbf16>) -> tensor<3x2x1x3xf32>
  %acc = stablehlo.convolution(%l, %r) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<1x5x6x3xf32>, tensor<3x2x1x3xf32>) -> tensor<1x3x5x3xf32>
  %o = stablehlo.convert %acc : (tensor<1x3x5x3xf32>) -> tensor<1x3x5x3xbf16>
  func.return %o : tensor<1x3x5x3xbf16>
}
