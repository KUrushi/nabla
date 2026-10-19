// IREE 3.11.0 で bf16 の畳み込みが "expects lhs and rhs to have compatible element type" で落ちる最小の再現
// （nabla issue #203）。bf16 の2入力を f32 に convert して f32 で畳み込み、結果を bf16 に戻す形で、
// カーネルは transpose を経ている（convolution の lhs の転置が出す形）。
// 期待: コンパイルが通る。実際: convolution の operand の一方が bf16 のまま（f32 と不一致）になりエラー。
// 回避策: conv-convert-operand-type-barrier.mlir（2つの f32 を optimization_barrier に通す）。
func.func @main(%a: tensor<2x5x6x4xbf16>, %w: tensor<3x3x3x4xbf16>) -> tensor<2x5x6x3xbf16> {
  %9 = stablehlo.transpose %w, dims = [0, 1, 3, 2] : (tensor<3x3x3x4xbf16>) -> tensor<3x3x4x3xbf16>
  %l = stablehlo.convert %a : (tensor<2x5x6x4xbf16>) -> tensor<2x5x6x4xf32>
  %r = stablehlo.convert %9 : (tensor<3x3x4x3xbf16>) -> tensor<3x3x4x3xf32>
  %acc = stablehlo.convolution(%l, %r) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x5x6x4xf32>, tensor<3x3x4x3xf32>) -> tensor<2x5x6x3xf32>
  %o = stablehlo.convert %acc : (tensor<2x5x6x3xf32>) -> tensor<2x5x6x3xbf16>
  func.return %o : tensor<2x5x6x3xbf16>
}
