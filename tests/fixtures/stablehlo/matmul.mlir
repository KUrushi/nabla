// 手書きの matmul サンプル。scripts/verify-iree.sh が iree-compile / iree-run-module
// にかけて、CPU（と GPU があれば CUDA）で同じ結果になることを確かめるのに使う。
//
// dot_general の一般形（"stablehlo.dot_general"(...) { dot_dimension_numbers = ... }）を
// 使う。third_party/stablehlo/stablehlo/tests/ops_stablehlo.mlir の @dot_general の書き方
// をそのまま流用した、バージョンに左右されにくい書き方。
//
// 入力（行優先、iree-run-module の --input=2x3xf32=1,2,3,4,5,6 のように渡す）:
//   a = [[1, 2, 3],
//        [4, 5, 6]]                     ; shape 2x3
//   b = [[ 7,  8],
//        [ 9, 10],
//        [11, 12]]                      ; shape 3x2
//
// 期待される出力 a @ b（shape 2x2）:
//   [[1*7+2*9+3*11,  1*8+2*10+3*12],      = [[ 58,  64],
//    [4*7+5*9+6*11,  4*8+5*10+6*12]]        [139, 154]]
//
// scripts/verify-iree.sh はこの期待値 "58 64 139 154" を iree-run-module の出力から
// 抜き出して比較する。

func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
