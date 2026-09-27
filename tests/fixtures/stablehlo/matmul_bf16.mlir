// matmul.mlir の bf16 版。issue #12（local/cuda 数値一致）で bf16 の経路も
// 確かめるために用意する。f32 を bf16 に置き換えただけ。契約 §0 事実4で、
// 入力 a=[[1,2,3],[4,5,6]]、b=[[7,8],[9,10],[11,12]] を渡すと
// [[58, 64], [139, 154]] を返すことを、この環境で llvm-cpu / cuda の
// 両方へのコンパイルとあわせて確認済み。
//
// dot_general の一般形は matmul.mlir と同じ書き方（バージョンに左右され
// にくい third_party/stablehlo/stablehlo/tests/ops_stablehlo.mlir 由来）。

func.func @main(%a: tensor<2x3xbf16>, %b: tensor<3x2xbf16>) -> tensor<2x2xbf16> {
  %0 = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<2x3xbf16>, tensor<3x2xbf16>) -> tensor<2x2xbf16>
  func.return %0 : tensor<2x2xbf16>
}
