// 手書きの要素ごとの加算サンプル。matmul.mlir と同じ理由（scripts/verify-iree.sh
// と tests/iree/execute-test.lisp の両方から使う小さな固定フィクスチャ）で
// 用意する。
//
// 入力（行優先、shape はどちらも 4x8）:
//   a, b はどちらも 4x8xf32 の任意の値
//
// 期待される出力 a + b（shape 4x8）: 要素ごとの和。
// tests/support/reference.lisp の reference-add がこの期待値を計算する。

func.func @main(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.add %a, %b : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
