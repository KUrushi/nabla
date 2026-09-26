// add.mlir の bf16 版。issue #12（local/cuda 数値一致）で bf16 の経路も
// 確かめるために用意する。f32 を bf16 に置き換えただけで、この環境で
// llvm-cpu と cuda の両方にコンパイルできることを確認済み（契約 §0 事実4）。
//
// 入力（行優先、shape はどちらも 4x8）:
//   a, b はどちらも 4x8xbf16 の任意の値
//
// 期待される出力 a + b（shape 4x8）: 要素ごとの和。
// tests/support/reference.lisp の reference-add がこの期待値を計算する
// （bf16 のビット列は tests/support/random-array.lisp の decode-element /
// decode-array で DOUBLE-FLOAT に戻してから比べる）。

func.func @main(%a: tensor<4x8xbf16>, %b: tensor<4x8xbf16>) -> tensor<4x8xbf16> {
  %0 = stablehlo.add %a, %b : tensor<4x8xbf16>
  func.return %0 : tensor<4x8xbf16>
}
