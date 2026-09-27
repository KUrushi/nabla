// stablehlo.reduce（add）の bf16 フィクスチャ（pretty form）。issue #63:
// IREE（llvm-cpu）は bf16 の stablehlo.reduce（add）を入力 dtype のまま
// 累積するので（dot_general の issue #54 と同じ理由）、f32 に convert
// してから reduce し、convert で bf16 に戻す。init は f32 の 0.0。
func.func @main(%a: tensor<4x8xbf16>) -> tensor<4xbf16> {
  %in32 = stablehlo.convert %a : (tensor<4x8xbf16>) -> tensor<4x8xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %acc = stablehlo.reduce(%in32 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  %0 = stablehlo.convert %acc : (tensor<4xf32>) -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}
