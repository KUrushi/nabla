// stablehlo.constant の bf16 フィクスチャ。dense 属性を16進ビット列で書く形
// （emitter は bf16/f16 の (unsigned-byte 16) をこの形でそのまま出せる）。
// 0x3F80 = 1.0（bf16）、0x4020 = 2.5（bf16）。
func.func @main() -> tensor<2xbf16> {
  %c = stablehlo.constant dense<[0x3F80, 0x4020]> : tensor<2xbf16>
  func.return %c : tensor<2xbf16>
}
