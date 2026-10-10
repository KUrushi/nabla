func.func @main(%a: tensor<2x0xf32>, %b: tensor<0x3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x0xf32>, tensor<0x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
