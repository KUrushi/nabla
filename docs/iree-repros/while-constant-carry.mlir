func.func @main(%x: tensor<3xf32>, %y: tensor<f32>) -> tensor<3xf32> {
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %r:3 = "stablehlo.while"(%c0, %x, %y) ({
    ^bb0(%i: tensor<f32>, %v: tensor<3xf32>, %s: tensor<f32>):
      %n = stablehlo.constant dense<4.0> : tensor<f32>
      %p = stablehlo.compare LT, %i, %n : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
  }, {
    ^bb0(%i: tensor<f32>, %v: tensor<3xf32>, %s: tensor<f32>):
      %one = stablehlo.constant dense<1.0> : tensor<f32>
      %i1 = stablehlo.add %i, %one : tensor<f32>
      %v1 = stablehlo.add %v, %v : tensor<3xf32>
      %s1 = stablehlo.add %s, %one : tensor<f32>
      stablehlo.return %i1, %v1, %s1 : tensor<f32>, tensor<3xf32>, tensor<f32>
  }) : (tensor<f32>, tensor<3xf32>, tensor<f32>) -> (tensor<f32>, tensor<3xf32>, tensor<f32>)
  return %r#1 : tensor<3xf32>
}
