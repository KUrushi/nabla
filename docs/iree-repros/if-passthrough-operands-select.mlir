// if-passthrough-operands.mlir の対照。pass-through の結果を stablehlo.if から外し、if の外の
// stablehlo.select にした形（nabla の回避策。src/primitives/cond.lisp の %cond-emit）。
// pass-through の枝は 0 の定数を返す。p = 0.0 / 1.0 のどちらも実行できる。
// INPUT: --input=f32=1.0
// INPUT: --input=f32=1.0
// INPUT: --input=f32=2.0
// INPUT: --input=f32=3.0
// INPUT: --input=f32=4.0
module {
  func.func @main(%p: tensor<f32>, %a: tensor<f32>, %b: tensor<f32>, %c: tensor<f32>, %d: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %half = stablehlo.constant dense<0.5> : tensor<f32>
    %pred = stablehlo.compare LT, %p, %half : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %k = stablehlo.constant dense<0.9> : tensor<f32>
    %r0, %y1, %y2, %y3 = "stablehlo.if"(%pred) ({
      %t0 = stablehlo.multiply %a, %k : tensor<f32>
      %z1 = stablehlo.constant dense<0.0> : tensor<f32>
      %z2 = stablehlo.constant dense<0.0> : tensor<f32>
      %z3 = stablehlo.constant dense<0.0> : tensor<f32>
      stablehlo.return %t0, %z1, %z2, %z3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
    }, {
      %e0 = stablehlo.multiply %a, %k : tensor<f32>
      %e1 = stablehlo.multiply %b, %k : tensor<f32>
      %e2 = stablehlo.multiply %c, %k : tensor<f32>
      %e3 = stablehlo.multiply %d, %k : tensor<f32>
      stablehlo.return %e0, %e1, %e2, %e3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
    }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
    %r1 = stablehlo.select %pred, %b, %y1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %r2 = stablehlo.select %pred, %c, %y2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %r3 = stablehlo.select %pred, %d, %y3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %r0, %r1, %r2, %r3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
