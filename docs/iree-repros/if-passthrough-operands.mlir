// A stablehlo.if whose then region returns three operands unchanged ("pass-through") next to one
// computed result fails at run time with
//   INVALID_ARGUMENT; ref is null; while invoking native function hal.buffer_view.create
// when the *other* (else) region is taken (p = 1.0). The then region (p = 0.0) works.
// Both --device=local-task and --device=local-sync fail. The compiler accepts the module.
// if-passthrough-operands-select.mlir computes the same values without a pass-through result.
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
    %r0, %r1, %r2, %r3 = "stablehlo.if"(%pred) ({
      %t0 = stablehlo.multiply %a, %k : tensor<f32>
      stablehlo.return %t0, %b, %c, %d : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
    }, {
      %e0 = stablehlo.multiply %a, %k : tensor<f32>
      %e1 = stablehlo.multiply %b, %k : tensor<f32>
      %e2 = stablehlo.multiply %c, %k : tensor<f32>
      %e3 = stablehlo.multiply %d, %k : tensor<f32>
      stablehlo.return %e0, %e1, %e2, %e3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
    }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
    return %r0, %r1, %r2, %r3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
