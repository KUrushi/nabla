// IREE 3.11 local-task: nested-scan-ys-barrier-race.mlir の内側の while を stablehlo.if（cond*）に替えた形。
// 本体が ys を optimization_barrier に通し、その結果へ if をはさんで dynamic_update_slice すると、
// 負荷のもとで ys の1行以上が 0 のまま返る（docs/iree-upstream-bugs.md の 7）。
// 対照は nested-scan-ys-if-no-barrier.mlir（%bk の barrier を外した形。nabla の回避策）。
// INPUT: --input=3xf32=0.5,-0.25,0.9
module {
  func.func @main(%h: tensor<3xf32>) -> (tensor<3xf32>, tensor<4x3xf32>) {
    %i0c = stablehlo.constant dense<0> : tensor<i32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %salt = stablehlo.constant dense<4> : tensor<i32>
    %i0, %zs, %u = stablehlo.optimization_barrier %i0c, %z, %salt : tensor<i32>, tensor<f32>, tensor<i32>
    %yb = stablehlo.broadcast_in_dim %zs, dims = [] : (tensor<f32>) -> tensor<4x3xf32>
    %y0 = stablehlo.optimization_barrier %yb : tensor<4x3xf32>
    %n, %c, %ys = "stablehlo.while"(%i0, %h, %y0) ({
    ^bb0(%ci: tensor<i32>, %cc: tensor<3xf32>, %cy: tensor<4x3xf32>):
      %len = stablehlo.constant dense<4> : tensor<i32>
      %lt = stablehlo.compare LT, %ci, %len : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    }, {
    ^bb0(%bi: tensor<i32>, %bc: tensor<3xf32>, %by: tensor<4x3xf32>):
      %bk = stablehlo.optimization_barrier %by : tensor<4x3xf32>
      %last = stablehlo.constant dense<3> : tensor<i32>
      %idx = stablehlo.subtract %last, %bi : tensor<i32>
      %zi = stablehlo.constant dense<0> : tensor<i32>
      %j0c = stablehlo.constant dense<0> : tensor<i32>
      %jz = stablehlo.constant dense<0.0> : tensor<f32>
      %jsalt = stablehlo.constant dense<2> : tensor<i32>
      %j0, %jzs, %ju = stablehlo.optimization_barrier %j0c, %jz, %jsalt : tensor<i32>, tensor<f32>, tensor<i32>
      %two = stablehlo.constant dense<2> : tensor<i32>
      %pr = stablehlo.compare LT, %bi, %two : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %ic = "stablehlo.if"(%pr) ({
        %t = stablehlo.tanh %bc : tensor<3xf32>
        stablehlo.return %t : tensor<3xf32>
      }, {
        %t2 = stablehlo.tanh %bc : tensor<3xf32>
        %t3 = stablehlo.tanh %t2 : tensor<3xf32>
        stablehlo.return %t3 : tensor<3xf32>
      }) : (tensor<i1>) -> tensor<3xf32>
      %u0 = stablehlo.reshape %ic : (tensor<3xf32>) -> tensor<1x3xf32>
      %w = stablehlo.dynamic_update_slice %bk, %u0, %idx, %zi : (tensor<4x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next = stablehlo.add %bi, %one : tensor<i32>
      stablehlo.return %next, %ic, %w : tensor<i32>, tensor<3xf32>, tensor<4x3xf32>
    }) : (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>) -> (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>)
    func.return %c, %ys : tensor<3xf32>, tensor<4x3xf32>
  }
}
