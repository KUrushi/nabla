// #233 の候補 (b) の race 用の形（if 版）。制御構造の結果 %ic と ys のバッファ %by を1つの
// optimization_barrier に一緒に通し、その結果の組へ dynamic_update_slice する（%icb, %byb）。
// コピーは消える（scripts/scan-ys-candidates/joint-barrier.mlir.in。長さに線形）。
// nested-scan-ys-if-race.mlir と同じ負荷のもとで、local-sync と比べて 0 の行が返らないかを確かめるためのもの。
// 未検証: Rosetta の VM では race がほとんど再現しない（while 版 0/200、if 版 1/200）。x86 ネイティブで確かめる（docs/iree-upstream-bugs.md の 7）。
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
      %icb, %byb = stablehlo.optimization_barrier %ic, %by : tensor<3xf32>, tensor<4x3xf32>
      %u0 = stablehlo.reshape %icb : (tensor<3xf32>) -> tensor<1x3xf32>
      %w = stablehlo.dynamic_update_slice %byb, %u0, %idx, %zi : (tensor<4x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next = stablehlo.add %bi, %one : tensor<i32>
      stablehlo.return %next, %icb, %w : tensor<i32>, tensor<3xf32>, tensor<4x3xf32>
    }) : (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>) -> (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>)
    func.return %c, %ys : tensor<3xf32>, tensor<4x3xf32>
  }
}
