// #233 の候補 (b) を一般化した「連鎖の形」の race 用の形（while 版）。形 (c): ys が 2 つ。
// 内側の while の直後に、その全出力と 2 つの ys のバッファ（%by0, %by1）を 1 つの optimization_barrier に通し、
// 以後は barrier の出力（%icb, %by0b, %by1b）だけを使う。y0_t は %icb、y1_t は %icb から作る派生値。
// ys の初期値は nabla の emit と同じく、ys ごとに別のスカラーを broadcast して別の barrier に通す。
// nested-scan-ys-joint-barrier-race.mlir と同じ負荷のもとで、local-sync と比べて 0 の行が返らないかを確かめる
// （docs/iree-upstream-bugs.md の 7）。
// INPUT: --input=3xf32=0.5,-0.25,0.9
module {
  func.func @main(%h: tensor<3xf32>) -> (tensor<3xf32>, tensor<4x3xf32>, tensor<4x3xf32>) {
    %i0c = stablehlo.constant dense<0> : tensor<i32>
    %z0 = stablehlo.constant dense<0.0> : tensor<f32>
    %z1 = stablehlo.constant dense<0.0> : tensor<f32>
    %salt = stablehlo.constant dense<4> : tensor<i32>
    %i0, %zs0, %zs1, %u = stablehlo.optimization_barrier %i0c, %z0, %z1, %salt : tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>
    %yb0 = stablehlo.broadcast_in_dim %zs0, dims = [] : (tensor<f32>) -> tensor<4x3xf32>
    %y00 = stablehlo.optimization_barrier %yb0 : tensor<4x3xf32>
    %yb1 = stablehlo.broadcast_in_dim %zs1, dims = [] : (tensor<f32>) -> tensor<4x3xf32>
    %y10 = stablehlo.optimization_barrier %yb1 : tensor<4x3xf32>
    %n, %c, %ys0, %ys1 = "stablehlo.while"(%i0, %h, %y00, %y10) ({
    ^bb0(%ci: tensor<i32>, %cc: tensor<3xf32>, %cy0: tensor<4x3xf32>, %cy1: tensor<4x3xf32>):
      %len = stablehlo.constant dense<4> : tensor<i32>
      %lt = stablehlo.compare LT, %ci, %len : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    }, {
    ^bb0(%bi: tensor<i32>, %bc: tensor<3xf32>, %by0: tensor<4x3xf32>, %by1: tensor<4x3xf32>):
      %last = stablehlo.constant dense<3> : tensor<i32>
      %idx = stablehlo.subtract %last, %bi : tensor<i32>
      %zi = stablehlo.constant dense<0> : tensor<i32>
      %j0c = stablehlo.constant dense<0> : tensor<i32>
      %jz = stablehlo.constant dense<0.0> : tensor<f32>
      %jsalt = stablehlo.constant dense<2> : tensor<i32>
      %j0, %jzs, %ju = stablehlo.optimization_barrier %j0c, %jz, %jsalt : tensor<i32>, tensor<f32>, tensor<i32>
      %jn, %ic = "stablehlo.while"(%j0, %bc) ({
      ^bb0(%ji: tensor<i32>, %jc: tensor<3xf32>):
        %jlen = stablehlo.constant dense<2> : tensor<i32>
        %jlt = stablehlo.compare LT, %ji, %jlen : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %jlt : tensor<i1>
      }, {
      ^bb0(%jbi: tensor<i32>, %jbc: tensor<3xf32>):
        %t = stablehlo.tanh %jbc : tensor<3xf32>
        %jone = stablehlo.constant dense<1> : tensor<i32>
        %jnext = stablehlo.add %jbi, %jone : tensor<i32>
        stablehlo.return %jnext, %t : tensor<i32>, tensor<3xf32>
      }) : (tensor<i32>, tensor<3xf32>) -> (tensor<i32>, tensor<3xf32>)
      %icb, %by0b, %by1b = stablehlo.optimization_barrier %ic, %by0, %by1 : tensor<3xf32>, tensor<4x3xf32>, tensor<4x3xf32>
      %y1t = stablehlo.negate %icb : tensor<3xf32>
      %u0 = stablehlo.reshape %icb : (tensor<3xf32>) -> tensor<1x3xf32>
      %w0 = stablehlo.dynamic_update_slice %by0b, %u0, %idx, %zi : (tensor<4x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
      %u1 = stablehlo.reshape %y1t : (tensor<3xf32>) -> tensor<1x3xf32>
      %w1 = stablehlo.dynamic_update_slice %by1b, %u1, %idx, %zi : (tensor<4x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next = stablehlo.add %bi, %one : tensor<i32>
      stablehlo.return %next, %icb, %w0, %w1 : tensor<i32>, tensor<3xf32>, tensor<4x3xf32>, tensor<4x3xf32>
    }) : (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>, tensor<4x3xf32>) -> (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>, tensor<4x3xf32>)
    func.return %c, %ys0, %ys1 : tensor<3xf32>, tensor<4x3xf32>, tensor<4x3xf32>
  }
}
