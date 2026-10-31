// #233 の候補 (b) を一般化した「連鎖の形」の race 用の形（if 版）。形 (d): 入れ子の scan。
// 外側の scan の本体に、ys を持つ内側の scan（長さ 2、ys 2x3）があり、内側の本体に stablehlo.if がある。
// 連鎖の barrier は 2 段に出る: 内側の本体では if の全出力と内側の ys のバッファ %jby を 1 つの barrier に通し、
// 外側の本体では内側の while の全出力（carry %ic と ys %iys）と外側の ys のバッファ %by を 1 つの barrier に通す。
// 以後はどちらも barrier の出力だけを使う。外側の y_t は内側の ys そのもの（外側の ys は 4x2x3）。
// nested-scan-ys-if-joint-barrier-race.mlir と同じ負荷のもとで、local-sync と比べて 0 の行が返らないかを確かめる
// （docs/iree-upstream-bugs.md の 7）。
// INPUT: --input=3xf32=0.5,-0.25,0.9
module {
  func.func @main(%h: tensor<3xf32>) -> (tensor<3xf32>, tensor<4x2x3xf32>) {
    %i0c = stablehlo.constant dense<0> : tensor<i32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %salt = stablehlo.constant dense<4> : tensor<i32>
    %i0, %zs, %u = stablehlo.optimization_barrier %i0c, %z, %salt : tensor<i32>, tensor<f32>, tensor<i32>
    %yb = stablehlo.broadcast_in_dim %zs, dims = [] : (tensor<f32>) -> tensor<4x2x3xf32>
    %y0 = stablehlo.optimization_barrier %yb : tensor<4x2x3xf32>
    %n, %c, %ys = "stablehlo.while"(%i0, %h, %y0) ({
    ^bb0(%ci: tensor<i32>, %cc: tensor<3xf32>, %cy: tensor<4x2x3xf32>):
      %len = stablehlo.constant dense<4> : tensor<i32>
      %lt = stablehlo.compare LT, %ci, %len : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    }, {
    ^bb0(%bi: tensor<i32>, %bc: tensor<3xf32>, %by: tensor<4x2x3xf32>):
      %last = stablehlo.constant dense<3> : tensor<i32>
      %idx = stablehlo.subtract %last, %bi : tensor<i32>
      %zi = stablehlo.constant dense<0> : tensor<i32>
      %j0c = stablehlo.constant dense<0> : tensor<i32>
      %jz = stablehlo.constant dense<0.0> : tensor<f32>
      %jsalt = stablehlo.constant dense<2> : tensor<i32>
      %j0, %jzs, %ju = stablehlo.optimization_barrier %j0c, %jz, %jsalt : tensor<i32>, tensor<f32>, tensor<i32>
      %jyb = stablehlo.broadcast_in_dim %jzs, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
      %jy0 = stablehlo.optimization_barrier %jyb : tensor<2x3xf32>
      %jn, %ic, %iys = "stablehlo.while"(%j0, %bc, %jy0) ({
      ^bb0(%ji: tensor<i32>, %jc: tensor<3xf32>, %jcy: tensor<2x3xf32>):
        %jlen = stablehlo.constant dense<2> : tensor<i32>
        %jlt = stablehlo.compare LT, %ji, %jlen : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %jlt : tensor<i1>
      }, {
      ^bb0(%jbi: tensor<i32>, %jbc: tensor<3xf32>, %jby: tensor<2x3xf32>):
        %jzi = stablehlo.constant dense<0> : tensor<i32>
        %jodd = stablehlo.constant dense<1> : tensor<i32>
        %jbit = stablehlo.and %jbi, %jodd : tensor<i32>
        %jpr = stablehlo.compare EQ, %jbit, %jzi : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %t = "stablehlo.if"(%jpr) ({
          %t1 = stablehlo.tanh %jbc : tensor<3xf32>
          stablehlo.return %t1 : tensor<3xf32>
        }, {
          %t2 = stablehlo.tanh %jbc : tensor<3xf32>
          %t3 = stablehlo.tanh %t2 : tensor<3xf32>
          stablehlo.return %t3 : tensor<3xf32>
        }) : (tensor<i1>) -> tensor<3xf32>
        %tb, %jbyb = stablehlo.optimization_barrier %t, %jby : tensor<3xf32>, tensor<2x3xf32>
        %ju0 = stablehlo.reshape %tb : (tensor<3xf32>) -> tensor<1x3xf32>
        %jw = stablehlo.dynamic_update_slice %jbyb, %ju0, %jbi, %jzi : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<2x3xf32>
        %jone = stablehlo.constant dense<1> : tensor<i32>
        %jnext = stablehlo.add %jbi, %jone : tensor<i32>
        stablehlo.return %jnext, %tb, %jw : tensor<i32>, tensor<3xf32>, tensor<2x3xf32>
      }) : (tensor<i32>, tensor<3xf32>, tensor<2x3xf32>) -> (tensor<i32>, tensor<3xf32>, tensor<2x3xf32>)
      %icb, %iysb, %byb = stablehlo.optimization_barrier %ic, %iys, %by : tensor<3xf32>, tensor<2x3xf32>, tensor<4x2x3xf32>
      %u0 = stablehlo.reshape %iysb : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
      %w = stablehlo.dynamic_update_slice %byb, %u0, %idx, %zi, %zi : (tensor<4x2x3xf32>, tensor<1x2x3xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x2x3xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next = stablehlo.add %bi, %one : tensor<i32>
      stablehlo.return %next, %icb, %w : tensor<i32>, tensor<3xf32>, tensor<4x2x3xf32>
    }) : (tensor<i32>, tensor<3xf32>, tensor<4x2x3xf32>) -> (tensor<i32>, tensor<3xf32>, tensor<4x2x3xf32>)
    func.return %c, %ys : tensor<3xf32>, tensor<4x2x3xf32>
  }
}
