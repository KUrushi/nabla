// #233 の候補 (b) を一般化した「連鎖の形」の race 用の形（while 版）。形 (a): y_t が制御構造の結果でない。
// 内側の while の結果 %ic は新しい carry にだけ効き、y_t（%yt）は本体の入力 %bc から作る派生値
// （dropout の rng の while が carry にだけ効く scan の形）。barrier には制御構造の全出力と ys のバッファ %by を
// 通し、以後は barrier の出力（%icb, %byb）だけを使う。y_t は barrier に通さない。
// nested-scan-ys-joint-barrier-race.mlir と同じ負荷のもとで、local-sync と比べて 0 の行が返らないかを確かめる
// （docs/iree-upstream-bugs.md の 7）。x86 のランナーでは 806/200000（0.4%）で ys の行が 0 のまま返った（競合する）。
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
      %yt = stablehlo.multiply %bc, %bc : tensor<3xf32>
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
      %icb, %byb = stablehlo.optimization_barrier %ic, %by : tensor<3xf32>, tensor<4x3xf32>
      %u0 = stablehlo.reshape %yt : (tensor<3xf32>) -> tensor<1x3xf32>
      %w = stablehlo.dynamic_update_slice %byb, %u0, %idx, %zi : (tensor<4x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<4x3xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next = stablehlo.add %bi, %one : tensor<i32>
      stablehlo.return %next, %icb, %w : tensor<i32>, tensor<3xf32>, tensor<4x3xf32>
    }) : (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>) -> (tensor<i32>, tensor<3xf32>, tensor<4x3xf32>)
    func.return %c, %ys : tensor<3xf32>, tensor<4x3xf32>
  }
}
