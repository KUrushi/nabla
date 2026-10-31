// group-norm-grad-multiplier.mlir の回避策の形（docs/iree-upstream-bugs.md の 9）。順伝播を
// y = ((x - mean) * rsqrt(var + eps)) * scale + bias（正規化してから scale をかける）にした grad。
// scale の勾配の縮約の被演算子が rsqrt の broadcast でなく正規化した値になる。iree-compile 単体で数百回落ちない。
module {
  func.func @main(%0: tensor<4xf32>, %1: tensor<4xf32>, %2: tensor<2x5x4xf32>, %3: tensor<2x5x4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<2x5x4xf32>) {
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.constant dense<5.0> : tensor<f32>
    %6 = stablehlo.constant dense<5.0> : tensor<f32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.0e-6> : tensor<f32>
    %9 = stablehlo.constant dense<0.5> : tensor<f32>
    %10 = stablehlo.constant dense<1.0> : tensor<f32>
    %11 = stablehlo.constant dense<0.0> : tensor<f32>
    %12 = stablehlo.constant dense<-0.5> : tensor<f32>
    %13 = stablehlo.reshape %2 : (tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
    %14 = stablehlo.transpose %13, dims = [0, 2, 1] : (tensor<2x5x4xf32>) -> tensor<2x4x5xf32>
    %15 = stablehlo.reshape %14 : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %init_16 = stablehlo.constant dense<0.0> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %init_16) applies stablehlo.add across dimensions = [2] : (tensor<2x4x5xf32>, tensor<f32>) -> tensor<2x4xf32>
    %17 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %18 = stablehlo.divide %16, %17 : tensor<2x4xf32>
    %19 = stablehlo.multiply %15, %15 : tensor<2x4x5xf32>
    %init_20 = stablehlo.constant dense<0.0> : tensor<f32>
    %20 = stablehlo.reduce(%19 init: %init_20) applies stablehlo.add across dimensions = [2] : (tensor<2x4x5xf32>, tensor<f32>) -> tensor<2x4xf32>
    %21 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %22 = stablehlo.divide %20, %21 : tensor<2x4xf32>
    %23 = stablehlo.multiply %18, %18 : tensor<2x4xf32>
    %24 = stablehlo.subtract %22, %23 : tensor<2x4xf32>
    %25 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %26 = stablehlo.maximum %24, %25 : tensor<2x4xf32>
    %27 = stablehlo.compare EQ, %24, %26 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
    %28 = stablehlo.compare EQ, %25, %26 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
    %29 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %30 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %31 = stablehlo.select %28, %29, %30 : tensor<2x4xi1>, tensor<2x4xf32>
    %32 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %33 = stablehlo.select %27, %31, %32 : tensor<2x4xi1>, tensor<2x4xf32>
    %34 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>
    %35 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %36 = stablehlo.add %26, %35 : tensor<2x4xf32>
    %37 = stablehlo.rsqrt %36 : tensor<2x4xf32>
    %38 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %39 = stablehlo.divide %37, %36 : tensor<2x4xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<2x4xf32>
    %41 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>
    %42 = stablehlo.reshape %34 : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %43 = stablehlo.transpose %42, dims = [0, 2, 1] : (tensor<2x4x5xf32>) -> tensor<2x5x4xf32>
    %44 = stablehlo.reshape %43 : (tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
    %45 = stablehlo.reshape %41 : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %46 = stablehlo.transpose %45, dims = [0, 2, 1] : (tensor<2x4x5xf32>) -> tensor<2x5x4xf32>
    %47 = stablehlo.reshape %46 : (tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
    %48 = stablehlo.subtract %2, %44 : tensor<2x5x4xf32>
    %49 = stablehlo.multiply %48, %47 : tensor<2x5x4xf32>
    %50 = stablehlo.broadcast_in_dim %0, dims = [2] : (tensor<4xf32>) -> tensor<2x5x4xf32>
    %51 = stablehlo.multiply %49, %50 : tensor<2x5x4xf32>
    %52 = stablehlo.broadcast_in_dim %1, dims = [2] : (tensor<4xf32>) -> tensor<2x5x4xf32>
    %53 = stablehlo.add %51, %52 : tensor<2x5x4xf32>
    %54 = stablehlo.multiply %53, %3 : tensor<2x5x4xf32>
    %init_55 = stablehlo.constant dense<0.0> : tensor<f32>
    %55 = stablehlo.reduce(%54 init: %init_55) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<2x5x4xf32>, tensor<f32>) -> tensor<f32>
    %56 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<2x5x4xf32>
    %57 = stablehlo.multiply %3, %56 : tensor<2x5x4xf32>
    %init_58 = stablehlo.constant dense<0.0> : tensor<f32>
    %58 = stablehlo.reduce(%57 init: %init_58) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x5x4xf32>, tensor<f32>) -> tensor<4xf32>
    %59 = stablehlo.multiply %49, %57 : tensor<2x5x4xf32>
    %60 = stablehlo.multiply %50, %57 : tensor<2x5x4xf32>
    %init_61 = stablehlo.constant dense<0.0> : tensor<f32>
    %61 = stablehlo.reduce(%59 init: %init_61) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x5x4xf32>, tensor<f32>) -> tensor<4xf32>
    %62 = stablehlo.multiply %48, %60 : tensor<2x5x4xf32>
    %63 = stablehlo.multiply %47, %60 : tensor<2x5x4xf32>
    %64 = stablehlo.negate %63 : tensor<2x5x4xf32>
    %65 = stablehlo.reshape %62 : (tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
    %66 = stablehlo.transpose %65, dims = [0, 2, 1] : (tensor<2x5x4xf32>) -> tensor<2x4x5xf32>
    %67 = stablehlo.reshape %66 : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %68 = stablehlo.reshape %64 : (tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
    %69 = stablehlo.transpose %68, dims = [0, 2, 1] : (tensor<2x5x4xf32>) -> tensor<2x4x5xf32>
    %70 = stablehlo.reshape %69 : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %init_71 = stablehlo.constant dense<0.0> : tensor<f32>
    %71 = stablehlo.reduce(%67 init: %init_71) applies stablehlo.add across dimensions = [2] : (tensor<2x4x5xf32>, tensor<f32>) -> tensor<2x4xf32>
    %72 = stablehlo.multiply %40, %71 : tensor<2x4xf32>
    %init_73 = stablehlo.constant dense<0.0> : tensor<f32>
    %73 = stablehlo.reduce(%70 init: %init_73) applies stablehlo.add across dimensions = [2] : (tensor<2x4x5xf32>, tensor<f32>) -> tensor<2x4xf32>
    %74 = stablehlo.multiply %33, %72 : tensor<2x4xf32>
    %75 = stablehlo.negate %74 : tensor<2x4xf32>
    %76 = stablehlo.multiply %18, %75 : tensor<2x4xf32>
    %77 = stablehlo.add %73, %76 : tensor<2x4xf32>
    %78 = stablehlo.multiply %18, %75 : tensor<2x4xf32>
    %79 = stablehlo.add %77, %78 : tensor<2x4xf32>
    %80 = stablehlo.divide %74, %21 : tensor<2x4xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>
    %82 = stablehlo.multiply %15, %81 : tensor<2x4x5xf32>
    %83 = stablehlo.multiply %15, %81 : tensor<2x4x5xf32>
    %84 = stablehlo.add %82, %83 : tensor<2x4x5xf32>
    %85 = stablehlo.divide %79, %17 : tensor<2x4xf32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4x5xf32>
    %87 = stablehlo.add %84, %86 : tensor<2x4x5xf32>
    %88 = stablehlo.reshape %87 : (tensor<2x4x5xf32>) -> tensor<2x4x5xf32>
    %89 = stablehlo.transpose %88, dims = [0, 2, 1] : (tensor<2x4x5xf32>) -> tensor<2x5x4xf32>
    %90 = stablehlo.reshape %89 : (tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
    %91 = stablehlo.add %63, %90 : tensor<2x5x4xf32>
    func.return %61, %58, %91 : tensor<4xf32>, tensor<4xf32>, tensor<2x5x4xf32>
  }
}