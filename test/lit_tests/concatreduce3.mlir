module @reactant_Boltz.L... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<6x2xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4x2xf32>, %arg6: tensor<2xf32>, %arg7: tensor<2xf32>) -> tensor<6x1xf32> {
    %cst = stablehlo.constant dense<1.59576917> : tensor<4x6xf32>
    %cst_0 = stablehlo.constant dense<4.471500e-02> : tensor<4x6xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<4x6xf32>
    %cst_2 = stablehlo.constant dense<1.59576917> : tensor<4x1xf32>
    %cst_3 = stablehlo.constant dense<4.471500e-02> : tensor<4x1xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<4x1xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x2xf32>) -> tensor<2x6xf32>
    %1 = stablehlo.reshape %arg7 : (tensor<2xf32>) -> tensor<2x1xf32>
    %2 = stablehlo.dot_general %arg1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %3 = stablehlo.reshape %arg2 : (tensor<4xf32>) -> tensor<4x1xf32>
    %4 = stablehlo.add %2, %3 : tensor<4x1xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<4x1xf32>
    %6 = stablehlo.multiply %5, %cst_3 : tensor<4x1xf32>
    %7 = stablehlo.add %6, %cst_4 : tensor<4x1xf32>
    %8 = stablehlo.multiply %cst_2, %4 : tensor<4x1xf32>
    %9 = stablehlo.multiply %8, %7 : tensor<4x1xf32>
    %10 = stablehlo.logistic %9 : tensor<4x1xf32>
    %11 = stablehlo.multiply %4, %10 : tensor<4x1xf32>
    %12 = stablehlo.dot_general %arg3, %11, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %13 = stablehlo.reshape %arg4 : (tensor<4xf32>) -> tensor<4x1xf32>
    %14 = stablehlo.add %12, %13 : tensor<4x1xf32>
    %15 = stablehlo.multiply %14, %14 : tensor<4x1xf32>
    %16 = stablehlo.multiply %15, %cst_3 : tensor<4x1xf32>
    %17 = stablehlo.add %16, %cst_4 : tensor<4x1xf32>
    %18 = stablehlo.multiply %cst_2, %14 : tensor<4x1xf32>
    %19 = stablehlo.multiply %18, %17 : tensor<4x1xf32>
    %20 = stablehlo.logistic %19 : tensor<4x1xf32>
    %21 = stablehlo.multiply %14, %20 : tensor<4x1xf32>
    %22 = stablehlo.dot_general %arg5, %21, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<4x1xf32>) -> tensor<2x1xf32>
    %23 = stablehlo.reshape %arg6 : (tensor<2xf32>) -> tensor<2x1xf32>
    %24 = stablehlo.add %22, %23 : tensor<2x1xf32>
    %25 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<6x2xf32>) -> tensor<4x6xf32>
    %26 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<4xf32>) -> tensor<4x6xf32>
    %27 = stablehlo.add %25, %26 : tensor<4x6xf32>
    %28 = stablehlo.multiply %27, %27 : tensor<4x6xf32>
    %29 = stablehlo.multiply %28, %cst_0 : tensor<4x6xf32>
    %30 = stablehlo.add %29, %cst_1 : tensor<4x6xf32>
    %31 = stablehlo.multiply %cst, %27 : tensor<4x6xf32>
    %32 = stablehlo.multiply %31, %30 : tensor<4x6xf32>
    %33 = stablehlo.logistic %32 : tensor<4x6xf32>
    %34 = stablehlo.multiply %27, %33 : tensor<4x6xf32>
    %35 = stablehlo.dot_general %arg3, %34, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x6xf32>) -> tensor<4x6xf32>
    %36 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<4xf32>) -> tensor<4x6xf32>
    %37 = stablehlo.add %35, %36 : tensor<4x6xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<4x6xf32>
    %39 = stablehlo.multiply %38, %cst_0 : tensor<4x6xf32>
    %40 = stablehlo.add %39, %cst_1 : tensor<4x6xf32>
    %41 = stablehlo.multiply %cst, %37 : tensor<4x6xf32>
    %42 = stablehlo.multiply %41, %40 : tensor<4x6xf32>
    %43 = stablehlo.logistic %42 : tensor<4x6xf32>
    %44 = stablehlo.multiply %37, %43 : tensor<4x6xf32>
    %45 = stablehlo.dot_general %arg5, %44, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<4x6xf32>) -> tensor<2x6xf32>
    %46 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<2xf32>) -> tensor<2x6xf32>
    %47 = stablehlo.add %45, %46 : tensor<2x6xf32>
    %48 = stablehlo.slice %47 [0:2, 0:1] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %49 = stablehlo.subtract %48, %24 : tensor<2x1xf32>
    %50 = stablehlo.multiply %49, %49 : tensor<2x1xf32>
    %51 = stablehlo.reduce(%50 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %52 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %53 = stablehlo.reshape %52 : (tensor<2x1xf32>) -> tensor<2xf32>
    %54 = stablehlo.subtract %53, %arg7 : tensor<2xf32>
    %55 = stablehlo.multiply %54, %54 : tensor<2xf32>
    %56 = stablehlo.reduce(%55 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %57 = stablehlo.reshape %51 : (tensor<f32>) -> tensor<1x1xf32>
    %58 = stablehlo.reshape %56 : (tensor<f32>) -> tensor<1x1xf32>
    %59 = stablehlo.add %57, %58 : tensor<1x1xf32>
    %60 = stablehlo.slice %47 [0:2, 1:2] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %61 = stablehlo.subtract %60, %24 : tensor<2x1xf32>
    %62 = stablehlo.multiply %61, %61 : tensor<2x1xf32>
    %63 = stablehlo.reduce(%62 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %64 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %65 = stablehlo.reshape %64 : (tensor<2x1xf32>) -> tensor<2xf32>
    %66 = stablehlo.subtract %65, %arg7 : tensor<2xf32>
    %67 = stablehlo.multiply %66, %66 : tensor<2xf32>
    %68 = stablehlo.reduce(%67 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %69 = stablehlo.reshape %63 : (tensor<f32>) -> tensor<1x1xf32>
    %70 = stablehlo.reshape %68 : (tensor<f32>) -> tensor<1x1xf32>
    %71 = stablehlo.add %69, %70 : tensor<1x1xf32>
    %72 = stablehlo.slice %47 [0:2, 2:3] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %73 = stablehlo.subtract %72, %24 : tensor<2x1xf32>
    %74 = stablehlo.multiply %73, %73 : tensor<2x1xf32>
    %75 = stablehlo.reduce(%74 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %76 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %77 = stablehlo.reshape %76 : (tensor<2x1xf32>) -> tensor<2xf32>
    %78 = stablehlo.subtract %77, %arg7 : tensor<2xf32>
    %79 = stablehlo.multiply %78, %78 : tensor<2xf32>
    %80 = stablehlo.reduce(%79 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %81 = stablehlo.reshape %75 : (tensor<f32>) -> tensor<1x1xf32>
    %82 = stablehlo.reshape %80 : (tensor<f32>) -> tensor<1x1xf32>
    %83 = stablehlo.add %81, %82 : tensor<1x1xf32>
    %84 = stablehlo.slice %47 [0:2, 3:4] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %85 = stablehlo.subtract %84, %24 : tensor<2x1xf32>
    %86 = stablehlo.multiply %85, %85 : tensor<2x1xf32>
    %87 = stablehlo.reduce(%86 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %88 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %89 = stablehlo.reshape %88 : (tensor<2x1xf32>) -> tensor<2xf32>
    %90 = stablehlo.subtract %89, %arg7 : tensor<2xf32>
    %91 = stablehlo.multiply %90, %90 : tensor<2xf32>
    %92 = stablehlo.reduce(%91 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %93 = stablehlo.reshape %87 : (tensor<f32>) -> tensor<1x1xf32>
    %94 = stablehlo.reshape %92 : (tensor<f32>) -> tensor<1x1xf32>
    %95 = stablehlo.add %93, %94 : tensor<1x1xf32>
    %96 = stablehlo.slice %47 [0:2, 4:5] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %97 = stablehlo.subtract %96, %24 : tensor<2x1xf32>
    %98 = stablehlo.multiply %97, %97 : tensor<2x1xf32>
    %99 = stablehlo.reduce(%98 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %100 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %101 = stablehlo.reshape %100 : (tensor<2x1xf32>) -> tensor<2xf32>
    %102 = stablehlo.subtract %101, %arg7 : tensor<2xf32>
    %103 = stablehlo.multiply %102, %102 : tensor<2xf32>
    %104 = stablehlo.reduce(%103 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %105 = stablehlo.reshape %99 : (tensor<f32>) -> tensor<1x1xf32>
    %106 = stablehlo.reshape %104 : (tensor<f32>) -> tensor<1x1xf32>
    %107 = stablehlo.add %105, %106 : tensor<1x1xf32>
    %108 = stablehlo.slice %47 [0:2, 5:6] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %109 = stablehlo.subtract %108, %24 : tensor<2x1xf32>
    %110 = stablehlo.multiply %109, %109 : tensor<2x1xf32>
    %111 = stablehlo.reduce(%110 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %112 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %113 = stablehlo.reshape %112 : (tensor<2x1xf32>) -> tensor<2xf32>
    %114 = stablehlo.subtract %113, %arg7 : tensor<2xf32>
    %115 = stablehlo.multiply %114, %114 : tensor<2xf32>
    %116 = stablehlo.reduce(%115 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %117 = stablehlo.reshape %111 : (tensor<f32>) -> tensor<1x1xf32>
    %118 = stablehlo.reshape %116 : (tensor<f32>) -> tensor<1x1xf32>
    %119 = stablehlo.add %117, %118 : tensor<1x1xf32>
    %120 = stablehlo.concatenate %59, %71, %83, %95, %107, %119, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<6x1xf32>
    return %120 : tensor<6x1xf32>
  }
}