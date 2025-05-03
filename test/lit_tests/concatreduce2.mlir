module @reactant_Boltz.L... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4x2xf32>, %arg6: tensor<2xf32>, %arg7: tensor<2xf32>) -> tensor<2x1xf32> {
    %cst = stablehlo.constant dense<1.59576917> : tensor<4x2xf32>
    %cst_0 = stablehlo.constant dense<4.471500e-02> : tensor<4x2xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<4x2xf32>
    %cst_2 = stablehlo.constant dense<1.59576917> : tensor<4x1xf32>
    %cst_3 = stablehlo.constant dense<4.471500e-02> : tensor<4x1xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<4x1xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
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
    %25 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
    %26 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<4xf32>) -> tensor<4x2xf32>
    %27 = stablehlo.add %25, %26 : tensor<4x2xf32>
    %28 = stablehlo.multiply %27, %27 : tensor<4x2xf32>
    %29 = stablehlo.multiply %28, %cst_0 : tensor<4x2xf32>
    %30 = stablehlo.add %29, %cst_1 : tensor<4x2xf32>
    %31 = stablehlo.multiply %cst, %27 : tensor<4x2xf32>
    %32 = stablehlo.multiply %31, %30 : tensor<4x2xf32>
    %33 = stablehlo.logistic %32 : tensor<4x2xf32>
    %34 = stablehlo.multiply %27, %33 : tensor<4x2xf32>
    %35 = stablehlo.dot_general %arg3, %34, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
    %36 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<4xf32>) -> tensor<4x2xf32>
    %37 = stablehlo.add %35, %36 : tensor<4x2xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<4x2xf32>
    %39 = stablehlo.multiply %38, %cst_0 : tensor<4x2xf32>
    %40 = stablehlo.add %39, %cst_1 : tensor<4x2xf32>
    %41 = stablehlo.multiply %cst, %37 : tensor<4x2xf32>
    %42 = stablehlo.multiply %41, %40 : tensor<4x2xf32>
    %43 = stablehlo.logistic %42 : tensor<4x2xf32>
    %44 = stablehlo.multiply %37, %43 : tensor<4x2xf32>
    %45 = stablehlo.dot_general %arg5, %44, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    %46 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<2xf32>) -> tensor<2x2xf32>
    %47 = stablehlo.add %45, %46 : tensor<2x2xf32>
    %48 = stablehlo.slice %47 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %49 = stablehlo.subtract %48, %24 : tensor<2x1xf32>
    %50 = stablehlo.multiply %49, %49 : tensor<2x1xf32>
    %51 = stablehlo.reduce(%50 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %52 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %53 = stablehlo.reshape %52 : (tensor<2x1xf32>) -> tensor<2xf32>
    %54 = stablehlo.subtract %53, %arg7 : tensor<2xf32>
    %55 = stablehlo.multiply %54, %54 : tensor<2xf32>
    %56 = stablehlo.reduce(%55 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %57 = stablehlo.reshape %51 : (tensor<f32>) -> tensor<1x1xf32>
    %58 = stablehlo.reshape %56 : (tensor<f32>) -> tensor<1x1xf32>
    %59 = stablehlo.slice %47 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %60 = stablehlo.subtract %59, %24 : tensor<2x1xf32>
    %61 = stablehlo.multiply %60, %60 : tensor<2x1xf32>
    %62 = stablehlo.reduce(%61 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %63 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %64 = stablehlo.reshape %63 : (tensor<2x1xf32>) -> tensor<2xf32>
    %65 = stablehlo.subtract %64, %arg7 : tensor<2xf32>
    %66 = stablehlo.multiply %65, %65 : tensor<2xf32>
    %67 = stablehlo.reduce(%66 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %68 = stablehlo.reshape %62 : (tensor<f32>) -> tensor<1x1xf32>
    %69 = stablehlo.reshape %67 : (tensor<f32>) -> tensor<1x1xf32>
    %70 = stablehlo.concatenate %57, %68, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
    %71 = stablehlo.concatenate %58, %69, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
    %72 = stablehlo.add %70, %71 : tensor<2x1xf32>
    return %72 : tensor<2x1xf32>
  }
}