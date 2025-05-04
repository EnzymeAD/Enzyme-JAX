// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s

module @reactant_Boltz.L... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2xf32>) -> tensor<3x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x3xf32>) -> tensor<2x1xf32>
    %2 = stablehlo.reshape %1 : (tensor<2x1xf32>) -> tensor<2xf32>
    %3 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<2x1xf32>
    %4 = stablehlo.subtract %1, %3 : tensor<2x1xf32>
    %5 = stablehlo.multiply %4, %4 : tensor<2x1xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.subtract %2, %arg1 : tensor<2xf32>
    %8 = stablehlo.multiply %7, %7 : tensor<2xf32>
    %9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.reshape %6 : (tensor<f32>) -> tensor<1x1xf32>
    %11 = stablehlo.reshape %9 : (tensor<f32>) -> tensor<1x1xf32>
    %12 = stablehlo.add %10, %11 : tensor<1x1xf32>
    %13 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x3xf32>) -> tensor<2x1xf32>
    %14 = stablehlo.reshape %13 : (tensor<2x1xf32>) -> tensor<2xf32>
    %15 = stablehlo.subtract %13, %3 : tensor<2x1xf32>
    %16 = stablehlo.multiply %15, %15 : tensor<2x1xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %18 = stablehlo.subtract %14, %arg1 : tensor<2xf32>
    %19 = stablehlo.multiply %18, %18 : tensor<2xf32>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %21 = stablehlo.reshape %17 : (tensor<f32>) -> tensor<1x1xf32>
    %22 = stablehlo.reshape %20 : (tensor<f32>) -> tensor<1x1xf32>
    %23 = stablehlo.add %21, %22 : tensor<1x1xf32>
    %24 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x3xf32>) -> tensor<2x1xf32>
    %25 = stablehlo.reshape %24 : (tensor<2x1xf32>) -> tensor<2xf32>
    %26 = stablehlo.subtract %24, %3 : tensor<2x1xf32>
    %27 = stablehlo.multiply %26, %26 : tensor<2x1xf32>
    %28 = stablehlo.reduce(%27 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.subtract %25, %arg1 : tensor<2xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<2xf32>
    %31 = stablehlo.reduce(%30 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %32 = stablehlo.reshape %28 : (tensor<f32>) -> tensor<1x1xf32>
    %33 = stablehlo.reshape %31 : (tensor<f32>) -> tensor<1x1xf32>
    %34 = stablehlo.add %32, %33 : tensor<1x1xf32>
    %35 = stablehlo.concatenate %12, %23, %34, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<3x1xf32>
    return %35 : tensor<3x1xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2xf32>) -> tensor<3x1xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x3xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<2x1xf32>) -> tensor<2xf32>
// CHECK-NEXT:     %3 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:     %4 = stablehlo.subtract %1, %3 : tensor<2x1xf32>
// CHECK-NEXT:     %5 = stablehlo.multiply %4, %4 : tensor<2x1xf32>
// CHECK-NEXT:     %6 = stablehlo.subtract %2, %arg1 : tensor<2xf32>
// CHECK-NEXT:     %7 = stablehlo.multiply %6, %6 : tensor<2xf32>
// CHECK-NEXT:     %8 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x3xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:     %9 = stablehlo.reshape %8 : (tensor<2x1xf32>) -> tensor<2xf32>
// CHECK-NEXT:     %10 = stablehlo.subtract %8, %3 : tensor<2x1xf32>
// CHECK-NEXT:     %11 = stablehlo.multiply %10, %10 : tensor<2x1xf32>
// CHECK-NEXT:     %12 = stablehlo.subtract %9, %arg1 : tensor<2xf32>
// CHECK-NEXT:     %13 = stablehlo.multiply %12, %12 : tensor<2xf32>
// CHECK-NEXT:     %14 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x3xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:     %15 = stablehlo.reshape %14 : (tensor<2x1xf32>) -> tensor<2xf32>
// CHECK-NEXT:     %16 = stablehlo.subtract %14, %3 : tensor<2x1xf32>
// CHECK-NEXT:     %17 = stablehlo.multiply %16, %16 : tensor<2x1xf32>
// CHECK-NEXT:     %18 = stablehlo.subtract %15, %arg1 : tensor<2xf32>
// CHECK-NEXT:     %19 = stablehlo.multiply %18, %18 : tensor<2xf32>
// CHECK-NEXT:     %20 = stablehlo.reshape %5 : (tensor<2x1xf32>) -> tensor<1x2x1xf32>
// CHECK-NEXT:     %21 = stablehlo.reshape %11 : (tensor<2x1xf32>) -> tensor<1x2x1xf32>
// CHECK-NEXT:     %22 = stablehlo.reshape %17 : (tensor<2x1xf32>) -> tensor<1x2x1xf32>
// CHECK-NEXT:     %23 = stablehlo.concatenate %20, %21, %22, dim = 0 : (tensor<1x2x1xf32>, tensor<1x2x1xf32>, tensor<1x2x1xf32>) -> tensor<3x2x1xf32>
// CHECK-NEXT:     %24 = stablehlo.reduce(%23 init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<3x2x1xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     %25 = stablehlo.reshape %24 : (tensor<3xf32>) -> tensor<3x1xf32>
// CHECK-NEXT:     %26 = stablehlo.reshape %7 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:     %27 = stablehlo.reshape %13 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:     %28 = stablehlo.reshape %19 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:     %29 = stablehlo.concatenate %26, %27, %28, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<3x2xf32>
// CHECK-NEXT:     %30 = stablehlo.reduce(%29 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x2xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     %31 = stablehlo.reshape %30 : (tensor<3xf32>) -> tensor<3x1xf32>
// CHECK-NEXT:     %32 = stablehlo.add %25, %31 : tensor<3x1xf32>
// CHECK-NEXT:     return %32 : tensor<3x1xf32>
// CHECK-NEXT: }
