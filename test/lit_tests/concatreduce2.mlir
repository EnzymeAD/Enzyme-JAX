// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s

module @reactant_Boltz.L... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2xf32>) -> tensor<2x1xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
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
    %12 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<2x1xf32>) -> tensor<2xf32>
    %14 = stablehlo.subtract %12, %3 : tensor<2x1xf32>
    %15 = stablehlo.multiply %14, %14 : tensor<2x1xf32>
    %16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x1xf32>, tensor<f32>) -> tensor<f32>
    %17 = stablehlo.subtract %13, %arg1 : tensor<2xf32>
    %18 = stablehlo.multiply %17, %17 : tensor<2xf32>
    %19 = stablehlo.reduce(%18 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %20 = stablehlo.reshape %16 : (tensor<f32>) -> tensor<1x1xf32>
    %21 = stablehlo.reshape %19 : (tensor<f32>) -> tensor<1x1xf32>
    %22 = stablehlo.concatenate %10, %20, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
    %23 = stablehlo.concatenate %11, %21, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<2x1xf32>
    %24 = stablehlo.add %22, %23 : tensor<2x1xf32>
    return %24 : tensor<2x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2xf32>) -> tensor<2x1xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.slice %[[i0]] [0:2, 0:1] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.reshape %[[i1]] : (tensor<2x1xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %[[i4:.+]] = stablehlo.slice %[[i0]] [0:2, 1:2] : (tensor<2x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %[[i5:.+]] = stablehlo.reshape %[[i4]] : (tensor<2x1xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %[[i6:.+]] = stablehlo.concatenate %[[i1]], %[[i4]], dim = 0 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:    %[[i14:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<2x2x1xf32>
// CHECK-NEXT:    %[[i7:.+]] = stablehlo.reshape %[[i14]] : (tensor<2x2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:    %[[i8:.+]] = stablehlo.subtract %[[i6]], %[[i7]] : tensor<4x1xf32>
// CHECK-NEXT:    %[[i9:.+]] = stablehlo.multiply %[[i8]], %[[i8]] : tensor<4x1xf32>
// CHECK-NEXT:    %[[i10:.+]] = stablehlo.reshape %[[i9]] : (tensor<4x1xf32>) -> tensor<2x2x1xf32>
// CHECK-NEXT:    %[[i11:.+]] = stablehlo.reduce(%[[i10]] init: %cst) applies stablehlo.add across dimensions = [1, 2] : (tensor<2x2x1xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:    %[[i12:.+]] = stablehlo.reshape %[[i11]] : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %[[i13:.+]] = stablehlo.concatenate %[[i2]], %[[i5]], dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %[[i142:.+]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    %[[i143:.+]] = stablehlo.reshape %[[i142:.+]] : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK-NEXT:    %[[i15:.+]] = stablehlo.subtract %[[i13]], %[[i143]] : tensor<4xf32>
// CHECK-NEXT:    %[[i16:.+]] = stablehlo.multiply %[[i15]], %[[i15]] : tensor<4xf32>
// CHECK-NEXT:    %[[i17:.+]] = stablehlo.reshape %[[i16]] : (tensor<4xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:    %[[i18:.+]] = stablehlo.reduce(%[[i17]] init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:    %[[i19:.+]] = stablehlo.reshape %[[i18]] : (tensor<2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %[[i20:.+]] = stablehlo.add %[[i12]], %[[i19]] : tensor<2x1xf32>
// CHECK-NEXT:    return %[[i20]] : tensor<2x1xf32>
// CHECK-NEXT:  }
