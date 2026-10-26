// RUN: enzymexlamlir-opt %s -raise-affine-to-stablehlo | FileCheck %s

func.func @test_if_0(%arg0: memref<10xf32>, %arg1: memref<i1>) {
  %cond = affine.load %arg1[] : memref<i1>
  scf.if %cond {
    %v = affine.load %arg0[0] : memref<10xf32>
    %v2 = arith.addf %v, %v : f32
    affine.store %v2, %arg0[0] : memref<10xf32>
  }
  return
}

// CHECK:    func.func private @test_if_0_raised(%[[a1:.+]]: tensor<10xf32>, %[[a2:.+]]: tensor<i1>) -> (tensor<10xf32>, tensor<i1>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.slice %[[a1]] [0:1] : (tensor<10xf32>) -> tensor<1xf32>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.reshape %[[a3]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %[[a5:.+]] = arith.addf %[[a4]], %[[a4]] : tensor<f32>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.broadcast_in_dim %[[a5]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.slice %[[a1]] [0:1] : (tensor<10xf32>) -> tensor<1xf32>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.reshape %[[a12]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.reshape %[[a13]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.select %[[a2]], %[[a14]], %[[a15]] : tensor<i1>, tensor<f32>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a17]], %[[a11]] : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK-NEXT:    return %[[a18]], %[[a2]] : tensor<10xf32>, tensor<i1>
// CHECK-NEXT:  }
