// RUN: enzymexlamlir-opt %s --enzyme-hlo-unroll | FileCheck %s

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<2.0> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0:2 = stablehlo.while(%iterArg = %cst_1, %iterArg_2 = %arg0) : tensor<f32>, tensor<10xf32>
     cond {
      %1 = stablehlo.compare  LE, %iterArg, %cst_0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.broadcast_in_dim %iterArg, dims = [] : (tensor<f32>) -> tensor<10xf32>
      %2 = stablehlo.add %iterArg_2, %1 : tensor<10xf32>
      %3 = stablehlo.add %iterArg, %cst : tensor<f32>
      stablehlo.return %3, %2 : tensor<f32>, tensor<10xf32>
    }
    return %0#1 : tensor<10xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<10xf32>
// CHECK-NEXT:    %1 = stablehlo.add %arg0, %0 : tensor<10xf32>
// CHECK-NEXT:    %2 = stablehlo.add %cst_0, %cst : tensor<f32>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<10xf32>
// CHECK-NEXT:    %4 = stablehlo.add %1, %3 : tensor<10xf32>
// CHECK-NEXT:    %5 = stablehlo.add %2, %cst : tensor<f32>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<10xf32>
// CHECK-NEXT:    %7 = stablehlo.add %4, %6 : tensor<10xf32>
// CHECK-NEXT:    return %7 : tensor<10xf32>
// CHECK-NEXT:  }
