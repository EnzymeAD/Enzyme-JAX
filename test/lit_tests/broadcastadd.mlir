// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<f32>) -> tensor<2xf32> {
    %bc = stablehlo.broadcast_in_dim %a, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %cst0 = arith.constant dense<2.000000e+00> : tensor<2xf32>
    %add = stablehlo.add %bc, %cst0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %add : tensor<2xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>) -> tensor<2xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.add %arg0, %[[i0]] : tensor<f32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.broadcast_in_dim %[[i1]], dims = [] : (tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<2xf32>
// CHECK-NEXT:  }
