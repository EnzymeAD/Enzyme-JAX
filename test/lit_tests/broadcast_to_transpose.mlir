// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  func.func @main(%a : tensor<20x85x180xf64>) -> tensor<85x180x20xf64> {
    %456 = "stablehlo.broadcast_in_dim"(%a) <{broadcast_dimensions = array<i64: 2, 0, 1>}> : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
    return %456 : tensor<85x180x20xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>) -> tensor<2xf32> {
// CHECK-NEXT:    %[[i0:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.add %arg0, %[[i0]] : tensor<f32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.broadcast_in_dim %[[i1]], dims = [] : (tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<2xf32>
// CHECK-NEXT:  }
