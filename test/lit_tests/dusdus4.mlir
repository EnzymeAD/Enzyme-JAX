// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%X: tensor<60x28x47xf32>, %u: tensor<1x18x37xf32>) -> (tensor<60x28x47xf32>, tensor<2x18x37xf32>) {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c50 = stablehlo.constant dense<50> : tensor<i32>
    %a = stablehlo.dynamic_update_slice %X, %u, %c1, %c0, %c0 : (tensor<60x28x47xf32>, tensor<1x18x37xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<60x28x47xf32>
    %b = stablehlo.dynamic_update_slice %a, %u, %c50, %c0, %c0 : (tensor<60x28x47xf32>, tensor<1x18x37xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<60x28x47xf32>
    %s = stablehlo.slice %a [0:2, 0:18, 0:37] : (tensor<60x28x47xf32>) -> tensor<2x18x37xf32>
    return %b, %s : tensor<60x28x47xf32>, tensor<2x18x37xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<60x28x47xf32>, %arg1: tensor<1x18x37xf32>) -> (tensor<60x28x47xf32>, tensor<2x18x37xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<50> : tensor<i32>
// CHECK-NEXT:    %[[DUS0:.+]] = stablehlo.dynamic_update_slice %arg0, %arg1, %c_0, %c, %c : (tensor<60x28x47xf32>, tensor<1x18x37xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<60x28x47xf32>
// CHECK-NEXT:    %[[DUS1:.+]] = stablehlo.dynamic_update_slice %[[DUS0]], %arg1, %c_1, %c, %c : (tensor<60x28x47xf32>, tensor<1x18x37xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<60x28x47xf32>
// CHECK-NEXT:    %[[SLICE:.+]] = stablehlo.slice %[[DUS1]] [0:2, 0:18, 0:37] : (tensor<60x28x47xf32>) -> tensor<2x18x37xf32>
// CHECK-NEXT:    return %[[DUS1]], %[[SLICE]] : tensor<60x28x47xf32>, tensor<2x18x37xf32>
// CHECK-NEXT:  }
