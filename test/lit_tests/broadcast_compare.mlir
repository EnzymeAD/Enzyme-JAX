// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcast_compare" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

  func.func @mid1() -> tensor<6128x12272xi1> {
    %c_254 = stablehlo.constant dense<12271> : tensor<12272xi64>
    %556 = stablehlo.iota dim = 0 : tensor<12272xi64>
    %613 = stablehlo.compare  EQ, %556, %c_254 : (tensor<12272xi64>, tensor<12272xi64>) -> tensor<12272xi1>
    %768 = stablehlo.broadcast_in_dim %613, dims = [1] : (tensor<12272xi1>) -> tensor<6128x12272xi1>
    return %768 : tensor<6128x12272xi1>
  }

// CHECK:  func.func @mid1() -> tensor<6128x12272xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<12271> : tensor<12272xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<12272xi64>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<12272xi64>) -> tensor<6128x12272xi64>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<12272xi64>) -> tensor<6128x12272xi64>
// CHECK-NEXT:    %3 = stablehlo.compare  EQ, %1, %2 : (tensor<6128x12272xi64>, tensor<6128x12272xi64>) -> tensor<6128x12272xi1>
// CHECK-NEXT:    return %3 : tensor<6128x12272xi1>
// CHECK-NEXT:  }