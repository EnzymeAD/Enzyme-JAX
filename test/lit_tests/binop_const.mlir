// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  func.func @add(%arg0: tensor<3xi64>) -> tensor<3xi64> {
    %c = stablehlo.constant dense<1> : tensor<3xi64>
    %0 = stablehlo.add %arg0, %c : tensor<3xi64>
    %1 = stablehlo.add %0, %c : tensor<3xi64>
    %2 = stablehlo.add %1, %c : tensor<3xi64>
    return %2 : tensor<3xi64>
  }
  func.func @mul(%arg0: tensor<3xi64>) -> tensor<3xi64> {
    %c = stablehlo.constant dense<2> : tensor<3xi64>
    %0 = stablehlo.multiply %arg0, %c : tensor<3xi64>
    %1 = stablehlo.multiply %0, %c : tensor<3xi64>
    %2 = stablehlo.multiply %1, %c : tensor<3xi64>
    return %2 : tensor<3xi64>
  }
}

// CHECK:  func.func @add(%arg0: tensor<3xi64>) -> tensor<3xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<3> : tensor<3xi64>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %c : tensor<3xi64>
// CHECK-NEXT:    return %0 : tensor<3xi64>
// CHECK-NEXT:  }

// CHECK:  func.func @mul(%arg0: tensor<3xi64>) -> tensor<3xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<3xi64>
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %c : tensor<3xi64>
// CHECK-NEXT:    return %0 : tensor<3xi64>
// CHECK-NEXT:  }
