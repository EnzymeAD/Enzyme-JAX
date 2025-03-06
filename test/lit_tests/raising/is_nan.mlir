// RUN: enzymexlamlir-opt %s --arith-raise | FileCheck %s

module {
  func.func @isnan(%arg0: tensor<4xf32>) -> tensor<4xi1> {
    %0 = "math.isnan"(%arg0) : (tensor<4xf32>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}

// CHECK:  func.func @isnan(%arg0: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.is_finite %arg0 : (tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    %1 = stablehlo.not %0 : tensor<4xi1>
// CHECK-NEXT:    %2 = chlo.is_inf %arg0 : tensor<4xf32> -> tensor<4xi1>
// CHECK-NEXT:    %3 = stablehlo.not %2 : tensor<4xi1>
// CHECK-NEXT:    %4 = stablehlo.and %1, %3 : tensor<4xi1>
// CHECK-NEXT:    return %4 : tensor<4xi1>
// CHECK-NEXT:  }
