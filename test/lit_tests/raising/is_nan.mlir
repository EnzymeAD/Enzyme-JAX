// RUN: enzymexlamlir-opt %s --arith-raise | FileCheck %s
// RUN: enzymexlamlir-opt %s --arith-raise | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @isnan(%arg0: tensor<4xf32>) -> tensor<4xi1> {
    %0 = "math.isnan"(%arg0) : (tensor<4xf32>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}

// CHECK:  func.func @isnan(%arg0: tensor<4xf32>) -> tensor<4xi1> {
// CHECK-NEXT:    %0 = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT:    return %0 : tensor<4xi1>
// CHECK-NEXT:  }

func.func @isnan_numeric(%arg0: tensor<10xf32>) -> tensor<10xi1> {
  %0 = "math.isnan"(%arg0) : (tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

func.func @main() {
  // quiet NaN, signaling NaN, negative quiet NaN, negative signaling NaN,
  // +inf, -inf, finite positive, finite negative, +0.0, -0.0
  %input = stablehlo.constant dense<[0x7FC00000, 0x7F800001, 0xFFC00000, 0xFF800001,
                                      0x7F800000, 0xFF800000,
                                      3.140000e+00, -3.140000e+00,
                                      0.000000e+00, -0.000000e+00]> : tensor<10xf32>
  %res = func.call @isnan_numeric(%input) : (tensor<10xf32>) -> tensor<10xi1>
  %exp = stablehlo.constant dense<[true, true, true, true,
                                    false, false,
                                    false, false,
                                    false, false]> : tensor<10xi1>
  "check.expect_eq"(%res, %exp) : (tensor<10xi1>, tensor<10xi1>) -> ()
  return
}
