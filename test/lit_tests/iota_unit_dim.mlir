// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=iota_simplify<16>(1024)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @unit_dim() -> tensor<1x2048xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<1x2048xi64>
    return %0 : tensor<1x2048xi64>
  }

  func.func @unit_dim_float() -> tensor<4096x1xf32> {
    %0 = stablehlo.iota dim = 1 : tensor<4096x1xf32>
    return %0 : tensor<4096x1xf32>
  }

  func.func @large() -> tensor<2048xi64> {
    %0 = stablehlo.iota dim = 0 : tensor<2048xi64>
    return %0 : tensor<2048xi64>
  }
}

// CHECK:  func.func @unit_dim() -> tensor<1x2048xi64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<1x2048xi64>
// CHECK-NEXT:    return %c : tensor<1x2048xi64>
// CHECK-NEXT:  }
// CHECK:  func.func @unit_dim_float() -> tensor<4096x1xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4096x1xf32>
// CHECK-NEXT:    return %cst : tensor<4096x1xf32>
// CHECK-NEXT:  }
// CHECK:  func.func @large() -> tensor<2048xi64> {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<2048xi64>
// CHECK-NEXT:    return %0 : tensor<2048xi64>
// CHECK-NEXT:  }
