// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt)" %s | FileCheck %s

// Test 1: Reverse a non-splat constant along one dimension
module {
  func.func @main() -> tensor<4xf64> {
    %cst = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
    %0 = stablehlo.reverse %cst, dims = [0] : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}

// CHECK:  func.func @main() -> tensor<4xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]> : tensor<4xf64>
// CHECK-NEXT:    return %cst : tensor<4xf64>
// CHECK-NEXT:  }

// Test 2: Reverse a 2D constant along both dimensions
module {
  func.func @main() -> tensor<2x3xf64> {
    %cst = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %0 = stablehlo.reverse %cst, dims = [0, 1] : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}

// CHECK:  func.func @main() -> tensor<2x3xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<2x3xf64>
// CHECK-NEXT:    return %cst : tensor<2x3xf64>
// CHECK-NEXT:  }

// Test 3: Reverse a 2D constant along just the first dimension
module {
  func.func @main() -> tensor<2x3xf64> {
    %cst = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %0 = stablehlo.reverse %cst, dims = [0] : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}

// CHECK:  func.func @main() -> tensor<2x3xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[4.000000e+00, 5.000000e+00, 6.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<2x3xf64>
// CHECK-NEXT:    return %cst : tensor<2x3xf64>
// CHECK-NEXT:  }

// Test 4: Reverse a 2D constant along just the second dimension
module {
  func.func @main() -> tensor<2x3xf64> {
    %cst = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %0 = stablehlo.reverse %cst, dims = [1] : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}

// CHECK:  func.func @main() -> tensor<2x3xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[3.000000e+00, 2.000000e+00, 1.000000e+00], [6.000000e+00, 5.000000e+00, 4.000000e+00]]> : tensor<2x3xf64>
// CHECK-NEXT:    return %cst : tensor<2x3xf64>
// CHECK-NEXT:  }
