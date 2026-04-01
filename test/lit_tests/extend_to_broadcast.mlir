// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td=patterns=extend_to_broadcast --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x1xf32>) -> tensor<3x1xf32> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    return %0 : tensor<3x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x1xf32>) -> tensor<3x1xf32> {
// CHECK-NEXT:    %[[BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<3x1xf32>
// CHECK-NEXT:    return %[[BCAST]] : tensor<3x1xf32>
// CHECK-NEXT:  }

module {
  func.func @test_extend_dim1(%arg0: tensor<2x1xf32>) -> tensor<2x5xf32> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 1 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<2x1xf32>) -> tensor<2x5xf32>
    return %0 : tensor<2x5xf32>
  }
}

// CHECK:  func.func @test_extend_dim1(%arg0: tensor<2x1xf32>) -> tensor<2x5xf32> {
// CHECK-NEXT:    %[[BCAST:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x5xf32>
// CHECK-NEXT:    return %[[BCAST]] : tensor<2x5xf32>
// CHECK-NEXT:  }

module {
  func.func @test_no_opt_non_singleton(%arg0: tensor<2x3xf32>) -> tensor<2x7xf32> {
    %0 = "enzymexla.extend"(%arg0) <{dimension = 1 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<2x3xf32>) -> tensor<2x7xf32>
    return %0 : tensor<2x7xf32>
  }
}

// CHECK:  func.func @test_no_opt_non_singleton(%arg0: tensor<2x3xf32>) -> tensor<2x7xf32> {
// CHECK-NEXT:    %[[EXT:.+]] = "enzymexla.extend"(%arg0) <{dimension = 1 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<2x3xf32>) -> tensor<2x7xf32>
// CHECK-NEXT:    return %[[EXT]] : tensor<2x7xf32>
// CHECK-NEXT:  }
