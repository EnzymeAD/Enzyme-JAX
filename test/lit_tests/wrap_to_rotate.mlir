// RUN: enzymexlamlir-opt %s --optimize-communication="wrap_to_rotate=1" | FileCheck %s

module {
  func.func @test_wrap_to_rotate_symmetric(%arg0: tensor<10xf32>) -> tensor<14xf32> {
    %wrap = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
    return %wrap : tensor<14xf32>
  }

  func.func @test_wrap_to_rotate_asymmetric(%arg0: tensor<10xf32>) -> tensor<15xf32> {
    %wrap = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<15xf32>
    return %wrap : tensor<15xf32>
  }
}

// CHECK:  func.func @test_wrap_to_rotate_symmetric(%[[ARG0:.+]]: tensor<10xf32>) -> tensor<14xf32> {
// CHECK-NEXT:    %[[EXTEND:.+]] = "enzymexla.extend"(%[[ARG0]]) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
// CHECK-NEXT:    %[[ROTATE:.+]] = "enzymexla.rotate"(%[[EXTEND]]) <{amount = 2 : si32, dimension = 0 : si32}> : (tensor<14xf32>) -> tensor<14xf32>
// CHECK-NEXT:    return %[[ROTATE]] : tensor<14xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @test_wrap_to_rotate_asymmetric(%[[ARG1:.+]]: tensor<10xf32>) -> tensor<15xf32> {
// CHECK-NEXT:    %[[EXTEND2:.+]] = "enzymexla.extend"(%[[ARG1]]) <{dimension = 0 : i64, lhs = 3 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<15xf32>
// CHECK-NEXT:    %[[ROTATE2:.+]] = "enzymexla.rotate"(%[[EXTEND2]]) <{amount = 3 : si32, dimension = 0 : si32}> : (tensor<15xf32>) -> tensor<15xf32>
// CHECK-NEXT:    return %[[ROTATE2]] : tensor<15xf32>
// CHECK-NEXT:  }
