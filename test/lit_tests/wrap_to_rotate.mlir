// RUN: enzymexlamlir-opt %s --optimize-communication="wrap_to_rotate=1" | FileCheck %s

module {
  func.func @test_wrap_to_rotate(%arg0: tensor<10xf32>) -> tensor<14xf32> {
    %wrap = "enzymexla.wrap"(%arg0) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
    return %wrap : tensor<14xf32>
  }
}

// CHECK:  func.func @test_wrap_to_rotate(%[[ARG0:.+]]: tensor<10xf32>) -> tensor<14xf32> {
// CHECK-NEXT:    %[[EXTEND:.+]] = "enzymexla.extend"(%[[ARG0]]) <{dimension = 0 : i64, lhs = 2 : i64, rhs = 2 : i64}> : (tensor<10xf32>) -> tensor<14xf32>
// CHECK-NEXT:    %[[ROTATE:.+]] = "enzymexla.rotate"(%[[EXTEND]]) <{amount = 2 : si32, dimension = 0 : si32}> : (tensor<14xf32>) -> tensor<14xf32>
// CHECK-NEXT:    return %[[ROTATE]] : tensor<14xf32>
// CHECK-NEXT:  }
