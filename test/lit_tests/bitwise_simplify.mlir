// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @and_i64_with_one(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.and %arg0, %c1 : tensor<i64>
    return %0 : tensor<i64>
  }

  func.func @and_i1_with_true(%arg0: tensor<i1>) -> tensor<i1> {
    %true = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.and %true, %arg0 : tensor<i1>
    return %0 : tensor<i1>
  }

  func.func @or_i64_with_one(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.or %arg0, %c1 : tensor<i64>
    return %0 : tensor<i64>
  }

  func.func @or_i1_with_true(%arg0: tensor<i1>) -> tensor<i1> {
    %true = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.or %true, %arg0 : tensor<i1>
    return %0 : tensor<i1>
  }

  func.func @xor_i64_with_one(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.xor %arg0, %c1 : tensor<i64>
    return %0 : tensor<i64>
  }

  func.func @xor_i1_with_true(%arg0: tensor<i1>) -> tensor<i1> {
    %true = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.xor %true, %arg0 : tensor<i1>
    return %0 : tensor<i1>
  }
}

// CHECK:  func.func @and_i64_with_one(%[[ARG0:.+]]: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.and %[[ARG0]], %[[C]] : tensor<i64>
// CHECK-NEXT:    return %[[RES]] : tensor<i64>
// CHECK-NEXT:  }

// CHECK:  func.func @and_i1_with_true(%[[ARG0:.+]]: tensor<i1>) -> tensor<i1> {
// CHECK-NEXT:    return %[[ARG0]] : tensor<i1>
// CHECK-NEXT:  }

// CHECK:  func.func @or_i64_with_one(%[[ARG0:.+]]: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.or %[[ARG0]], %[[C]] : tensor<i64>
// CHECK-NEXT:    return %[[RES]] : tensor<i64>
// CHECK-NEXT:  }

// CHECK:  func.func @or_i1_with_true(%[[ARG0:.+]]: tensor<i1>) -> tensor<i1> {
// CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    return %[[C]] : tensor<i1>
// CHECK-NEXT:  }

// CHECK:  func.func @xor_i64_with_one(%[[ARG0:.+]]: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    %[[C:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.xor %[[ARG0]], %[[C]] : tensor<i64>
// CHECK-NEXT:    return %[[RES]] : tensor<i64>
// CHECK-NEXT:  }

// CHECK:  func.func @xor_i1_with_true(%[[ARG0:.+]]: tensor<i1>) -> tensor<i1> {
// CHECK-NEXT:    %[[RES:.+]] = stablehlo.not %[[ARG0]] : tensor<i1>
// CHECK-NEXT:    return %[[RES]] : tensor<i1>
// CHECK-NEXT:  }
