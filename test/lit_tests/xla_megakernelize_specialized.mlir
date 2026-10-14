// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s

// Only the first call specializes a scalar. Keep it after the buffer inputs,
// and feed the first call's updated buffer into the second call.
module {
  func.func @first_specialized(%a: memref<i32>, %n: i32) {
    enzymexla.xla_wrapper @first (%a, %n) <{num_specialized = 1 : i64}> : (memref<i32>, i32) -> ()
    enzymexla.xla_wrapper @second (%a) : (memref<i32>) -> ()
    return
  }
  func.func private @first(%a: tensor<i32>, %n: tensor<i32>) -> tensor<i32> {
    %sum = stablehlo.add %a, %n : tensor<i32>
    return %sum : tensor<i32>
  }
  func.func private @second(%a: tensor<i32>) -> tensor<i32> {
    %negated = stablehlo.negate %a : tensor<i32>
    return %negated : tensor<i32>
  }
}
// CHECK-LABEL: func.func @first_specialized(
// CHECK-SAME: %[[HOST_A:.*]]: memref<i32>, %[[HOST_N:.*]]: i32)
// CHECK-NEXT: enzymexla.xla_wrapper @first (%[[HOST_A]], %[[HOST_N]]) <num_specialized = 1> : (memref<i32>, i32) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[A:.*]]: tensor<i32>, %[[N:.*]]: tensor<i32>) -> tensor<i32>
// CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[A]], %[[N]] : tensor<i32>
// CHECK-NEXT: %[[NEGATED:.*]] = stablehlo.negate %[[SUM]] : tensor<i32>
// CHECK-NEXT: return %[[NEGATED]] : tensor<i32>
// CHECK-NOT: func.func private @second

// -----

// Only the second call specializes a scalar. Its scalar tensor is a new
// argument of the first function, without adding a corresponding result.
module {
  func.func @second_specialized(%a: memref<i64>, %n: i64) {
    enzymexla.xla_wrapper @first (%a) : (memref<i64>) -> ()
    enzymexla.xla_wrapper @second (%a, %n) <{num_specialized = 1 : i64}> : (memref<i64>, i64) -> ()
    return
  }
  func.func private @first(%a: tensor<i64>) -> tensor<i64> {
    %negated = stablehlo.negate %a : tensor<i64>
    return %negated : tensor<i64>
  }
  func.func private @second(%a: tensor<i64>, %n: tensor<i64>) -> tensor<i64> {
    %sum = stablehlo.add %a, %n : tensor<i64>
    return %sum : tensor<i64>
  }
}
// CHECK-LABEL: func.func @second_specialized(
// CHECK-SAME: %[[HOST_A:.*]]: memref<i64>, %[[HOST_N:.*]]: i64)
// CHECK-NEXT: enzymexla.xla_wrapper @first (%[[HOST_A]], %[[HOST_N]]) <num_specialized = 1> : (memref<i64>, i64) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[A:.*]]: tensor<i64>, %[[N:.*]]: tensor<i64>) -> tensor<i64>
// CHECK-NEXT: %[[NEGATED:.*]] = stablehlo.negate %[[A]] : tensor<i64>
// CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[NEGATED]], %[[N]] : tensor<i64>
// CHECK-NEXT: return %[[SUM]] : tensor<i64>
// CHECK-NOT: func.func private @second

// -----

// first(A, n), second(B, A, m, k) becomes first(A, B, n, m, k).
// Here n and m receive the same host SSA value. Canonicalization merges those
// arguments, so the final call is first_without_duplicates(A, B, n, k).
// Only A and B have results, in the union's buffer order.
module {
  func.func @different_specialized_counts(%a: memref<i32>, %b: memref<i32>,
                                          %n: i32, %k: i64) {
    enzymexla.xla_wrapper @first (%a, %n) <{num_specialized = 1 : i64}> : (memref<i32>, i32) -> ()
    enzymexla.xla_wrapper @second (%b, %a, %n, %k) <{num_specialized = 2 : i64}> : (memref<i32>, memref<i32>, i32, i64) -> ()
    return
  }
  func.func private @first(%a: tensor<i32>, %n: tensor<i32>) -> tensor<i32> {
    %sum = stablehlo.add %a, %n : tensor<i32>
    return %sum : tensor<i32>
  }
  func.func private @second(%b: tensor<i32>, %a: tensor<i32>,
                            %m: tensor<i32>, %k: tensor<i64>)
      -> (tensor<i32>, tensor<i32>) {
    %scaled = stablehlo.multiply %b, %m : tensor<i32>
    %k32 = stablehlo.convert %k : (tensor<i64>) -> tensor<i32>
    %sum = stablehlo.add %a, %k32 : tensor<i32>
    return %scaled, %sum : tensor<i32>, tensor<i32>
  }
}
// CHECK-LABEL: func.func @different_specialized_counts(
// CHECK-SAME: %[[HOST_A:.*]]: memref<i32>, %[[HOST_B:.*]]: memref<i32>, %[[HOST_N:.*]]: i32, %[[HOST_K:.*]]: i64)
// CHECK-NEXT: enzymexla.xla_wrapper @first_without_duplicates (%[[HOST_A]], %[[HOST_B]], %[[HOST_N]], %[[HOST_K]]) <num_specialized = 2> : (memref<i32>, memref<i32>, i32, i64) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first_without_duplicates(
// CHECK-SAME: %[[A:.*]]: tensor<i32>, %[[B:.*]]: tensor<i32>, %[[N:.*]]: tensor<i32>, %[[K:.*]]: tensor<i64>) -> (tensor<i32>, tensor<i32>)
// CHECK-NEXT: %[[FIRST_SUM:.*]] = stablehlo.add %[[A]], %[[N]] : tensor<i32>
// CHECK-NEXT: %[[SCALED:.*]] = stablehlo.multiply %[[B]], %[[N]] : tensor<i32>
// CHECK-NEXT: %[[K32:.*]] = stablehlo.convert %[[K]] : (tensor<i64>) -> tensor<i32>
// CHECK-NEXT: %[[SECOND_SUM:.*]] = stablehlo.add %[[FIRST_SUM]], %[[K32]] : tensor<i32>
// CHECK-NEXT: return %[[SECOND_SUM]], %[[SCALED]] : tensor<i32>, tensor<i32>
// CHECK-NOT: func.func private @second

// -----

// A scalar computed between the calls cannot be an input at the first call.
// Leave the calls separate even though the intervening arithmetic is pure.
module {
  func.func @late_specialized_input(%a: memref<i32>, %n: i32) {
    enzymexla.xla_wrapper @first (%a) : (memref<i32>) -> ()
    %late = arith.addi %n, %n : i32
    enzymexla.xla_wrapper @second (%a, %late) <{num_specialized = 1 : i64}> : (memref<i32>, i32) -> ()
    return
  }
  func.func private @first(%a: tensor<i32>) -> tensor<i32> {
    %negated = stablehlo.negate %a : tensor<i32>
    return %negated : tensor<i32>
  }
  func.func private @second(%a: tensor<i32>, %n: tensor<i32>) -> tensor<i32> {
    %sum = stablehlo.add %a, %n : tensor<i32>
    return %sum : tensor<i32>
  }
}
// CHECK-LABEL: func.func @late_specialized_input(
// CHECK-SAME: %[[HOST_A:.*]]: memref<i32>, %[[HOST_N:.*]]: i32)
// CHECK-NEXT: enzymexla.xla_wrapper @first (%[[HOST_A]]) : (memref<i32>) -> ()
// CHECK-NEXT: %[[LATE:.*]] = arith.addi %[[HOST_N]], %[[HOST_N]] : i32
// CHECK-NEXT: enzymexla.xla_wrapper @second (%[[HOST_A]], %[[LATE]]) <num_specialized = 1> : (memref<i32>, i32) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[A:.*]]: tensor<i32>) -> tensor<i32>
// CHECK-NEXT: %[[NEGATED:.*]] = stablehlo.negate %[[A]] : tensor<i32>
// CHECK-NEXT: return %[[NEGATED]] : tensor<i32>
// CHECK-LABEL: func.func private @second(
// CHECK-SAME: %[[A:.*]]: tensor<i32>, %[[N:.*]]: tensor<i32>) -> tensor<i32>
// CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[A]], %[[N]] : tensor<i32>
// CHECK-NEXT: return %[[SUM]] : tensor<i32>
