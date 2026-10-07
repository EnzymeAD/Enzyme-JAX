// RUN: enzymexlamlir-opt %s --allow-unregistered-dialect --canonicalize | FileCheck %s

// The unknown region can contain symbol uses that cannot be enumerated.
// Keep the original private function. Rewrite the known wrapper to a clone.
// Before: wrapper @negate(data, spare)
// After:  wrapper @negate_without_unused(data)
module {
  func.func @call_with_unknown_uses(%data: memref<4xf32>, %spare: memref<4xf32>) {
    enzymexla.xla_wrapper @negate (%data, %spare) :
        (memref<4xf32>, memref<4xf32>) -> ()
    return
  }
  func.func private @negate(%data: tensor<4xf32>, %spare: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %negative = stablehlo.negate %data : tensor<4xf32>
    return %negative, %spare : tensor<4xf32>, tensor<4xf32>
  }
  "test.unknown_scope"() ({}) : () -> ()
}

// CHECK-LABEL: func.func @call_with_unknown_uses(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<4xf32>, %[[SPARE:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @negate_without_unused (%[[DATA]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>, %[[SPARE:[^:]+]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]], %[[SPARE]] : tensor<4xf32>, tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @negate_without_unused(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[NEGATIVE:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[NEGATIVE]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: "test.unknown_scope"() ({
// CHECK-NEXT: }) : () -> ()
// CHECK-NEXT: }
