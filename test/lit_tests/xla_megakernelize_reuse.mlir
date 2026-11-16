// RUN: enzymexlamlir-opt %s --allow-unregistered-dialect --split-input-file --xla-megakernelize | FileCheck %s

// Public functions can have callers outside this module. Keep their bodies.
module {
  func.func @keep_public_function(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @first (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @second (%buffer) : (memref<4xf32>) -> ()
    return
  }
  func.func @first(%a: tensor<4xf32>) -> tensor<4xf32> {
    %negated = stablehlo.negate %a : tensor<4xf32>
    return %negated : tensor<4xf32>
  }
  func.func private @second(%a: tensor<4xf32>) -> tensor<4xf32> {
    %doubled = stablehlo.add %a, %a : tensor<4xf32>
    return %doubled : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keep_public_function(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[BUFFER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func @first(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[N:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: return %[[N]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[N:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: %[[D:[^ ]+]] = stablehlo.add %[[N]], %[[N]] : tensor<4xf32>
// CHECK-NEXT: return %[[D]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// A private function with another caller must keep its original behavior.
module {
  func.func @keep_other_caller(%buffer: memref<4xf32>, %value: tensor<4xf32>)
      -> tensor<4xf32> {
    enzymexla.xla_wrapper @first (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @second (%buffer) : (memref<4xf32>) -> ()
    %result = func.call @first(%value) : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
  func.func private @first(%a: tensor<4xf32>) -> tensor<4xf32> {
    %negated = stablehlo.negate %a : tensor<4xf32>
    return %negated : tensor<4xf32>
  }
  func.func private @second(%a: tensor<4xf32>) -> tensor<4xf32> {
    %doubled = stablehlo.add %a, %a : tensor<4xf32>
    return %doubled : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @keep_other_caller(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>, %[[VALUE:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[BUFFER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: %[[RESULT:[^ ]+]] = call @first(%[[VALUE]]) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[N:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: return %[[N]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[N:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: %[[D:[^ ]+]] = stablehlo.add %[[N]], %[[N]] : tensor<4xf32>
// CHECK-NEXT: return %[[D]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

// Unknown regions can hide symbol uses. Keep the original private function.
module {
  func.func @keep_unknown_users(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @first (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @second (%buffer) : (memref<4xf32>) -> ()
    return
  }
  func.func private @first(%a: tensor<4xf32>) -> tensor<4xf32> {
    %negated = stablehlo.negate %a : tensor<4xf32>
    return %negated : tensor<4xf32>
  }
  func.func private @second(%a: tensor<4xf32>) -> tensor<4xf32> {
    %doubled = stablehlo.add %a, %a : tensor<4xf32>
    return %doubled : tensor<4xf32>
  }
  "test.unknown_scope"() ({}) : () -> ()
}

// CHECK-LABEL: func.func @keep_unknown_users(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @rxla$megakernel_0 (%[[BUFFER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[N:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: return %[[N]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-LABEL: func.func private @rxla$megakernel_0(
// CHECK-SAME: %[[A:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[N:[^ ]+]] = stablehlo.negate %[[A]] : tensor<4xf32>
// CHECK-NEXT: %[[D:[^ ]+]] = stablehlo.add %[[N]], %[[N]] : tensor<4xf32>
// CHECK-NEXT: return %[[D]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: }
