// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s

// The first wrapper has no inputs. Keep its effect before the buffer update.
module {
  func.func @first_empty(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @first () : () -> ()
    enzymexla.xla_wrapper @second (%buffer) : (memref<4xf32>) -> ()
    return
  }
  func.func private @first() {
    stablehlo.custom_call @first_effect() {has_side_effect = true} : () -> ()
    return
  }
  func.func private @second(%buffer: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.custom_call @second_effect(%buffer) {has_side_effect = true} : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @first_empty(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @first (%[[BUFFER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[BUFFER:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: stablehlo.custom_call @first_effect() {has_side_effect = true} : () -> ()
// CHECK-NEXT: %[[RESULT:[^ ]+]] = stablehlo.custom_call @second_effect(%[[BUFFER]]) {has_side_effect = true}
// CHECK-NEXT: return %[[RESULT]] : tensor<4xf32>
// CHECK-NOT: func.func private @second

// -----

// The second wrapper has no inputs. Keep its effect after the buffer update.
module {
  func.func @second_empty(%buffer: memref<4xf32>) {
    enzymexla.xla_wrapper @first (%buffer) : (memref<4xf32>) -> ()
    enzymexla.xla_wrapper @second () : () -> ()
    return
  }
  func.func private @first(%buffer: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.custom_call @first_effect(%buffer) {has_side_effect = true} : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
  func.func private @second() {
    stablehlo.custom_call @second_effect() {has_side_effect = true} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @second_empty(
// CHECK-SAME: %[[BUFFER:[^:]+]]: memref<4xf32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @first (%[[BUFFER]]) : (memref<4xf32>) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first(
// CHECK-SAME: %[[BUFFER:[^:]+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT: %[[RESULT:[^ ]+]] = stablehlo.custom_call @first_effect(%[[BUFFER]]) {has_side_effect = true}
// CHECK-NEXT: stablehlo.custom_call @second_effect() {has_side_effect = true} : () -> ()
// CHECK-NEXT: return %[[RESULT]] : tensor<4xf32>
// CHECK-NOT: func.func private @second

// -----

// Both wrappers have no inputs. Fuse them and keep their effects in order.
module {
  func.func @both_empty() {
    enzymexla.xla_wrapper @first () : () -> ()
    enzymexla.xla_wrapper @second () : () -> ()
    return
  }
  func.func private @first() {
    stablehlo.custom_call @first_effect() {has_side_effect = true} : () -> ()
    return
  }
  func.func private @second() {
    stablehlo.custom_call @second_effect() {has_side_effect = true} : () -> ()
    return
  }
}

// CHECK-LABEL: func.func @both_empty() {
// CHECK-NEXT: enzymexla.xla_wrapper @first () : () -> ()
// CHECK-NEXT: return
// CHECK-LABEL: func.func private @first() {
// CHECK-NEXT: stablehlo.custom_call @first_effect() {has_side_effect = true} : () -> ()
// CHECK-NEXT: stablehlo.custom_call @second_effect() {has_side_effect = true} : () -> ()
// CHECK-NEXT: return
// CHECK-NOT: func.func private @second
