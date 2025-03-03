// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @throw() {
llvm.func @throw() {
  llvm.unreachable
}

// CHECK-LABEL: func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: enzymexla.jit_call @throw
  // CHECK-SAME: has_side_effect = true
  // CHECK-SAME: backend_config = {bar = 42 : i32}
  enzymexla.jit_call @throw () {
      has_side_effect = true,
      backend_config = {bar = 42 : i32}
    } : () -> ()
  return %arg0 : tensor<4xf32>
}
