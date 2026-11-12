// RUN: enzymexlamlir-opt --fuse-jit="dag=true" %s | FileCheck %s

// A fused call must keep `enzymexla.require_jit_ordering` when any of the calls
// it replaces had it, so `lower-jit` still orders it.

module {
  llvm.func @first(%arg0: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    llvm.store %0, %arg0 : f32, !llvm.ptr
    llvm.return
  }
  llvm.func @second(%arg0: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    llvm.store %0, %arg0 : f32, !llvm.ptr
    llvm.return
  }

  // CHECK-LABEL: func.func @all_ordered
  // CHECK-NEXT: %[[R:.+]] = enzymexla.jit_call @fused__first_second{{[_0-9]*}}
  // CHECK-SAME: {enzymexla.require_jit_ordering}
  // CHECK-NEXT: return %[[R]]
  func.func @all_ordered(%x: tensor<f32>) -> tensor<f32> {
    %0 = enzymexla.jit_call @first (%x) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> {enzymexla.require_jit_ordering} : (tensor<f32>) -> tensor<f32>
    %1 = enzymexla.jit_call @second (%0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> {enzymexla.require_jit_ordering} : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @one_ordered
  // CHECK-NEXT: %[[R:.+]] = enzymexla.jit_call @fused__first_second{{[_0-9]*}}
  // CHECK-SAME: {enzymexla.require_jit_ordering}
  // CHECK-NEXT: return %[[R]]
  func.func @one_ordered(%x: tensor<f32>) -> tensor<f32> {
    %0 = enzymexla.jit_call @first (%x) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<f32>) -> tensor<f32>
    %1 = enzymexla.jit_call @second (%0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> {enzymexla.require_jit_ordering} : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @none_ordered
  // CHECK-NEXT: %[[R:.+]] = enzymexla.jit_call @fused__first_second{{[_0-9]*}}
  // CHECK-NOT: enzymexla.require_jit_ordering
  // CHECK-NEXT: return %[[R]]
  func.func @none_ordered(%x: tensor<f32>) -> tensor<f32> {
    %0 = enzymexla.jit_call @first (%x) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<f32>) -> tensor<f32>
    %1 = enzymexla.jit_call @second (%0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}
