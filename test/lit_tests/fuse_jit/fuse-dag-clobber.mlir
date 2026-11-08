// RUN: enzymexlamlir-opt --fuse-jit="dag=true" %s | FileCheck %s

// An in-place write to an earlier group result clobbers that result's fused
// slot. Fusion is only legal when no reader of the old value runs after the
// write.

module {
  llvm.func @k1(%a: !llvm.ptr) {
    llvm.return
  }
  llvm.func @k2(%a: !llvm.ptr, %b: !llvm.ptr) {
    llvm.return
  }

  // CHECK-LABEL: func.func @escaping_reader
  // CHECK: enzymexla.jit_call @k1
  // CHECK: stablehlo.transpose
  // CHECK: enzymexla.jit_call @k2
  // CHECK-NOT: @fused
  func.func @escaping_reader(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> (tensor<3x2xf32>, tensor<2x3xf32>) {
    %0 = enzymexla.jit_call @k1 (%x) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %t = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %1 = enzymexla.jit_call @k2 (%y, %0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %t, %1 : tensor<3x2xf32>, tensor<2x3xf32>
  }

  // CHECK-LABEL: func.func @later_member_reader
  // CHECK: enzymexla.jit_call @k1
  // CHECK: enzymexla.jit_call @k1
  // CHECK: enzymexla.jit_call @k2
  // CHECK-NOT: @fused
  func.func @later_member_reader(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = enzymexla.jit_call @k1 (%x) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = enzymexla.jit_call @k1 (%0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = enzymexla.jit_call @k2 (%y, %0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }

  // CHECK-LABEL: func.func @writer_reads_twice
  // CHECK: enzymexla.jit_call @k1
  // CHECK: enzymexla.jit_call @k2
  // CHECK-NOT: @fused
  func.func @writer_reads_twice(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = enzymexla.jit_call @k1 (%x) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = enzymexla.jit_call @k2 (%0, %0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }

  // The only other reader of %0 runs before the write, so fusion is legal.
  // CHECK-LABEL: func.func @earlier_member_reader
  // CHECK-NEXT: %[[RES:.*]]:2 = enzymexla.jit_call @fused__k1_k2_k1
  // CHECK-NEXT: return
  func.func @earlier_member_reader(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = enzymexla.jit_call @k1 (%x) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = enzymexla.jit_call @k2 (%y, %0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %2 = enzymexla.jit_call @k1 (%0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>
  }
}
