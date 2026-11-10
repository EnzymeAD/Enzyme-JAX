// RUN: enzymexlamlir-opt %s --map-symbol="symbols=shared_runtime=0x1" --pass-pipeline="builtin.module(lower-jit{dump_final_module=true})" 2>&1 | FileCheck %s

// Two jitted calls, each with a constructor that stores the address of the
// same runtime declaration into the slot its call reads. Each call's module
// must carry its own constructor only: the declaration copied along with one
// constructor must not pull the other constructor into the call.

module {
  llvm.func @shared_runtime(!llvm.ptr) -> i32
  llvm.mlir.global internal @slot_a() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
  llvm.mlir.global internal @slot_b() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.ptr
  llvm.func internal @ctor_a() {
    %0 = llvm.mlir.addressof @shared_runtime : !llvm.ptr
    %1 = llvm.mlir.addressof @slot_a : !llvm.ptr
    llvm.store %0, %1 : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func internal @ctor_b() {
    %0 = llvm.mlir.addressof @shared_runtime : !llvm.ptr
    %1 = llvm.mlir.addressof @slot_b : !llvm.ptr
    llvm.store %0, %1 : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@ctor_a], priorities = [65535 : i32], data = [#llvm.zero]
  llvm.mlir.global_ctors ctors = [@ctor_b], priorities = [65535 : i32], data = [#llvm.zero]
  llvm.func @wrapper_a(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.addressof @slot_a : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    llvm.store %1, %arg0 : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func @wrapper_b(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.addressof @slot_b : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    llvm.store %1, %arg0 : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<i64>) {
    %0 = enzymexla.jit_call @wrapper_a (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<i64>) -> tensor<i64>
    %1 = enzymexla.jit_call @wrapper_b (%arg1) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<i64>) -> tensor<i64>
    return %0, %1 : tensor<i64>, tensor<i64>
  }
}

// CHECK:      final_llvm_module before jit
// CHECK:      define internal void @ctor_a()
// CHECK-NOT:  @ctor_b
// CHECK:      final_llvm_module before jit
// CHECK-NOT:  @ctor_a
// CHECK:      define internal void @ctor_b()
