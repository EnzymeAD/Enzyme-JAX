// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<i64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<i64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %0 = enzymexla.mpi.allreduce(%arg0, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT>, op = #enzymexla.op<MPI_LAND>} : (tensor<i64>, tensor<i64>, tensor<i32>) -> tensor<i64>
    return %0 : tensor<i64>
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.mlir.global external constant @MPI_LAND() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.mlir.global external constant @MPI_INT() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Allreduce_MPI_LAND_MPI_INT(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.mlir.addressof @MPI_LAND : !llvm.ptr
// CPU-NEXT:      %1 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CPU-NEXT:      %2 = llvm.mlir.addressof @MPI_INT : !llvm.ptr
// CPU-NEXT:      %3 = llvm.load %arg2 : !llvm.ptr -> i32
// CPU-NEXT:      %4 = llvm.call @MPI_Allreduce(%arg0, %arg1, %3, %2, %0, %1) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main(%arg0: tensor<i64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<i64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %c = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:      %0 = enzymexla.jit_call @enzymexla_wrapper_MPI_Allreduce_MPI_LAND_MPI_INT (%arg0, %c, %c_0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]} : (tensor<i64>, tensor<i64>, tensor<i32>) -> tensor<i64>
// CPU-NEXT:      return %0 : tensor<i64>
// CPU-NEXT:    }
// CPU-NEXT:  }
