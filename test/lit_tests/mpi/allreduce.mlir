// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cuda ncclCommPtr=1})" %s | FileCheck %s --check-prefix=CUDA

module {
  func.func @main(%arg0: tensor<i64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<i64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %0 = enzymexla.mpi.allreduce(%arg0, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT64_T>, op = #enzymexla.op<MPI_SUM>} : (tensor<i64>, tensor<i64>, tensor<i32>) -> tensor<i64>
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


// CUDA:  module {
// CUDA-NEXT:    llvm.func @ncclAllReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CUDA-NEXT:    llvm.func @enzymexla_wrapper_ncclAllReduce_MPI_SUM_MPI_INT64_T(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CUDA-NEXT:      %0 = llvm.mlir.constant(0 : i32) : i32
// CUDA-NEXT:      %1 = llvm.mlir.constant(1 : i64) : i64
// CUDA-NEXT:      %2 = llvm.mlir.constant(4 : i32) : i32
// CUDA-NEXT:      %3 = llvm.load %arg2 : !llvm.ptr -> i32
// CUDA-NEXT:      %4 = llvm.zext %3 : i32 to i64
// CUDA-NEXT:      %5 = llvm.inttoptr %1 : i64 to !llvm.ptr
// CUDA-NEXT:      %6 = "enzymexla.get_stream"() : () -> !llvm.ptr
// CUDA-NEXT:      %7 = llvm.call @ncclAllReduce(%arg0, %arg1, %4, %2, %0, %5, %6) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CUDA-NEXT:      llvm.return
// CUDA-NEXT:    }
// CUDA-NEXT:    func.func @main(%arg0: tensor<i64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<i64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CUDA-NEXT:      %c = stablehlo.constant dense<0> : tensor<i64>
// CUDA-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CUDA-NEXT:      %0 = enzymexla.jit_call @enzymexla_wrapper_ncclAllReduce_MPI_SUM_MPI_INT64_T (%arg0, %c, %c_0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]} : (tensor<i64>, tensor<i64>, tensor<i32>) -> tensor<i64>
// CUDA-NEXT:      return %0 : tensor<i64>
// CUDA-NEXT:    }
// CUDA-NEXT:  }
