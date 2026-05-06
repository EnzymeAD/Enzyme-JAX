// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cuda ncclCommPtr=1})" %s | FileCheck %s --check-prefix=CUDA

module {
  func.func @main() -> tensor<i32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = enzymexla.mpi.comm_rank : tensor<i32>
    return %0 : tensor<i32>
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Comm_rank(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CPU-NEXT:      %1 = llvm.call @MPI_Comm_rank(%0, %arg0) : (!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main() -> tensor<i32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %c = stablehlo.constant dense<-1> : tensor<i32>
// CPU-NEXT:      %0 = enzymexla.jit_call @enzymexla_wrapper_MPI_Comm_rank (%c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i32>) -> tensor<i32>
// CPU-NEXT:      return %0 : tensor<i32>
// CPU-NEXT:    }
// CPU-NEXT:  }

// CUDA:  module {
// CUDA-NEXT:    llvm.func @ncclCommUserRank(!llvm.ptr, !llvm.ptr) -> i32
// CUDA-NEXT:    llvm.func @enzymexla_wrapper_ncclCommUserRank(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CUDA-NEXT:      %0 = llvm.mlir.constant(1 : i64) : i64
// CUDA-NEXT:      %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
// CUDA-NEXT:      %2 = llvm.call @ncclCommUserRank(%1, %arg0) : (!llvm.ptr, !llvm.ptr) -> i32
// CUDA-NEXT:      llvm.return
// CUDA-NEXT:    }
// CUDA-NEXT:    func.func @main() -> tensor<i32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CUDA-NEXT:      %c = stablehlo.constant dense<-1> : tensor<i32>
// CUDA-NEXT:      %0 = enzymexla.jit_call @enzymexla_wrapper_ncclCommUserRank (%c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i32>) -> tensor<i32>
// CUDA-NEXT:      return %0 : tensor<i32>
// CUDA-NEXT:    }
// CUDA-NEXT:  }
