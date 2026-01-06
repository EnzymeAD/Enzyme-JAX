// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main() -> tensor<i32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = enzymexla.mpi.comm_rank : tensor<i32>
    return %0 : tensor<i32>
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Comm_rank() attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CPU-NEXT:      %c1_i32 = arith.constant 1 : i32
// CPU-NEXT:      %1 = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
// CPU-NEXT:      %2 = llvm.call @MPI_Comm_rank(%0, %1) : (!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main() -> tensor<i32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = enzymexla.jit_call @enzymexla_wrapper_MPI_Comm_rank () : () -> tensor<i32>
// CPU-NEXT:      return %0 : tensor<i32>
// CPU-NEXT:    }
// CPU-NEXT:  }
