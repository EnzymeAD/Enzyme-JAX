// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main() attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    enzymexla.mpi_barrier
    return
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.func @MPI_Barrier(!llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Barrier() attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CPU-NEXT:      %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main() attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
// CPU-NEXT:      return
// CPU-NEXT:    }
// CPU-NEXT:  }
