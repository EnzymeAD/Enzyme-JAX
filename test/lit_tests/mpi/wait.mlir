// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main() {
    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
    enzymexla.wait(%c_2) : tensor<i64>
    return
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Wait(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %c1_i32 = arith.constant 1 : i32
// CPU-NEXT:      %0 = llvm.alloca %c1_i32 x !llvm.array<6 x i32> : (i32) -> !llvm.ptr
// CPU-NEXT:      %1 = llvm.call @MPI_Wait(%arg0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main() {
// CPU-NEXT:      %c = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:      enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%c) : (tensor<i64>) -> ()
// CPU-NEXT:      return
// CPU-NEXT:    }
// CPU-NEXT:  }
