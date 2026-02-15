// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main() {
    %request = stablehlo.constant dense<-1> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %request, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %5, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    enzymexla.mpi.waitall(%c_2, %6) : tensor<i32>, tensor<1xi32>
    return
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.func @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Waitall(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.load %arg0 : !llvm.ptr -> i32
// CPU-NEXT:      %1 = llvm.alloca %0 x !llvm.array<6 x i32> : (i32) -> !llvm.ptr
// CPU-NEXT:      %2 = llvm.call @MPI_Waitall(%0, %arg1, %1) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main() {
// CPU-NEXT:      %c = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:      %c_0 = stablehlo.constant dense<-1> : tensor<i32>
// CPU-NEXT:      %0 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
// CPU-NEXT:      %1 = stablehlo.concatenate %0, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
// CPU-NEXT:      enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall (%c, %1) : (tensor<i32>, tensor<1xi32>) -> ()
// CPU-NEXT:      return
// CPU-NEXT:    }
// CPU-NEXT:  }
