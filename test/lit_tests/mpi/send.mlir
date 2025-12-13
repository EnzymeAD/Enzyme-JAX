// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<43> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    enzymexla.send(%0, %c_1, %c_0, %c) : tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>
    %1 = stablehlo.transpose %0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}


// CPU:  module {
// CPU-NEXT:    llvm.mlir.global external constant @MPI_DOUBLE() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.func @MPI_Send(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_SendMPI_DOUBLE(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg3: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CPU-NEXT:      %1 = llvm.mlir.addressof @MPI_DOUBLE : !llvm.ptr
// CPU-NEXT:      %2 = llvm.load %arg2 : !llvm.ptr -> i32
// CPU-NEXT:      %3 = llvm.load %arg3 : !llvm.ptr -> i32
// CPU-NEXT:      %4 = llvm.load %arg1 : !llvm.ptr -> i32
// CPU-NEXT:      %5 = llvm.call @MPI_Send(%arg0, %4, %1, %2, %3, %0) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %c = stablehlo.constant dense<5> : tensor<i32>
// CPU-NEXT:      %c_0 = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:      %c_1 = stablehlo.constant dense<43> : tensor<i32>
// CPU-NEXT:      %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CPU-NEXT:      enzymexla.jit_call @enzymexla_wrapper_MPI_SendMPI_DOUBLE (%0, %c, %c_0, %c_1) : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
// CPU-NEXT:      %1 = stablehlo.transpose %0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CPU-NEXT:      return %1 : tensor<5xf64>
// CPU-NEXT:    }
// CPU-NEXT:  }
