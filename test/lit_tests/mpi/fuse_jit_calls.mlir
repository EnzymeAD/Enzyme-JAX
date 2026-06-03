// RUN: enzymexlamlir-opt --fuse-jit-calls %s | FileCheck %s

module {
  llvm.func @enzymexla_wrapper_MPI_Irecv(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
  llvm.func @enzymexla_wrapper_MPI_Wait(%arg0: !llvm.ptr) {
    llvm.return
  }
  func.func @main(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %c_0 = stablehlo.constant dense<5> : tensor<i32>
    %1:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv (%arg0, %c_0) : (tensor<5xf64>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%1#1) : (tensor<i32>) -> ()
    return %1#0 : tensor<5xf64>
  }
}

// CHECK: llvm.func @enzymexla_wrapper_MPI_Irecv_Wait(
// CHECK: func.func @main
// CHECK-NEXT: %[[C:.*]] = stablehlo.constant
// CHECK-NEXT: %[[RES:.*]] = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_Wait
// CHECK-NEXT: return %[[RES]]
