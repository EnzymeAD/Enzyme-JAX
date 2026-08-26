// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%request0: tensor<i32>, %request1: tensor<i32>) {
    enzymexla.mpi.waitall(%request0, %request1) : tensor<i32>, tensor<i32>
    return
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.func @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
// CPU-LABEL:   llvm.func @enzymexla_wrapper_MPI_Waitall_2(
// CPU-SAME:        %arg0: !llvm.ptr
// CPU-SAME:        %arg1: !llvm.ptr
// CPU:           %[[COUNT:.*]] = arith.constant 2 : i32
// CPU:           %[[REQUESTS:.*]] = llvm.alloca %[[COUNT]] x i32 : (i32) -> !llvm.ptr
// CPU:           llvm.call @MPI_Waitall(%[[COUNT]], %[[REQUESTS]], {{.*}})
// CPU:           llvm.store {{.*}}, %arg0
// CPU:           llvm.store {{.*}}, %arg1
// CPU-LABEL:   func.func @main(
// CPU-SAME:        %[[REQUEST0:[^,]+]]: tensor<i32>,
// CPU-SAME:        %[[REQUEST1:[^)]+]]: tensor<i32>)
// CPU:           enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall_2 (%[[REQUEST0]], %[[REQUEST1]]) : (tensor<i32>, tensor<i32>) -> ()
// CPU-NOT:       stablehlo.broadcast_in_dim
// CPU-NOT:       stablehlo.concatenate
