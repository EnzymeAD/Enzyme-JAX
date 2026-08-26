// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(fuse-jit)" %s | FileCheck %s

// This is the complete IR emitted by lower-enzymexla-mpi for Irecv + Wait.
// Starting after MPI lowering isolates the generic JIT fusion pass.

module {
  llvm.mlir.global external constant @MPI_INT() {addr_space = 0 : i32} : !llvm.ptr
  llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Irecv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_wrapper_MPI_Irecv_MPI_INT(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg3: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg4: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
    %1 = llvm.mlir.addressof @MPI_INT : !llvm.ptr
    %2 = llvm.load %arg1 : !llvm.ptr -> i32
    %3 = llvm.load %arg2 : !llvm.ptr -> i32
    %4 = llvm.load %arg3 : !llvm.ptr -> i32
    %5 = llvm.call @MPI_Irecv(%arg0, %2, %1, %3, %4, %0, %arg4) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_wrapper_MPI_Wait(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.alloca %c1_i32 x !llvm.array<6 x i32> : (i32) -> !llvm.ptr
    %1 = llvm.call @MPI_Wait(%arg0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<-1> : tensor<i32>
    %c_0 = stablehlo.constant dense<5> : tensor<i32>
    %c_1 = stablehlo.constant dense<42> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %1:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT (%0, %c_0, %c_2, %c_1, %c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>]} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%1#1) : (tensor<i32>) -> ()
    %2 = stablehlo.transpose %1#0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    return %2 : tensor<5xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait
// CHECK-SAME: %[[BUFFER:[^ ,)]+]]: !llvm.ptr
// CHECK-SAME: %[[REQUEST:[^ ,)]+]]: !llvm.ptr)
// CHECK:      llvm.call @MPI_Irecv({{.*}}, %[[REQUEST]])
// CHECK:      %[[STATUS:.*]] = llvm.alloca
// CHECK:      llvm.call @MPI_Wait(%[[REQUEST]], %[[STATUS]])
// CHECK:      llvm.return

// CHECK-LABEL: func.func @main
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK:         %[[FUSED:.*]] = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Wait
// CHECK:         %[[OUT:.*]] = stablehlo.transpose %[[FUSED]]
// CHECK:         return %[[OUT]]
