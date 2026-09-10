// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

// Exercise the real MPI lowering so Waitall receives direct JIT results from
// both Irecv calls before the generic JIT fusion pass runs.

module {
  func.func @main(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> (tensor<5xi32>, tensor<5xi32>) {
    %count = stablehlo.constant dense<5> : tensor<i32>
    %source = stablehlo.constant dense<1> : tensor<i32>
    %tag0 = stablehlo.constant dense<41> : tensor<i32>
    %tag1 = stablehlo.constant dense<42> : tensor<i32>
    %out0, %request0 = enzymexla.mpi.irecv(%arg0, %count, %source, %tag0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xi32>, tensor<i32>)
    %out1, %request1 = enzymexla.mpi.irecv(%arg1, %count, %source, %tag1) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xi32>, tensor<i32>)
    enzymexla.mpi.waitall(%request0, %request1) : tensor<i32>, tensor<i32>
    return %out0, %out1 : tensor<5xi32>, tensor<5xi32>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Waitall_2
// CHECK:         llvm.call @MPI_Irecv({{.*}}, %[[REQUEST0:[^ ,)]+]])
// CHECK:         llvm.call @MPI_Irecv({{.*}}, %[[REQUEST1:[^ ,)]+]])
// CHECK:         %[[REQUEST0_VALUE:.*]] = llvm.load %[[REQUEST0]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[REQUEST0_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         %[[REQUEST1_VALUE:.*]] = llvm.load %[[REQUEST1]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[REQUEST1_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Waitall
// CHECK:         llvm.return

// CHECK-LABEL: func.func @main
// CHECK:         %[[FUSED:.*]]:2 = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Waitall_2
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 5, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall_2
// CHECK:         return %[[FUSED]]#0, %[[FUSED]]#1 : tensor<5xi32>, tensor<5xi32>
