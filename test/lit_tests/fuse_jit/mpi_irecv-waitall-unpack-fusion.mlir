// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

// Regression test for https://github.com/EnzymeAD/Reactant.jl/issues/2591.
// The completed receive buffer must be returned by the fused Irecv/Waitall JIT
// call before it is unpacked into the destination buffer.

module {
  func.func @main(%recv_buf: tensor<1xf64>, %data_buf: tensor<2xf64>) -> tensor<2xf64> {
    %count = stablehlo.constant dense<1> : tensor<i32>
    %source = stablehlo.constant dense<0> : tensor<i32>
    %tag = stablehlo.constant dense<43> : tensor<i32>
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %recv_ready, %request = enzymexla.mpi.irecv(%recv_buf, %count, %source, %tag) {datatype = #enzymexla.datatype<MPI_DOUBLE>} : (tensor<1xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<1xf64>, tensor<i32>)
    enzymexla.mpi.waitall(%request) : tensor<i32>
    %unpacked = stablehlo.dynamic_update_slice %data_buf, %recv_ready, %zero : (tensor<2xf64>, tensor<1xf64>, tensor<i32>) -> tensor<2xf64>
    return %unpacked : tensor<2xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_DOUBLE_enzymexla_wrapper_MPI_Waitall_1
// CHECK:         llvm.call @MPI_Irecv({{.*}}, %[[REQUEST:[^ ,)]+]])
// CHECK:         %[[REQUESTS:.*]] = llvm.alloca {{.*}} x i32 : (i32) -> !llvm.ptr
// CHECK:         %[[REQUEST_VALUE:.*]] = llvm.load %[[REQUEST]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[REQUEST_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Waitall({{.*}}, %[[REQUESTS]], {{.*}})
// CHECK:         llvm.return

// CHECK-LABEL: func.func @main
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_DOUBLE
// CHECK:         %[[RECV_READY:.*]] = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_DOUBLE_enzymexla_wrapper_MPI_Waitall_1
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall_1
// CHECK:         %[[UNPACKED:.*]] = stablehlo.dynamic_update_slice %{{.*}}, %[[RECV_READY]], %{{.*}} : (tensor<2xf64>, tensor<1xf64>, tensor<i32>) -> tensor<2xf64>
// CHECK:         return %[[UNPACKED]] : tensor<2xf64>
