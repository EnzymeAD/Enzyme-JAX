// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<42> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %outbuf, %request = enzymexla.mpi.irecv(%0, %c_1, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    %request_1 = enzymexla.mpi.isend(%outbuf, %c_1, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    enzymexla.mpi.waitall(%request, %request_1) : tensor<i32>, tensor<i32>
    %outbuf_1 = stablehlo.transpose %outbuf, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %1 = stablehlo.add %outbuf, %outbuf_1 : tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Waitall_2
// CHECK:         llvm.call @MPI_Irecv({{.*}}, %[[IRECV_REQUEST:[^ ,)]+]])
// CHECK:         llvm.call @MPI_Isend({{.*}}, %[[ISEND_REQUEST:[^ ,)]+]])
// CHECK:         %[[IRECV_REQUEST_VALUE:.*]] = llvm.load %[[IRECV_REQUEST]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[IRECV_REQUEST_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         %[[ISEND_REQUEST_VALUE:.*]] = llvm.load %[[ISEND_REQUEST]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[ISEND_REQUEST_VALUE]], %{{.*}} : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Waitall
// CHECK:         llvm.return

// CHECK-LABEL: func.func @main
// CHECK:         %[[TRANSPOSE:.*]] = stablehlo.transpose %{{.*}}, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT
// CHECK:         %[[FUSED:.*]] = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Waitall_2
// CHECK-SAME:    output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT
// CHECK-NOT:     enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall_2
// CHECK-NOT:     stablehlo.concatenate
// CHECK:         %[[OUTBUF_1:.*]] = stablehlo.transpose %[[FUSED]], dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CHECK:         %[[RESULT:.*]] = stablehlo.add %[[FUSED]], %[[OUTBUF_1]] : tensor<5xf64>
// CHECK:         return %[[RESULT]] : tensor<5xf64>
