// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit{strategy=dependencies})" %s | FileCheck %s

// Lower two Irecv calls, one Isend, and Waitall, then fuse them into one JIT.

module {
  func.func @mpi_waitall(%recvbuf0: tensor<4xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}, %recvbuf1: tensor<4xi32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %sendbuf: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %count = stablehlo.constant dense<4> : tensor<i32>
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %tag0 = stablehlo.constant dense<10> : tensor<i32>
    %tag1 = stablehlo.constant dense<20> : tensor<i32>
    %tag2 = stablehlo.constant dense<30> : tensor<i32>
    %outbuf0, %request0 = enzymexla.mpi.irecv(%recvbuf0, %count, %peer, %tag0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
    %outbuf1, %request1 = enzymexla.mpi.irecv(%recvbuf1, %count, %peer, %tag1) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<4xi32>, tensor<i32>)
    %send_request = enzymexla.mpi.isend(%sendbuf, %count, %peer, %tag2) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<4xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    enzymexla.mpi.waitall(%request0, %request1, %send_request) : tensor<i32>, tensor<i32>, tensor<i32>
    return %outbuf0, %outbuf1 : tensor<4xi32>, tensor<4xi32>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Waitall_3(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %[[REQUEST0:[^ :]+]]: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %[[REQUEST1:[^ :]+]]: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %[[REQUEST2:[^ :]+]]: !llvm.ptr)
// CHECK: %[[COUNT:.*]] = arith.constant 3 : i32
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Irecv({{.*}}, %[[REQUEST0]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Irecv({{.*}}, %[[REQUEST1]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Isend({{.*}}, %[[REQUEST2]])
// CHECK-NOT: llvm.call
// CHECK: %[[REQUESTS:.*]] = llvm.alloca %[[COUNT]] x i32
// CHECK-NEXT: %[[VALUE0:.*]] = llvm.load %[[REQUEST0]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.store %[[VALUE0]], %[[REQUESTS]] : i32, !llvm.ptr
// CHECK-NEXT: %[[VALUE1:.*]] = llvm.load %[[REQUEST1]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[ELEMENT1:.*]] = llvm.getelementptr %[[REQUESTS]][1]
// CHECK-NEXT: llvm.store %[[VALUE1]], %[[ELEMENT1]] : i32, !llvm.ptr
// CHECK-NEXT: %[[VALUE2:.*]] = llvm.load %[[REQUEST2]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[ELEMENT2:.*]] = llvm.getelementptr %[[REQUESTS]][2]
// CHECK-NEXT: llvm.store %[[VALUE2]], %[[ELEMENT2]] : i32, !llvm.ptr
// CHECK-NEXT: %[[STATUSES:.*]] = llvm.alloca %[[COUNT]] x !llvm.array<6 x i32>
// CHECK-NEXT: llvm.call @MPI_Waitall(%[[COUNT]], %[[REQUESTS]], %[[STATUSES]])
// CHECK-NEXT: %[[DONE0:.*]] = llvm.load %[[REQUESTS]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.store %[[DONE0]], %[[REQUEST0]] : i32, !llvm.ptr
// CHECK-NEXT: %[[DONE1:.*]] = llvm.load %[[ELEMENT1]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.store %[[DONE1]], %[[REQUEST1]] : i32, !llvm.ptr
// CHECK-NEXT: %[[DONE2:.*]] = llvm.load %[[ELEMENT2]] : !llvm.ptr -> i32
// CHECK-NEXT: llvm.store %[[DONE2]], %[[REQUEST2]] : i32, !llvm.ptr
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @mpi_waitall(
// CHECK-NOT: enzymexla.jit_call
// CHECK: %[[WAITALL:.*]]:5 = enzymexla.jit_call @fused__enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Isend_MPI_INT_enzymexla_wrapper_MPI_Waitall_3
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 5, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 10, operand_tuple_indices = []>]}
// CHECK-NEXT: return %[[WAITALL]]#0, %[[WAITALL]]#1 : tensor<4xi32>, tensor<4xi32>
// CHECK-NEXT: }

