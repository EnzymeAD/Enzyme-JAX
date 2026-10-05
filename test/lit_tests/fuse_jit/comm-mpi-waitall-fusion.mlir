// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_DOUBLE=0x3,MPI_Irecv=0x1,MPI_Isend=0x2,MPI_Waitall=0x4,MPI_STATUSES_IGNORE=0x5" --lower-comm-to-jit --cse --fuse-jit="strategy=dependencies" %s | FileCheck %s

// Lower comm.mpi nonblocking operations and variadic Waitall, then fuse their
// JIT calls.
module {
  func.func @comm_mpi_waitall(
      %sendbuf: tensor<4xf64>,
      %src0: tensor<i32>, %src1: tensor<i32>, %dst: tensor<i32>,
      %tag0: tensor<i32>, %tag1: tensor<i32>, %tag2: tensor<i32>,
      %comm: !comm.mpi.comm
  ) -> (tensor<4xf64>, tensor<4xf64>) {
    %recvbuf0, %request0 = comm.mpi.irecv %src0, %tag0, %comm : (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> (tensor<4xf64>, !comm.mpi.request)
    %recvbuf1, %request1 = comm.mpi.irecv %src1, %tag1, %comm : (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> (tensor<4xf64>, !comm.mpi.request)
    %send_request = comm.mpi.isend %sendbuf, %dst, %tag2, %comm : (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
    comm.mpi.waitall %request0, %request1, %send_request : !comm.mpi.request, !comm.mpi.request, !comm.mpi.request
    return %recvbuf0, %recvbuf1 : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Waitall_3(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: !llvm.ptr) {
// CHECK: %[[THREE:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: %[[IGNORE:.*]] = llvm.mlir.addressof @MPI_STATUSES_IGNORE : !llvm.ptr
// CHECK: llvm.call @MPI_Irecv(%arg0, {{.*}}, %[[REQ0:[^ )]+]]) :
// CHECK-NOT: llvm.call @MPI_Irecv({{.*}}, %[[REQ0]])
// CHECK: llvm.call @MPI_Irecv(%arg7, {{.*}}, %[[REQ1:[^ )]+]]) :
// CHECK-NOT: llvm.call @MPI_Isend({{.*}}, %[[REQ1]])
// CHECK: llvm.call @MPI_Isend(%arg11, {{.*}}, %[[REQ2:[^ )]+]]) :
// CHECK-NOT: llvm.call @MPI_Isend({{.*}}, %[[REQ0]])
// CHECK-NOT: llvm.call @MPI_Isend({{.*}}, %[[REQ1]])
// CHECK: %[[REQARRAY:.*]] = llvm.alloca %[[THREE]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT: llvm.store %[[REQ0]], %[[REQARRAY]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[REQARRAY1:.*]] = llvm.getelementptr %[[REQARRAY]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: llvm.store %[[REQ1]], %[[REQARRAY1]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[REQARRAY2:.*]] = llvm.getelementptr %[[REQARRAY]][2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: llvm.store %[[REQ2]], %[[REQARRAY2]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT: %[[REQARRAYBASE:.*]] = llvm.getelementptr %[[REQARRAY]][] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT: llvm.call @MPI_Waitall(%[[THREE]], %[[REQARRAYBASE]], %[[IGNORE]]) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @comm_mpi_waitall(
// CHECK: %[[FUSED:.*]]:5 = enzymexla.jit_call @fused__enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Waitall_3
// CHECK-SAME: (%cst, %c, %c_1, %arg1, %arg4, %arg7, %c_0, %cst, %arg2, %arg5, %c_0, %arg0, %arg3, %arg6, %c_0)
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 6, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 10, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [4], operand_index = 14, operand_tuple_indices = []>]
// CHECK-NOT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Irecv
// CHECK-NOT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Isend
// CHECK-NOT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall
// CHECK-NEXT: return %[[FUSED]]#0, %[[FUSED]]#1 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT: }
