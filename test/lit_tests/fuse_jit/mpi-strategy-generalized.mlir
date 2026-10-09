// RUN: enzymexlamlir-opt --fuse-jit="strategy=generalized" %s | FileCheck %s

// Fuse two independent lowered request chains and a control send into one JIT
// call.

module {
  llvm.func @MPI_Send(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Send(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> i32
    %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.load %arg3 : !llvm.ptr -> i32
    %3 = llvm.load %arg4 : !llvm.ptr -> i32
    %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.call @MPI_Send(%arg0, %0, %1, %2, %3, %4) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.mlir.global external constant @MPI_STATUS_IGNORE() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Wait(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.addressof @MPI_STATUS_IGNORE : !llvm.ptr
    %1 = llvm.call @MPI_Wait(%arg0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @MPI_Irecv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Irecv(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> i32
    %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.load %arg3 : !llvm.ptr -> i32
    %3 = llvm.load %arg4 : !llvm.ptr -> i32
    %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.call @MPI_Irecv(%arg0, %0, %1, %2, %3, %4, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @MPI_Isend(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Isend(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> i32
    %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.load %arg3 : !llvm.ptr -> i32
    %3 = llvm.load %arg4 : !llvm.ptr -> i32
    %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.call @MPI_Isend(%arg0, %0, %1, %2, %3, %4, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  func.func @mpi_order(%arg0: tensor<5xi32>, %arg1: tensor<1xi32>, %arg2: tensor<i64>) -> tensor<5xi32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<10> : tensor<i32>
    %c_1 = stablehlo.constant dense<20> : tensor<i32>
    %c_2 = stablehlo.constant dense<30> : tensor<i32>
    %c_3 = stablehlo.constant dense<5> : tensor<i32>
    %c_4 = stablehlo.constant dense<-1> : tensor<i64>
    %c_5 = stablehlo.constant dense<3> : tensor<i64>
    %0 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Isend (%arg0, %c_3, %c_5, %c, %c_0, %arg2, %c_4) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 6, operand_tuple_indices = []>]> : (tensor<5xi32>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %c_6 = stablehlo.constant dense<0> : tensor<5xi32>
    %c_7 = stablehlo.constant dense<5> : tensor<i32>
    %c_8 = stablehlo.constant dense<-1> : tensor<i64>
    %c_9 = stablehlo.constant dense<3> : tensor<i64>
    %1:2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Irecv (%c_6, %c_7, %c_9, %c, %c_1, %arg2, %c_8) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>]> : (tensor<5xi32>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> (tensor<5xi32>, tensor<i64>)
    enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait (%0) : (tensor<i64>) -> ()
    %c_10 = stablehlo.constant dense<1> : tensor<i32>
    %c_11 = stablehlo.constant dense<3> : tensor<i64>
    enzymexla.jit_call @enzymexla_jitwrap_MPI_Send (%arg1, %c_10, %c_11, %c, %c_2, %arg2) : (tensor<1xi32>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>) -> ()
    enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait (%1#1) : (tensor<i64>) -> ()
    return %1#0 : tensor<5xi32>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Wait_enzymexla_jitwrap_MPI_Send_enzymexla_jitwrap_MPI_Wait(
// CHECK-SAME: %[[SEND_BUF:[^ :]+]]: !llvm.ptr, %[[COUNT_PTR:[^ :]+]]: !llvm.ptr, %[[DATATYPE_PTR:[^ :]+]]: !llvm.ptr, %[[PEER_PTR:[^ :]+]]: !llvm.ptr, %[[TAG_A_PTR:[^ :]+]]: !llvm.ptr, %[[COMM_PTR:[^ :]+]]: !llvm.ptr, %[[SEND_SLOT:[^ :]+]]: !llvm.ptr, %[[RECV_BUF:[^ :]+]]: !llvm.ptr, %[[TAG_B_PTR:[^ :]+]]: !llvm.ptr, %[[RECV_SLOT:[^ :]+]]: !llvm.ptr, %[[CONTROL_BUF:[^ :]+]]: !llvm.ptr, %[[CONTROL_TAG_PTR:[^ :]+]]: !llvm.ptr) {
// CHECK-NOT: llvm.call
// CHECK: %[[SEND_COUNT:.*]] = llvm.load %[[COUNT_PTR]]
// CHECK-NEXT: %[[SEND_DATATYPE:.*]] = llvm.load %[[DATATYPE_PTR]]
// CHECK-NEXT: %[[SEND_PEER:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[SEND_TAG:.*]] = llvm.load %[[TAG_A_PTR]]
// CHECK-NEXT: %[[SEND_COMM:.*]] = llvm.load %[[COMM_PTR]]
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Isend(%[[SEND_BUF]], %[[SEND_COUNT]], %[[SEND_DATATYPE]], %[[SEND_PEER]], %[[SEND_TAG]], %[[SEND_COMM]], %[[SEND_SLOT]])
// CHECK-NOT: llvm.call
// CHECK: %[[RECV_COUNT:.*]] = llvm.load %[[COUNT_PTR]]
// CHECK-NEXT: %[[RECV_DATATYPE:.*]] = llvm.load %[[DATATYPE_PTR]]
// CHECK-NEXT: %[[RECV_PEER:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[RECV_TAG:.*]] = llvm.load %[[TAG_B_PTR]]
// CHECK-NEXT: %[[RECV_COMM:.*]] = llvm.load %[[COMM_PTR]]
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Irecv(%[[RECV_BUF]], %[[RECV_COUNT]], %[[RECV_DATATYPE]], %[[RECV_PEER]], %[[RECV_TAG]], %[[RECV_COMM]], %[[RECV_SLOT]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[SEND_SLOT]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: %[[CONTROL_COUNT:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[CONTROL_DATATYPE:.*]] = llvm.load %[[DATATYPE_PTR]]
// CHECK-NEXT: %[[CONTROL_PEER:.*]] = llvm.load %[[PEER_PTR]]
// CHECK-NEXT: %[[CONTROL_TAG:.*]] = llvm.load %[[CONTROL_TAG_PTR]]
// CHECK-NEXT: %[[CONTROL_COMM:.*]] = llvm.load %[[COMM_PTR]]
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Send(%[[CONTROL_BUF]], %[[CONTROL_COUNT]], %[[CONTROL_DATATYPE]], %[[CONTROL_PEER]], %[[CONTROL_TAG]], %[[CONTROL_COMM]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[RECV_SLOT]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @mpi_order(
// CHECK-SAME: %[[SEND:[^ :]+]]: tensor<5xi32>, %[[CONTROL:[^ :]+]]: tensor<1xi32>, %[[COMM:[^ :]+]]: tensor<i64>)
// CHECK-NOT: enzymexla.jit_call
// CHECK-DAG: %[[PEER:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[COUNT:.*]] = stablehlo.constant dense<5> : tensor<i32>
// CHECK-DAG: %[[DATATYPE:.*]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[REQUEST:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-DAG: %[[RECV:.*]] = stablehlo.constant dense<0> : tensor<5xi32>
// CHECK-DAG: %[[TAG_A:.*]] = stablehlo.constant dense<10> : tensor<i32>
// CHECK-DAG: %[[TAG_B:.*]] = stablehlo.constant dense<20> : tensor<i32>
// CHECK-DAG: %[[CONTROL_TAG:.*]] = stablehlo.constant dense<30> : tensor<i32>
// CHECK-NOT: enzymexla.jit_call
// CHECK: %[[MPI:.*]]:3 = enzymexla.jit_call @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Wait_enzymexla_jitwrap_MPI_Send_enzymexla_jitwrap_MPI_Wait (%[[SEND]], %[[COUNT]], %[[DATATYPE]], %[[PEER]], %[[TAG_A]], %[[COMM]], %[[REQUEST]], %[[RECV]], %[[TAG_B]], %[[REQUEST]], %[[CONTROL]], %[[CONTROL_TAG]])
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 9, operand_tuple_indices = []>]
// CHECK-SAME: : {{.*}} -> (tensor<5xi32>, tensor<i64>, tensor<i64>)
// CHECK-NEXT: return %[[MPI]]#0 : tensor<5xi32>
// CHECK-NEXT: }
