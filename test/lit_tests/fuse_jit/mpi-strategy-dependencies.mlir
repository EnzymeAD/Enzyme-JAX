// RUN: enzymexlamlir-opt --fuse-jit="strategy=dependencies" %s | FileCheck %s

// Fuse the lowered Irecv/Wait and Isend/Wait calls. Keep the add between them:
// the send needs its result.

module {
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
  func.func @computed_send(%arg0: tensor<4xi32>, %arg1: tensor<i64>) -> tensor<4xi32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<10> : tensor<i32>
    %c_1 = stablehlo.constant dense<20> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<4xi32>
    %c_3 = stablehlo.constant dense<4> : tensor<i32>
    %c_4 = stablehlo.constant dense<-1> : tensor<i64>
    %c_5 = stablehlo.constant dense<3> : tensor<i64>
    %0:2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Irecv (%c_2, %c_3, %c_5, %c, %c_0, %arg1, %c_4) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>]> : (tensor<4xi32>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> (tensor<4xi32>, tensor<i64>)
    enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait (%0#1) : (tensor<i64>) -> ()
    %1 = stablehlo.add %0#0, %arg0 : tensor<4xi32>
    %c_6 = stablehlo.constant dense<4> : tensor<i32>
    %c_7 = stablehlo.constant dense<-1> : tensor<i64>
    %c_8 = stablehlo.constant dense<3> : tensor<i64>
    %2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Isend (%1, %c_6, %c_8, %c, %c_1, %arg1, %c_7) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 6, operand_tuple_indices = []>]> : (tensor<4xi32>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i64>
    enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait (%2) : (tensor<i64>) -> ()
    return %1 : tensor<4xi32>
  }
}


// CHECK-LABEL: llvm.func @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Wait(
// CHECK-SAME: %[[SEND_BUFFER:[^ :]+]]: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %[[SEND_REQUEST:[^ :]+]]: !llvm.ptr) {
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Isend(%[[SEND_BUFFER]], {{.*}}, %[[SEND_REQUEST]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[SEND_REQUEST]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: llvm.func @fused__enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Wait(
// CHECK-SAME: %[[RECV_BUFFER:[^ :]+]]: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %{{[^ :]+}}: !llvm.ptr, %[[RECV_REQUEST:[^ :]+]]: !llvm.ptr) {
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Irecv(%[[RECV_BUFFER]], {{.*}}, %[[RECV_REQUEST]])
// CHECK-NOT: llvm.call
// CHECK: llvm.call @MPI_Wait(%[[RECV_REQUEST]], %{{[^ ,)]+}})
// CHECK-NOT: llvm.call
// CHECK: llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @computed_send(
// CHECK-SAME: %[[DELTA:[^ :]+]]: tensor<4xi32>, %[[COMM:[^ :]+]]: tensor<i64>)
// CHECK-NOT: enzymexla.jit_call
// CHECK-DAG: %[[BUFFER:.*]] = stablehlo.constant dense<0> : tensor<4xi32>
// CHECK-DAG: %[[COUNT:.*]] = stablehlo.constant dense<4> : tensor<i32>
// CHECK-DAG: %[[DATATYPE:.*]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[PEER:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-DAG: %[[RECV_TAG:.*]] = stablehlo.constant dense<10> : tensor<i32>
// CHECK-DAG: %[[SEND_TAG:.*]] = stablehlo.constant dense<20> : tensor<i32>
// CHECK-DAG: %[[REQUEST:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NOT: enzymexla.jit_call
// CHECK: %[[RECV:.*]]:2 = enzymexla.jit_call @fused__enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Wait (%[[BUFFER]], %[[COUNT]], %[[DATATYPE]], %[[PEER]], %[[RECV_TAG]], %[[COMM]], %[[REQUEST]])
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>]
// CHECK-SAME: : {{.*}} -> (tensor<4xi32>, tensor<i64>)
// CHECK-NEXT: %[[COMPUTED:.*]] = stablehlo.add %[[RECV]]#0, %[[DELTA]] : tensor<4xi32>
// CHECK-NEXT: %{{[^ ]+}} = enzymexla.jit_call @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Wait (%[[COMPUTED]], %[[COUNT]], %[[DATATYPE]], %[[PEER]], %[[SEND_TAG]], %[[COMM]], %[[REQUEST]])
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 6, operand_tuple_indices = []>]
// CHECK-SAME: : {{.*}} -> tensor<i64>
// CHECK-NEXT: return %[[COMPUTED]] : tensor<4xi32>
// CHECK-NEXT: }
