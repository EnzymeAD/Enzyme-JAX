// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%src: tensor<i32>, %tag: tensor<i32>, %comm: !comm.mpi.comm) -> (tensor<4xf64>, !comm.mpi.request) {
    %0:2 = comm.mpi.irecv %src, %tag, %comm : (tensor<i32>, tensor<i32>, !comm.mpi.comm) -> (tensor<4xf64>, !comm.mpi.request)
    return %0#0, %0#1 : tensor<4xf64>, !comm.mpi.request
}

// SHLO: func.func @main(%[[DST:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> (tensor<4xf64>, tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiIrecv(%[[DST]], %[[TAG]], %[[COMM]]) {has_side_effect = true} : (tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<4xf64>, tensor<i64>)

// JIT-LABEL: llvm.func @MPI_Irecv
// JIT-SAME:                     (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Irecv
// JIT-SAME:                                       (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.load %arg1 : !llvm.ptr -> i32
// JIT-NEXT:   %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %2 = llvm.load %arg3 : !llvm.ptr -> i32
// JIT-NEXT:   %3 = llvm.load %arg4 : !llvm.ptr -> i32
// JIT-NEXT:   %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %5 = llvm.call @MPI_Irecv(%arg0, %0, %1, %2, %3, %4, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[BUFFER:.*]]: tensor<4xf64>, %[[SRC:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> (tensor<4xf64>, tensor<i64>) {
// JIT-NEXT: %[[BUFFER:.*]] = stablehlo.constant dense<0> : tensor<4xf64>
// JIT-NEXT: %[[COUNT:.*]] = stablehlo.constant dense<4> : tensor<i32>
// JIT-NEXT: %[[DATATYPE:.*]] = stablehlo.constant dense<0> : tensor<i64>
// JIT-NEXT: %[[REQUEST:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// JIT-NEXT: %[[v0:.*]] = stablehlo.jit_call @enzymexla_jitwrap_MPI_Irecv(%[[BUFFER]], %[[COUNT]], %[[DATATYPE]], %[[SRC]], %[[TAG]], %[[COMM]], %[[REQUEST]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 6, operand_tuple_indices = []>]} : (tensor<4xf64>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> (tensor<4xf64>, tensor<i64>)
// JIT-NEXT: return %[[v0]]#0, %[[v0]]#1 : tensor<4xf64>, tensor<i64>
