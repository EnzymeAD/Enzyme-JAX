// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%buffer: tensor<4xf64>, %dst: tensor<i32>, %tag: tensor<i32>, %comm: !comm.mpi.comm) -> !comm.mpi.request {
    %0 = comm.mpi.isend %buffer, %dst, %tag, %comm : (tensor<4xf64>, tensor<i32>, tensor<i32>, !comm.mpi.comm) -> !comm.mpi.request
    return %0 : !comm.mpi.request
}

// SHLO: func.func @main(%[[BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) -> tensor<i64> {
// SHLO-NEXT: stablehlo.custom_call @MpiIsend(%[[BUFFER]], %[[DST]], %[[TAG]], %[[COMM]]) {has_side_effect = true} : (tensor<4xf64>, tensor<i32>, tensor<i32>, tensor<i64>) -> tensor<i64>

// JIT-LABEL: llvm.func @MPI_Isend
// JIT-SAME:                     (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Isend
// JIT-SAME:                                       (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.load %arg1 : !llvm.ptr -> i32
// JIT-NEXT:   %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %2 = llvm.load %arg3 : !llvm.ptr -> i32
// JIT-NEXT:   %3 = llvm.load %arg4 : !llvm.ptr -> i32
// JIT-NEXT:   %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %5 = llvm.call @MPI_Isend(%arg0, %0, %1, %2, %3, %4, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[BUFFER:.*]]: tensor<4xf64>, %[[DST:.*]]: tensor<i32>, %[[TAG:.*]]: tensor<i32>, %[[COMM:.*]]: tensor<i64>) {
// JIT-NEXT: %[[COUNT:.*]] = stablehlo.constant dense<4> : tensor<i32>
// JIT-NEXT: %[[DATATYPE:.*]] = stablehlo.constant dense<0> : tensor<i64>
// JIT-NEXT: %[[REQUEST:.*]] = stablehlo.constant dense<-1> : tensor<i64>
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Isend(%[[BUFFER]], %[[COUNT]], %[[DATATYPE]], %[[DST]], %[[TAG]], %[[COMM]], %[[REQUEST]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 6, operand_tuple_indices = []>]} : (tensor<4xf64>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> ()
// JIT-NEXT: return
