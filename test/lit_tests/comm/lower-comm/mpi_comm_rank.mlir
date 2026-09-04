// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%comm : !comm.mpi.comm) -> tensor<i32> {
    %0 = comm.mpi.comm_rank %comm : tensor<i32>
    return %0 : tensor<i32>
}

// SHLO: func.func @main(%[[COMM:.*]]: tensor<i64>) -> tensor<i32> {
// SHLO-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiCommRank(%[[COMM]]) : (tensor<i64>) -> tensor<i32>
// SHLO-NEXT: return %[[v0]] : tensor<i32>

// JIT-LABEL: llvm.func @MPI_Comm_rank
// JIT-SAME:                          (!llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Comm_rank
// JIT-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %1 = llvm.call @MPI_Comm_rank(%0, %arg0) : (!llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[COMM:.*]]: tensor<i64>) -> tensor<i32> {
// JIT-NEXT: %[[rank_pl:.*]] = stablehlo.constant dense<0> : tensor<i32>
// JIT-NEXT: %[[v0:.*]] = stablehlo.jit_call @enzymexla_jitwrap_MPI_Comm_rank(%[[COMM]], %[[rank_pl]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], xla_side_effect_free} : (tensor<i64>, tensor<i32>) -> tensor<i32>
// JIT-NEXT: return %[[v0]] : tensor<i32>
