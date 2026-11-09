// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUS_IGNORE=0x0" --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%req : !comm.mpi.request) {
    comm.mpi.wait %req : !comm.mpi.request
    return
}
// SHLO: func.func @main(%[[REQ:.*]]: tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiWait(%[[REQ]]) {has_side_effect = true} : (tensor<i64>) -> ()

// JIT-LABEL: llvm.func @MPI_Wait
// JIT-SAME:                     (!llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Wait
// JIT-SAME:                                       (%arg0: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.mlir.constant(0 : i64) : i64
// JIT-NEXT:   %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
// JIT-NEXT:   %2 = llvm.call @MPI_Wait(%arg0, %1) : (!llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[REQ:.*]]: tensor<i64>) {
// JIT-NEXT: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// JIT-NEXT: %[[CALL:.+]] = enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait (%[[REQ]], %[[TOKEN]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> : (tensor<i64>, !stablehlo.token) -> !stablehlo.token
