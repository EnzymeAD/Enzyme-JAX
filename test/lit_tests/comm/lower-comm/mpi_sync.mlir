// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUS_IGNORE=0x0" --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUS_IGNORE=0x0" --lower-comm-to-jit --canonicalize %s | FileCheck %s --check-prefix=CANON

// The buffer is not touched in the wrapper, but its alias must survive
// canonicalization so that its uses stay ordered after `MPI_Wait`.

func.func @main(%req : !comm.mpi.request, %buf : tensor<4xf32>) -> tensor<4xf32> {
    %0 = comm.mpi.sync %req, %buf : tensor<4xf32>
    return %0 : tensor<4xf32>
}

// JIT-LABEL: llvm.func @MPI_Wait
// JIT-SAME:                     (!llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Wait_sync
// JIT-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}) attributes {enzymexla.comm} {
// JIT-NEXT:   %0 = llvm.mlir.constant(0 : i64) : i64
// JIT-NEXT:   %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
// JIT-NEXT:   %2 = llvm.call @MPI_Wait(%arg0, %1) : (!llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[REQ:.*]]: tensor<i64>, %[[BUF:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// JIT-NEXT: %[[v0:.*]] = enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait_sync (%[[REQ]], %[[BUF]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> {enzymexla.require_jit_ordering} : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
// JIT-NEXT: return %[[v0]] : tensor<4xf32>

// CANON-LABEL: llvm.func @enzymexla_jitwrap_MPI_Wait_sync
// CANON-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]})
// CANON-LABEL: func.func @main
// CANON-SAME:                 (%[[REQ:.*]]: tensor<i64>, %[[BUF:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CANON-NEXT: %[[v0:.*]] = enzymexla.jit_call @enzymexla_jitwrap_MPI_Wait_sync (%[[REQ]], %[[BUF]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]>
// CANON-NEXT: return %[[v0]] : tensor<4xf32>
