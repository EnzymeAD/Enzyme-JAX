// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_Barrier=0x1" --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_Barrier=0x1" --lower-comm-to-jit --lower-jit="jit=false" %s | FileCheck %s --check-prefix=LOWERED
// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s --check-prefix=NCCL

// LOWERED-LABEL: func.func @main(
// LOWERED-SAME: %[[COMM:.+]]: tensor<i64>) {
// LOWERED-NEXT: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// LOWERED-NEXT: %[[CALL:.+]] = stablehlo.custom_call @enzymexla_compile_cpu(%[[COMM]], %[[TOKEN]])
// LOWERED-SAME: operand_index = 1
// LOWERED-SAME: : (tensor<i64>, !stablehlo.token) -> !stablehlo.token
// LOWERED-NEXT: return
func.func @main(%comm : !comm.mpi.comm) {
    comm.mpi.barrier %comm : !comm.mpi.comm
    return
}
// SHLO: func.func @main(%[[COMM:.*]]: tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiBarrier(%[[COMM]]) {has_side_effect = true} : (tensor<i64>) -> ()

// JIT-LABEL: llvm.func @MPI_Barrier
// JIT-SAME:                        (!llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Barrier
// JIT-SAME:                                          (%arg0: !llvm.ptr) attributes {enzymexla.comm} {
// JIT-NEXT:   %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[COMM:.*]]: tensor<i64>) {
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Barrier (%[[COMM]]) {enzymexla.require_jit_ordering} : (tensor<i64>) -> ()

// NCCL-LABEL: func.func @main
// NCCL-SAME: (%[[COMM:.*]]: !comm.nccl.comm) {
// NCCL-NEXT: comm.nccl.barrier %[[COMM]] : !comm.nccl.comm
// NCCL-NEXT: return

llvm.func @jitwrap_unrelated() {
  llvm.return
}

// JIT-LABEL: func.func @two_barriers(
// JIT-SAME: %[[COMM:.+]]: tensor<i64>) {
// JIT-NEXT: enzymexla.jit_call @jitwrap_unrelated () : () -> ()
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Barrier (%[[COMM]]) {enzymexla.require_jit_ordering} : (tensor<i64>) -> ()
// JIT-NEXT: enzymexla.jit_call @jitwrap_unrelated () : () -> ()
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Barrier (%[[COMM]]) {enzymexla.require_jit_ordering} : (tensor<i64>) -> ()
// JIT-NEXT: return

// Only the marked calls are chained, the unrelated calls stay untouched.
// LOWERED-LABEL: func.func @two_barriers(
// LOWERED-SAME: %[[COMM:.+]]: tensor<i64>) {
// LOWERED-NEXT: %[[TOKEN:.+]] = stablehlo.after_all : !stablehlo.token
// LOWERED-NEXT: stablehlo.custom_call @enzymexla_compile_cpu()
// LOWERED-SAME: : () -> ()
// LOWERED-NEXT: %[[FIRST:.+]] = stablehlo.custom_call @enzymexla_compile_cpu(%[[COMM]], %[[TOKEN]])
// LOWERED-SAME: operand_index = 1
// LOWERED-SAME: : (tensor<i64>, !stablehlo.token) -> !stablehlo.token
// LOWERED-NEXT: stablehlo.custom_call @enzymexla_compile_cpu()
// LOWERED-SAME: : () -> ()
// LOWERED-NEXT: %[[SECOND:.+]] = stablehlo.custom_call @enzymexla_compile_cpu(%[[COMM]], %[[FIRST]])
// LOWERED-SAME: operand_index = 1
// LOWERED-SAME: : (tensor<i64>, !stablehlo.token) -> !stablehlo.token
// LOWERED-NEXT: return
func.func @two_barriers(%comm: !comm.mpi.comm) {
  enzymexla.jit_call @jitwrap_unrelated () : () -> ()
  comm.mpi.barrier %comm : !comm.mpi.comm
  enzymexla.jit_call @jitwrap_unrelated () : () -> ()
  comm.mpi.barrier %comm : !comm.mpi.comm
  return
}
