// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%buf : tensor<4xf64>, %comm : !comm.mpi.comm) -> tensor<4xf64> {
    %0 = comm.mpi.allreduce %buf, #comm.mpi.op<MPI_OP_NULL>, %comm : (tensor<4xf64>, !comm.mpi.comm) -> tensor<4xf64>
    return %0 : tensor<4xf64>
}

// SHLO: func.func @main(%[[BUF:.*]]: tensor<4xf64>, %[[COMM:.*]]: tensor<i64>)  -> tensor<4xf64> {
// SHLO-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiAllreduce(%[[BUF]], %[[COMM]]) {api_version = 4 : i32, backend_config = {op = "MPI_OP_NULL"}, has_side_effect = true} : (tensor<4xf64>, tensor<i64>) -> tensor<4xf64>

// JIT-LABEL: llvm.func @MPI_Allreduce
// JIT-SAME:                          (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Allreduce
// JIT-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.load %arg2 : !llvm.ptr -> i32
// JIT-NEXT:   %1 = llvm.load %arg3 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %2 = llvm.load %arg4 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %3 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %4 = llvm.call @MPI_Allreduce(%arg0, %arg1, %0, %1, %2, %3) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[BUF:.*]]: tensor<4xf64, %[[COMM:.*]]: tensor<i64>) -> tensor<4xf64> {
// JIT-NEXT: %[[COUNT:.*]] = stablehlo.constant dense<4> : tensor<i32>
// JIT-NEXT: %[[DATATYPE:.*]] = stablehlo.constant dense<0> : tensor<i64>
// JIT-NEXT: %[[OP:.*]] = stablehlo.constant dense<0> : tensor<i32>
// JIT-NEXT: %[[RECVBUF:.*]] = stablehlo.constant dense<0> : tensor<4xf64>
// JIT-NEXT: %[[v0:.*]] = stablehlo.jit_call @enzymexla_jitwrap_MPI_Allreduce(%[[BUF]], %[[RECVBUF]], %[[COUNT]], %[[DATATYPE]], %[[OP]], %[[COMM]]) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]} : (tensor<4xf64>, tensor<4xf64>, tensor<i32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4xf64>
// JIT-NEXT: return %[[v0]] : tensor<4xf64>
