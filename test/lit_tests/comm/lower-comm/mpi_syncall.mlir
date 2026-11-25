// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUSES_IGNORE=0x0" --lower-comm-to-jit %s -split-input-file | FileCheck %s --check-prefix=JIT
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUSES_IGNORE=0x0" --lower-comm-to-jit="mpi_request_width=32" %s -split-input-file | FileCheck %s --check-prefix=JIT32
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUSES_IGNORE=0x0" --lower-comm-to-jit --canonicalize %s -split-input-file | FileCheck %s --check-prefix=CANON

// The buffer is not touched in the wrapper, but its alias must survive
// canonicalization so that its uses stay ordered after `MPI_Waitall`.

func.func @single(%req : !comm.mpi.request, %buf : tensor<4xf32>) -> tensor<4xf32> {
    %0 = comm.mpi.syncall requests = (%req), buffers = (%buf) : tensor<4xf32>
    return %0 : tensor<4xf32>
}

// JIT-LABEL: llvm.func @MPI_Waitall
// JIT-SAME:                        (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_sync_1
// JIT-SAME:                                                 (%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}) attributes {enzymexla.comm} {
// JIT-LABEL: func.func @single
// JIT-SAME:                   (%[[REQ:.*]]: tensor<i64>, %[[BUF:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// JIT-NEXT: %[[v0:.*]] = enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_sync_1 (%[[REQ]], %[[BUF]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]> {enzymexla.require_jit_ordering} : (tensor<i64>, tensor<4xf32>) -> tensor<4xf32>
// JIT-NEXT: return %[[v0]] : tensor<4xf32>

// CANON-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_sync_1
// CANON-SAME:                                                 (%arg0: !llvm.ptr, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]})
// CANON-LABEL: func.func @single
// CANON-SAME:                   (%[[REQ:.*]]: tensor<i64>, %[[BUF:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CANON-NEXT: %[[v0:.*]] = enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_sync_1 (%[[REQ]], %[[BUF]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]>
// CANON-NEXT: return %[[v0]] : tensor<4xf32>

// -----

func.func @multiple(%req1 : !comm.mpi.request, %req2 : !comm.mpi.request, %buf1 : tensor<4xf32>, %buf2 : tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>) {
    %0:2 = comm.mpi.syncall requests = (%req1, %req2), buffers = (%buf1, %buf2) : tensor<4xf32>, tensor<8xi32>
    return %0#0, %0#1 : tensor<4xf32>, tensor<8xi32>
}

// JIT-LABEL: llvm.func @MPI_Waitall
// JIT-SAME:                        (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_sync_2
// JIT-SAME:                                                 (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}, %arg3: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}) attributes {enzymexla.comm} {
// JIT-NEXT:   %0 = llvm.mlir.constant(2 : i32) : i32
// JIT-NEXT:   %1 = llvm.alloca %0 x i64 : (i32) -> !llvm.ptr
// JIT-NEXT:   %2 = llvm.load %arg0 : !llvm.ptr -> i64
// JIT-NEXT:   %3 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, i64
// JIT-NEXT:   llvm.store %2, %3 : i64, !llvm.ptr
// JIT-NEXT:   %4 = llvm.load %arg1 : !llvm.ptr -> i64
// JIT-NEXT:   %5 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, i64
// JIT-NEXT:   llvm.store %4, %5 : i64, !llvm.ptr
// JIT-NEXT:   %6 = llvm.mlir.constant(0 : i64) : i64
// JIT-NEXT:   %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
// JIT-NEXT:   %8 = llvm.call @MPI_Waitall(%0, %1, %7) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @multiple
// JIT-SAME:                     (%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>, %[[BUF1:.*]]: tensor<4xf32>, %[[BUF2:.*]]: tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>) {
// JIT-NEXT: %[[v0:.*]]:2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_sync_2 (%[[REQ1]], %[[REQ2]], %[[BUF1]], %[[BUF2]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>]> {enzymexla.require_jit_ordering} : (tensor<i64>, tensor<i64>, tensor<4xf32>, tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>)
// JIT-NEXT: return %[[v0]]#0, %[[v0]]#1 : tensor<4xf32>, tensor<8xi32>

// CANON-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_sync_2
// CANON-SAME:                                                 (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]}, %arg3: !llvm.ptr {enzymexla.memory_effects = ["read", "write"]})
// CANON-LABEL: func.func @multiple
// CANON-SAME:                     (%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>, %[[BUF1:.*]]: tensor<4xf32>, %[[BUF2:.*]]: tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>) {
// CANON-NEXT: %[[v0:.*]]:2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_sync_2 (%[[REQ1]], %[[REQ2]], %[[BUF1]], %[[BUF2]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 3, operand_tuple_indices = []>]>
// CANON-NEXT: return %[[v0]]#0, %[[v0]]#1 : tensor<4xf32>, tensor<8xi32>

// JIT32-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_sync_2
// JIT32:         %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
// JIT32-NEXT:    %2 = llvm.load %arg0 : !llvm.ptr -> i32
// JIT32-LABEL: func.func @multiple
// JIT32-SAME:                     (%[[REQ1:.*]]: tensor<i32>, %[[REQ2:.*]]: tensor<i32>, %[[BUF1:.*]]: tensor<4xf32>, %[[BUF2:.*]]: tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>) {
// JIT32-NEXT: %{{.*}}:2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_sync_2 (%[[REQ1]], %[[REQ2]], %[[BUF1]], %[[BUF2]]) {{.*}} : (tensor<i32>, tensor<i32>, tensor<4xf32>, tensor<8xi32>) -> (tensor<4xf32>, tensor<8xi32>)
