// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main() -> tensor<i32> {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  %rank = comm.nccl.comm_user_rank %comm : tensor<i32>
  return %rank : tensor<i32>
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclCommUserRank
// CHECK-SAME: (%[[WRAP_COMM:[^:]*]]: !llvm.ptr, %[[WRAP_RANK:[^:]*]]: !llvm.ptr)
// CHECK-SAME: attributes {enzymexla.comm}
// CHECK: %[[STREAM:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// CHECK: %[[HOST_COMM:.*]] = llvm.alloca {{.*}} x !llvm.ptr
// CHECK: %[[HOST_RANK:.*]] = llvm.alloca {{.*}} x i32
// CHECK: %[[COMM_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_COMM]], %[[WRAP_COMM]], %[[COMM_BYTES]], %[[STREAM]])
// CHECK: llvm.call @cuStreamSynchronize(%[[STREAM]])
// CHECK: %[[COMM:.*]] = llvm.load %[[HOST_COMM]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @ncclCommUserRank(%[[COMM]], %[[HOST_RANK]])
// CHECK: %[[RANK_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[WRAP_RANK]], %[[HOST_RANK]], %[[RANK_BYTES]], %[[STREAM]])
// CHECK-NEXT: llvm.return
// CHECK-DAG: llvm.func @cuMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @cuStreamSynchronize(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclCommUserRank(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @main
// CHECK: %[[COMM_ARG:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: %[[PLACEHOLDER:.*]] = stablehlo.constant dense<-1> : tensor<i32>
// CHECK: %[[RESULT:.*]] = enzymexla.jit_call @enzymexla_jitwrap_ncclCommUserRank (%[[COMM_ARG]], %[[PLACEHOLDER]])
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>]
// CHECK-SAME: xla_side_effect_free
// CHECK-SAME: : (tensor<i64>, tensor<i32>) -> tensor<i32>
// CHECK: return %[[RESULT]] : tensor<i32>
// CHECK-NOT: comm.nccl
