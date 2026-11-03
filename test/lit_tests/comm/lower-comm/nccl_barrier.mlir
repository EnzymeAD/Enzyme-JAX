// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main() {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  comm.nccl.barrier %comm : !comm.nccl.comm
  return
}

// The barrier is an in-place allreduce on a scratch buffer, so it uses the
// same wrapper as comm.nccl.allreduce; see nccl_allreduce.mlir.
// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclAllReduce
// CHECK-SAME: (%[[WRAP_BUF:[^:]*]]: !llvm.ptr
// CHECK: llvm.call @ncclAllReduce(%[[WRAP_BUF]], %[[WRAP_BUF]],
// CHECK-DAG: llvm.func @ncclAllReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @main
// CHECK: %[[COMM:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: %[[SCRATCH:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: %[[REDOP:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK: %[[COUNT:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK: %[[DATATYPE:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK: enzymexla.jit_call @enzymexla_jitwrap_ncclAllReduce (%[[SCRATCH]], %[[COUNT]], %[[DATATYPE]], %[[REDOP]], %[[COMM]])
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
// CHECK-NOT: xla_side_effect_free
// CHECK-SAME: : (tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>) -> tensor<i32>
// CHECK-NEXT: return
// CHECK-NOT: comm.nccl
