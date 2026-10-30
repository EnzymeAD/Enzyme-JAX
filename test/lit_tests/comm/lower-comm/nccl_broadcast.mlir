// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main(%buf: tensor<5xf32>, %root: tensor<i32>) -> tensor<5xf32> {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  %result = comm.nccl.broadcast %buf, %root, %comm : (tensor<5xf32>, tensor<i32>, !comm.nccl.comm) -> tensor<5xf32>
  return %result : tensor<5xf32>
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclBroadcast
// CHECK-SAME: (%[[WRAP_BUF:[^:]*]]: !llvm.ptr
// CHECK-SAME: attributes {enzymexla.comm}
// CHECK: %[[STREAM:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// CHECK: %[[HOST_COUNT:.*]] = llvm.alloca {{.*}} x i64
// CHECK: %[[HOST_DATATYPE:.*]] = llvm.alloca {{.*}} x i32
// CHECK: %[[HOST_ROOT:.*]] = llvm.alloca {{.*}} x i32
// CHECK: %[[HOST_COMM:.*]] = llvm.alloca {{.*}} x !llvm.ptr
// CHECK: %[[COUNT_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_COUNT]], {{.*}}, %[[COUNT_BYTES]], %[[STREAM]])
// CHECK: %[[DATATYPE_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_DATATYPE]], {{.*}}, %[[DATATYPE_BYTES]], %[[STREAM]])
// CHECK: %[[ROOT_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_ROOT]], {{.*}}, %[[ROOT_BYTES]], %[[STREAM]])
// CHECK: %[[COMM_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_COMM]], {{.*}}, %[[COMM_BYTES]], %[[STREAM]])
// CHECK: llvm.call @cuStreamSynchronize(%[[STREAM]])
// CHECK: %[[COUNT:.*]] = llvm.load %[[HOST_COUNT]] : !llvm.ptr -> i64
// CHECK: %[[DATATYPE:.*]] = llvm.load %[[HOST_DATATYPE]] : !llvm.ptr -> i32
// CHECK: %[[ROOT:.*]] = llvm.load %[[HOST_ROOT]] : !llvm.ptr -> i32
// CHECK: %[[WRAP_COMM:.*]] = llvm.load %[[HOST_COMM]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @ncclBroadcast(%[[WRAP_BUF]], %[[WRAP_BUF]], %[[COUNT]], %[[DATATYPE]], %[[ROOT]], %[[WRAP_COMM]], %[[STREAM]])
// CHECK: llvm.return
// CHECK-DAG: llvm.func @cuMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @cuStreamSynchronize(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclBroadcast(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[BUF:.*]]: tensor<5xf32>, %[[ROOT_ARG:.*]]: tensor<i32>) -> tensor<5xf32> {
// CHECK: %[[COMM:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: %[[COUNT_ARG:.*]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK: %[[DATATYPE_ARG:.*]] = stablehlo.constant {{.*}} : tensor<i32>
// CHECK: %[[RESULT:.*]] = enzymexla.jit_call @enzymexla_jitwrap_ncclBroadcast (%[[BUF]], %[[COUNT_ARG]], %[[DATATYPE_ARG]], %[[ROOT_ARG]], %[[COMM]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<5xf32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>) -> tensor<5xf32>
// CHECK: return %[[RESULT]] : tensor<5xf32>
// CHECK-NOT: comm.nccl
