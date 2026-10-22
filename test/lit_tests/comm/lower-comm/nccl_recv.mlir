// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main(%peer: tensor<i32>) -> tensor<5xi32> {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  %result = comm.nccl.recv %peer, %comm : (tensor<i32>, !comm.nccl.comm) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclRecv
// CHECK: %[[STREAM:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// CHECK: %[[HOST_COUNT:.*]] = llvm.alloca {{.*}} x i64
// CHECK: %[[HOST_DATATYPE:.*]] = llvm.alloca {{.*}} x i32
// CHECK: %[[HOST_PEER:.*]] = llvm.alloca {{.*}} x i32
// CHECK: %[[HOST_COMM:.*]] = llvm.alloca {{.*}} x !llvm.ptr
// CHECK: %[[COUNT_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_COUNT]], {{.*}}, %[[COUNT_BYTES]], {{.*}}, %[[STREAM]])
// CHECK: %[[DATATYPE_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_DATATYPE]], {{.*}}, %[[DATATYPE_BYTES]], {{.*}}, %[[STREAM]])
// CHECK: %[[PEER_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_PEER]], {{.*}}, %[[PEER_BYTES]], {{.*}}, %[[STREAM]])
// CHECK: %[[COMM_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_COMM]], {{.*}}, %[[COMM_BYTES]], {{.*}}, %[[STREAM]])
// CHECK: llvm.call @cuStreamSynchronize(%[[STREAM]])
// CHECK: %[[COUNT:.*]] = llvm.load %[[HOST_COUNT]] : !llvm.ptr -> i64
// CHECK: %[[DATATYPE:.*]] = llvm.load %[[HOST_DATATYPE]] : !llvm.ptr -> i32
// CHECK: %[[PEER:.*]] = llvm.load %[[HOST_PEER]] : !llvm.ptr -> i32
// CHECK: %[[WRAP_COMM:.*]] = llvm.load %[[HOST_COMM]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @ncclRecv({{.*}}, %[[COUNT]], %[[DATATYPE]], %[[PEER]], %[[WRAP_COMM]], %[[STREAM]])
// CHECK: llvm.return
// CHECK-DAG: llvm.func @cuMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @cuStreamSynchronize(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclRecv(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[PEER_ARG:.*]]: tensor<i32>) -> tensor<5xi32> {
// CHECK: %[[COMM:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: %[[BUFFER:.*]] = stablehlo.constant dense<0> : tensor<5xi32>
// CHECK: %[[COUNT_ARG:.*]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK: %[[DATATYPE_ARG:.*]] = stablehlo.constant {{.*}} : tensor<i32>
// CHECK: %[[RESULT:.*]] = enzymexla.jit_call @enzymexla_jitwrap_ncclRecv (%[[BUFFER]], %[[COUNT_ARG]], %[[DATATYPE_ARG]], %[[PEER_ARG]], %[[COMM]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<5xi32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>) -> tensor<5xi32>
// CHECK: return %[[RESULT]] : tensor<5xi32>
// CHECK-NOT: comm.nccl.get_comm_world
