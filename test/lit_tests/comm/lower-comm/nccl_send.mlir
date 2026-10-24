// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main(%buf: tensor<5xi32>, %peer: tensor<i32>) {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  comm.nccl.send %buf, %peer, %comm : tensor<5xi32>, tensor<i32>, !comm.nccl.comm
  return
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclSend
// CHECK-SAME: enzymexla.device_abi = "cuda"
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
// CHECK: llvm.call @ncclSend({{.*}}, %[[COUNT]], %[[DATATYPE]], %[[PEER]], %[[WRAP_COMM]], %[[STREAM]])
// CHECK: llvm.return
// CHECK-DAG: llvm.func @cuMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @cuStreamSynchronize(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclSend(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[BUF:.*]]: tensor<5xi32>, %[[PEER_ARG:.*]]: tensor<i32>) {
// CHECK: %[[COMM:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: %[[COUNT_ARG:.*]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK: %[[DATATYPE_ARG:.*]] = stablehlo.constant {{.*}} : tensor<i32>
// CHECK: enzymexla.jit_call @enzymexla_jitwrap_ncclSend (%[[BUF]], %[[COUNT_ARG]], %[[DATATYPE_ARG]], %[[PEER_ARG]], %[[COMM]]) : (tensor<5xi32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>) -> ()
// CHECK-NOT: comm.nccl.get_comm_world
