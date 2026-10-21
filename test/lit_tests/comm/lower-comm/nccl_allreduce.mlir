// RUN: enzymexlamlir-opt --lower-comm-to-jit="ncclCommPtr=4660" %s | FileCheck %s --check-prefix=JIT
// RUN: not enzymexlamlir-opt --lower-comm-to-jit %s 2>&1 | FileCheck %s --check-prefix=NO-COMM

func.func @main(%buf: tensor<5xi32>) -> tensor<5xi32> {
  %comm = comm.nccl.get_comm : !comm.nccl.comm
  %result = comm.nccl.allreduce %buf, #comm.nccl.red_op<ncclSum>, %comm : (tensor<5xi32>, !comm.nccl.comm) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

// JIT-LABEL: llvm.func @enzymexla_jitwrap_ncclAllReduce
// JIT: %[[STREAM:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// JIT: %[[HOST_COUNT:.*]] = llvm.alloca
// JIT: %[[HOST_DATATYPE:.*]] = llvm.alloca
// JIT: %[[HOST_REDOP:.*]] = llvm.alloca
// JIT: %[[HOST_COMM:.*]] = llvm.alloca
// JIT: %[[COUNT_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// JIT: llvm.call @cudaMemcpyAsync(%[[HOST_COUNT]], {{.*}}, %[[COUNT_BYTES]], {{.*}}, %[[STREAM]])
// JIT: %[[REDOP_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// JIT: llvm.call @cudaMemcpyAsync(%[[HOST_REDOP]], {{.*}}, %[[REDOP_BYTES]], {{.*}}, %[[STREAM]])
// JIT: %[[DATATYPE_BYTES:.*]] = llvm.mlir.constant(4 : i64) : i64
// JIT: llvm.call @cudaMemcpyAsync(%[[HOST_DATATYPE]], {{.*}}, %[[DATATYPE_BYTES]], {{.*}}, %[[STREAM]])
// JIT: %[[COMM_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// JIT: llvm.call @cudaMemcpyAsync(%[[HOST_COMM]], {{.*}}, %[[COMM_BYTES]], {{.*}}, %[[STREAM]])
// JIT: llvm.call @cuStreamSynchronize(%[[STREAM]])
// JIT: %[[COUNT:.*]] = llvm.load %[[HOST_COUNT]] : !llvm.ptr -> i64
// JIT: %[[DATATYPE:.*]] = llvm.load %[[HOST_DATATYPE]] : !llvm.ptr -> i32
// JIT: %[[REDOP:.*]] = llvm.load %[[HOST_REDOP]] : !llvm.ptr -> i32
// JIT: %[[WRAP_COMM:.*]] = llvm.load %[[HOST_COMM]] : !llvm.ptr -> !llvm.ptr
// JIT: llvm.call @ncclAllReduce({{.*}}, %[[COUNT]], %[[DATATYPE]], %[[REDOP]], %[[WRAP_COMM]], %[[STREAM]])
// JIT: llvm.return
// JIT-DAG: llvm.func @cudaMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr) -> i32
// JIT-DAG: llvm.func @cuStreamSynchronize(!llvm.ptr) -> i32
// JIT-DAG: llvm.func @ncclAllReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: func.func @main
// JIT-SAME: (%[[BUF:.*]]: tensor<5xi32>) -> tensor<5xi32> {
// JIT: %[[COMM:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// JIT: stablehlo.constant dense<5> : tensor<i64>
// JIT: enzymexla.jit_call @enzymexla_jitwrap_ncclAllReduce ({{.*}}%[[COMM]])
// JIT-NOT: comm.nccl.get_comm

// NO-COMM: 'comm.nccl.get_comm' op requires a nonzero ncclCommPtr pass option
