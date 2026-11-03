// UNSUPPORTED: true
// RUN: enzymexlamlir-opt --map-symbol="symbols=Reactant_NCCL_COMM_WORLD=0x1234" --lower-comm-to-jit %s | FileCheck %s

func.func @main() {
  %comm = comm.nccl.get_comm_world : !comm.nccl.comm
  comm.nccl.barrier %comm : !comm.nccl.comm
  return
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclBarrier
// CHECK-SAME: (%[[WRAP_COMM:[^:]*]]: !llvm.ptr)
// CHECK-SAME: attributes {enzymexla.comm}
// CHECK: %[[STREAM:.*]] = "enzymexla.get_stream"() : () -> !llvm.ptr
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[HOST_COMM:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr
// CHECK: %[[COMM_BYTES:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @cuMemcpyAsync(%[[HOST_COMM]], %[[WRAP_COMM]], %[[COMM_BYTES]], %[[STREAM]])
// CHECK: llvm.call @cuStreamSynchronize(%[[STREAM]])
// CHECK: %[[COMM:.*]] = llvm.load %[[HOST_COMM]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[HOST_NRANKS:.*]] = llvm.alloca %[[ONE]] x i32
// CHECK: llvm.call @ncclCommCount(%[[COMM]], %[[HOST_NRANKS]])
// CHECK: %[[NRANKS:.*]] = llvm.load %[[HOST_NRANKS]] : !llvm.ptr -> i32
// CHECK: %[[DESCS:.*]] = llvm.alloca %[[NRANKS]] x !llvm.struct<(i32, i32, i32, i32)>
// CHECK: llvm.br ^[[FILL_COND:.*]](%[[ZERO]] : i32)
// CHECK: ^[[FILL_COND]](%[[FILL_PEER:.*]]: i32):
// CHECK: %[[FILL_IN_BOUNDS:.*]] = llvm.icmp "slt" %[[FILL_PEER]], %[[NRANKS]] : i32
// CHECK: llvm.cond_br %[[FILL_IN_BOUNDS]], ^[[FILL_LOOP:.*]], ^[[FILL_EXIT:.*]]
// CHECK: ^[[FILL_LOOP]]:
// CHECK: %[[OPCNT_PTR:.*]] = llvm.getelementptr %[[DESCS]][%[[FILL_PEER]], 0]
// CHECK: llvm.store %[[ONE]], %[[OPCNT_PTR]] : i32, !llvm.ptr
// CHECK: %[[PEER_PTR:.*]] = llvm.getelementptr %[[DESCS]][%[[FILL_PEER]], 1]
// CHECK: llvm.store %[[FILL_PEER]], %[[PEER_PTR]] : i32, !llvm.ptr
// CHECK: %[[SIGIDX_PTR:.*]] = llvm.getelementptr %[[DESCS]][%[[FILL_PEER]], 2]
// CHECK: llvm.store %[[ZERO]], %[[SIGIDX_PTR]] : i32, !llvm.ptr
// CHECK: %[[CTX_PTR:.*]] = llvm.getelementptr %[[DESCS]][%[[FILL_PEER]], 3]
// CHECK: llvm.store %[[ZERO]], %[[CTX_PTR]] : i32, !llvm.ptr
// CHECK-NOT: llvm.call
// CHECK: %[[FILL_NEXT:.*]] = llvm.add %[[FILL_PEER]], %[[ONE]] : i32
// CHECK: llvm.br ^[[FILL_COND]](%[[FILL_NEXT]] : i32)
// CHECK: ^[[FILL_EXIT]]:
// CHECK-NEXT: llvm.call @ncclGroupStart()
// CHECK-NEXT: llvm.br ^[[SIGNAL_COND:.*]](%[[ZERO]] : i32)
// CHECK: ^[[SIGNAL_COND]](%[[SIGNAL_PEER:.*]]: i32):
// CHECK: %[[SIGNAL_IN_BOUNDS:.*]] = llvm.icmp "slt" %[[SIGNAL_PEER]], %[[NRANKS]] : i32
// CHECK: llvm.cond_br %[[SIGNAL_IN_BOUNDS]], ^[[SIGNAL_LOOP:.*]], ^[[SIGNAL_EXIT:.*]]
// CHECK: ^[[SIGNAL_LOOP]]:
// CHECK-NEXT: llvm.call @ncclSignal(%[[SIGNAL_PEER]], %[[ZERO]], %[[ZERO]], %[[ZERO]], %[[COMM]], %[[STREAM]])
// CHECK-NEXT: %[[SIGNAL_NEXT:.*]] = llvm.add %[[SIGNAL_PEER]], %[[ONE]] : i32
// CHECK-NEXT: llvm.br ^[[SIGNAL_COND]](%[[SIGNAL_NEXT]] : i32)
// CHECK: ^[[SIGNAL_EXIT]]:
// CHECK-NEXT: llvm.call @ncclGroupEnd()
// CHECK: llvm.call @ncclWaitSignal(%[[NRANKS]], %[[DESCS]], %[[COMM]], %[[STREAM]])
// CHECK-NEXT: llvm.return
// CHECK-DAG: llvm.func @cuMemcpyAsync(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @cuStreamSynchronize(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclCommCount(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclGroupStart() -> i32
// CHECK-DAG: llvm.func @ncclGroupEnd() -> i32
// CHECK-DAG: llvm.func @ncclSignal(i32, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @ncclWaitSignal(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @main
// CHECK: %[[COMM_ARG:.*]] = stablehlo.constant dense<4660> : tensor<i64>
// CHECK: enzymexla.jit_call @enzymexla_jitwrap_ncclBarrier (%[[COMM_ARG]])
// CHECK-SAME: : (tensor<i64>) -> ()
// CHECK-NOT: comm.nccl
