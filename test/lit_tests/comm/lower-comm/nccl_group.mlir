// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s

func.func @main() {
  comm.nccl.group_start
  comm.nccl.group_end
  return
}

// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclGroupEnd() attributes {enzymexla.device_abi = "cuda"} {
// CHECK-NEXT: %{{.*}} = llvm.call @ncclGroupEnd() : () -> i32
// CHECK-NEXT: llvm.return
// CHECK-LABEL: llvm.func @enzymexla_jitwrap_ncclGroupStart() attributes {enzymexla.device_abi = "cuda"} {
// CHECK-NEXT: %{{.*}} = llvm.call @ncclGroupStart() : () -> i32
// CHECK-NEXT: llvm.return
// CHECK-LABEL: func.func @main() {
// CHECK-NEXT: enzymexla.jit_call @enzymexla_jitwrap_ncclGroupStart () : () -> ()
// CHECK-NEXT: enzymexla.jit_call @enzymexla_jitwrap_ncclGroupEnd () : () -> ()
// CHECK-NEXT: return
