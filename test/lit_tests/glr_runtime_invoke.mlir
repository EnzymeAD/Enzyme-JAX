// RUN: enzymexlamlir-opt %s --gpu-launch-recognition | FileCheck %s

// With exception handling preserved, CUDA runtime calls arrive in invoke
// form; they cannot throw, so they normalize to calls the runtime rewrites
// see — here an invoked cudaMalloc still becomes a gpu.alloc.
module {
  llvm.func @cudaMalloc(!llvm.ptr, i64) -> i32
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func @alloc(%slot: !llvm.ptr, %n: i64) -> i32 attributes {personality = @__gxx_personality_v0} {
    %e = llvm.invoke @cudaMalloc(%slot, %n) to ^ok unwind ^lp : (!llvm.ptr, i64) -> i32
  ^ok:
    llvm.return %e : i32
  ^lp:
    %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.return %one : i32
  }
}

// CHECK-LABEL: llvm.func @alloc(
// CHECK-NOT: llvm.invoke @cudaMalloc
// CHECK: %[[M:.+]] = gpu.alloc (%{{.+}}) : memref<?xi8, 1>
// CHECK: %[[P:.+]] = "enzymexla.memref2pointer"(%[[M]])
// CHECK: llvm.store %[[P]], %arg0
// CHECK: %[[Z:.+]] = llvm.mlir.zero : i32
// CHECK: llvm.br ^[[OK:.+]]{{$}}
// CHECK: ^[[OK]]:
// CHECK: llvm.return %[[Z]] : i32
