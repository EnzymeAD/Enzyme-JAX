// RUN: enzymexlamlir-opt %s --parallel-serialization | FileCheck %s

// Merging identical blocks threads the differing values through new block
// arguments on every predecessor's terminator. An llvm.invoke is such a
// terminator, and its successor operands only take LLVM types -- handing it
// the index that distinguishes these two stores is not a merge but a
// verifier error. The pass's greedy driver must leave the blocks alone.

module {
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func @maythrow()

  llvm.func @unmerged(%p: !llvm.ptr, %c: i1) attributes {personality = @__gxx_personality_v0} {
    %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<4xf32>
    %cst = arith.constant 1.000000e+00 : f32
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    llvm.cond_br %c, ^direct, ^inv
  ^inv:
    llvm.invoke @maythrow() to ^a unwind ^lp : () -> ()
  ^a:
    memref.store %cst, %m[%i0] : memref<4xf32>
    llvm.return
  ^direct:
    memref.store %cst, %m[%i1] : memref<4xf32>
    llvm.return
  ^lp:
    %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.resume %1 : !llvm.struct<(ptr, i32)>
  }
}

// Both stores survive in their own blocks, and the invoke still carries no
// successor operands.
// CHECK-LABEL: llvm.func @unmerged
// CHECK:         llvm.invoke @maythrow() to ^{{.+}} unwind
// CHECK:         memref.store %{{.+}}, %{{.+}}[%[[A:.+]]] : memref<4xf32>
// CHECK:         memref.store %{{.+}}, %{{.+}}[%[[B:.+]]] : memref<4xf32>
// CHECK:         llvm.landingpad
