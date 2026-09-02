// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A branch choosing between scratch buffers, read back through a pointer: the
// access names neither the branch's result nor a view of it, but a view of a
// pointer taken of it. Each access moves into the arms, onto that arm's own
// buffer, so the branch is left with nothing to raise -- which is what lets
// the function raise at all, since scratch that is written elsewhere can
// never be selected between as whole tensors.

// CHECK-LABEL: @pick_scratch
// CHECK-NOT: affine.if
// CHECK-NOT: enzymexla.memref2pointer

#set = affine_set<(d0) : (d0 - 2 == 0)>
module {
  func.func private @pick_scratch(%out: memref<4xf64, 1>, %in: memref<4xf64, 1>) {
    %a = memref.alloca() : memref<4xf64>
    %b = memref.alloca() : memref<4xf64>
    affine.parallel (%t) = (0) to (4) {
      %v = affine.load %in[%t] : memref<4xf64, 1>
      affine.store %v, %a[%t] : memref<4xf64>
      affine.store %v, %b[%t] : memref<4xf64>
      %sel = affine.if #set(%t) -> memref<4xf64> {
        affine.yield %a : memref<4xf64>
      } else {
        affine.yield %b : memref<4xf64>
      }
      %p = "enzymexla.memref2pointer"(%sel) : (memref<4xf64>) -> !llvm.ptr
      %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
      %ld = affine.load %m[%t] : memref<?xf64>
      affine.store %ld, %out[%t] : memref<4xf64, 1>
    }
    return
  }
}
