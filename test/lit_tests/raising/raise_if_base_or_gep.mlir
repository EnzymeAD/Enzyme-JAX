// RUN: enzymexlamlir-opt %s --canonicalize-parallel --llvm-to-affine-access --canonicalize-parallel --raise-affine-to-stablehlo | FileCheck %s

// A coefficient ternary yields either a pointer or a step off it from an
// affine.if; the branch becomes a choice of index and the access raises.

// CHECK-LABEL: @ifptr_raised
// CHECK: stablehlo.select
// CHECK: stablehlo.gather
// CHECK-NOT: affine.if
// CHECK-NOT: llvm.getelementptr

module {
  func.func private @ifptr(%out: memref<16xf64, 1>, %in: memref<64xf64, 1>, %cbuf: memref<1xi32, 1>) {
    %n = affine.load %cbuf[0] : memref<1xi32, 1>
    %ni = arith.index_cast %n : i32 to index
    %p = "enzymexla.memref2pointer"(%in) : (memref<64xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%t) = (0) to (16) {
      %sel = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%ni] -> !llvm.ptr<1> {
        affine.yield %p : !llvm.ptr<1>
      } else {
        %g = llvm.getelementptr %p[16] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
        affine.yield %g : !llvm.ptr<1>
      }
      %view = "enzymexla.pointer2memref"(%sel) : (!llvm.ptr<1>) -> memref<?xf64, 1>
      %v = affine.load %view[%t] : memref<?xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
