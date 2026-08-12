// RUN: enzymexlamlir-opt %s --llvm-to-affine-access -split-input-file | FileCheck %s

// A truncation that drops set bits is not a cast to see through: CUDA packs a
// dim3's x and y into one i64 and truncates to recover x, and taking the packed
// value in its place indexes at y*2^32 + x. The raised load has to keep reading
// what the unraised store writes.
module {
  func.func @packed(%lo: i32, %hi: i32, %p: !llvm.ptr, %q: !llvm.ptr) {
    %c32 = arith.constant 32 : i64
    %0 = arith.extui %lo : i32 to i64
    %1 = arith.extui %hi : i32 to i64
    %2 = arith.shli %1, %c32 overflow<nuw> : i64
    %3 = arith.ori %2, %0 {isDisjoint} : i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = arith.index_cast %4 : i32 to index
    affine.parallel (%i) = (0) to (10) {
      %6 = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
      %7 = arith.index_castui %5 : index to i32
      %8 = arith.index_castui %i : index to i32
      %9 = arith.addi %8, %7 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %6[%10] : memref<?xf64>
      %12 = "enzymexla.pointer2memref"(%q) : (!llvm.ptr) -> memref<?xf64>
      memref.store %11, %12[%10] : memref<?xf64>
    }
    return
  }
}

// The packed word must not become an index of its own: every symbol standing
// for x has to come from the truncation.
// CHECK-LABEL: func.func @packed
// CHECK: %[[PACKED:.+]] = arith.ori
// CHECK-NOT: arith.index_cast %[[PACKED]] : i64 to index
// CHECK: arith.trunci %[[PACKED]] : i64 to i32
// CHECK: affine.apply

// -----

// A truncation the loop bounds show is exact is still worth seeing through.
module {
  func.func @small(%p: !llvm.ptr) {
    affine.parallel (%i) = (0) to (10) {
      %0 = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
      %1 = arith.index_castui %i : index to i64
      %2 = arith.trunci %1 : i64 to i16
      %3 = arith.index_cast %2 : i16 to index
      %4 = memref.load %0[%3] : memref<?xf64>
      memref.store %4, %0[%3] : memref<?xf64>
    }
    return
  }
}

// CHECK-LABEL: func.func @small
// CHECK: affine.parallel (%[[IV:.+]]) =
// CHECK: memref.load %{{.+}}[%[[IV]]]
