// RUN: enzymexlamlir-opt %s -sort-block-memory | FileCheck %s

// A run of (load, add, store) triples to disjoint locations, as remove-atomics
// leaves behind. Emitted in place each load waits on the previous store; the
// loads should all be hoisted so they can be in flight together, and their
// original relative order preserved.
// CHECK-LABEL: @disjoint
// CHECK: affine.load %arg0[%arg2 + 1]
// CHECK-NEXT: affine.load %arg0[%arg2 + 2]
// CHECK-NEXT: affine.load %arg0[%arg2 + 3]
// CHECK: affine.store {{.*}}, %arg0[%arg2 + 1]
// CHECK-NEXT: affine.store {{.*}}, %arg0[%arg2 + 2]
// CHECK-NEXT: affine.store {{.*}}, %arg0[%arg2 + 3]
func.func @disjoint(%m: memref<100xf32>, %v: f32) {
  affine.parallel (%i) = (0) to (10) {
    %0 = affine.load %m[%i + 1] : memref<100xf32>
    %1 = arith.addf %0, %v : f32
    affine.store %1, %m[%i + 1] : memref<100xf32>
    %2 = affine.load %m[%i + 2] : memref<100xf32>
    %3 = arith.addf %2, %v : f32
    affine.store %3, %m[%i + 2] : memref<100xf32>
    %4 = affine.load %m[%i + 3] : memref<100xf32>
    %5 = arith.addf %4, %v : f32
    affine.store %5, %m[%i + 3] : memref<100xf32>
  }
  return
}

// A load must not be hoisted above a store it may overlap: every access here is
// to the same element, so the original order has to survive.
// CHECK-LABEL: @overlapping
// CHECK: affine.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: affine.store
// CHECK-NEXT: affine.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: affine.store
func.func @overlapping(%m: memref<100xf32>, %v: f32) {
  affine.parallel (%i) = (0) to (10) {
    %0 = affine.load %m[%i] : memref<100xf32>
    %1 = arith.addf %0, %v : f32
    affine.store %1, %m[%i] : memref<100xf32>
    %2 = affine.load %m[%i] : memref<100xf32>
    %3 = arith.addf %2, %v : f32
    affine.store %3, %m[%i] : memref<100xf32>
  }
  return
}

// A load may not be hoisted above the op defining the memref it reads, even
// when that op is the very first one in the block.
// CHECK-LABEL: @operand_defined_in_block
// CHECK: %[[M0:.+]] = "enzymexla.pointer2memref"(%arg0)
// CHECK-NEXT: affine.load %[[M0]]
// CHECK: %[[M1:.+]] = "enzymexla.pointer2memref"(%arg1)
// CHECK: affine.load %[[M1]]
func.func @operand_defined_in_block(%p0: !llvm.ptr, %p1: !llvm.ptr, %v: f32) {
  %m0 = "enzymexla.pointer2memref"(%p0) : (!llvm.ptr) -> memref<?xf32>
  %0 = affine.load %m0[1] : memref<?xf32>
  %1 = arith.addf %0, %v : f32
  affine.store %1, %m0[1] : memref<?xf32>
  %m1 = "enzymexla.pointer2memref"(%p1) : (!llvm.ptr) -> memref<?xf32>
  %2 = affine.load %m1[2] : memref<?xf32>
  %3 = arith.addf %2, %v : f32
  affine.store %3, %m1[2] : memref<?xf32>
  return
}

// Distinct buffers never overlap whatever the indices are, so the second load
// hoists above the first store.
// CHECK-LABEL: @distinct_memrefs
// CHECK: affine.load %arg0
// CHECK-NEXT: affine.load %arg1
func.func @distinct_memrefs(%a: memref<100xf32>, %b: memref<100xf32>, %v: f32) {
  affine.parallel (%i) = (0) to (10) {
    %0 = affine.load %a[%i] : memref<100xf32>
    %1 = arith.addf %0, %v : f32
    affine.store %1, %a[%i] : memref<100xf32>
    %2 = affine.load %b[%i] : memref<100xf32>
    %3 = arith.addf %2, %v : f32
    affine.store %3, %b[%i] : memref<100xf32>
  }
  return
}
