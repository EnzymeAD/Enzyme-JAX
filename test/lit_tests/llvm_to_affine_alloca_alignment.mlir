// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --allow-unregistered-dialect --split-input-file | FileCheck %s

// An llvm.alloca is often over-aligned past its element type -- a stack array
// gets 16 for vectorization. Converting it to a memref.alloca must carry that
// alignment: without it the memref falls back to the element's natural
// alignment (4 for i32), and lowered back the under-aligned slot shifts the
// frame, leaving an adjacent aligned alloca on a misaligned address (an
// aligned SSE load there faults -- MFEM's NCMesh::CountSplits did).

// CHECK-LABEL: func.func @over_aligned
func.func @over_aligned() {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %a = llvm.alloca %c1 x !llvm.array<12 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  %m = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
  %i = arith.constant 0 : index
  %v = memref.load %m[%i] : memref<?xi32>
  "test.use"(%v) : (i32) -> ()
  return
}

// CHECK: memref.alloca() {alignment = 16 : i64} : memref<12xi32>

// -----

// A natural-alignment alloca carries no alignment attribute, so none is added.

// CHECK-LABEL: func.func @natural_aligned
func.func @natural_aligned() {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %a = llvm.alloca %c1 x !llvm.array<12 x i32> : (i32) -> !llvm.ptr
  %m = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi32>
  %i = arith.constant 0 : index
  %v = memref.load %m[%i] : memref<?xi32>
  "test.use"(%v) : (i32) -> ()
  return
}

// CHECK: memref.alloca() : memref<12xi32>
// CHECK-NOT: alignment
