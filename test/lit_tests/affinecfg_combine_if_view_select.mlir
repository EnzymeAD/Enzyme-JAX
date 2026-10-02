// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

#set = affine_set<()[s0] : (s0 - 1 >= 0)>

// A pure if choosing between pointers is a select between views; it is not
// merged into the preceding if whose arms store.
func.func @keep_view_select(%n: index, %a: !llvm.ptr, %b: !llvm.ptr, %out: memref<f64>) {
  %cst = arith.constant 1.0 : f64
  affine.if #set()[%n] {
    affine.store %cst, %out[] : memref<f64>
  }
  %p = affine.if #set()[%n] -> !llvm.ptr {
    affine.yield %a : !llvm.ptr
  } else {
    affine.yield %b : !llvm.ptr
  }
  llvm.store %cst, %p : f64, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @keep_view_select(
// CHECK: affine.if #{{.+}}()[%{{.+}}] {
// CHECK-NEXT: affine.store
// CHECK-NEXT: }
// CHECK-NEXT: %[[p:.+]] = affine.if #{{.+}}()[%{{.+}}] -> !llvm.ptr {
// CHECK: llvm.store %{{.+}}, %[[p]]

// -----

#set = affine_set<()[s0] : (s0 - 1 >= 0)>

// A scalar choice merges into the if with effects as before.
func.func @merge_scalar_select(%n: index, %a: f64, %b: f64, %out: memref<f64>, %out2: memref<f64>) {
  %cst = arith.constant 1.0 : f64
  affine.if #set()[%n] {
    affine.store %cst, %out[] : memref<f64>
  }
  %v = affine.if #set()[%n] -> f64 {
    affine.yield %a : f64
  } else {
    affine.yield %b : f64
  }
  affine.store %v, %out2[] : memref<f64>
  return
}

// CHECK-LABEL: func.func @merge_scalar_select(
// CHECK: affine.if
// CHECK-NOT: affine.if

// -----

#set = affine_set<()[s0] : (s0 - 1 >= 0)>

// Two pure pointer choices on the same condition still merge.
func.func @merge_pure_views(%n: index, %a: !llvm.ptr, %b: !llvm.ptr, %c: !llvm.ptr, %d: !llvm.ptr) -> (!llvm.ptr, !llvm.ptr) {
  %p = affine.if #set()[%n] -> !llvm.ptr {
    affine.yield %a : !llvm.ptr
  } else {
    affine.yield %b : !llvm.ptr
  }
  %q = affine.if #set()[%n] -> !llvm.ptr {
    affine.yield %c : !llvm.ptr
  } else {
    affine.yield %d : !llvm.ptr
  }
  return %p, %q : !llvm.ptr, !llvm.ptr
}

// CHECK-LABEL: func.func @merge_pure_views(
// CHECK: affine.if
// CHECK-NOT: affine.if
