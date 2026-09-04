// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s

#set = affine_set<()[s0] : (s0 - 1 >= 0)>

// A load through a pointer an affine.if chose becomes the if choosing between
// the loads, so no pointer crosses the if.
func.func @load_of_if_ptr(%a: !llvm.ptr, %b: !llvm.ptr, %n: index, %i: index) -> f64 {
  %p = affine.if #set()[%n] -> !llvm.ptr {
    affine.yield %a : !llvm.ptr
  } else {
    affine.yield %b : !llvm.ptr
  }
  %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
  %v = affine.load %m[symbol(%i)] : memref<?xf64>
  return %v : f64
}

// CHECK-LABEL: func.func @load_of_if_ptr(
// CHECK-SAME: %[[A:[a-z0-9]+]]: !llvm.ptr, %[[B:[a-z0-9]+]]: !llvm.ptr
// CHECK-NOT: !llvm.ptr, f64
// CHECK: %[[R:.+]] = affine.if #set{{.*}} -> f64 {
// CHECK: %[[MA:.+]] = "enzymexla.pointer2memref"(%[[A]])
// CHECK: %[[VA:.+]] = affine.load %[[MA]][symbol(%{{.+}})]
// CHECK: affine.yield %[[VA]] : f64
// CHECK: } else {
// CHECK: %[[MB:.+]] = "enzymexla.pointer2memref"(%[[B]])
// CHECK: %[[VB:.+]] = affine.load %[[MB]][symbol(%{{.+}})]
// CHECK: affine.yield %[[VB]] : f64
// CHECK: }
// CHECK: return %[[R]] : f64
