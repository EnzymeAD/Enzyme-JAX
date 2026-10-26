// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s
#set = affine_set<()[s0] : (s0 - 1 >= 0)>

// A result nothing reads is not yielded, which is what lets a pointer the if
// no longer needs disappear.
func.func @dead_if_result(%a: !llvm.ptr, %m: memref<?xf64>, %n: index) -> f64 {
  %r:2 = affine.if #set()[%n] -> (!llvm.ptr, f64) {
    %v = affine.load %m[0] : memref<?xf64>
    affine.yield %a, %v : !llvm.ptr, f64
  } else {
    %v = affine.load %m[1] : memref<?xf64>
    affine.yield %a, %v : !llvm.ptr, f64
  }
  return %r#1 : f64
}

// CHECK-LABEL: func.func @dead_if_result(
// CHECK-NOT: !llvm.ptr, f64
// CHECK: %[[R:.+]] = affine.if #set{{.*}} -> f64 {
// CHECK: affine.yield %{{.+}} : f64
// CHECK: return %[[R]] : f64
