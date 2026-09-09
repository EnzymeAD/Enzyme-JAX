// RUN: enzymexlamlir-opt %s --canonicalize-parallel | FileCheck %s

// A coefficient ternary that steps the pointer on one side only: the arm
// yielding the base itself stepped zero elements of the other's type.
// CHECK-LABEL: func.func @affine_if_bare_then
// CHECK-SAME: (%[[p:.+]]: !llvm.ptr<1>, %[[n:.+]]: index)
// CHECK: %[[i:.+]] = affine.if #{{.+}}()[%[[n]]] -> i64 {
// CHECK-NEXT:   affine.yield %c0_i64 : i64
// CHECK-NEXT: } else {
// CHECK-NEXT:   affine.yield %c16_i64 : i64
// CHECK-NEXT: }
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr inbounds %[[p]][%[[i]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
// CHECK-NEXT: return %[[g]]
func.func @affine_if_bare_then(%p: !llvm.ptr<1>, %n: index) -> !llvm.ptr<1> {
  %sel = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%n] -> !llvm.ptr<1> {
    affine.yield %p : !llvm.ptr<1>
  } else {
    %g = llvm.getelementptr inbounds %p[16] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
    affine.yield %g : !llvm.ptr<1>
  }
  return %sel : !llvm.ptr<1>
}

// The gep side may index dynamically; the bare side then chooses a zero of
// that index's type.
// CHECK-LABEL: func.func @scf_if_bare_else
// CHECK-SAME: (%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1, %[[k:.+]]: i32)
// CHECK: %[[i:.+]] = arith.select %[[c]], %[[k]], %c0_i32 : i32
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %[[p]][%[[i]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
// CHECK-NEXT: return %[[g]]
func.func @scf_if_bare_else(%p: !llvm.ptr, %c: i1, %k: i32) -> !llvm.ptr {
  %sel = scf.if %c -> !llvm.ptr {
    %g = llvm.getelementptr %p[%k] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    scf.yield %g : !llvm.ptr
  } else {
    scf.yield %p : !llvm.ptr
  }
  return %sel : !llvm.ptr
}

// CHECK-LABEL: func.func @select_bare
// CHECK-SAME: (%[[p:.+]]: !llvm.ptr, %[[c:.+]]: i1)
// CHECK: %[[i:.+]] = arith.select %[[c]], %c0_i64, %c3_i64 : i64
// CHECK-NEXT: %[[g:.+]] = llvm.getelementptr %[[p]][%[[i]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT: return %[[g]]
func.func @select_bare(%p: !llvm.ptr, %c: i1) -> !llvm.ptr {
  %g = llvm.getelementptr %p[3] : (!llvm.ptr) -> !llvm.ptr, f64
  %sel = arith.select %c, %p, %g : !llvm.ptr
  return %sel : !llvm.ptr
}

// Two different pointers are not a base and a step off it.
// CHECK-LABEL: func.func @select_unrelated
// CHECK: arith.select %{{.+}}, %{{.+}}, %{{.+}} : !llvm.ptr
func.func @select_unrelated(%p: !llvm.ptr, %q: !llvm.ptr, %c: i1) -> !llvm.ptr {
  %g = llvm.getelementptr %q[3] : (!llvm.ptr) -> !llvm.ptr, f64
  %sel = arith.select %c, %p, %g : !llvm.ptr
  return %sel : !llvm.ptr
}
