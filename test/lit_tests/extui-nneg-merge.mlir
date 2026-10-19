// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

// Merging extui(extui(x)) must not keep the outer op's nneg flag: the flag
// asserts the SOURCE is non-negative as a signed value, and the merge changes
// the source. extui nneg i8->i32 over extui i1->i8 is satisfied for every
// bool byte, but the merged extui nneg i1->i32 is poison whenever the bool
// is true -- LLVM then folds the bool to false wherever the merged value
// flows, which is how MFEM's trialHcurl=true kernels lost their loop bounds.

func.func @merge(%b: i1) -> i32 {
  %w = arith.extui %b : i1 to i8
  %r = arith.extui %w nneg : i8 to i32
  return %r : i32
}

// CHECK-LABEL: func.func @merge(
// CHECK-NOT: nneg
// CHECK: arith.extui %{{.*}} : i1 to i32
// CHECK-NOT: nneg

// When the inner extension itself carries nneg, the merged one may keep it.
func.func @keep(%v: i8) -> i64 {
  %w = arith.extui %v nneg : i8 to i32
  %r = arith.extui %w nneg : i32 to i64
  return %r : i64
}

// CHECK-LABEL: func.func @keep(
// CHECK: arith.extui %{{.*}} nneg : i8 to i64
