// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise | FileCheck %s

// minimumnum/maximumnum arrive as llvm.call_intrinsic (no first-class llvm
// dialect op) and treat NaN as missing data, i.e. arith.minnumf/maxnumf.
func.func @minmax(%a: f64, %b: f64, %c: f32, %d: f32) -> (f64, f32) {
  %0 = llvm.call_intrinsic "llvm.minimumnum.f64"(%a, %b) : (f64, f64) -> f64
  %1 = llvm.call_intrinsic "llvm.maximumnum.f32"(%c, %d) : (f32, f32) -> f32
  return %0, %1 : f64, f32
}

// CHECK-LABEL: func.func @minmax(
// CHECK-SAME: %[[A:.+]]: f64, %[[B:.+]]: f64, %[[C:.+]]: f32, %[[D:.+]]: f32
// CHECK: %[[MIN:.+]] = arith.minnumf %[[A]], %[[B]] : f64
// CHECK: %[[MAX:.+]] = arith.maxnumf %[[C]], %[[D]] : f32
// CHECK: return %[[MIN]], %[[MAX]] : f64, f32
