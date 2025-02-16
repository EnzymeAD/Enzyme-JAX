// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm  | FileCheck %s

func.func @foo(%arg0 : f64) -> f64 {
  %0 = math.cbrt %arg0 : f64
  return %0 : f64
}

// CHECK: llvm.func @cbrt(f64) -> f64 attributes {llvm.readnone, sym_visibility = "private"}
// CHECK-LABEL: foo
// CHECK-SAME: %[[ARG0:.+]]: f64
// CHECK: llvm.call @cbrt(%[[ARG0]]) : (f64) -> f64
