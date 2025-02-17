// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm -split-input-file  | FileCheck %s

func.func @foo(%arg0 : f64) -> f64 {
  %0 = math.cbrt %arg0 : f64
  return %0 : f64
}

// CHECK: llvm.func @cbrt(f64) -> f64 attributes {llvm.readnone, sym_visibility = "private"}
// CHECK-LABEL: foo
// CHECK-SAME: %[[ARG0:.+]]: f64
// CHECK: llvm.call @cbrt(%[[ARG0]]) : (f64) -> f64

// -----

func.func @bar(%arg0: i32) -> i32 {
  return %arg0: i32
}

// CHECK-LABEL: foo
func.func @foo(%arg0: i32) -> i32 {
  // CHECK: llvm.call @bar(%{{.+}}) {my_attribute = "florence"} : (i32) -> i32
  %0 = func.call @bar(%arg0) {my_attribute = "florence"} : (i32) -> i32
  return %0 : i32
}
