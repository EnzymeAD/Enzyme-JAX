// RUN: enzymexlamlir-opt %s --lower-enzymexla-math --convert-polygeist-to-llvm | FileCheck %s

module {
  func.func @main(%a: f64, %b: f64, %c: f64) -> f64 {
    %0 = enzymexla.math.fmuladd %a, %b, %c : f64
    return %0 : f64
  }
}

// CHECK:  llvm.func @main(%[[A:.+]]: f64, %[[B:.+]]: f64, %[[C:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[RES:.+]] = llvm.intr.fmuladd(%[[A]], %[[B]], %[[C]]) : (f64, f64, f64) -> f64
// CHECK-NEXT:    llvm.return %[[RES]] : f64
// CHECK-NEXT:  }

