// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm='backend=cpu' --split-input-file | FileCheck %s

// enzymexla.math.fmuladd lowers back to the intrinsic it was raised from:
// llvm.intr.fmuladd, which only permits fusing. It must not become
// llvm.intr.fma, whose required single rounding a target without FMA units
// honors with a libm call per multiply-add.

// CHECK-LABEL: @permission
func.func @permission(%a: f64, %b: f64, %c: f64) -> f64 {
  %r = enzymexla.math.fmuladd %a, %b, %c : f64
  return %r : f64
}

// CHECK: llvm.intr.fmuladd(%arg0, %arg1, %arg2) : (f64, f64, f64) -> f64
// CHECK-NOT: llvm.intr.fma(

// -----

// The strict form stays strict: math.fma still lowers to llvm.intr.fma.

// CHECK-LABEL: @requirement
func.func @requirement(%a: f64, %b: f64, %c: f64) -> f64 {
  %r = math.fma %a, %b, %c : f64
  return %r : f64
}

// CHECK: llvm.intr.fma(%arg0, %arg1, %arg2) : (f64, f64, f64) -> f64
