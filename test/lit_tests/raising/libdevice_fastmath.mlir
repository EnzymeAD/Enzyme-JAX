// RUN: enzymexlamlir-opt --libdevice-funcs-raise -split-input-file %s | FileCheck %s

// Raising an LLVM op to arith has to translate its flags, not carry the LLVM
// spelling across as a discardable attribute: `#llvm.fastmath` on an arith op
// means nothing to anyone downstream, so the flags are silently lost when the
// op is lowered back.

llvm.func @fastmath(%a: f64, %b: f64, %c: f64) -> f64 {
  %0 = llvm.fmul %a, %b {fastmathFlags = #llvm.fastmath<fast>} : f64
  %1 = llvm.fadd %0, %c {fastmathFlags = #llvm.fastmath<fast>} : f64
  %2 = llvm.fsub %1, %a {fastmathFlags = #llvm.fastmath<nsz>} : f64
  %3 = llvm.fdiv %2, %b {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.return %3 : f64
}

// CHECK-LABEL: llvm.func @fastmath
// CHECK:         arith.mulf %arg0, %arg1 fastmath<fast>
// CHECK:         arith.addf %{{.*}}, %arg2 fastmath<fast>
// CHECK:         arith.subf %{{.*}}, %arg0 fastmath<nsz>
// CHECK:         arith.divf %{{.*}}, %arg1 fastmath<fast>
// CHECK-NOT:     fastmathFlags

// -----

// Absent flags stay absent rather than becoming an explicit `none`.

llvm.func @no_flags(%a: f64, %b: f64) -> f64 {
  %0 = llvm.fmul %a, %b : f64
  llvm.return %0 : f64
}

// CHECK-LABEL: llvm.func @no_flags
// CHECK:         arith.mulf %arg0, %arg1 : f64

// -----

// The same goes for integer overflow flags.

llvm.func @overflow(%a: i32, %b: i32) -> i32 {
  %0 = llvm.mul %a, %b overflow<nsw> : i32
  %1 = llvm.add %0, %a overflow<nsw, nuw> : i32
  %2 = llvm.sub %1, %b overflow<nuw> : i32
  llvm.return %2 : i32
}

// CHECK-LABEL: llvm.func @overflow
// CHECK:         arith.muli %arg0, %arg1 overflow<nsw>
// CHECK:         arith.addi %{{.*}}, %arg0 overflow<nsw, nuw>
// CHECK:         arith.subi %{{.*}}, %arg1 overflow<nuw>
