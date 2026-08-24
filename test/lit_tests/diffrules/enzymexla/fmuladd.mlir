// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_dup,enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_active,enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s --check-prefix=REVERSE

// d(fmuladd(x, y, z)) = dx*y + x*dy + dz -- the same rule as math.fma, whose
// value the op shares; only the lowering contract differs.

func.func @f(%x: f64, %y: f64, %z: f64) -> f64 {
  %r = enzymexla.math.fmuladd %x, %y, %z : f64
  return %r : f64
}

// FORWARD:  func.func @f(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64, %arg5: f64) -> (f64, f64) {
// FORWARD-NEXT:    %0 = arith.mulf %arg1, %arg2 fastmath<fast> : f64
// FORWARD-NEXT:    %1 = arith.mulf %arg3, %arg0 fastmath<fast> : f64
// FORWARD-NEXT:    %2 = arith.addf %0, %1 fastmath<fast> : f64
// FORWARD-NEXT:    %3 = arith.addf %2, %arg5 fastmath<fast> : f64
// FORWARD-NEXT:    %4 = enzymexla.math.fmuladd %arg0, %arg2, %arg4 : f64
// FORWARD-NEXT:    return %4, %3 : f64, f64
// FORWARD-NEXT:  }

// REVERSE:  func.func @f(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) -> (f64, f64, f64) {
// REVERSE-NEXT:    %0 = arith.mulf %arg3, %arg1 fastmath<fast> : f64
// REVERSE-NEXT:    %1 = arith.mulf %arg3, %arg0 fastmath<fast> : f64
// REVERSE-NEXT:    return %0, %1, %arg3 : f64, f64, f64
// REVERSE-NEXT:  }
