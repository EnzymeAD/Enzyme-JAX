// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file | FileCheck %s

// Lowering a tessera.guard whose condition is a scalar comparison: synthesize
// the test, then replace the guard with an ordinary conditional branch. In the
// LLVM dialect the phi node is a block argument on the continuation block.

module {
  tessera.define @lib.qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.tiled_qux(%a: i512, %n: i64) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  // CHECK-LABEL: llvm.func @int_compare
  llvm.func @int_compare(%x: i512, %n: i64) -> f32 {
    // The literal is built at the type of the value it is compared against,
    // not at a width guessed from its magnitude.
    // CHECK: %[[C:.*]] = llvm.mlir.constant(64 : i64) : i64
    // CHECK: %[[P:.*]] = llvm.icmp "sgt" %arg1, %[[C]] : i64
    // CHECK: llvm.cond_br %[[P]], ^[[THEN:.*]], ^[[ELSE:.*]]
    // CHECK: ^[[THEN]]:
    // CHECK: %[[S:.*]] = tessera.call @lib.tiled_qux
    // CHECK: llvm.br ^[[TAIL:.*]](%[[S]] : f32)
    // CHECK: ^[[ELSE]]:
    // CHECK: %[[O:.*]] = tessera.call @lib.qux
    // CHECK: llvm.br ^[[TAIL]](%[[O]] : f32)
    // CHECK: ^[[TAIL]](%[[PHI:.*]]: f32):
    // CHECK: llvm.return %[[PHI]]
    %0 = tessera.guard "n > 64" args(%x, %n) {argNames = ["x", "n"]} : (i512, i64) -> f32 {
      %1 = tessera.call @lib.tiled_qux(%x, %n) : (i512, i64) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.qux(%x, %n) : (i512, i64) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }

  // Nothing tessera-specific is left behind by the lowering itself.
  // CHECK-NOT: tessera.guard
  // CHECK-NOT: tessera.yield
}

// -----

module {
  tessera.define @lib.f(%a: f64) -> f64 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f64
  }
  tessera.define @lib.g(%a: f64) -> f64 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f64
  }

  // Float comparisons are ordered, following C.
  // CHECK-LABEL: llvm.func @float_compare
  llvm.func @float_compare(%a: f64) -> f64 {
    // CHECK: %[[C:.*]] = llvm.mlir.constant(5.000000e-01 : f64) : f64
    // CHECK: %[[P:.*]] = llvm.fcmp "oge" %arg0, %[[C]] : f64
    // CHECK: llvm.cond_br %[[P]]
    %0 = tessera.guard "a >= 0.5" args(%a) {argNames = ["a"]} : (f64) -> f64 {
      %1 = tessera.call @lib.g(%a) : (f64) -> f64
      tessera.yield %1 : f64
    } else {
      %2 = tessera.call @lib.f(%a) : (f64) -> f64
      tessera.yield %2 : f64
    }
    llvm.return %0 : f64
  }
}

// -----

module {
  tessera.define @lib.f(%a: i64, %b: i64) -> i64 attributes {byRefTypes = [unit, unit], pure = true} {
    tessera.return %a : i64
  }
  tessera.define @lib.g(%a: i64, %b: i64) -> i64 attributes {byRefTypes = [unit, unit], pure = true} {
    tessera.return %a : i64
  }

  // The connectives lower to bitwise ops on i1, and '!' to an xor with true.
  // CHECK-LABEL: llvm.func @connectives
  llvm.func @connectives(%m: i64, %n: i64) -> i64 {
    // Emitted left to right, with no short-circuiting: both sides of every
    // connective are evaluated.
    // CHECK: %[[A:.*]] = llvm.icmp "sgt"
    // CHECK: %[[B:.*]] = llvm.icmp "ne"
    // CHECK: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
    // CHECK: %[[NOTB:.*]] = llvm.xor %[[B]], %[[TRUE]] : i1
    // CHECK: %[[AND:.*]] = llvm.and %[[A]], %[[NOTB]] : i1
    // CHECK: %[[C:.*]] = llvm.icmp "slt"
    // CHECK: %[[OR:.*]] = llvm.or %[[AND]], %[[C]] : i1
    // CHECK: llvm.cond_br %[[OR]]
    %0 = tessera.guard "(m > 0 && !(n != 0)) || m < -3" args(%m, %n) {argNames = ["m", "n"]} : (i64, i64) -> i64 {
      %1 = tessera.call @lib.g(%m, %n) : (i64, i64) -> i64
      tessera.yield %1 : i64
    } else {
      %2 = tessera.call @lib.f(%m, %n) : (i64, i64) -> i64
      tessera.yield %2 : i64
    }
    llvm.return %0 : i64
  }
}

// -----

module {
  tessera.define @lib.f(%a: i64) -> () attributes {byRefTypes = [unit], pure = false} {
    tessera.return
  }
  tessera.define @lib.g(%a: i64) -> () attributes {byRefTypes = [unit], pure = false} {
    tessera.return
  }

  // A guard with no results: the continuation takes no block arguments.
  // CHECK-LABEL: llvm.func @no_results
  llvm.func @no_results(%n: i64) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.br ^[[TAIL:.*]]{{$}}
    // CHECK: llvm.br ^[[TAIL]]{{$}}
    // CHECK: ^[[TAIL]]:
    // CHECK-NEXT: llvm.return
    tessera.guard "n > 0" args(%n) {argNames = ["n"]} : (i64) -> () {
      tessera.call @lib.g(%n) : (i64) -> ()
      tessera.yield
    } else {
      tessera.call @lib.f(%n) : (i64) -> ()
      tessera.yield
    }
    llvm.return
  }
}
