// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -split-input-file | FileCheck %s

// A rule that requires more goes before one that requires less for the same
// call, whichever the header lists first: a catch-all applies only where the
// specific rule doesn't, and a rule requiring two facts goes before one
// requiring one of them.

// The catch-all listed first.
module {
  tessera.define private @lib.solve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.solve_spd(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.solve_try(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  llvm.func @fill_spd(!llvm.ptr {tessera.establishes = ["SPD"], tessera.noescape})

  tessera.optimizations {
    tessera.optimization "lib.solve(a, b) -> lib.solve_try(a, b)"
    tessera.optimization "if SPD(a), lib.solve(a, b) -> lib.solve_spd(a, b)"
  }

  // CHECK-LABEL: llvm.func @known_spd
  // CHECK: tessera.call @lib.solve_spd
  llvm.func @known_spd(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr) -> i32 {
    llvm.call @fill_spd(%a) : (!llvm.ptr) -> ()
    %r = tessera.call @lib.solve(%a, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // CHECK-LABEL: llvm.func @unknown
  // CHECK: tessera.call @lib.solve_try
  llvm.func @unknown(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr) -> i32 {
    %r = tessera.call @lib.solve(%a, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// -----

// The catch-all listed last.
module {
  tessera.define private @lib.solve(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.solve_spd(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.solve_try(!llvm.ptr, !llvm.ptr) -> i32 attributes {argModes = [unit, unit], pure = false}
  llvm.func @fill_spd(!llvm.ptr {tessera.establishes = ["SPD"], tessera.noescape})

  tessera.optimizations {
    tessera.optimization "if SPD(a), lib.solve(a, b) -> lib.solve_spd(a, b)"
    tessera.optimization "lib.solve(a, b) -> lib.solve_try(a, b)"
  }

  // CHECK-LABEL: llvm.func @known_spd_2
  // CHECK: tessera.call @lib.solve_spd
  llvm.func @known_spd_2(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr) -> i32 {
    llvm.call @fill_spd(%a) : (!llvm.ptr) -> ()
    %r = tessera.call @lib.solve(%a, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }

  // CHECK-LABEL: llvm.func @unknown_2
  // CHECK: tessera.call @lib.solve_try
  llvm.func @unknown_2(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr) -> i32 {
    %r = tessera.call @lib.solve(%a, %b) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %r : i32
  }
}

// -----

// Two rules with conditions: the one requiring both facts goes first.
module {
  tessera.define private @vec.dedup(!llvm.ptr) attributes {argModes = [unit], pure = false}
  tessera.define private @vec.dedup_adjacent(!llvm.ptr) attributes {argModes = [unit], pure = false}
  tessera.define private @vec.noop(!llvm.ptr) attributes {argModes = [unit], pure = false}
  llvm.func @fill_sorted_unique(!llvm.ptr {tessera.establishes = ["sorted", "unique"], tessera.noescape})
  llvm.func @fill_sorted(!llvm.ptr {tessera.establishes = ["sorted"], tessera.noescape})

  tessera.optimizations {
    tessera.optimization "if sorted(v), vec.dedup(v) -> vec.dedup_adjacent(v)"
    tessera.optimization "if sorted(v) && unique(v), vec.dedup(v) -> vec.noop(v)"
  }

  // CHECK-LABEL: llvm.func @sorted_unique
  // CHECK: tessera.call @vec.noop
  llvm.func @sorted_unique(%v: !llvm.ptr {llvm.noalias}) {
    llvm.call @fill_sorted_unique(%v) : (!llvm.ptr) -> ()
    tessera.call @vec.dedup(%v) : (!llvm.ptr) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func @sorted_only
  // CHECK: tessera.call @vec.dedup_adjacent
  llvm.func @sorted_only(%v: !llvm.ptr {llvm.noalias}) {
    llvm.call @fill_sorted(%v) : (!llvm.ptr) -> ()
    tessera.call @vec.dedup(%v) : (!llvm.ptr) -> ()
    llvm.return
  }
}
