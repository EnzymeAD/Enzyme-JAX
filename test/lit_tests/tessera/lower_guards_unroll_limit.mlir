// RUN: enzymexlamlir-opt %s -tessera-lower-guards="max-unrolled-elems=4" -split-input-file -verify-diagnostics

// The comparisons a property check needs are emitted straight-line, so a large
// matrix would turn into an unreasonable amount of code. Above the limit the
// predicate declines rather than emitting it. A loop form would lift this; it
// is not implemented, so this is an error and not a silent miscompile.

// 2x2 is four elements, exactly at the limit, so it still lowers.
module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @at_the_limit(%x: !llvm.ptr) -> f32 {
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f32 {
      %1 = tessera.call @lib.sym_foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

// 3x3 is nine, above it.
module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 3 : i64, cols = 3 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @over_the_limit(%x: !llvm.ptr) -> f32 {
    // expected-error @+1 {{'symmetric' needs 9 elements checked, above the max-unrolled-elems limit of 4; a loop form is not implemented yet}}
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f32 {
      %1 = tessera.call @lib.sym_foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}
