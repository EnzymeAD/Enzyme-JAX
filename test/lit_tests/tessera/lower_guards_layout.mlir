// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file -verify-diagnostics

// How a matrix operand's layout is found, and what happens when it cannot be.
//
// After llvm-to-tessera a matrix is usually a flat integer, which records its
// size but neither its element type nor its shape. The callee's declaration is
// what knows, so the lookup goes through the call the guard kept in its else
// region and reads the layout off the matching tessera.define argument.

// An explicit tessera.layout is taken at face value.
module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @explicit(%x: !llvm.ptr) -> f32 {
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

// With no explicit layout, the by-reference type is walked down to the array
// holding the elements and *assumed* square and row-major. That is a guess, and
// a wrong guess miscompiles quietly rather than failing, so it says so.
module {
  tessera.define @lib.foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [!llvm.struct<"Outer", (struct<"Inner", (array<4 x f32>)>)>], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @inferred(%x: !llvm.ptr) -> f32 {
    // expected-remark @+1 {{assuming argument 0 of 'lib.foo' is a 2x2 row-major matrix of 'f32'; declare tessera.layout on the tessera.define to be certain}}
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

// Neither an explicit layout nor a by-reference type to infer from.
module {
  tessera.define @lib.foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @no_layout(%x: !llvm.ptr) -> f32 {
    // expected-error @+1 {{cannot determine the layout of the operand of 'symmetric'; declare tessera.layout on the corresponding tessera.define argument}}
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

// A predicate name that is not in the registry.
module {
  tessera.define @lib.foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @unknown_predicate(%x: !llvm.ptr) -> f32 {
    // expected-error @+1 {{unknown predicate 'orthogonal'; known predicates are symmetric, diagonal, triangular_upper, triangular_lower, identity}}
    %0 = tessera.guard "orthogonal(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f32 {
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

// Right name, wrong number of arguments.
module {
  tessera.define @lib.foo(%a: !llvm.ptr, %b: !llvm.ptr) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr, %b: !llvm.ptr) -> f32 attributes {byRefTypes = [unit, unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @wrong_arity(%x: !llvm.ptr, %y: !llvm.ptr) -> f32 {
    // expected-error @+1 {{predicate 'symmetric' takes 1 argument(s), but got 2}}
    %0 = tessera.guard "symmetric(x, y)" args(%x, %y) {argNames = ["x", "y"]} : (!llvm.ptr, !llvm.ptr) -> f32 {
      %1 = tessera.call @lib.sym_foo(%x, %y) : (!llvm.ptr, !llvm.ptr) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x, %y) : (!llvm.ptr, !llvm.ptr) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

// Symmetry is only defined for a square matrix.
module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 2 : i64, cols = 3 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.sym_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  llvm.func @not_square(%x: !llvm.ptr) -> f32 {
    // expected-error @+1 {{'symmetric' needs a square matrix, but the operand is 2x3}}
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
