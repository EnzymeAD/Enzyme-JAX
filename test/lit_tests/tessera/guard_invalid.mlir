// RUN: enzymexlamlir-opt %s -split-input-file -verify-diagnostics

// Invariants tessera.guard relies on when it is lowered. argNames is how a
// variable named in the condition is resolved to an SSA value, so it has to
// line up with args and be unambiguous; the two regions both feed the guard's
// results, so each yield has to agree with them.

module {
  tessera.define @f(%a: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f32
  }
  llvm.func @t(%x: f32) -> f32 {
    // expected-error @+1 {{'tessera.guard' op argNames size (2) must match number of args (1)}}
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x", "y"]} : (f32) -> f32 {
      %1 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @f(%a: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f32
  }
  llvm.func @t(%x: f32) -> f32 {
    // expected-error @+1 {{'tessera.guard' op argNames entry 1 duplicates the name 'x'}}
    %0 = tessera.guard "symmetric(x)" args(%x, %x) {argNames = ["x", "x"]} : (f32, f32) -> f32 {
      %1 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @f(%a: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f32
  }
  llvm.func @t(%x: f32) -> f32 {
    // expected-error @+1 {{'tessera.guard' op argNames entry 0 must be a StringAttr, but got 42 : i32}}
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = [42 : i32]} : (f32) -> f32 {
      %1 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @f(%a: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f32
  }
  llvm.func @t(%x: f32) -> f32 {
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (f32) -> f32 {
      // expected-error @+1 {{'tessera.yield' op has 0 operands, but the enclosing tessera.guard returns 1}}
      tessera.yield
    } else {
      %2 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @f(%a: f32) -> f32 attributes {byRefTypes = [unit], pure = true} {
    tessera.return %a : f32
  }
  llvm.func @t(%x: f32) -> f32 {
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (f32) -> f32 {
      %c = llvm.mlir.constant(1 : i32) : i32
      // expected-error @+1 {{type of yield operand 0 ('i32') doesn't match the enclosing tessera.guard result type ('f32')}}
      tessera.yield %c : i32
    } else {
      %2 = tessera.call @f(%x) : (f32) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  llvm.func @t() {
    // expected-error @+1 {{'tessera.yield' op expects parent op 'tessera.guard'}}
    tessera.yield
  }
}
