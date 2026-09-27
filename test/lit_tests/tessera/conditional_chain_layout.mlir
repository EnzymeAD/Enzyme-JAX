// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-apply-pdl -tessera-lower-guards -split-input-file | FileCheck %s

// A predicate's operand is laid out however the callee taking it declares. In
// a chain rule that callee may be the nested producer rather than the root, so
// the layout is looked up across every call the guard's else region holds.

// The predicate names a variable only the producer takes.
module {
  tessera.define private @lib.g(!llvm.ptr {tessera.layout = {elem = f64, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> !llvm.ptr attributes {argModes = [unit], pure = false}
  tessera.define private @lib.f(!llvm.ptr, !llvm.ptr) -> f64 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.fast(!llvm.ptr, !llvm.ptr) -> f64 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(a), lib.f(lib.g(a), b) -> lib.fast(a, b)"
  }

  // CHECK-LABEL: llvm.func @producer_operand
  // CHECK: llvm.getelementptr %arg0[
  // CHECK: llvm.getelementptr %arg0[
  // CHECK: %[[P:.*]] = llvm.fcmp "oeq"
  // CHECK: llvm.cond_br %[[P]], ^[[THEN:.*]], ^[[ELSE:.*]]
  // CHECK: ^[[THEN]]:
  // CHECK-NEXT: tessera.call @lib.fast(%arg0, %arg1)
  // CHECK: ^[[ELSE]]:
  // CHECK-NEXT: %[[G:.*]] = tessera.call @lib.g(%arg0)
  // CHECK-NEXT: tessera.call @lib.f(%[[G]], %arg1)
  llvm.func @producer_operand(%a: !llvm.ptr, %b: !llvm.ptr) -> f64 {
    %0 = tessera.call @lib.g(%a) : (!llvm.ptr) -> !llvm.ptr
    %1 = tessera.call @lib.f(%0, %b) : (!llvm.ptr, !llvm.ptr) -> f64
    llvm.return %1 : f64
  }
}

// -----

// The predicate names a variable only the root takes. With the producer now
// ahead of the root in the else region, looking at the first call alone would
// miss it.
module {
  tessera.define private @lib.g(!llvm.ptr) -> !llvm.ptr attributes {argModes = [unit], pure = false}
  tessera.define private @lib.f(!llvm.ptr, !llvm.ptr {tessera.layout = {elem = f64, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f64 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @lib.fast(!llvm.ptr, !llvm.ptr) -> f64 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if symmetric(b), lib.f(lib.g(a), b) -> lib.fast(a, b)"
  }

  // CHECK-LABEL: llvm.func @root_operand
  // CHECK: llvm.getelementptr %arg1[
  // CHECK: llvm.getelementptr %arg1[
  // CHECK: llvm.fcmp "oeq"
  // CHECK: llvm.cond_br
  llvm.func @root_operand(%a: !llvm.ptr, %b: !llvm.ptr) -> f64 {
    %0 = tessera.call @lib.g(%a) : (!llvm.ptr) -> !llvm.ptr
    %1 = tessera.call @lib.f(%0, %b) : (!llvm.ptr, !llvm.ptr) -> f64
    llvm.return %1 : f64
  }
}
