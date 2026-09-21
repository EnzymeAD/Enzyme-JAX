// RUN: enzymexlamlir-opt %s -tessera-lower-guards -split-input-file | FileCheck %s

// The matrix property checks are synthesized by the compiler from the
// operand's layout -- no user-written predicate function is involved.
//
// Which form the element access takes follows from what the callee itself
// reads, so that the check and the call can never disagree: an integer operand
// carries the matrix by value and is taken apart with shifts, a pointer
// operand is read through the pointer.

module {
  tessera.define @lib.foo(%a: i128 {tessera.layout = {elem = f32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.symmetric_foo(%a: i128) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  // A 2x2 needs exactly one comparison: A[0][1] against A[1][0].
  // CHECK-LABEL: llvm.func @symmetric_packed
  llvm.func @symmetric_packed(%x: i128) -> f32 {
    // CHECK: %[[S1:.*]] = llvm.mlir.constant(32 : i128) : i128
    // CHECK: %[[R1:.*]] = llvm.lshr %arg0, %[[S1]] : i128
    // CHECK: %[[T1:.*]] = llvm.trunc %[[R1]] : i128 to i32
    // CHECK: %[[E1:.*]] = llvm.bitcast %[[T1]] : i32 to f32
    // CHECK: %[[S2:.*]] = llvm.mlir.constant(64 : i128) : i128
    // CHECK: %[[R2:.*]] = llvm.lshr %arg0, %[[S2]] : i128
    // CHECK: %[[T2:.*]] = llvm.trunc %[[R2]] : i128 to i32
    // CHECK: %[[E2:.*]] = llvm.bitcast %[[T2]] : i32 to f32
    // CHECK: %[[P:.*]] = llvm.fcmp "oeq" %[[E1]], %[[E2]] : f32
    // CHECK: llvm.cond_br %[[P]]
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (i128) -> f32 {
      %1 = tessera.call @lib.symmetric_foo(%x) : (i128) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (i128) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.symmetric_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  // A pointer operand is read through the pointer, since that is the same
  // memory the callee will read.
  // CHECK-LABEL: llvm.func @symmetric_pointer
  llvm.func @symmetric_pointer(%x: !llvm.ptr) -> f32 {
    // CHECK: %[[I1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[G1:.*]] = llvm.getelementptr %arg0[%[[I1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[E1:.*]] = llvm.load %[[G1]] : !llvm.ptr -> f32
    // CHECK: %[[I2:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[G2:.*]] = llvm.getelementptr %arg0[%[[I2]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[E2:.*]] = llvm.load %[[G2]] : !llvm.ptr -> f32
    // CHECK: llvm.fcmp "oeq" %[[E1]], %[[E2]] : f32
    %0 = tessera.guard "symmetric(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f32 {
      %1 = tessera.call @lib.symmetric_foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.diagonal_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  // Both off-diagonal entries compared against zero.
  // CHECK-LABEL: llvm.func @diagonal
  llvm.func @diagonal(%x: !llvm.ptr) -> f32 {
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK: llvm.getelementptr %arg0[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[C1:.*]] = llvm.fcmp "oeq" %{{.*}}, %[[ZERO]] : f32
    // CHECK: llvm.getelementptr %arg0[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[C2:.*]] = llvm.fcmp "oeq" %{{.*}}, %[[ZERO]] : f32
    // CHECK: %[[AND:.*]] = llvm.and %[[C1]], %[[C2]] : i1
    // CHECK: llvm.cond_br %[[AND]]
    %0 = tessera.guard "diagonal(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f32 {
      %1 = tessera.call @lib.diagonal_foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = f32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }
  tessera.define @lib.tri_foo(%a: !llvm.ptr) -> f32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0.0 : f32) : f32
    tessera.return %c : f32
  }

  // Upper triangular zeroes what is strictly below the diagonal, which for a
  // 2x2 is the single entry A[1][0] at flat index 2.
  // CHECK-LABEL: llvm.func @triangular_upper
  llvm.func @triangular_upper(%x: !llvm.ptr) -> f32 {
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK: %[[I:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.getelementptr %arg0[%[[I]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK: %[[P:.*]] = llvm.fcmp "oeq" %{{.*}}, %[[ZERO]] : f32
    // CHECK: llvm.cond_br %[[P]]
    %0 = tessera.guard "triangular_upper(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> f32 {
      %1 = tessera.call @lib.tri_foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %1 : f32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> f32
      tessera.yield %2 : f32
    }
    llvm.return %0 : f32
  }
}

// -----

module {
  tessera.define @lib.foo(%a: !llvm.ptr {tessera.layout = {elem = i32, rows = 2 : i64, cols = 2 : i64, row_major = true}}) -> i32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i32) : i32
    tessera.return %c : i32
  }
  tessera.define @lib.id_foo(%a: !llvm.ptr) -> i32 attributes {byRefTypes = [unit], pure = true} {
    %c = llvm.mlir.constant(0 : i32) : i32
    tessera.return %c : i32
  }

  // An integer element type compares with icmp rather than fcmp, and identity
  // checks every entry: ones on the diagonal, zeros off it.
  // CHECK-LABEL: llvm.func @identity_integer
  llvm.func @identity_integer(%x: !llvm.ptr) -> i32 {
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: llvm.icmp "eq" %{{.*}}, %[[ONE]] : i32
    // CHECK: llvm.icmp "eq" %{{.*}}, %[[ZERO]] : i32
    // CHECK: llvm.icmp "eq" %{{.*}}, %[[ZERO]] : i32
    // CHECK: llvm.icmp "eq" %{{.*}}, %[[ONE]] : i32
    // CHECK: llvm.cond_br
    %0 = tessera.guard "identity(x)" args(%x) {argNames = ["x"]} : (!llvm.ptr) -> i32 {
      %1 = tessera.call @lib.id_foo(%x) : (!llvm.ptr) -> i32
      tessera.yield %1 : i32
    } else {
      %2 = tessera.call @lib.foo(%x) : (!llvm.ptr) -> i32
      tessera.yield %2 : i32
    }
    llvm.return %0 : i32
  }
}
