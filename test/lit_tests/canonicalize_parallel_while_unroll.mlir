// RUN: enzymexlamlir-opt %s --canonicalize-parallel --allow-unregistered-dialect | FileCheck %s

// The strided copy loop after the range analysis folded k += stride to a
// constant: every yield is loop-invariant, so the loop runs at most twice.
// CHECK-LABEL: func.func @strided
// CHECK-SAME: (%[[buf:.+]]: memref<32xf64>, %[[n:.+]]: i32)
// CHECK: affine.parallel (%[[i:.+]]) = (0) to (16) {
// CHECK-NEXT:   %[[k:.+]] = arith.index_cast %[[i]] : index to i32
// CHECK-NEXT:   %[[idx:.+]] = arith.index_cast %[[k]] : i32 to index
// CHECK-NEXT:   memref.store %{{.+}}, %[[buf]][%[[idx]]] : memref<32xf64>
// CHECK-NEXT:   %[[c1:.+]] = arith.cmpi ult, %[[k]], %[[n]] : i32
// CHECK-NEXT:   scf.if %[[c1]] {
// CHECK-NEXT:     memref.store %{{.+}}, %[[buf]][%[[c16:.+]]] : memref<32xf64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NOT: scf.while
func.func @strided(%buf: memref<32xf64>, %n: i32) {
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  affine.parallel (%i) = (0) to (16) {
    %k = arith.index_cast %i : index to i32
    %r = scf.while (%arg = %k) : (i32) -> i32 {
      %idx = arith.index_cast %arg : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %cond = arith.cmpi ult, %arg, %n : i32
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg: i32):
      scf.yield %c16 : i32
    }
  }
  return
}

// A pass-through of a forwarded value only counts as invariant when the
// forwarded value itself is.
// CHECK-LABEL: func.func @forwarded_invariant
// CHECK-NOT: scf.while
// CHECK: scf.if
// CHECK-NOT: scf.while
func.func @forwarded_invariant(%buf: memref<32xf64>, %n: i32, %j: i32) {
  %v = arith.constant 2.0 : f64
  affine.parallel (%i) = (0) to (16) {
    %k = arith.index_cast %i : index to i32
    scf.while (%arg = %k) : (i32) -> i32 {
      %idx = arith.index_cast %arg : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %cond = arith.cmpi ult, %arg, %n : i32
      scf.condition(%cond) %j : i32
    } do {
    ^bb0(%arg: i32):
      scf.yield %arg : i32
    }
  }
  return
}

// The done flag: the after region's side effect lands between the two copies
// of the before region, and the constant condition then dissolves the if.
// CHECK-LABEL: func.func @sideflag
// CHECK-SAME: (%[[buf:.+]]: memref<8xf64>)
// CHECK-NOT: scf.while
// CHECK: affine.store %{{.+}}, %[[buf]][0] : memref<8xf64>
// CHECK-NEXT: affine.store %{{.+}}, %[[buf]][1] : memref<8xf64>
// CHECK-NEXT: affine.store %{{.+}}, %[[buf]][0] : memref<8xf64>
// CHECK-NEXT: return
func.func @sideflag(%buf: memref<8xf64>) {
  %true = arith.constant true
  %false = arith.constant false
  %v = arith.constant 2.0 : f64
  %w = arith.constant 3.0 : f64
  scf.while (%flag = %true) : (i1) -> () {
    affine.store %v, %buf[0] : memref<8xf64>
    scf.condition(%flag)
  } do {
    affine.store %w, %buf[1] : memref<8xf64>
    scf.yield %false : i1
  }
  return
}

// A real strided loop: the yield varies with the carried state.
// CHECK-LABEL: func.func @varying
// CHECK: scf.while
func.func @varying(%buf: memref<32xf64>, %n: i32) {
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  affine.parallel (%i) = (0) to (16) {
    %k = arith.index_cast %i : index to i32
    scf.while (%arg = %k) : (i32) -> i32 {
      %idx = arith.index_cast %arg : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %cond = arith.cmpi ult, %arg, %n : i32
      %next = arith.addi %arg, %c16 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%arg: i32):
      scf.yield %arg : i32
    }
  }
  return
}

// The exit test reads memory the body writes, so the second iteration's
// decision is not the third's.
// CHECK-LABEL: func.func @cond_reads_memory
// CHECK: scf.while
func.func @cond_reads_memory(%buf: memref<32xf64>, %flags: memref<32xi1>) {
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  affine.parallel (%i) = (0) to (16) {
    %k = arith.index_cast %i : index to i32
    scf.while (%arg = %k) : (i32) -> i32 {
      %idx = arith.index_cast %arg : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %cond = memref.load %flags[%idx] : memref<32xi1>
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg: i32):
      scf.yield %c16 : i32
    }
  }
  return
}

// The yield is computed in the after region from a value that varies.
// CHECK-LABEL: func.func @after_varying
// CHECK: scf.while
func.func @after_varying(%buf: memref<32xf64>, %n: i32) {
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  affine.parallel (%i) = (0) to (16) {
    %k = arith.index_cast %i : index to i32
    scf.while (%arg = %k) : (i32) -> i32 {
      %idx = arith.index_cast %arg : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %cond = arith.cmpi ult, %arg, %n : i32
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg: i32):
      %next = arith.addi %arg, %c16 : i32
      scf.yield %next : i32
    }
  }
  return
}
