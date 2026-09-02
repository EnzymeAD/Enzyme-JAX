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

// The accumulator is carried along, but the exit test reads only the counter,
// whose next value is invariant.
// CHECK-LABEL: func.func @accumulating
// CHECK-SAME: (%[[buf:.+]]: memref<32xf64>, %[[n:.+]]: i32, %[[k0:.+]]: i32)
// CHECK: %[[i0:.+]] = arith.index_cast %[[k0]] : i32 to index
// CHECK-NEXT: %[[v0:.+]] = memref.load %[[buf]][%[[i0]]] : memref<32xf64>
// CHECK-NEXT: %[[s0:.+]] = arith.addf %[[v0]], %{{.+}} : f64
// CHECK-NEXT: %[[c0:.+]] = arith.cmpi ult, %[[k0]], %[[n]] : i32
// CHECK-NEXT: %[[r:.+]] = scf.if %[[c0]] -> (f64) {
// CHECK-NEXT:   %[[v1:.+]] = memref.load %[[buf]][%[[c16:.+]]] : memref<32xf64>
// CHECK-NEXT:   %[[s1:.+]] = arith.addf %[[s0]], %[[v1]] : f64
// CHECK-NEXT:   scf.yield %[[s1]] : f64
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[s0]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: return %[[r]] : f64
func.func @accumulating(%buf: memref<32xf64>, %n: i32, %k0: i32) -> f64 {
  %c16 = arith.constant 16 : i32
  %zero = arith.constant 0.0 : f64
  %r = scf.while (%k = %k0, %acc = %zero) : (i32, f64) -> f64 {
    %idx = arith.index_cast %k : i32 to index
    %v = memref.load %buf[%idx] : memref<32xf64>
    %sum = arith.addf %acc, %v : f64
    %cond = arith.cmpi ult, %k, %n : i32
    scf.condition(%cond) %sum : f64
  } do {
  ^bb0(%sum: f64):
    scf.yield %c16, %sum : i32, f64
  }
  return %r : f64
}

// The exit test reads the carried accumulator, which varies.
// CHECK-LABEL: func.func @cond_reads_carried
// CHECK: scf.while
func.func @cond_reads_carried(%buf: memref<32xf64>, %n: i32, %k0: i32) -> f64 {
  %c16 = arith.constant 16 : i32
  %zero = arith.constant 0.0 : f64
  %lim = arith.constant 1.0 : f64
  %r = scf.while (%k = %k0, %acc = %zero) : (i32, f64) -> f64 {
    %idx = arith.index_cast %k : i32 to index
    %v = memref.load %buf[%idx] : memref<32xf64>
    %sum = arith.addf %acc, %v : f64
    %cond = arith.cmpf olt, %sum, %lim : f64
    scf.condition(%cond) %sum : f64
  } do {
  ^bb0(%sum: f64):
    scf.yield %c16, %sum : i32, f64
  }
  return %r : f64
}

// A rotated single-trip loop: the flag is yielded false, and on it the exit
// test folds to false whatever the body reads; the second body then folds
// away entirely.
// CHECK-LABEL: func.func @trip_flag
// CHECK-SAME: (%[[buf:.+]]: memref<32xf64>, %[[flags:.+]]: memref<32xi1>, %[[go:.+]]: i1)
// CHECK-NOT: scf.while
// CHECK: scf.if %[[go]] {
// CHECK-NEXT:   memref.store %{{.+}}, %[[buf]][%{{.+}}] : memref<32xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @trip_flag(%buf: memref<32xf64>, %flags: memref<32xi1>, %go: i1) {
  %false = arith.constant false
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %v = arith.constant 2.0 : f64
  scf.while (%run = %go, %i = %c0) : (i1, i32) -> i32 {
    %r:2 = scf.if %run -> (i1, i32) {
      %idx = arith.index_cast %i : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %again = memref.load %flags[%idx] : memref<32xi1>
      %next = arith.addi %i, %c1 : i32
      scf.yield %again, %next : i1, i32
    } else {
      scf.yield %false, %i : i1, i32
    }
    scf.condition(%r#0) %r#1 : i32
  } do {
  ^bb0(%i: i32):
    scf.yield %false, %i : i1, i32
  }
  return
}

// The flag is yielded true, so the exit test stays with the body's reads.
// CHECK-LABEL: func.func @trip_flag_true
// CHECK: scf.while
func.func @trip_flag_true(%buf: memref<32xf64>, %flags: memref<32xi1>, %go: i1) {
  %false = arith.constant false
  %true = arith.constant true
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %v = arith.constant 2.0 : f64
  scf.while (%run = %go, %i = %c0) : (i1, i32) -> i32 {
    %r:2 = scf.if %run -> (i1, i32) {
      %idx = arith.index_cast %i : i32 to index
      memref.store %v, %buf[%idx] : memref<32xf64>
      %again = memref.load %flags[%idx] : memref<32xi1>
      %next = arith.addi %i, %c1 : i32
      scf.yield %again, %next : i1, i32
    } else {
      scf.yield %false, %i : i1, i32
    }
    scf.condition(%r#0) %r#1 : i32
  } do {
  ^bb0(%i: i32):
    scf.yield %true, %i : i1, i32
  }
  return
}

// The counter starts at zero and steps by one, and the exit test ors it with
// another value before comparing to zero: the second iteration's counter has
// its low bit set, so there is no third.
// CHECK-LABEL: func.func @low_bit_set
// CHECK-NOT: scf.while
// CHECK: %[[c1:.+]] = arith.constant 1 : index
// CHECK: %[[go:.+]] = arith.cmpi eq, %{{.+}}, %[[c0:.+]] : i32
// CHECK: scf.if %[[go]] -> (f64) {
// CHECK: memref.load %{{.+}}[%[[c1]]] : memref<8xf64>
// CHECK: } else {
// CHECK: return
func.func @low_bit_set(%buf: memref<8xf64>, %odd: i32) -> f64 {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%acc = %zero, %k = %c0) : (f64, i32) -> (f64, i32) {
    %idx = arith.index_cast %k : i32 to index
    %x = memref.load %buf[%idx] : memref<8xf64>
    %sum = arith.addf %acc, %x : f64
    %next = arith.addi %k, %c1 : i32
    %bits = arith.ori %k, %odd : i32
    %again = arith.cmpi eq, %bits, %c0 : i32
    scf.condition(%again) %sum, %next : f64, i32
  } do {
  ^bb0(%acc: f64, %k: i32):
    scf.yield %acc, %k : f64, i32
  }
  return %r#0 : f64
}

// Masking the counter instead may leave it zero, so the loop may run on.
// CHECK-LABEL: func.func @low_bit_masked
// CHECK: scf.while
func.func @low_bit_masked(%buf: memref<8xf64>, %mask: i32) -> f64 {
  %zero = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r:2 = scf.while (%acc = %zero, %k = %c0) : (f64, i32) -> (f64, i32) {
    %idx = arith.index_cast %k : i32 to index
    %x = memref.load %buf[%idx] : memref<8xf64>
    %sum = arith.addf %acc, %x : f64
    %next = arith.addi %k, %c1 : i32
    %bits = arith.andi %k, %mask : i32
    %again = arith.cmpi eq, %bits, %c0 : i32
    scf.condition(%again) %sum, %next : f64, i32
  } do {
  ^bb0(%acc: f64, %k: i32):
    scf.yield %acc, %k : f64, i32
  }
  return %r#0 : f64
}
