// RUN: enzymexlamlir-opt %s --canonicalize-parallel --allow-unregistered-dialect | FileCheck %s

// The reduction over one or two dofs: clang writes `dx + 1 < 2 - odd` as
// `(dx | odd) == 0`, and the counter is one on the second evaluation.
func.func @ored_counter(%x: memref<8xf64>, %b: memref<4xf64>, %out: memref<f64>, %odd: i32) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %r:2 = scf.while (%acc = %zero, %dx = %c0_i32) : (f64, i32) -> (f64, i32) {
      %i = arith.index_cast %dx : i32 to index
      %xv = memref.load %x[%i] : memref<8xf64>
      %bv = memref.load %b[%i] : memref<4xf64>
      %p = arith.mulf %xv, %bv : f64
      %sum = arith.addf %acc, %p : f64
      %next = arith.addi %dx, %c1_i32 : i32
      %or = arith.ori %dx, %odd : i32
      %again = arith.cmpi eq, %or, %c0_i32 : i32
      scf.condition(%again) %sum, %next : f64, i32
    } do {
    ^bb0(%sum: f64, %next: i32):
      scf.yield %sum, %next : f64, i32
    }
    memref.store %r#0, %out[] : memref<f64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The counter starts at one: the pattern only knows counters from zero.
func.func @starts_at_one(%x: memref<8xf64>, %b: memref<4xf64>, %out: memref<f64>, %odd: i32) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %r:2 = scf.while (%acc = %zero, %dx = %c1_i32) : (f64, i32) -> (f64, i32) {
      %i = arith.index_cast %dx : i32 to index
      %xv = memref.load %x[%i] : memref<8xf64>
      %sum = arith.addf %acc, %xv : f64
      %next = arith.addi %dx, %c1_i32 : i32
      %or = arith.ori %dx, %odd : i32
      %again = arith.cmpi eq, %or, %c0_i32 : i32
      scf.condition(%again) %sum, %next : f64, i32
    } do {
    ^bb0(%sum: f64, %next: i32):
      scf.yield %sum, %next : f64, i32
    }
    memref.store %r#0, %out[] : memref<f64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// Stepping by two keeps the low bit clear: the loop may run on.
func.func @steps_by_two(%x: memref<8xf64>, %out: memref<f64>, %odd: i32) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %zero = arith.constant 0.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %r:2 = scf.while (%acc = %zero, %dx = %c0_i32) : (f64, i32) -> (f64, i32) {
      %i = arith.index_cast %dx : i32 to index
      %xv = memref.load %x[%i] : memref<8xf64>
      %sum = arith.addf %acc, %xv : f64
      %next = arith.addi %dx, %c2_i32 : i32
      %or = arith.ori %dx, %odd : i32
      %again = arith.cmpi eq, %or, %c0_i32 : i32
      scf.condition(%again) %sum, %next : f64, i32
    } do {
    ^bb0(%sum: f64, %next: i32):
      scf.yield %sum, %next : f64, i32
    }
    memref.store %r#0, %out[] : memref<f64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// Masking instead of oring may leave the test true.
func.func @masked(%x: memref<8xf64>, %out: memref<f64>, %mask: i32) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %r:2 = scf.while (%acc = %zero, %dx = %c0_i32) : (f64, i32) -> (f64, i32) {
      %i = arith.index_cast %dx : i32 to index
      %xv = memref.load %x[%i] : memref<8xf64>
      %sum = arith.addf %acc, %xv : f64
      %next = arith.addi %dx, %c1_i32 : i32
      %and = arith.andi %dx, %mask : i32
      %again = arith.cmpi eq, %and, %c0_i32 : i32
      scf.condition(%again) %sum, %next : f64, i32
    } do {
    ^bb0(%sum: f64, %next: i32):
      scf.yield %sum, %next : f64, i32
    }
    memref.store %r#0, %out[] : memref<f64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The bound does not rest on forward progress, so a host loop unrolls too.
func.func @host(%x: memref<8xf64>, %out: memref<f64>, %odd: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %zero = arith.constant 0.0 : f64
  %r:2 = scf.while (%acc = %zero, %dx = %c0_i32) : (f64, i32) -> (f64, i32) {
    %i = arith.index_cast %dx : i32 to index
    %xv = memref.load %x[%i] : memref<8xf64>
    %sum = arith.addf %acc, %xv : f64
    %next = arith.addi %dx, %c1_i32 : i32
    %or = arith.ori %dx, %odd : i32
    %again = arith.cmpi eq, %or, %c0_i32 : i32
    scf.condition(%again) %sum, %next : f64, i32
  } do {
  ^bb0(%sum: f64, %next: i32):
    scf.yield %sum, %next : f64, i32
  }
  memref.store %r#0, %out[] : memref<f64>
  return
}

// CHECK:    func.func @ored_counter(%arg0: memref<8xf64>, %arg1: memref<4xf64>, %arg2: memref<f64>, %arg3: i32) {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      %1 = memref.load %arg0[%c0] : memref<8xf64>
// CHECK-NEXT:      %2 = memref.load %arg1[%c0] : memref<4xf64>
// CHECK-NEXT:      %3 = arith.mulf %1, %2 : f64
// CHECK-NEXT:      %4 = arith.addf %3, %cst : f64
// CHECK-NEXT:      %5 = arith.cmpi eq, %arg3, %c0_i32 : i32
// CHECK-NEXT:      %6 = scf.if %5 -> (f64) {
// CHECK-NEXT:        %7 = memref.load %arg0[%c1] : memref<8xf64>
// CHECK-NEXT:        %8 = memref.load %arg1[%c1] : memref<4xf64>
// CHECK-NEXT:        %9 = arith.mulf %7, %8 : f64
// CHECK-NEXT:        %10 = arith.addf %4, %9 : f64
// CHECK-NEXT:        scf.yield %10 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %4 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %6, %arg2[] : memref<f64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @starts_at_one(%arg0: memref<8xf64>, %arg1: memref<4xf64>, %arg2: memref<f64>, %arg3: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      %1:2 = scf.while (%arg4 = %cst, %arg5 = %c1_i32) : (f64, i32) -> (f64, i32) {
// CHECK-NEXT:        %2 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:        %3 = memref.load %arg0[%2] : memref<8xf64>
// CHECK-NEXT:        %4 = arith.addf %arg4, %3 : f64
// CHECK-NEXT:        %5 = arith.addi %arg5, %c1_i32 : i32
// CHECK-NEXT:        %6 = arith.ori %arg5, %arg3 : i32
// CHECK-NEXT:        %7 = arith.cmpi eq, %6, %c0_i32 : i32
// CHECK-NEXT:        scf.condition(%7) %4, %5 : f64, i32
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg4: f64, %arg5: i32):
// CHECK-NEXT:        scf.yield %arg4, %arg5 : f64, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1#0, %arg2[] : memref<f64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @steps_by_two(%arg0: memref<8xf64>, %arg1: memref<f64>, %arg2: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      %1:2 = scf.while (%arg3 = %cst, %arg4 = %c0_i32) : (f64, i32) -> (f64, i32) {
// CHECK-NEXT:        %2 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:        %3 = memref.load %arg0[%2] : memref<8xf64>
// CHECK-NEXT:        %4 = arith.addf %arg3, %3 : f64
// CHECK-NEXT:        %5 = arith.addi %arg4, %c2_i32 : i32
// CHECK-NEXT:        %6 = arith.ori %arg4, %arg2 : i32
// CHECK-NEXT:        %7 = arith.cmpi eq, %6, %c0_i32 : i32
// CHECK-NEXT:        scf.condition(%7) %4, %5 : f64, i32
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg3: f64, %arg4: i32):
// CHECK-NEXT:        scf.yield %arg3, %arg4 : f64, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1#0, %arg1[] : memref<f64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @masked(%arg0: memref<8xf64>, %arg1: memref<f64>, %arg2: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      %1:2 = scf.while (%arg3 = %cst, %arg4 = %c0_i32) : (f64, i32) -> (f64, i32) {
// CHECK-NEXT:        %2 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:        %3 = memref.load %arg0[%2] : memref<8xf64>
// CHECK-NEXT:        %4 = arith.addf %arg3, %3 : f64
// CHECK-NEXT:        %5 = arith.addi %arg4, %c1_i32 : i32
// CHECK-NEXT:        %6 = arith.andi %arg4, %arg2 : i32
// CHECK-NEXT:        %7 = arith.cmpi eq, %6, %c0_i32 : i32
// CHECK-NEXT:        scf.condition(%7) %4, %5 : f64, i32
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg3: f64, %arg4: i32):
// CHECK-NEXT:        scf.yield %arg3, %arg4 : f64, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1#0, %arg1[] : memref<f64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @host(%arg0: memref<8xf64>, %arg1: memref<f64>, %arg2: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = memref.load %arg0[%c0] : memref<8xf64>
// CHECK-NEXT:    %1 = arith.addf %0, %cst : f64
// CHECK-NEXT:    %2 = arith.cmpi eq, %arg2, %c0_i32 : i32
// CHECK-NEXT:    %3 = scf.if %2 -> (f64) {
// CHECK-NEXT:      %4 = memref.load %arg0[%c1] : memref<8xf64>
// CHECK-NEXT:      %5 = arith.addf %1, %4 : f64
// CHECK-NEXT:      scf.yield %5 : f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %1 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.store %3, %arg1[] : memref<f64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
