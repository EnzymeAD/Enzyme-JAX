// RUN: enzymexlamlir-opt %s --canonicalize-parallel --allow-unregistered-dialect | FileCheck %s

// The strided copy loop after the range analysis folded k += stride to a
// constant: the exit test reads a carried argument yielded a constant, so
// the second evaluation decides.
func.func @strided(%buf: memref<32xf64>, %n: i32) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The same loop on the host may run forever, so it stays.
func.func @host(%buf: memref<32xf64>, %n: i32) {
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
func.func @forwarded_invariant(%buf: memref<32xf64>, %n: i32, %j: i32) {
  %c1 = arith.constant 1 : index
  %v = arith.constant 2.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The done flag: the after region's side effect lands between the two copies
// of the before region, and the constant condition then dissolves the if.
func.func @sideflag(%buf: memref<8xf64>) {
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %false = arith.constant false
  %v = arith.constant 2.0 : f64
  %w = arith.constant 3.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    scf.while (%flag = %true) : (i1) -> () {
      affine.store %v, %buf[0] : memref<8xf64>
      scf.condition(%flag)
    } do {
      affine.store %w, %buf[1] : memref<8xf64>
      scf.yield %false : i1
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// A real strided loop: the yield varies with the carried state.
func.func @varying(%buf: memref<32xf64>, %n: i32) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The exit test reads memory the body writes, so the second evaluation's
// decision is not the third's.
func.func @cond_reads_memory(%buf: memref<32xf64>, %flags: memref<32xi1>) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The yield is computed in the after region from a value that varies.
func.func @after_varying(%buf: memref<32xf64>, %n: i32) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : i32
  %v = arith.constant 2.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The accumulator is carried along, but the exit test reads only the counter,
// whose next value is invariant.
func.func @accumulating(%buf: memref<32xf64>, %out: memref<f64>, %n: i32, %k0: i32) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : i32
  %zero = arith.constant 0.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    memref.store %r, %out[] : memref<f64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// The exit test reads the carried accumulator, which varies.
func.func @cond_reads_carried(%buf: memref<32xf64>, %out: memref<f64>, %n: i32, %k0: i32) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : i32
  %zero = arith.constant 0.0 : f64
  %lim = arith.constant 1.0 : f64
  %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
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
    memref.store %r, %out[] : memref<f64>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:    func.func @strided(%arg0: memref<32xf64>, %arg1: i32) {
// CHECK-NEXT:    %c16 = arith.constant 16 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:        %1 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:        %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:        memref.store %cst, %arg0[%2] : memref<32xf64>
// CHECK-NEXT:        %3 = arith.cmpi ult, %1, %arg1 : i32
// CHECK-NEXT:        scf.if %3 {
// CHECK-NEXT:          memref.store %cst, %arg0[%c16] : memref<32xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @host(%arg0: memref<32xf64>, %arg1: i32) {
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:      %0 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:      scf.while (%arg3 = %0) : (i32) -> () {
// CHECK-NEXT:        %1 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:        memref.store %cst, %arg0[%1] : memref<32xf64>
// CHECK-NEXT:        %2 = arith.cmpi ult, %arg3, %arg1 : i32
// CHECK-NEXT:        scf.condition(%2)
// CHECK-NEXT:      } do {
// CHECK-NEXT:        scf.yield %c16_i32 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @forwarded_invariant(%arg0: memref<32xf64>, %arg1: i32, %arg2: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg3) = (0) to (16) {
// CHECK-NEXT:        %1 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:        %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:        memref.store %cst, %arg0[%2] : memref<32xf64>
// CHECK-NEXT:        %3 = arith.cmpi ult, %1, %arg1 : i32
// CHECK-NEXT:        scf.if %3 {
// CHECK-NEXT:          %4 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:          memref.store %cst, %arg0[%4] : memref<32xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @sideflag(%arg0: memref<8xf64>) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.store %cst, %arg0[0] : memref<8xf64>
// CHECK-NEXT:      affine.store %cst_0, %arg0[1] : memref<8xf64>
// CHECK-NEXT:      affine.store %cst, %arg0[0] : memref<8xf64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @varying(%arg0: memref<32xf64>, %arg1: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:        %1 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:        %2 = scf.while (%arg3 = %1) : (i32) -> i32 {
// CHECK-NEXT:          %3 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:          memref.store %cst, %arg0[%3] : memref<32xf64>
// CHECK-NEXT:          %4 = arith.cmpi ult, %arg3, %arg1 : i32
// CHECK-NEXT:          %5 = arith.addi %arg3, %c16_i32 : i32
// CHECK-NEXT:          scf.condition(%4) %5 : i32
// CHECK-NEXT:        } do {
// CHECK-NEXT:        ^bb0(%arg3: i32):
// CHECK-NEXT:          scf.yield %arg3 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @cond_reads_memory(%arg0: memref<32xf64>, %arg1: memref<32xi1>) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:        %1 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:        scf.while (%arg3 = %1) : (i32) -> () {
// CHECK-NEXT:          %2 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:          memref.store %cst, %arg0[%2] : memref<32xf64>
// CHECK-NEXT:          %3 = memref.load %arg1[%2] : memref<32xi1>
// CHECK-NEXT:          scf.condition(%3)
// CHECK-NEXT:        } do {
// CHECK-NEXT:          scf.yield %c16_i32 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @after_varying(%arg0: memref<32xf64>, %arg1: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (16) {
// CHECK-NEXT:        %1 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:        %2 = scf.while (%arg3 = %1) : (i32) -> i32 {
// CHECK-NEXT:          %3 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:          memref.store %cst, %arg0[%3] : memref<32xf64>
// CHECK-NEXT:          %4 = arith.cmpi ult, %arg3, %arg1 : i32
// CHECK-NEXT:          scf.condition(%4) %arg3 : i32
// CHECK-NEXT:        } do {
// CHECK-NEXT:        ^bb0(%arg3: i32):
// CHECK-NEXT:          %3 = arith.addi %arg3, %c16_i32 : i32
// CHECK-NEXT:          scf.yield %3 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @accumulating(%arg0: memref<32xf64>, %arg1: memref<f64>, %arg2: i32, %arg3: i32) {
// CHECK-NEXT:    %c16 = arith.constant 16 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      %1 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:      %2 = memref.load %arg0[%1] : memref<32xf64>
// CHECK-NEXT:      %3 = arith.addf %2, %cst : f64
// CHECK-NEXT:      %4 = arith.cmpi ult, %arg3, %arg2 : i32
// CHECK-NEXT:      %5 = scf.if %4 -> (f64) {
// CHECK-NEXT:        %6 = memref.load %arg0[%c16] : memref<32xf64>
// CHECK-NEXT:        %7 = arith.addf %3, %6 : f64
// CHECK-NEXT:        scf.yield %7 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %3 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %5, %arg1[] : memref<f64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @cond_reads_carried(%arg0: memref<32xf64>, %arg1: memref<f64>, %arg2: i32, %arg3: i32) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
// CHECK-NEXT:      %1 = scf.while (%arg4 = %arg3, %arg5 = %cst) : (i32, f64) -> f64 {
// CHECK-NEXT:        %2 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:        %3 = memref.load %arg0[%2] : memref<32xf64>
// CHECK-NEXT:        %4 = arith.addf %arg5, %3 : f64
// CHECK-NEXT:        %5 = arith.cmpf olt, %4, %cst_0 : f64
// CHECK-NEXT:        scf.condition(%5) %4 : f64
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg4: f64):
// CHECK-NEXT:        scf.yield %c16_i32, %arg4 : i32, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.store %1, %arg1[] : memref<f64>
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }
