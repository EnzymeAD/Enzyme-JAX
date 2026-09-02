// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// A rotated do-while loop counts to max(n, 1) + 1. The min/max of valid
// symbols is itself a symbol: it is hoisted out of the affine scope like the
// rest of the bound arithmetic, so the loop raises with the max as a symbol
// operand rather than staying scf.for.

func.func @dowhile(%e : i32, %n : i32, %A : memref<?xf64>) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f64
  %ei = arith.index_cast %e : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%ei, %c1, %c1, %c256, %c1, %c1) ({
    affine.parallel (%b) = (0) to (symbol(%ei)) {
      %mx = arith.maxsi %n, %c1_i32 : i32
      %ub = arith.addi %mx, %c1_i32 : i32
      scf.for %i = %c1_i32 to %ub step %c1_i32 : i32 {
        %bi = arith.index_cast %b : index to i32
        %k = arith.muli %bi, %n : i32
        %k1 = arith.addi %k, %i : i32
        %idx = arith.index_cast %k1 : i32 to index
        memref.store %cst, %A[%idx] : memref<?xf64>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL:   func.func @dowhile(
// CHECK-SAME:      %[[e:.*]]: i32, %[[n:.*]]: i32, %[[A:.*]]: memref<?xf64>) {
// CHECK-DAG:       %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[ei:.*]] = arith.index_cast %[[e]] : i32 to index
// CHECK-DAG:       %[[mx:.*]] = arith.maxsi %[[n]], %[[c1_i32]] : i32
// CHECK-DAG:       %[[ni:.*]] = arith.index_cast %[[n]] : i32 to index
// CHECK:           "enzymexla.gpu_wrapper"
// CHECK:             %[[mxi:.*]] = arith.index_cast %[[mx]] : i32 to index
// CHECK:             affine.parallel (%[[b:.*]]) = (0) to (symbol(%[[ei]])) {
// CHECK:               affine.for %[[i:.*]] = 0 to %[[mxi]] {
// CHECK:                 affine.store %[[cst]], %[[A]][%[[i]] + %[[b]] * symbol(%[[ni]]) + 1] : memref<?xf64>
// CHECK:               }
// CHECK:             }
