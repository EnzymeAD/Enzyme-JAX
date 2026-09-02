// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// Clang's short-circuit lowering of `n >= 1 && m >= 1` arrives as
// !(n < 1 || (n >= 1 && m < 1)). The middle test is the negation of the
// first and absorbs: or(!a, and(a, b)) -> or(!a, b), which the affine.if
// conversion then reads as a conjunction of the two negated comparisons.

func.func @absorb(%e : i32, %n : i32, %m : i32, %A : memref<?xf64>) {
  %true = arith.constant true
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f64
  %ei = arith.index_cast %e : i32 to index
  %0 = "enzymexla.gpu_wrapper"(%ei, %c1, %c1, %c256, %c1, %c1) ({
    affine.parallel (%b) = (0) to (symbol(%ei)) {
      %a = arith.cmpi slt, %n, %c1_i32 : i32
      %na = arith.cmpi sge, %n, %c1_i32 : i32
      %bl = arith.cmpi slt, %m, %c1_i32 : i32
      %and = arith.andi %na, %bl : i1
      %or = arith.ori %a, %and : i1
      %cond = arith.xori %or, %true : i1
      scf.if %cond {
        memref.store %cst, %A[%b] : memref<?xf64>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK: #[[set:.+]] = affine_set<()[s0, s1] : (s0 - 1 >= 0, s1 - 1 >= 0)>
// CHECK-LABEL:   func.func @absorb(
// CHECK-SAME:      %[[e:.*]]: i32, %[[n:.*]]: i32, %[[m:.*]]: i32, %[[A:.*]]: memref<?xf64>) {
// CHECK-DAG:       %[[ni:.*]] = arith.index_cast %[[n]] : i32 to index
// CHECK-DAG:       %[[mi:.*]] = arith.index_cast %[[m]] : i32 to index
// CHECK:           affine.parallel (%[[b:.*]]) = (0) to (symbol(%{{.*}})) {
// CHECK-NOT:         scf.if
// CHECK:             affine.if #[[set]]()[%[[ni]], %[[mi]]] {
// CHECK:               affine.store %{{.*}}, %[[A]][%[[b]]] : memref<?xf64>
