// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// An affine.if yielding 1/0 is folded into the constraint system built for the
// select which consumes it. That system is symbol-only and expects two applies
// per constraint, while the if brings its own dim/symbol numbering and an
// unrelated number of constraints and operands, so both must be rebased.

// CHECK-DAG: #[[SET0:.+]] = affine_set<(d0)[s0] : (d0 - 5 >= 0, -d0 + s0 - 1 >= 0)>
// CHECK-DAG: #[[SET1:.+]] = affine_set<(d0)[s0, s1] : (d0 - s0 >= 0, -d0 + s1 - 1 >= 0)>
// CHECK-DAG: #[[SET2:.+]] = affine_set<(d0)[s0] : (d0 - 5 >= 0, -d0 + 90 >= 0, -d0 + s0 - 1 >= 0)>
// CHECK-DAG: #[[SET3:.+]] = affine_set<(d0, d1)[s0, s1] : (d0 + d1 - s0 >= 0, -d0 + s1 - 1 >= 0)>

module {
  // One dim, no symbols, one constraint: the dim must survive the rebase onto
  // the symbol space, and the comparison must keep referring to its own
  // operands rather than being shifted onto the if's.
  func.func @if_cond_dim(%out: memref<100xf64>, %a: f64, %b: f64, %n: index) {
    %true = arith.constant true
    %false = arith.constant false
    affine.for %i = 0 to 100 {
      %c = affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) -> i1 {
        affine.yield %true : i1
      } else {
        affine.yield %false : i1
      }
      %d = arith.cmpi slt, %i, %n : index
      %e = arith.andi %c, %d : i1
      %v = arith.select %e, %a, %b : f64
      affine.store %v, %out[%i] : memref<100xf64>
    }
    return
  }

// CHECK-LABEL:   func.func @if_cond_dim(
// CHECK-SAME:      %[[OUT:[^:]+]]: memref<100xf64>, %[[A:[^:]+]]: f64, %[[B:[^:]+]]: f64, %[[N:[^:]+]]: index
// CHECK:           affine.parallel (%[[I:.+]]) = (0) to (100) {
// CHECK-NEXT:        %[[V:.+]] = affine.if #[[SET0]](%[[I]])[%[[N]]] -> f64 {
// CHECK-NEXT:          affine.yield %[[A]] : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[B]] : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.store %[[V]], %[[OUT]][%[[I]]]

  // One dim and one symbol.
  func.func @if_cond_dim_symbol(%out: memref<100xf64>, %a: f64, %b: f64, %n: index, %m: index) {
    %true = arith.constant true
    %false = arith.constant false
    affine.for %i = 0 to 100 {
      %c = affine.if affine_set<(d0)[s0] : (d0 - s0 >= 0)>(%i)[%m] -> i1 {
        affine.yield %true : i1
      } else {
        affine.yield %false : i1
      }
      %d = arith.cmpi slt, %i, %n : index
      %e = arith.andi %c, %d : i1
      %v = arith.select %e, %a, %b : f64
      affine.store %v, %out[%i] : memref<100xf64>
    }
    return
  }

// CHECK-LABEL:   func.func @if_cond_dim_symbol(
// CHECK-SAME:      %[[OUT:[^:]+]]: memref<100xf64>, %[[A:[^:]+]]: f64, %[[B:[^:]+]]: f64, %[[N:[^:]+]]: index, %[[M:[^:]+]]: index
// CHECK:           affine.parallel (%[[I:.+]]) = (0) to (100) {
// CHECK-NEXT:        %[[V:.+]] = affine.if #[[SET1]](%[[I]])[%[[M]], %[[N]]] -> f64 {
// CHECK-NEXT:          affine.yield %[[A]] : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[B]] : f64
// CHECK-NEXT:        }

  // Two constraints but a single operand, so the system needs padding applies.
  func.func @if_cond_two_constraints(%out: memref<100xf64>, %a: f64, %b: f64, %n: index) {
    %true = arith.constant true
    %false = arith.constant false
    affine.for %i = 0 to 100 {
      %c = affine.if affine_set<(d0) : (d0 - 5 >= 0, -d0 + 90 >= 0)>(%i) -> i1 {
        affine.yield %true : i1
      } else {
        affine.yield %false : i1
      }
      %d = arith.cmpi slt, %i, %n : index
      %e = arith.andi %c, %d : i1
      %v = arith.select %e, %a, %b : f64
      affine.store %v, %out[%i] : memref<100xf64>
    }
    return
  }

// CHECK-LABEL:   func.func @if_cond_two_constraints(
// CHECK-SAME:      %[[OUT:[^:]+]]: memref<100xf64>, %[[A:[^:]+]]: f64, %[[B:[^:]+]]: f64, %[[N:[^:]+]]: index
// CHECK:           affine.parallel (%[[I:.+]]) = (0) to (100) {
// CHECK-NEXT:        %[[V:.+]] = affine.if #[[SET2]](%[[I]])[%[[N]]] -> f64 {
// CHECK-NEXT:          affine.yield %[[A]] : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[B]] : f64
// CHECK-NEXT:        }

  // More operands than two per constraint, so the system needs a padding
  // constraint as well.
  func.func @if_cond_many_operands(%out: memref<100x100xf64>, %a: f64, %b: f64, %n: index, %m: index) {
    %true = arith.constant true
    %false = arith.constant false
    affine.for %i = 0 to 100 {
      affine.for %j = 0 to 100 {
        %c = affine.if affine_set<(d0, d1)[s0] : (d0 + d1 - s0 >= 0)>(%i, %j)[%m] -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %d = arith.cmpi slt, %i, %n : index
        %e = arith.andi %c, %d : i1
        %v = arith.select %e, %a, %b : f64
        affine.store %v, %out[%i, %j] : memref<100x100xf64>
      }
    }
    return
  }

// CHECK-LABEL:   func.func @if_cond_many_operands(
// CHECK-SAME:      %[[OUT:[^:]+]]: memref<100x100xf64>, %[[A:[^:]+]]: f64, %[[B:[^:]+]]: f64, %[[N:[^:]+]]: index, %[[M:[^:]+]]: index
// CHECK:           affine.parallel (%[[I:.+]], %[[J:.+]]) = (0, 0) to (100, 100) {
// CHECK-NEXT:        %[[V:.+]] = affine.if #[[SET3]](%[[I]], %[[J]])[%[[M]], %[[N]]] -> f64 {
// CHECK-NEXT:          affine.yield %[[A]] : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[B]] : f64
// CHECK-NEXT:        }
}
