// RUN: enzymexlamlir-opt %s -strip-perfify | FileCheck %s

// Every Perfify op goes, including those nested in a loop body and the
// module-level symbols and assumptions; the loop and its other ops stay.

// CHECK-NOT: perfify.
// CHECK-LABEL: func.func @loop
// CHECK:         affine.for
// CHECK-NEXT:      memref.store
// CHECK-NEXT:    }
// CHECK-NOT: perfify.
module {
  perfify.define_symbol @N : i64
  func.func @loop(%n: index, %m: memref<?xf64>, %v: f64) {
    affine.for %i = 0 to %n {
      %a = perfify.arg 0
      perfify.estimated_cost @trip_cost {
        %s = perfify.sym_cost @N : i64
        perfify.yield %s : i64
      }
      %c = perfify.cmp le, %a, @trip_cost
      %h = perfify.hypothesis %c
      memref.store %v, %m[%i] : memref<?xf64>
    }
    return
  }
  perfify.assumptions {
    perfify.cost "*" {
      %c1 = arith.constant 1 : i64
      perfify.yield %c1 : i64
    }
  }
}
