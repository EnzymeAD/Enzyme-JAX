// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// One store and one load is not yet a forwarding: the store has to have run
// first on every path. Here the store sits on one side of a branch and the
// load below the merge -- on the other path the slot was never written, and
// the load must stay a load. (With exception handling kept in cf form, shapes
// like this reach the raising; the old test was whether the stored value was
// *available* at the load, which a value from a skipped path can be.)

module {
  func.func @no_forward(%c: i1, %v: f64) -> f64 {
    %a = memref.alloca() : memref<1xf64>
    %z = arith.constant 0 : index
    cf.cond_br %c, ^store, ^join
  ^store:
    memref.store %v, %a[%z] : memref<1xf64>
    cf.br ^join
  ^join:
    %r = memref.load %a[%z] : memref<1xf64>
    return %r : f64
  }

  func.func @forward(%v: f64) -> f64 {
    %a = memref.alloca() : memref<1xf64>
    %z = arith.constant 0 : index
    memref.store %v, %a[%z] : memref<1xf64>
    %r = memref.load %a[%z] : memref<1xf64>
    return %r : f64
  }
}

// CHECK-LABEL: func.func @no_forward
// CHECK:         memref.store
// CHECK:       ^{{.*}}:
// CHECK:         %[[R:.+]] = {{(affine|memref)\.load}}
// CHECK:         return %[[R]]

// CHECK-LABEL: func.func @forward
// CHECK-NOT:     memref.load
// CHECK:         return %arg0 : f64
