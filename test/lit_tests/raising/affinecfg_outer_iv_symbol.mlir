// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// The if inside the wrapper tests a cast of the enclosing loop's induction
// variable, a symbol from the wrapper's scope.

module {
  func.func @f(%n : i64, %m : memref<?xi64>) {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %s = arith.index_cast %n : i64 to index
    affine.for %i = 1 to %s {
      %ic = arith.index_cast %i : index to i64
      %j = arith.index_cast %ic : i64 to index
      %w = "enzymexla.gpu_wrapper"(%c1, %c1, %c256, %c1, %c1, %c1) ({
        affine.parallel (%t) = (0) to (256) {
          affine.if affine_set<()[s0] : (s0 - 3 >= 0)>()[%j] {
            affine.store %ic, %m[%t] : memref<?xi64>
          }
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index
    }
    return
  }
}

// CHECK: #[[SET:.+]] = affine_set<(d0) : (d0 - 3 >= 0)>
// CHECK-LABEL: func.func @f
// CHECK:         affine.for %[[I:.+]] = 1 to %{{.*}} {
// CHECK-NEXT:      %[[IC:.+]] = arith.index_cast %[[I]] : index to i64
// CHECK:           affine.parallel (%[[T:.+]]) = (0) to (256) {
// CHECK-NEXT:        affine.if #[[SET]](%[[I]]) {
// CHECK-NEXT:          affine.store %[[IC]], %{{.*}}[%[[T]]] : memref<?xi64>
