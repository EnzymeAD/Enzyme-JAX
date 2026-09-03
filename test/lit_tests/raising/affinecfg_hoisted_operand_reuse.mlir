// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// The select is hoisted (cloned and erased) while legalizing the first index;
// the other index must see the clone, not the erased op.

module {
  func.func @twice(%m: memref<?x?xf64>, %s: index, %t: index, %n: index, %c: i1) {
    affine.for %i = 0 to %n {
      %a = arith.select %c, %s, %t : index
      %v = memref.load %m[%a, %a] : memref<?x?xf64>
      affine.store %v, %m[%i, 0] : memref<?x?xf64>
    }
    return
  }

  func.func @derived(%m: memref<?x?xf64>, %s: index, %t: index, %n: index, %c: i1) {
    %c1 = arith.constant 1 : index
    affine.for %i = 0 to %n {
      %a = arith.select %c, %s, %t : index
      %b = arith.addi %a, %c1 : index
      %v = memref.load %m[%b, %a] : memref<?x?xf64>
      affine.store %v, %m[%i, 0] : memref<?x?xf64>
    }
    return
  }
}

// CHECK-LABEL:   func.func @twice(
// CHECK-SAME:      %[[M:.*]]: memref<?x?xf64>, %[[S:.*]]: index, %[[T:.*]]: index, %[[N:.*]]: index, %[[C:.*]]: i1) {
// CHECK:           %[[SEL:.*]] = arith.select %[[C]], %[[S]], %[[T]] : index
// CHECK:           affine.for %[[I:.*]] = 0 to %[[N]] {
// CHECK:             %[[V:.*]] = affine.load %[[M]][symbol(%[[SEL]]), symbol(%[[SEL]])] : memref<?x?xf64>
// CHECK:             affine.store %[[V]], %[[M]][%[[I]], 0] : memref<?x?xf64>

// CHECK-LABEL:   func.func @derived(
// CHECK-SAME:      %[[M:.*]]: memref<?x?xf64>, %[[S:.*]]: index, %[[T:.*]]: index, %[[N:.*]]: index, %[[C:.*]]: i1) {
// CHECK:           %[[SEL:.*]] = arith.select %[[C]], %[[S]], %[[T]] : index
// CHECK:           affine.for %[[I:.*]] = 0 to %[[N]] {
// CHECK:             %[[V:.*]] = affine.load %[[M]][symbol(%[[SEL]]) + 1, symbol(%[[SEL]])] : memref<?x?xf64>
// CHECK:             affine.store %[[V]], %[[M]][%[[I]], 0] : memref<?x?xf64>
