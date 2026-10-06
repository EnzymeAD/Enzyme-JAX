// RUN: enzymexlamlir-opt %s --affine-cfg 2>&1 | FileCheck %s

// The if simplification subtracts the domain inside an if from the one around
// it. The condition brings in a symbol, %nd, the domain around it lacks: isl
// aligns the two only by the names of their parameters, which are those of
// the values they stand for.
// CHECK-NOT: unnamed parameters

#set = affine_set<()[s0] : (s0 >= 0)>
func.func @guarded(%x: memref<?xf64>, %ne: index, %nd: index) {
  affine.if #set()[%nd] {
    affine.parallel (%e) = (0) to (symbol(%ne)) {
      %a = affine.load %x[%e] : memref<?xf64>
      affine.store %a, %x[%e + 1] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @guarded(%arg0: memref<?xf64>, %arg1: index, %arg2: index) {
// CHECK-NEXT:    affine.if #set()[%arg2] {
// CHECK-NEXT:      affine.parallel (%arg3) = (0) to (symbol(%arg1)) {
