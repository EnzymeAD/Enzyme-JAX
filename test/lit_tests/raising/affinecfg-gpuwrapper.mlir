// RUN: enzymexlamlir-opt %s --affine-cfg -allow-unregistered-dialect --mlir-print-local-scope | FileCheck %s

module {
  func.func @split_iv(%arg0: memref<10000xi64>, %23 : index, %tostore: i64) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %24 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %23, %c1, %c1) ({
      %119 = "test.op"() : () -> index
      affine.parallel (%i) = (0) to (10) {
        %124 = arith.shli %i, %c4 : index
        %125 = arith.ori %124, %119 {isDisjoint} : index
        memref.store %tostore, %arg0[%125] : memref<10000xi64>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK:  func.func @split_iv(%arg0: memref<10000xi64>, %arg1: index, %arg2: i64) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %arg1, %c1, %c1) ({
// CHECK-NEXT:      %1 = "test.op"() : () -> index
// CHECK-NEXT:      affine.parallel (%arg3) = (0) to (10) {
// CHECK-NEXT:        affine.store %arg2, %arg0[%arg3 * 16 + symbol(%1)] : memref<10000xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    return
// CHECK-NEXT:  }