// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @oneindex() {
    %c1_i64 = arith.constant 1 : i64
    %c51_i64 = arith.constant 51 : i64
    scf.for %arg34 = %c1_i64 to %c51_i64 step %c1_i64  : i64 {
      "test.use"(%arg34) : (i64) -> () 
    }
    return
  }
}

// CHECK:  func.func @oneindex() {
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    affine.for %arg0 = 0 to 50 {
// CHECK-NEXT:      %0 = arith.index_cast %arg0 : index to i64
// CHECK-NEXT:      %1 = arith.addi %0, %c1_i64 : i64
// CHECK-NEXT:      "test.use"(%1) : (i64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
