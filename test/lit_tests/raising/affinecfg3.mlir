// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @sd() {
    %c-180 = arith.constant -180 : index
    affine.parallel (%arg0, %arg1) = (0, 0) to (12, 256) {
      %cmp = arith.cmpi uge, %arg0, %c-180 : index
      scf.if %cmp {
        "test.op"() : () -> ()
      }
    }
    return
  }
}

// CHECK: affine_set<(d0) : (d0 + 180 >= 0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @sd() {
// CHECK-NEXT:     affine.parallel (%arg0, %arg1) = (0, 0) to (12, 256) {
// CHECK-NEXT:       affine.if #set(%arg0) {
// CHECK-NEXT:         "test.op"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

