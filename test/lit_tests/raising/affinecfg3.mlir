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

// CK: affine_set<(d0) : (d0 + 180 >= 0)>
// CK-NEXT: module {
// CK-NEXT:   func.func @sd() {
// CK-NEXT:     affine.parallel (%arg0, %arg1) = (0, 0) to (12, 256) {
// CK-NEXT:       affine.if #set(%arg0) {
// CK-NEXT:         "test.op"() : () -> ()
// CK-NEXT:       }
// CK-NEXT:     }
// CK-NEXT:     return
// CK-NEXT:   }
// CK-NEXT: }

// CHECK: module {
// CHECK-NEXT:   func.func @sd() {
// CHECK-NEXT:     affine.parallel (%arg0, %arg1) = (0, 0) to (12, 256) {
// CHECK-NEXT:       "test.op"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

