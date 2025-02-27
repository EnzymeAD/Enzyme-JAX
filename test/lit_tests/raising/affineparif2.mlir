// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @f() {
    affine.parallel (%arg0, %arg1, %arg2) = (0, 0, 0) to (6, 16, 192) {
      affine.if affine_set<(d0, d1, d2) : (-d0 - d1 * 16 + 86 >= 0, d2 >= 0, -d2 + 181 >= 0)>(%arg1, %arg0, %arg2) {
        "test.op"() : () -> ()
      }
    }
    func.return
  }
}

// CHECK:    affine.parallel (%arg0, %arg1) = (0, 0) to (87, 182) {
// CHECK-NEXT:      "test.op"() : () -> ()
// CHECK-NEXT:    }

