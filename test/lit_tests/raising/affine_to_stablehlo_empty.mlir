// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

module {
  func.func @myfunc() {
    return
  }
}
  
// CHECK:  func.func @myfunc() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
