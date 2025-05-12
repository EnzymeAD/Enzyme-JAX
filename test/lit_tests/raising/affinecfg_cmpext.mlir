// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s
module {
  func.func @_Z7runTestiPPc(%26 : i1) -> i1 {
      %c0_i64 = arith.constant 0 : i64
      %27 = arith.extui %26 : i1 to i64
      %28 = arith.cmpi eq, %27, %c0_i64 : i64
      return %28 : i1
  }
}

// CHECK:  func.func @_Z7runTestiPPc(%arg0: i1) -> i1 {
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %0 = arith.xori %arg0, %true : i1
// CHECK-NEXT:    return %0 : i1
// CHECK-NEXT:  }
