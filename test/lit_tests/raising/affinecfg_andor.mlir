// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s


module {
  func.func @_Z7runTestiPPc(%a : i1, %b : i1) -> i1 {
      %true = arith.constant true
      %nota = arith.xori %a, %true : i1
      %orab = arith.ori %a, %b : i1
      %res = arith.andi %nota, %orab : i1
      return %res : i1
  }
  func.func @_Z7runTestiPPc2(%a : i1, %b : i1) -> i1 {
      %orab = arith.ori %a, %b : i1
      %res = arith.andi %a, %orab : i1
      return %res : i1
  }
}

// CHECK:  func.func @_Z7runTestiPPc(%arg0: i1, %arg1: i1) -> i1 {
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %0 = arith.xori %arg0, %true : i1
// CHECK-NEXT:    %1 = arith.andi %0, %arg1 : i1
// CHECK-NEXT:    return %1 : i1
// CHECK-NEXT:  }

// CHECK:  func.func @_Z7runTestiPPc2(%arg0: i1, %arg1: i1) -> i1 {
// CHECK-NEXT:    return %arg0 : i1
// CHECK-NEXT:  }
