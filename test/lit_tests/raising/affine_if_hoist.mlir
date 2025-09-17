// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module {
  func.func @main(%i: index) -> f32 {
    %0 = arith.constant 1.0 : f32
    %1 = affine.if affine_set<(d0): (d0 >= 12, -d0 >= -15)>(%i) -> f32 {
      affine.yield %0 : f32
    } else {
      affine.yield %0 : f32
    }
    return %1 : f32
  }
}

// CHECK:  func.func @main(%[[ARG:.+]]: index) -> f32 {
// CHECK-NEXT:    %[[C1:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    return %[[C1]] : f32
// CHECK-NEXT:  }
