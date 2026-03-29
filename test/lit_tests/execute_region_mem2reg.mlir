// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(func.func(polygeist-mem2reg))' %s | FileCheck %s

module {
  func.func @test_execute_region_void() -> f32 {
    %A = memref.alloca() : memref<f32>
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %A[] : memref<f32>

    scf.execute_region {
      %cond = arith.constant true
      cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
      %c1 = arith.constant 1.0 : f32
      memref.store %c1, %A[] : memref<f32>
      scf.yield
    ^bb2:
      scf.yield
    }

    %val = memref.load %A[] : memref<f32>
    return %val : f32
  }
}

// CHECK:      func.func @test_execute_region_void() -> f32 {
// CHECK-NEXT:   %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[RES:.*]] = scf.execute_region -> f32 {
// CHECK-NEXT:     %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:     cf.cond_br %[[TRUE]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK-NEXT:   ^[[BB1]]:
// CHECK-NEXT:     %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     scf.yield %[[CST_1]] : f32
// CHECK-NEXT:   ^[[BB2]]:
// CHECK-NEXT:     scf.yield %[[CST_0]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[RES]] : f32
// CHECK-NEXT: }
