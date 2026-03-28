// RUN: enzymexlamlir-opt --affine-cfg %s 2>&1 | FileCheck %s

module {
  func.func @test_isl_named_params(%N: index, %buf: memref<?xf32>) {
    %cst = arith.constant 0.0 : f32
    affine.for %i = 0 to %N {
      affine.if affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>(%i)[%N] {
        affine.store %cst, %buf[%i] : memref<?xf32>
      }
    }
    return
  }
}

// CHECK-NOT: unexpected unnamed parameters

// CHECK-LABEL: func.func @test_isl_named_params
// CHECK-NOT: affine.if
// CHECK: affine.store
// CHECK: return
