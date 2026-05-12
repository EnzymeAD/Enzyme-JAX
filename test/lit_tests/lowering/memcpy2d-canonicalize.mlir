// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize))" | FileCheck %s

module {
  func.func @test_memcpy2d_to_memcpy(%dst: memref<?xi8>, %src: memref<?xi8, 1>, %width: index) {
    %c1 = arith.constant 1 : index
    // height = 1, should simplify
    enzymexla.memcpy2d %dst, %width, %src, %width, %width, %c1 : memref<?xi8>, memref<?xi8, 1>
    return
  }

  func.func @test_memcpy2d_to_memcpy_all_const(%dst: memref<?xi8>, %src: memref<?xi8, 1>) {
    %c10 = arith.constant 10 : index
    %c2 = arith.constant 2 : index
    // dpitch == width, spitch == width, height > 1
    enzymexla.memcpy2d %dst, %c10, %src, %c10, %c10, %c2 : memref<?xi8>, memref<?xi8, 1>
    return
  }

  func.func @test_memcpy2d_to_memcpy_mixed(%dst: memref<?xi8>, %src: memref<?xi8, 1>, %height: index) {
    %c10 = arith.constant 10 : index
    // dpitch == width, spitch == width, height is dynamic
    enzymexla.memcpy2d %dst, %c10, %src, %c10, %c10, %height : memref<?xi8>, memref<?xi8, 1>
    return
  }

  func.func @test_memcpy2d_no_simplify(%dst: memref<?xi8>, %src: memref<?xi8, 1>, %height: index) {
    %c10 = arith.constant 10 : index
    %c12 = arith.constant 12 : index
    // dpitch != width, cannot simplify
    enzymexla.memcpy2d %dst, %c12, %src, %c10, %c10, %height : memref<?xi8>, memref<?xi8, 1>
    return
  }
}

// CHECK-LABEL: func.func @test_memcpy2d_to_memcpy
// CHECK:         enzymexla.memcpy  %arg0, %arg1, %arg2 : memref<?xi8>, memref<?xi8, 1>

// CHECK-LABEL: func.func @test_memcpy2d_to_memcpy_all_const
// CHECK:         %[[C20:.+]] = arith.constant 20 : index
// CHECK:         enzymexla.memcpy  %arg0, %arg1, %[[C20]] : memref<?xi8>, memref<?xi8, 1>

// CHECK-LABEL: func.func @test_memcpy2d_to_memcpy_mixed
// CHECK:         %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[SIZE:.+]] = arith.muli %arg2, %[[C10]] : index
// CHECK:         enzymexla.memcpy  %arg0, %arg1, %[[SIZE]] : memref<?xi8>, memref<?xi8, 1>

// CHECK-LABEL: func.func @test_memcpy2d_no_simplify
// CHECK:         enzymexla.memcpy2d
