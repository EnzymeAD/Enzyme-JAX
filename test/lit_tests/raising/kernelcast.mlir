// RUN: enzymexlamlir-opt --pass-pipeline='builtin.module(func.func(kernelcast))' %s | FileCheck %s

// CHECK-LABEL: func.func @cast_f32_to_bf16
// CHECK-SAME:    (%arg0: memref<10xbf16>, %arg1: memref<10xbf16>)
// CHECK-NOT:     enzymexla.float_type
func.func @cast_f32_to_bf16(%arg0: memref<10xf32>, %arg1: memref<10xf32>)
    attributes {"enzymexla.float_type" = bf16} {
  affine.parallel (%i) = (0) to (10) {
    // CHECK:       %[[V:.+]] = affine.load %arg0[%arg2] : memref<10xbf16>
    %v = affine.load %arg0[%i] : memref<10xf32>
    // CHECK-NEXT:  %[[CST:.+]] = arith.constant 1.000000e+00 : f32
    // CHECK-NEXT:  %[[TC:.+]] = arith.truncf %[[CST]] : f32 to bf16
    %c = arith.constant 1.0 : f32
    // CHECK-NEXT:  %[[R:.+]] = arith.addf %[[V]], %[[TC]] : bf16
    %r = arith.addf %v, %c : f32
    // CHECK-NEXT:  affine.store %[[R]], %arg1[%arg2] : memref<10xbf16>
    affine.store %r, %arg1[%i] : memref<10xf32>
  }
  return
}

// Function without the attribute must be left unchanged.
// CHECK-LABEL: func.func @no_cast
// CHECK-SAME:    (%arg0: memref<10xbf16>)
func.func @no_cast(%arg0: memref<10xf32>) -> f32
    attributes {"enzymexla.float_type" = bf16} {
  // CHECK: %[[V:.+]] = affine.load %arg0[0] : memref<10xbf16>
  %v = affine.load %arg0[0] : memref<10xf32>
  // CHECK-NEXT: return %[[V]] : bf16
  return %v : f32
}

// Upcast from bf16 to f32: constant should use arith.extf.
// CHECK-LABEL: func.func @cast_bf16_to_f32
// CHECK-SAME:    (%arg0: memref<4xf32>, %arg1: memref<4xf32>)
// CHECK-NOT:     enzymexla.float_type
func.func @cast_bf16_to_f32(%arg0: memref<4xbf16>, %arg1: memref<4xbf16>)
    attributes {"enzymexla.float_type" = f32} {
  affine.parallel (%i) = (0) to (4) {
    // CHECK:       %[[V:.+]] = affine.load %arg0[%arg2] : memref<4xf32>
    %v = affine.load %arg0[%i] : memref<4xbf16>
    // CHECK-NEXT:  %[[CST:.+]] = arith.constant 2.000000e+00 : bf16
    // CHECK-NEXT:  %[[EC:.+]] = arith.extf %[[CST]] : bf16 to f32
    %c = arith.constant 2.0 : bf16
    // CHECK-NEXT:  %[[R:.+]] = arith.mulf %[[V]], %[[EC]] : f32
    %r = arith.mulf %v, %c : bf16
    // CHECK-NEXT:  affine.store %[[R]], %arg1[%arg2] : memref<4xf32>
    affine.store %r, %arg1[%i] : memref<4xbf16>
  }
  return
}

// Noop cast elision: converting f32 kernel to f32 must not leave extf/truncf.
// CHECK-LABEL: func.func @noop_cast
// CHECK-SAME:    (%arg0: memref<4xf32>, %arg1: memref<4xf32>)
// CHECK-NOT:     enzymexla.float_type
func.func @noop_cast(%arg0: memref<4xf32>, %arg1: memref<4xf32>)
    attributes {"enzymexla.float_type" = f32} {
  affine.parallel (%i) = (0) to (4) {
    // CHECK:       %[[V:.+]] = affine.load %arg0[%arg2] : memref<4xf32>
    %v = affine.load %arg0[%i] : memref<4xf32>
    // CHECK-NEXT:  %[[CST:.+]] = arith.constant 1.000000e+00 : f32
    %c = arith.constant 1.0 : f32
    // CHECK-NEXT:  %[[R:.+]] = arith.addf %[[V]], %[[CST]] : f32
    // CHECK-NOT:   arith.extf
    // CHECK-NOT:   arith.truncf
    %r = arith.addf %v, %c : f32
    // CHECK:       affine.store %[[R]], %arg1[%arg2] : memref<4xf32>
    affine.store %r, %arg1[%i] : memref<4xf32>
  }
  return
}
