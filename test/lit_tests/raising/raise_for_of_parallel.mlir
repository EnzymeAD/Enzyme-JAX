// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A sequential loop whose trip count is only known at runtime iterates as a
// stablehlo.while; the parallel body raises batched inside it and the buffers
// it writes are carried through the loop.
func.func @timeloop(%out: memref<100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %t = 0 to %ni {
    affine.parallel (%i) = (0) to (100) {
      %v = affine.load %out[%i] : memref<100xf64, 1>
      %c = arith.constant 1.0 : f64
      %s = arith.addf %v, %c : f64
      affine.store %s, %out[%i] : memref<100xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @timeloop_raised(
// CHECK-SAME: %[[OUT:.+]]: tensor<100xf64>, %[[N:.+]]: tensor<i64>
// CHECK: %[[WHILE:.+]]:3 = stablehlo.while(%[[IV:.+]] = %{{.+}}, %[[BUF:.+]] = %[[OUT]], %{{.+}} = %[[N]]) : tensor<i64>, tensor<100xf64>, tensor<i64>
// CHECK: cond {
// CHECK: stablehlo.compare LT, %[[IV]], %{{.+}} : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK: } do {
// CHECK: %[[CUR:.+]] = stablehlo.reshape %[[BUF]] : (tensor<100xf64>) -> tensor<100xf64>
// CHECK: arith.addf %[[CUR]], %{{.+}} : tensor<100xf64>
// CHECK: return %[[WHILE]]#1, %[[WHILE]]#2 : tensor<100xf64>, tensor<i64>

// -----

// Scratch allocated outside the loop is carried through the while like any
// other buffer, starting from a zero splat.
func.func @scratchloop(%out: memref<10xf64, 1>, %nbuf: memref<i64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %t = 0 to %ni {
    affine.parallel (%i) = (0) to (10) {
      %v = affine.load %out[%i] : memref<10xf64, 1>
      affine.store %v, %tmp[%i] : memref<10xf64>
    }
    affine.parallel (%i) = (0) to (10) {
      %v = affine.load %tmp[9 - %i] : memref<10xf64>
      affine.store %v, %out[%i] : memref<10xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @scratchloop_raised(
// CHECK-SAME: %[[OUT2:.+]]: tensor<10xf64>, %[[N2:.+]]: tensor<i64>
// CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: stablehlo.while
// CHECK: } do {
// CHECK: stablehlo.reverse
