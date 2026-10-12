// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A kernel reads and writes a complex number as its [2 x T] aggregate. Passed
// a complex buffer as interleaved reals, its aggregate view of that buffer
// becomes accesses of the reals, one per field, on the buffer itself where
// its shape spells out the aggregates.

func.func @real_to_complex(%out: memref<4x2xf64, 1>, %in: memref<4xf64, 1>) {
  %zero = arith.constant 0.000000e+00 : f64
  %c = llvm.mlir.constant(dense<0.000000e+00> : tensor<2xf64>) : !llvm.array<2 x f64>
  %p = "enzymexla.memref2pointer"(%out) : (memref<4x2xf64, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (4) {
    %x = affine.load %in[%i] : memref<4xf64, 1>
    %a = llvm.insertvalue %x, %c[0] : !llvm.array<2 x f64>
    %b = llvm.insertvalue %zero, %a[1] : !llvm.array<2 x f64>
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f64>, 1>
    affine.store %b, %v[%i] : memref<?x!llvm.array<2 x f64>, 1>
  }
  return
}

// CHECK-LABEL: func.func @real_to_complex
// CHECK-SAME: (%[[OUT:.+]]: memref<4x2xf64, 1>, %[[IN:.+]]: memref<4xf64, 1>)
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%[[I:.+]]) = (0) to (4) {
// CHECK-NEXT:      %[[X:.+]] = affine.load %[[IN]][%[[I]]] : memref<4xf64, 1>
// CHECK-NEXT:      affine.store %[[X]], %[[OUT]][%[[I]], 0] : memref<4x2xf64, 1>
// CHECK-NEXT:      affine.store %[[ZERO]], %[[OUT]][%[[I]], 1] : memref<4x2xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// A complex constant stored whole (out[i] = 1 + 2im) is a constant aggregate:
// the extractvalue of each field folds to a scalar llvm.mlir.constant, which
// the pass raises to an arith constant, the raising of the input's constants
// having run long before.
func.func @complex_constant(%out: memref<8x2xf32, 1>) {
  %c = llvm.mlir.constant(dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>) : !llvm.array<2 x f32>
  %p = "enzymexla.memref2pointer"(%out) : (memref<8x2xf32, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (8) {
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f32>, 1>
    affine.store %c, %v[%i] : memref<?x!llvm.array<2 x f32>, 1>
  }
  return
}

// CHECK-LABEL: func.func @complex_constant
// CHECK-SAME: (%[[OUT:.+]]: memref<8x2xf32, 1>)
// CHECK-NOT:     llvm.mlir.constant
// CHECK-DAG:     %[[RE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[IM:.+]] = arith.constant 2.000000e+00 : f32
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (8) {
// CHECK-NEXT:      affine.store %[[RE]], %[[OUT]][%[[I]], 0] : memref<8x2xf32, 1>
// CHECK-NEXT:      affine.store %[[IM]], %[[OUT]][%[[I]], 1] : memref<8x2xf32, 1>
// CHECK-NEXT:    }
// CHECK-NOT:     llvm.mlir.constant

// A complex zero (out[i] = 0) is a zero-initialized aggregate.
func.func @complex_zero(%out: memref<8x2xf32, 1>) {
  %c = llvm.mlir.zero : !llvm.array<2 x f32>
  %p = "enzymexla.memref2pointer"(%out) : (memref<8x2xf32, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (8) {
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f32>, 1>
    affine.store %c, %v[%i] : memref<?x!llvm.array<2 x f32>, 1>
  }
  return
}

// CHECK-LABEL: func.func @complex_zero
// CHECK-SAME: (%[[OUT:.+]]: memref<8x2xf32, 1>)
// CHECK-NOT:     llvm.mlir.zero
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (8) {
// CHECK-NEXT:      affine.store %[[ZERO]], %[[OUT]][%[[I]], 0] : memref<8x2xf32, 1>
// CHECK-NEXT:      affine.store %[[ZERO]], %[[OUT]][%[[I]], 1] : memref<8x2xf32, 1>
// CHECK-NEXT:    }
// CHECK-NOT:     llvm.mlir.zero

// A struct constant carries its fields as an array attribute.
func.func @struct_constant(%out: memref<8x2xf64, 1>) {
  %c = llvm.mlir.constant([3.000000e+00 : f64, 4.000000e+00 : f64]) : !llvm.struct<(f64, f64)>
  %p = "enzymexla.memref2pointer"(%out) : (memref<8x2xf64, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (8) {
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.struct<(f64, f64)>, 1>
    affine.store %c, %v[%i] : memref<?x!llvm.struct<(f64, f64)>, 1>
  }
  return
}

// CHECK-LABEL: func.func @struct_constant
// CHECK-SAME: (%[[OUT:.+]]: memref<8x2xf64, 1>)
// CHECK-NOT:     llvm.mlir.constant
// CHECK-DAG:     %[[RE:.+]] = arith.constant 3.000000e+00 : f64
// CHECK-DAG:     %[[IM:.+]] = arith.constant 4.000000e+00 : f64
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (8) {
// CHECK-NEXT:      affine.store %[[RE]], %[[OUT]][%[[I]], 0] : memref<8x2xf64, 1>
// CHECK-NEXT:      affine.store %[[IM]], %[[OUT]][%[[I]], 1] : memref<8x2xf64, 1>
// CHECK-NEXT:    }
// CHECK-NOT:     llvm.mlir.constant

// Scalar llvm constants become arith constants outright; aggregates are left.
func.func @scalars() -> (f32, i64, f64, !llvm.array<2 x f32>) {
  %f = llvm.mlir.constant(1.500000e+00 : f32) : f32
  %i = llvm.mlir.constant(7 : i64) : i64
  %z = llvm.mlir.zero : f64
  %a = llvm.mlir.constant(dense<0.000000e+00> : tensor<2xf32>) : !llvm.array<2 x f32>
  return %f, %i, %z, %a : f32, i64, f64, !llvm.array<2 x f32>
}

// CHECK-LABEL: func.func @scalars
// CHECK-DAG:     %[[F:.+]] = arith.constant 1.500000e+00 : f32
// CHECK-DAG:     %[[I:.+]] = arith.constant 7 : i64
// CHECK-DAG:     %[[Z:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[A:.+]] = llvm.mlir.constant(dense<0.000000e+00> : tensor<2xf32>) : !llvm.array<2 x f32>
// CHECK:         return %[[F]], %[[I]], %[[Z]], %[[A]]

func.func @complex_to_real(%out: memref<2xf32, 1>, %in: memref<2x2xf32, 1>) {
  %p = "enzymexla.memref2pointer"(%in) : (memref<2x2xf32, 1>) -> !llvm.ptr<1>
  affine.parallel (%i) = (0) to (2) {
    %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f32>, 1>
    %z = affine.load %v[%i] : memref<?x!llvm.array<2 x f32>, 1>
    %re = llvm.extractvalue %z[0] : !llvm.array<2 x f32>
    %im = llvm.extractvalue %z[1] : !llvm.array<2 x f32>
    %s = arith.addf %re, %im : f32
    affine.store %s, %out[%i] : memref<2xf32, 1>
  }
  return
}

// CHECK-LABEL: func.func @complex_to_real
// CHECK-SAME: (%[[OUT:.+]]: memref<2xf32, 1>, %[[IN:.+]]: memref<2x2xf32, 1>)
// CHECK-NEXT:    affine.parallel (%[[I:.+]]) = (0) to (2) {
// CHECK-NEXT:      %[[RE:.+]] = affine.load %[[IN]][%[[I]], 0] : memref<2x2xf32, 1>
// CHECK-NEXT:      %[[IM:.+]] = affine.load %[[IN]][%[[I]], 1] : memref<2x2xf32, 1>
// CHECK-NEXT:      %[[S:.+]] = arith.addf %[[RE]], %[[IM]] : f32
// CHECK-NEXT:      affine.store %[[S]], %[[OUT]][%[[I]]] : memref<2xf32, 1>
// CHECK-NEXT:    }

// The outer dimensions of a multi-dimensional buffer delinearize the index.
func.func @matrix(%in: memref<3x5x2xf64>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%in) : (memref<3x5x2xf64>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?x!llvm.array<2 x f64>>
  %z = affine.load %v[%i] : memref<?x!llvm.array<2 x f64>>
  %im = llvm.extractvalue %z[1] : !llvm.array<2 x f64>
  return %im : f64
}

// CHECK-LABEL: func.func @matrix
// CHECK-SAME: (%[[IN:.+]]: memref<3x5x2xf64>, %[[I:.+]]: index)
// CHECK:         affine.load %[[IN]][%[[I]] floordiv 5, %[[I]] mod 5, 1] : memref<3x5x2xf64>

// A buffer whose shape does not spell out the aggregates is viewed flat.
func.func @flat(%in: memref<2x3xf64>, %i: index) -> f64 {
  %p = "enzymexla.memref2pointer"(%in) : (memref<2x3xf64>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?x!llvm.array<2 x f64>>
  %z = affine.load %v[%i] : memref<?x!llvm.array<2 x f64>>
  %im = llvm.extractvalue %z[1] : !llvm.array<2 x f64>
  return %im : f64
}

// CHECK-LABEL: func.func @flat
// CHECK:         %[[V:.+]] = "enzymexla.pointer2memref"(%{{.+}}) : (!llvm.ptr) -> memref<?xf64>
// CHECK:         affine.load %[[V]][%{{.+}} * 2 + 1] : memref<?xf64>

// The view of a buffer of another scalar is left alone.
func.func @other_scalar(%in: memref<4xf64, 1>) -> f32 {
  %p = "enzymexla.memref2pointer"(%in) : (memref<4xf64, 1>) -> !llvm.ptr<1>
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr<1>) -> memref<?x!llvm.array<2 x f32>, 1>
  %z = affine.load %v[0] : memref<?x!llvm.array<2 x f32>, 1>
  %re = llvm.extractvalue %z[0] : !llvm.array<2 x f32>
  return %re : f32
}

// CHECK-LABEL: func.func @other_scalar
// CHECK: memref<?x!llvm.array<2 x f32>, 1>

// An integer aggregate is more likely a byte-wise move: left alone.
func.func @bytes(%in: memref<16xi8>) -> !llvm.array<8 x i8> {
  %p = "enzymexla.memref2pointer"(%in) : (memref<16xi8>) -> !llvm.ptr
  %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?x!llvm.array<8 x i8>>
  %z = affine.load %v[1] : memref<?x!llvm.array<8 x i8>>
  return %z : !llvm.array<8 x i8>
}

// CHECK-LABEL: func.func @bytes
// CHECK: affine.load %{{.+}}[1] : memref<?x!llvm.array<8 x i8>>
