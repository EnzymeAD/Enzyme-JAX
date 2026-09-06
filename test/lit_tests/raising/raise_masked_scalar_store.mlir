// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A masked store into a rank-0 buffer reads the previous value back as the
// buffer itself. A rank-0 stablehlo.dynamic_slice would print as
// `%x, , sizes = []`, which no parser reads back (the runtime re-parses the
// raised module), so none may be emitted.
func.func @scalar_masked(%out: memref<f64, 1>, %in: memref<4xf64, 1>, %flag: memref<1xi64, 1>) {
  %c1_i64 = arith.constant 1 : i64
  affine.parallel (%i) = (0) to (4) {
    %v = affine.load %in[0] : memref<4xf64, 1>
    %f = affine.load %flag[0] : memref<1xi64, 1>
    %c = arith.cmpi eq, %f, %c1_i64 : i64
    scf.if %c {
      affine.store %v, %out[] : memref<f64, 1>
    }
  }
  return
}

// CHECK:    func.func private @scalar_masked_raised(%arg0: tensor<f64>, %arg1: tensor<4xf64>, %arg2: tensor<1xi64>) -> (tensor<f64>, tensor<4xf64>, tensor<1xi64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<4xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<4xi64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %5 = stablehlo.reshape %arg2 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:    %6 = arith.cmpi eq, %5, %c : tensor<i64>
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %8 = stablehlo.reshape %7 : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %9 = stablehlo.reshape %arg0 : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %10 = stablehlo.select %6, %8, %9 : tensor<i1>, tensor<f64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %12 = stablehlo.dynamic_update_slice %arg0, %11 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    return %12, %arg1, %arg2 : tensor<f64>, tensor<4xf64>, tensor<1xi64>
// CHECK-NEXT:  }
