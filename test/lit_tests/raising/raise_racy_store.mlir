// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file --verify-diagnostics | FileCheck %s

// Every lane along %t stores its own value to out[%e]: the program leaves
// the winner undefined, so lane 0's value is a sound refinement.
func.func @racy(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (8, 32) {
    %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
    // expected-warning @below {{racy store: the stored value varies along a parallel axis the destination does not index; raising lane 0's write}}
    affine.store %v, %out[%e] : memref<8xf64, 1>
  }
  return
}

// CHECK:   func.func private @racy_raised(%arg0: tensor<8xf64>, %arg1: tensor<?xf64>) -> (tensor<8xf64>, tensor<?xf64>) {
// CHECK-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<8xi64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %c : tensor<8xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<8xi64>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %c_0 : tensor<8xi64>
// CHECK-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %c_1 : tensor<32xi64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:     %5 = stablehlo.multiply %4, %c_2 : tensor<32xi64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %6 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:     %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<256> : tensor<1xi64>
// CHECK-NEXT:     %9 = stablehlo.add %8, %c_3 : tensor<1xi64>
// CHECK-NEXT:     %10 = stablehlo.subtract %c_4, %9 : tensor<1xi64>
// CHECK-NEXT:     %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %11 = stablehlo.maximum %10, %c_5 : tensor<1xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %12 = stablehlo.dynamic_pad %arg1, %cst, %c_3, %11, %c_6 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:     %13 = stablehlo.slice %12 [0:256] : (tensor<?xf64>) -> tensor<256xf64>
// CHECK-NEXT:     %14 = stablehlo.reshape %13 : (tensor<256xf64>) -> tensor<8x32xf64>
// CHECK-NEXT:     %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_12 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %15 = stablehlo.slice %14 [0:8, 0:1] : (tensor<8x32xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:     %16 = stablehlo.reshape %15 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:     %17 = stablehlo.dynamic_update_slice %arg0, %16, %c_12 : (tensor<8xf64>, tensor<8xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:     return %17, %arg1 : tensor<8xf64>, tensor<?xf64>
// CHECK-NEXT:   }
// -----

// The same race through the scatter path (a strided destination).
func.func @racy_scatter(%out: memref<16xf64, 1>, %in: memref<?xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (8, 32) {
    %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
    // expected-warning @below {{racy store: the stored value varies along a parallel axis the destination does not index; raising one lane's write}}
    affine.store %v, %out[%e * 2] : memref<16xf64, 1>
  }
  return
}

// CHECK:   func.func private @racy_scatter_raised(%arg0: tensor<16xf64>, %arg1: tensor<?xf64>) -> (tensor<16xf64>, tensor<?xf64>) {
// CHECK-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<8xi64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %c : tensor<8xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<8xi64>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %c_0 : tensor<8xi64>
// CHECK-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %c_1 : tensor<32xi64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:     %5 = stablehlo.multiply %4, %c_2 : tensor<32xi64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %6 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?xf64>) -> tensor<i32>
// CHECK-NEXT:     %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:     %8 = stablehlo.reshape %7 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<256> : tensor<1xi64>
// CHECK-NEXT:     %9 = stablehlo.add %8, %c_3 : tensor<1xi64>
// CHECK-NEXT:     %10 = stablehlo.subtract %c_4, %9 : tensor<1xi64>
// CHECK-NEXT:     %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %11 = stablehlo.maximum %10, %c_5 : tensor<1xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %12 = stablehlo.dynamic_pad %arg1, %cst, %c_3, %11, %c_6 : (tensor<?xf64>, tensor<f64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf64>
// CHECK-NEXT:     %13 = stablehlo.slice %12 [0:256] : (tensor<?xf64>) -> tensor<256xf64>
// CHECK-NEXT:     %c_7 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %14 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:     %15 = stablehlo.multiply %2, %14 : tensor<8xi64>
// CHECK-NEXT:     %16 = stablehlo.reshape %13 : (tensor<256xf64>) -> tensor<8x32xf64>
// CHECK-NEXT:     %17 = stablehlo.reshape %15 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:     %18 = stablehlo.slice %16 [0:8, 0:1] : (tensor<8x32xf64>) -> tensor<8x1xf64>
// CHECK-NEXT:     %19 = stablehlo.reshape %18 : (tensor<8x1xf64>) -> tensor<8xf64>
// CHECK-NEXT:     %20 = "stablehlo.scatter"(%arg0, %17, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:     ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:       stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:     }) : (tensor<16xf64>, tensor<8x1xi64>, tensor<8xf64>) -> tensor<16xf64>
// CHECK-NEXT:     return %20, %arg1 : tensor<16xf64>, tensor<?xf64>
// CHECK-NEXT:   }
