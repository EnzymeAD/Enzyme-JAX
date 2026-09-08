// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A lane-guarded read of a shared scratch buffer whose lane range (3) is
// wider than the buffer's dimension (2). The guard (%i <= 1) keeps every
// executed access in bounds, but the raised load reads the buffer with a
// dynamic_slice sized to the lane range, which stablehlo rejects
// (`slice size 3 greater than dimension size 2`). The buffer is padded out
// to the lane range first; the masked lanes read the padding and the guard's
// select discards it.
#set = affine_set<(d0) : (-d0 + 1 >= 0)>
func.func @guarded_lane_read(%out: memref<9xf64, 1>, %in: memref<6xf64, 1>) {
  %cst = arith.constant 0.000000e+00 : f64
  %alloca = memref.alloca() : memref<2x3xf64>
  affine.parallel (%i) = (0) to (3) {
    affine.for %a = 0 to 6 {
      %v = affine.load %in[%a] : memref<6xf64, 1>
      affine.store %v, %alloca[%a floordiv 3, %a mod 3] : memref<2x3xf64>
    }
    "enzymexla.barrier"(%i) : (index) -> ()
    affine.for %k = 0 to 3 {
      %r = affine.if #set(%i) -> f64 {
        %v = affine.load %alloca[%i, %k] : memref<2x3xf64>
        affine.yield %v : f64
      } else {
        affine.yield %cst : f64
      }
      affine.store %r, %out[%i * 3 + %k] : memref<9xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @guarded_lane_read_raised(%arg0: tensor<9xf64>, %arg1: tensor<6xf64>) -> (tensor<9xf64>, tensor<6xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<3xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<3xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<6xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<6xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_2 : tensor<6xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<6xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_3 : tensor<6xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<6xi64>
// CHECK-NEXT:    %7 = stablehlo.compare LT, %5, %c_6 : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi1>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<6xi64>
// CHECK-NEXT:    %8 = stablehlo.negate %5 : tensor<6xi64>
// CHECK-NEXT:    %9 = stablehlo.subtract %8, %c_7 : tensor<6xi64>
// CHECK-NEXT:    %10 = stablehlo.select %7, %9, %5 : tensor<6xi1>, tensor<6xi64>
// CHECK-NEXT:    %11 = stablehlo.divide %10, %6 : tensor<6xi64>
// CHECK-NEXT:    %12 = stablehlo.negate %11 : tensor<6xi64>
// CHECK-NEXT:    %13 = stablehlo.subtract %12, %c_7 : tensor<6xi64>
// CHECK-NEXT:    %14 = stablehlo.select %7, %13, %11 : tensor<6xi1>, tensor<6xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %15 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:    %16 = stablehlo.remainder %5, %15 : tensor<6xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<6xi64>
// CHECK-NEXT:    %17 = stablehlo.compare LT, %16, %c_9 : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi1>
// CHECK-NEXT:    %18 = stablehlo.add %16, %15 : tensor<6xi64>
// CHECK-NEXT:    %19 = stablehlo.select %17, %18, %16 : tensor<6xi1>, tensor<6xi64>
// CHECK-NEXT:    %20 = stablehlo.reshape %14 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:    %22 = stablehlo.concatenate %20, %21, dim = 1 : (tensor<6x1xi64>, tensor<6x1xi64>) -> tensor<6x2xi64>
// CHECK-NEXT:    %23 = "stablehlo.scatter"(%cst_0, %22, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<2x3xf64>, tensor<6x2xi64>, tensor<6xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %24 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %25 = stablehlo.add %24, %c_10 : tensor<3xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %26 = stablehlo.multiply %25, %c_11 : tensor<3xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %27 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %28 = stablehlo.multiply %2, %27 : tensor<3xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %29 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %30 = stablehlo.add %28, %29 : tensor<3xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %31 = stablehlo.compare GE, %30, %c_14 : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %32 = stablehlo.pad %23, %cst_17, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %33 = stablehlo.not %31 : tensor<3xi1>
// CHECK-NEXT:    %34 = stablehlo.broadcast_in_dim %31, dims = [0] : (tensor<3xi1>) -> tensor<3x3xi1>
// CHECK-NEXT:    %35 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %36 = stablehlo.select %34, %32, %35 : tensor<3x3xi1>, tensor<3x3xf64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %37 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %38 = stablehlo.multiply %2, %37 : tensor<3xi64>
// CHECK-NEXT:    %39 = stablehlo.broadcast_in_dim %38, dims = [0] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %40 = stablehlo.broadcast_in_dim %26, dims = [1] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %41 = stablehlo.add %39, %40 : tensor<3x3xi64>
// CHECK-NEXT:    %42 = stablehlo.reshape %41 : (tensor<3x3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %43 = "stablehlo.scatter"(%arg0, %42, %36) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<9xf64>, tensor<3x3x1xi64>, tensor<3x3xf64>) -> tensor<9xf64>
// CHECK-NEXT:    return %43, %arg1 : tensor<9xf64>, tensor<6xf64>
// CHECK-NEXT:  }
