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
  affine.parallel (%i) = (0) to (3) {
    %alloca = memref.alloca() : memref<2x3xf64>
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
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<3xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<3xi64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<6xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<6xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_2 : tensor<6xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<6xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_3 : tensor<6xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.dynamic_slice %arg1, %c_4, sizes = [6] : (tensor<6xf64>, tensor<i64>) -> tensor<6xf64>
// CHECK-NEXT:    %7 = stablehlo.reshape %6 : (tensor<6xf64>) -> tensor<6xf64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<6xi64>
// CHECK-NEXT:    %9 = stablehlo.compare LT, %5, %c_6 : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi1>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<6xi64>
// CHECK-NEXT:    %10 = stablehlo.negate %5 : tensor<6xi64>
// CHECK-NEXT:    %11 = stablehlo.subtract %10, %c_7 : tensor<6xi64>
// CHECK-NEXT:    %12 = stablehlo.select %9, %11, %5 : tensor<6xi1>, tensor<6xi64>
// CHECK-NEXT:    %13 = stablehlo.divide %12, %8 : tensor<6xi64>
// CHECK-NEXT:    %14 = stablehlo.negate %13 : tensor<6xi64>
// CHECK-NEXT:    %15 = stablehlo.subtract %14, %c_7 : tensor<6xi64>
// CHECK-NEXT:    %16 = stablehlo.select %9, %15, %13 : tensor<6xi1>, tensor<6xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i64>) -> tensor<6xi64>
// CHECK-NEXT:    %18 = stablehlo.remainder %5, %17 : tensor<6xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<6xi64>
// CHECK-NEXT:    %19 = stablehlo.compare LT, %18, %c_9 : (tensor<6xi64>, tensor<6xi64>) -> tensor<6xi1>
// CHECK-NEXT:    %20 = stablehlo.add %18, %17 : tensor<6xi64>
// CHECK-NEXT:    %21 = stablehlo.select %19, %20, %18 : tensor<6xi1>, tensor<6xi64>
// CHECK-NEXT:    %22 = stablehlo.reshape %16 : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:    %23 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<6xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<6x1xi64>) -> tensor<6x1xi64>
// CHECK-NEXT:    %25 = stablehlo.concatenate %24, %23, dim = 1 : (tensor<6x1xi64>, tensor<6x1xi64>) -> tensor<6x2xi64>
// CHECK-NEXT:    %26 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<6xf64>) -> tensor<6xf64>
// CHECK-NEXT:    %27 = "stablehlo.scatter"(%cst_1, %25, %26) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<2x3xf64>, tensor<6x2xi64>, tensor<6xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %28 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %29 = stablehlo.add %28, %c_10 : tensor<3xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %30 = stablehlo.multiply %29, %c_11 : tensor<3xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %31 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %32 = stablehlo.multiply %2, %31 : tensor<3xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %33 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %34 = stablehlo.add %32, %33 : tensor<3xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %35 = stablehlo.compare GE, %34, %c_14 : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %36 = stablehlo.pad %27, %cst_17, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %37 = stablehlo.dynamic_slice %36, %c_15, %c_16, sizes = [3, 3] : (tensor<3x3xf64>, tensor<i64>, tensor<i64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %38 = stablehlo.reshape %37 : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %39 = stablehlo.not %35 : tensor<3xi1>
// CHECK-NEXT:    %40 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<3xi1>) -> tensor<3x3xi1>
// CHECK-NEXT:    %41 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %42 = stablehlo.select %40, %38, %41 : tensor<3x3xi1>, tensor<3x3xf64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %43 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %44 = stablehlo.multiply %2, %43 : tensor<3xi64>
// CHECK-NEXT:    %45 = stablehlo.broadcast_in_dim %44, dims = [0] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %46 = stablehlo.broadcast_in_dim %30, dims = [1] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %47 = stablehlo.add %45, %46 : tensor<3x3xi64>
// CHECK-NEXT:    %48 = stablehlo.reshape %47 : (tensor<3x3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %49 = stablehlo.broadcast_in_dim %42, dims = [0, 1] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %50 = "stablehlo.scatter"(%arg0, %48, %49) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<9xf64>, tensor<3x3x1xi64>, tensor<3x3xf64>) -> tensor<9xf64>
// CHECK-NEXT:    return %50, %arg1 : tensor<9xf64>, tensor<6xf64>
// CHECK-NEXT:  }
