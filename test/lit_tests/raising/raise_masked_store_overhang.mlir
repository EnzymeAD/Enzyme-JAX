// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A lane-guarded store into a scratch buffer narrower than the lane range:
// three lanes, two rows, the guard (%i <= 1) keeping every executed store in
// bounds. The store already pads the buffer out to the range before the
// dynamic_update_slice; the masked path's read-back of the previous value
// sliced the buffer before that padding with the range's extent, and
// `stablehlo.dynamic_slice` result inference aborted the pass
// (`has slice size 3 greater than dimension size 2`). The masked update now
// reads the padded buffer, and the pad rows slice away afterwards.
#set = affine_set<(d0) : (-d0 + 1 >= 0)>
func.func @guarded_lane_store(%out: memref<18xf64, 1>, %in: memref<9xf64, 1>) {
  %alloca = memref.alloca() : memref<2x3xf64>
  affine.parallel (%i) = (0) to (3) {
    affine.if #set(%i) {
      affine.for %b = 0 to 3 {
        %v = affine.load %in[%i * 3 + %b] : memref<9xf64, 1>
        affine.store %v, %alloca[%i, %b] : memref<2x3xf64>
      }
    }
    "enzymexla.barrier"(%i) : (index) -> ()
    affine.for %r = 0 to 2 {
      affine.for %b = 0 to 3 {
        %v = affine.load %alloca[%r, %b] : memref<2x3xf64>
        affine.store %v, %out[%i * 6 + %r * 3 + %b] : memref<18xf64, 1>
      }
    }
  }
  return
}

// CHECK:    func.func private @guarded_lane_store_raised(%arg0: tensor<18xf64>, %arg1: tensor<9xf64>) -> (tensor<18xf64>, tensor<9xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x3xf64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<3xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<3xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %4 = stablehlo.multiply %2, %3 : tensor<3xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %6 = stablehlo.add %4, %5 : tensor<3xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %7 = stablehlo.compare GE, %6, %c_3 : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:    %8 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:    %9 = stablehlo.add %8, %c_4 : tensor<3xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:    %10 = stablehlo.multiply %9, %c_5 : tensor<3xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:    %12 = stablehlo.multiply %2, %11 : tensor<3xi64>
// CHECK-NEXT:    %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %14 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:    %15 = stablehlo.add %13, %14 : tensor<3x3xi64>
// CHECK-NEXT:    %16 = stablehlo.reshape %15 : (tensor<3x3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:    %17 = "stablehlo.gather"(%arg1, %16) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf64>, tensor<3x3x1xi64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %18 = stablehlo.pad %cst, %cst_19, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<3xi1>) -> tensor<3x3xi1>
// CHECK-NEXT:    %20 = stablehlo.select %19, %17, %18 : tensor<3x3xi1>, tensor<3x3xf64>
// CHECK-NEXT:    %21 = stablehlo.dynamic_update_slice %18, %20, %c_12, %c_18 : (tensor<3x3xf64>, tensor<3x3xf64>, tensor<i64>, tensor<i64>) -> tensor<3x3xf64>
// CHECK-NEXT:    %22 = stablehlo.slice %21 [0:2, 0:3] : (tensor<3x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %23:4 = stablehlo.while(%iterArg = %c_20, %iterArg_23 = %arg0, %iterArg_24 = %arg1, %iterArg_25 = %22) : tensor<i64>, tensor<18xf64>, tensor<9xf64>, tensor<2x3xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %24 = stablehlo.compare LT, %iterArg, %c_21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %24 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %24 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:      %c_26 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:      %25 = stablehlo.add %24, %c_26 : tensor<3xi64>
// CHECK-NEXT:      %c_27 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:      %26 = stablehlo.multiply %25, %c_27 : tensor<3xi64>
// CHECK-NEXT:      %c_28 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %27 = stablehlo.dynamic_slice %iterArg_25, %iterArg, %c_28, sizes = [1, 3] : (tensor<2x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %28 = stablehlo.reshape %27 : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:      %c_29 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:      %29 = stablehlo.broadcast_in_dim %c_29, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %30 = stablehlo.multiply %2, %29 : tensor<3xi64>
// CHECK-NEXT:      %c_30 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:      %31 = stablehlo.multiply %iterArg, %c_30 : tensor<i64>
// CHECK-NEXT:      %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:      %33 = stablehlo.add %30, %32 : tensor<3xi64>
// CHECK-NEXT:      %34 = stablehlo.broadcast_in_dim %33, dims = [0] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:      %35 = stablehlo.broadcast_in_dim %26, dims = [1] : (tensor<3xi64>) -> tensor<3x3xi64>
// CHECK-NEXT:      %36 = stablehlo.add %34, %35 : tensor<3x3xi64>
// CHECK-NEXT:      %37 = stablehlo.reshape %36 : (tensor<3x3xi64>) -> tensor<3x3x1xi64>
// CHECK-NEXT:      %38 = stablehlo.broadcast_in_dim %28, dims = [1] : (tensor<3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:      %39 = "stablehlo.scatter"(%iterArg_23, %37, %38) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:      ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:        stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:      }) : (tensor<18xf64>, tensor<3x3x1xi64>, tensor<3x3xf64>) -> tensor<18xf64>
// CHECK-NEXT:      %40 = stablehlo.add %iterArg, %c_22 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %40, %39, %iterArg_24, %iterArg_25 : tensor<i64>, tensor<18xf64>, tensor<9xf64>, tensor<2x3xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %23#1, %23#2 : tensor<18xf64>, tensor<9xf64>
// CHECK-NEXT:  }
