// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false err_if_not_fully_raised=true" | FileCheck %s

// A store into per-lane scratch from a loop the raising iterates: the
// index's range (`k * 6 + 1` over three iterations) runs one past the buffer,
// so the buffer is padded for the update and sliced back afterwards. The
// slice must use the raised buffer's shape, lane dimensions included, not
// the memref's rank.

module {
  func.func private @padded(%out: memref<4xf64, 1>, %in: memref<4xf64, 1>) {
    %c0 = arith.constant 0 : index
    affine.parallel (%i, %j) = (0, 0) to (2, 2) {
      %a = memref.alloca() : memref<18xf64>
      %v = affine.load %in[%i * 2 + %j] : memref<4xf64, 1>
      affine.for %k = 0 to 3 {
        affine.store %v, %a[%k * 6 + 1] : memref<18xf64>
        "enzymexla.barrier"(%i, %j, %c0) : (index, index, index) -> ()
      }
      %r = affine.load %a[13] : memref<18xf64>
      affine.store %r, %out[%i * 2 + %j] : memref<4xf64, 1>
    }
    return
  }
}

// CHECK:    func.func private @padded_raised(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<2xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<2xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_2 : tensor<2xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_3 : tensor<2xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2x18xf64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %7 = stablehlo.multiply %2, %6 : tensor<2xi64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<2xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<2x2xi64>
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<2x2xi64>) -> tensor<2x2x1xi64>
// CHECK-NEXT:    %12 = "stablehlo.gather"(%arg1, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xf64>, tensor<2x2x1xi64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_19 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_21 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_23 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %c_24 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_25 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %13 = stablehlo.multiply %c_5, %c_25 : tensor<i64>
// CHECK-NEXT:    %c_26 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %14 = stablehlo.add %13, %c_26 : tensor<i64>
// CHECK-NEXT:    %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %16 = stablehlo.pad %cst, %cst_27, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<2x2x18xf64>, tensor<f64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:    %17 = stablehlo.dynamic_update_slice %16, %15, %c_13, %c_19, %14 : (tensor<2x2x19xf64>, tensor<2x2x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:    %18 = stablehlo.slice %17 [0:2, 0:2, 0:18] : (tensor<2x2x19xf64>) -> tensor<2x2x18xf64>
// CHECK-NEXT:    %19 = stablehlo.add %c_5, %c_7 : tensor<i64>
// CHECK-NEXT:    %c_28 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_29 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_30 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_31 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_32 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_33 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_34 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_35 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_36 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_37 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_38 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_39 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_40 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_41 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %c_42 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_43 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %c_44 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_45 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %20 = stablehlo.multiply %19, %c_45 : tensor<i64>
// CHECK-NEXT:    %c_46 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %21 = stablehlo.add %20, %c_46 : tensor<i64>
// CHECK-NEXT:    %22 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:    %cst_47 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %23 = stablehlo.pad %18, %cst_47, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<2x2x18xf64>, tensor<f64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:    %24 = stablehlo.dynamic_update_slice %23, %22, %c_33, %c_39, %21 : (tensor<2x2x19xf64>, tensor<2x2x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:    %25 = stablehlo.slice %24 [0:2, 0:2, 0:18] : (tensor<2x2x19xf64>) -> tensor<2x2x18xf64>
// CHECK-NEXT:    %26 = stablehlo.add %19, %c_7 : tensor<i64>
// CHECK-NEXT:    %c_48 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_49 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_50 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_51 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_52 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_53 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_54 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_55 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_56 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_57 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_58 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_59 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_60 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_61 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:    %c_62 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_63 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:    %c_64 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_65 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %27 = stablehlo.multiply %26, %c_65 : tensor<i64>
// CHECK-NEXT:    %c_66 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %28 = stablehlo.add %27, %c_66 : tensor<i64>
// CHECK-NEXT:    %29 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:    %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %30 = stablehlo.pad %25, %cst_67, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<2x2x18xf64>, tensor<f64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:    %31 = stablehlo.dynamic_update_slice %30, %29, %c_53, %c_59, %28 : (tensor<2x2x19xf64>, tensor<2x2x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:    %32 = stablehlo.slice %31 [0:2, 0:2, 0:18] : (tensor<2x2x19xf64>) -> tensor<2x2x18xf64>
// CHECK-NEXT:    %33 = stablehlo.add %26, %c_7 : tensor<i64>
// CHECK-NEXT:    %34 = stablehlo.slice %32 [0:2, 0:2, 13:14] : (tensor<2x2x18xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:    %35 = stablehlo.reshape %34 : (tensor<2x2x1xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %c_68 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %36 = stablehlo.broadcast_in_dim %c_68, dims = [] : (tensor<i64>) -> tensor<2xi64>
// CHECK-NEXT:    %37 = stablehlo.multiply %2, %36 : tensor<2xi64>
// CHECK-NEXT:    %38 = stablehlo.broadcast_in_dim %37, dims = [0] : (tensor<2xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %39 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<2xi64>) -> tensor<2x2xi64>
// CHECK-NEXT:    %40 = stablehlo.add %38, %39 : tensor<2x2xi64>
// CHECK-NEXT:    %41 = stablehlo.reshape %40 : (tensor<2x2xi64>) -> tensor<2x2x1xi64>
// CHECK-NEXT:    %42 = "stablehlo.scatter"(%arg0, %41, %35) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<4xf64>, tensor<2x2x1xi64>, tensor<2x2xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %42, %arg1 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }
