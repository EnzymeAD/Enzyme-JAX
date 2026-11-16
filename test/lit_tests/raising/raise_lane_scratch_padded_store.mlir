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

// CHECK:   func.func private @padded_raised(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %c_0 : tensor<2xi64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %c_1 : tensor<2xi64>
// CHECK-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %c_2 : tensor<2xi64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:     %5 = stablehlo.multiply %4, %c_3 : tensor<2xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<2x2x18xf64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_5 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %c_6 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %6 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:     %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_12 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_18 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_20 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:     %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_22 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:     %c_23 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_24 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:     %7 = stablehlo.multiply %c_4, %c_24 : tensor<i64>
// CHECK-NEXT:     %c_25 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %8 = stablehlo.add %7, %c_25 : tensor<i64>
// CHECK-NEXT:     %9 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:     %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %10 = stablehlo.pad %cst, %cst_26, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<2x2x18xf64>, tensor<f64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:     %11 = stablehlo.dynamic_update_slice %10, %9, %c_12, %c_18, %8 : (tensor<2x2x19xf64>, tensor<2x2x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:     %12 = stablehlo.slice %11 [0:2, 0:2, 0:18] : (tensor<2x2x19xf64>) -> tensor<2x2x18xf64>
// CHECK-NEXT:     %13 = stablehlo.add %c_4, %c_6 : tensor<i64>
// CHECK-NEXT:     %14 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:     %c_27 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_28 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_29 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_30 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_31 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_32 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_33 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_34 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_35 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_36 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_37 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_38 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_39 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_40 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:     %c_41 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_42 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:     %c_43 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_44 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:     %15 = stablehlo.multiply %13, %c_44 : tensor<i64>
// CHECK-NEXT:     %c_45 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %16 = stablehlo.add %15, %c_45 : tensor<i64>
// CHECK-NEXT:     %17 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:     %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %18 = stablehlo.pad %12, %cst_46, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<2x2x18xf64>, tensor<f64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:     %19 = stablehlo.dynamic_update_slice %18, %17, %c_32, %c_38, %16 : (tensor<2x2x19xf64>, tensor<2x2x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:     %20 = stablehlo.slice %19 [0:2, 0:2, 0:18] : (tensor<2x2x19xf64>) -> tensor<2x2x18xf64>
// CHECK-NEXT:     %21 = stablehlo.add %13, %c_6 : tensor<i64>
// CHECK-NEXT:     %22 = stablehlo.reshape %arg1 : (tensor<4xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:     %c_47 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_48 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_49 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_50 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_51 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_52 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_53 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_54 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_55 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_56 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_57 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_58 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %c_59 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_60 = stablehlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:     %c_61 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_62 = stablehlo.constant dense<-1> : tensor<1xi64>
// CHECK-NEXT:     %c_63 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_64 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:     %23 = stablehlo.multiply %21, %c_64 : tensor<i64>
// CHECK-NEXT:     %c_65 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %24 = stablehlo.add %23, %c_65 : tensor<i64>
// CHECK-NEXT:     %25 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:     %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %26 = stablehlo.pad %20, %cst_66, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<2x2x18xf64>, tensor<f64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:     %27 = stablehlo.dynamic_update_slice %26, %25, %c_52, %c_58, %24 : (tensor<2x2x19xf64>, tensor<2x2x1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x19xf64>
// CHECK-NEXT:     %28 = stablehlo.slice %27 [0:2, 0:2, 0:18] : (tensor<2x2x19xf64>) -> tensor<2x2x18xf64>
// CHECK-NEXT:     %29 = stablehlo.add %21, %c_6 : tensor<i64>
// CHECK-NEXT:     %30 = stablehlo.slice %28 [0:2, 0:2, 13:14] : (tensor<2x2x18xf64>) -> tensor<2x2x1xf64>
// CHECK-NEXT:     %31 = stablehlo.reshape %30 : (tensor<2x2x1xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:     %c_67 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_68 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_69 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_70 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_71 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %c_72 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %32 = stablehlo.reshape %31 : (tensor<2x2xf64>) -> tensor<4xf64>
// CHECK-NEXT:     %33 = stablehlo.dynamic_update_slice %arg0, %32, %c_72 : (tensor<4xf64>, tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:     return %33, %arg1 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:   }
