// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,arith-raise,enzyme-hlo-opt{max_constant_expansion=1})" | FileCheck %s

module {
  func.func private @ui(%arg0: memref<1x78x78xf32, 1>) {
    %c0 = arith.constant 0 : index
    %c62_i64 = arith.constant 62 : i64
    affine.parallel (%arg1, %arg2) = (0, 0) to (62, 62) {
      %0 = arith.index_castui %arg1 : index to i64
      %1 = arith.cmpi eq, %arg1, %c0 : index
      %2 = arith.cmpi sgt, %0, %c62_i64 : i64
      %3 = arith.ori %1, %2 : i1
      %4 = arith.uitofp %3 : i1 to f32
      affine.store %4, %arg0[0, %arg1 + 8, %arg2 + 8] : memref<1x78x78xf32, 1>
    }
    return
  }
  func.func private @si(%arg0: memref<1x78x78xf32, 1>) {
    %c0 = arith.constant 0 : index
    %c62_i64 = arith.constant 62 : i64
    affine.parallel (%arg1, %arg2) = (0, 0) to (62, 62) {
      %0 = arith.index_castui %arg1 : index to i64
      %1 = arith.cmpi eq, %arg1, %c0 : index
      %2 = arith.cmpi sgt, %0, %c62_i64 : i64
      %3 = arith.ori %1, %2 : i1
      %4 = arith.sitofp %3 : i1 to f32
      affine.store %4, %arg0[0, %arg1 + 8, %arg2 + 8] : memref<1x78x78xf32, 1>
    }
    return
  }
}

// CHECK:  func.func private @si_raised(%arg0: tensor<1x78x78xf32>) -> tensor<1x78x78xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<-1.000000e+00> : tensor<1x1x62xf32>
// CHECK-NEXT:    %0 = stablehlo.pad %cst_1, %cst, low = [0, 0, 0], high = [0, 61, 0], interior = [0, 0, 0] : (tensor<1x1x62xf32>, tensor<f32>) -> tensor<1x62x62xf32>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %c, %c : (tensor<1x78x78xf32>, tensor<1x62x62xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x78x78xf32>
// CHECK-NEXT:    return %1 : tensor<1x78x78xf32>
// CHECK-NEXT:  }

// CHECK:  func.func private @ui_raised(%arg0: tensor<1x78x78xf32>) -> tensor<1x78x78xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<1x1x62xf32>
// CHECK-NEXT:    %0 = stablehlo.pad %cst_1, %cst, low = [0, 0, 0], high = [0, 61, 0], interior = [0, 0, 0] : (tensor<1x1x62xf32>, tensor<f32>) -> tensor<1x62x62xf32>
// CHECK-NEXT:    %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %c, %c : (tensor<1x78x78xf32>, tensor<1x62x62xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x78x78xf32>
// CHECK-NEXT:    return %1 : tensor<1x78x78xf32>
// CHECK-NEXT:  }
