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
// CHECK-NEXT:    %c = stablehlo.constant dense<62> : tensor<62xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<62xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<62xi64>
// CHECK-NEXT:    %1 = stablehlo.compare  EQ, %0, %c_0,  SIGNED : (tensor<62xi64>, tensor<62xi64>) -> tensor<62xi1>
// CHECK-NEXT:    %2 = stablehlo.compare  GT, %0, %c,  SIGNED : (tensor<62xi64>, tensor<62xi64>) -> tensor<62xi1>
// CHECK-NEXT:    %3 = stablehlo.or %1, %2 : tensor<62xi1>
// CHECK-NEXT:    %4 = stablehlo.convert %3 : (tensor<62xi1>) -> tensor<62xf32>
// CHECK-NEXT:    %5 = stablehlo.negate %4 : tensor<62xf32>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %5, dims = [1] : (tensor<62xf32>) -> tensor<1x62x62xf32>
// CHECK-NEXT:    %7 = stablehlo.dynamic_update_slice %arg0, %6, %c_2, %c_1, %c_1 : (tensor<1x78x78xf32>, tensor<1x62x62xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x78x78xf32>
// CHECK-NEXT:    return %7 : tensor<1x78x78xf32>
// CHECK-NEXT:  }
// CHECK:  func.func private @ui_raised(%arg0: tensor<1x78x78xf32>) -> tensor<1x78x78xf32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<62> : tensor<62xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<62xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<62xi64>
// CHECK-NEXT:    %1 = stablehlo.compare  EQ, %0, %c_0,  SIGNED : (tensor<62xi64>, tensor<62xi64>) -> tensor<62xi1>
// CHECK-NEXT:    %2 = stablehlo.compare  GT, %0, %c,  SIGNED : (tensor<62xi64>, tensor<62xi64>) -> tensor<62xi1>
// CHECK-NEXT:    %3 = stablehlo.or %1, %2 : tensor<62xi1>
// CHECK-NEXT:    %4 = stablehlo.convert %3 : (tensor<62xi1>) -> tensor<62xf32>
// CHECK-NEXT:    %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<62xf32>) -> tensor<1x62x62xf32>
// CHECK-NEXT:    %6 = stablehlo.dynamic_update_slice %arg0, %5, %c_2, %c_1, %c_1 : (tensor<1x78x78xf32>, tensor<1x62x62xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x78x78xf32>
// CHECK-NEXT:    return %6 : tensor<1x78x78xf32>
// CHECK-NEXT:  }
