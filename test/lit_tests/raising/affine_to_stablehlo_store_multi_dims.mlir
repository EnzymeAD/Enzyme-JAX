// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s --check-prefix=SLICE

module {
  func.func @main(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (1, 1) {
      %0 = affine.load %arg1[%i * 10 + %j] : memref<100xf32>
      affine.store %0, %arg0[%i * 10 + %j] : memref<100xf32>
    }
    return
  }

  func.func @flattened_two_iv(%arg0: memref<20x1536x32xf32>,
                              %arg1: memref<20x1536x32xf32>) {
    affine.parallel (%i, %j, %k, %l) = (0, 0, 0, 0) to (4, 16, 95, 16) {
      %0 = affine.load %arg1[%i + 9, %j + %k * 16 + 9, %l + 8]
          : memref<20x1536x32xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg0[%i, %j + %k * 16 + 9, %l + 8]
          : memref<20x1536x32xf32>
    }
    return
  }
}

// CHECK-LABEL: @main
// CHECK: return %arg1, %arg1 : tensor<100xf32>, tensor<100xf32>

// SLICE:   func.func private @flattened_two_iv_raised(%arg0: tensor<20x1536x32xf32>, %arg1: tensor<20x1536x32xf32>) -> (tensor<20x1536x32xf32>, tensor<20x1536x32xf32>) {
// SLICE-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// SLICE-NEXT:     %c = stablehlo.constant dense<0> : tensor<4xi64>
// SLICE-NEXT:     %1 = stablehlo.add %0, %c : tensor<4xi64>
// SLICE-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<4xi64>
// SLICE-NEXT:     %2 = stablehlo.multiply %1, %c_0 : tensor<4xi64>
// SLICE-NEXT:     %3 = stablehlo.iota dim = 0 : tensor<16xi64>
// SLICE-NEXT:     %c_1 = stablehlo.constant dense<0> : tensor<16xi64>
// SLICE-NEXT:     %4 = stablehlo.add %3, %c_1 : tensor<16xi64>
// SLICE-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<16xi64>
// SLICE-NEXT:     %5 = stablehlo.multiply %4, %c_2 : tensor<16xi64>
// SLICE-NEXT:     %6 = stablehlo.iota dim = 0 : tensor<95xi64>
// SLICE-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<95xi64>
// SLICE-NEXT:     %7 = stablehlo.add %6, %c_3 : tensor<95xi64>
// SLICE-NEXT:     %c_4 = stablehlo.constant dense<1> : tensor<95xi64>
// SLICE-NEXT:     %8 = stablehlo.multiply %7, %c_4 : tensor<95xi64>
// SLICE-NEXT:     %9 = stablehlo.iota dim = 0 : tensor<16xi64>
// SLICE-NEXT:     %c_5 = stablehlo.constant dense<0> : tensor<16xi64>
// SLICE-NEXT:     %10 = stablehlo.add %9, %c_5 : tensor<16xi64>
// SLICE-NEXT:     %c_6 = stablehlo.constant dense<1> : tensor<16xi64>
// SLICE-NEXT:     %11 = stablehlo.multiply %10, %c_6 : tensor<16xi64>
// SLICE-NEXT:     %12 = stablehlo.slice %arg1 [9:13, 9:1529, 8:24] : (tensor<20x1536x32xf32>) -> tensor<4x1520x16xf32>
// SLICE-NEXT:     %13 = stablehlo.reshape %12 : (tensor<4x1520x16xf32>) -> tensor<4x95x16x16xf32>
// SLICE-NEXT:     %14 = stablehlo.reshape %12 : (tensor<4x1520x16xf32>) -> tensor<4x95x16x16xf32>
// SLICE-NEXT:     %15 = arith.mulf %13, %14 : tensor<4x95x16x16xf32>
// SLICE-NEXT:     %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_12 = stablehlo.constant dense<0> : tensor<i64>
// SLICE-NEXT:     %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_18 = stablehlo.constant dense<9> : tensor<i64>
// SLICE-NEXT:     %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_22 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_23 = stablehlo.constant dense<0> : tensor<1xi64>
// SLICE-NEXT:     %c_24 = stablehlo.constant dense<8> : tensor<i64>
// SLICE-NEXT:     %16 = stablehlo.reshape %15 : (tensor<4x95x16x16xf32>) -> tensor<4x1520x16xf32>
// SLICE-NEXT:     %17 = stablehlo.dynamic_update_slice %arg0, %16, %c_12, %c_18, %c_24 : (tensor<20x1536x32xf32>, tensor<4x1520x16xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<20x1536x32xf32>
// SLICE-NEXT:     return %17, %arg1 : tensor<20x1536x32xf32>, tensor<20x1536x32xf32>
// SLICE-NEXT:   }
