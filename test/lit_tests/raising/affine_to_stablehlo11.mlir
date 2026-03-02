// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(raise-affine-to-stablehlo,enzyme-hlo-opt{max_constant_expansion=0})" | FileCheck %s

#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 70 >= 0)>

module {
  func.func private @call__Z31gpu__fill_south_and_north_halo(%arg0: memref<194xf64, 1>) {
    affine.parallel (%arg1) = (0) to (180) {
      %1 = affine.load %arg0[-%arg1 + 186] : memref<194xf64, 1>
      %2 = affine.load %arg0[%arg1 + 7] : memref<194xf64, 1>
      %3 = affine.if #set(%arg1) -> f64 { // %arg1 < 89
        affine.yield %2 : f64
      } else {
        affine.yield %1 : f64
      }
      affine.store %3, %arg0[%arg1 + 7] : memref<194xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @call__Z31gpu__fill_south_and_north_halo_raised(%arg0: tensor<194xf64>) -> tensor<194xf64> {
// CHECK-NEXT:    %c = stablehlo.constant {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} dense<89> : tensor<180xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<180xi64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [7:187] : (tensor<194xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %2 = stablehlo.reverse %1, dims = [0] : tensor<180xf64>
// CHECK-NEXT:    %3 = stablehlo.subtract %c, %0 {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<180xi64>
// CHECK-NEXT:    %4 = stablehlo.compare  GE, %3, %c_0 : (tensor<180xi64>, tensor<180xi64>) -> tensor<180xi1>
// CHECK-NEXT:    %5 = stablehlo.select %4, %1, %2 : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg0 [0:7] : (tensor<194xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %7 = stablehlo.slice %arg0 [187:194] : (tensor<194xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %8 = stablehlo.concatenate %6, %5, %7, dim = 0 : (tensor<7xf64>, tensor<180xf64>, tensor<7xf64>) -> tensor<194xf64>
// CHECK-NEXT:    return %8 : tensor<194xf64>
// CHECK-NEXT:  }
