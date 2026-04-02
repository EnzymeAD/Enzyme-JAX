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
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [7:187] : (tensor<194xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %1 = stablehlo.reverse %0, dims = [0] : tensor<180xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [90:180] : (tensor<180xf64>) -> tensor<90xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg0 [187:194] : (tensor<194xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %4 = stablehlo.slice %arg0 [0:97] : (tensor<194xf64>) -> tensor<97xf64>
// CHECK-NEXT:    %5 = stablehlo.concatenate %4, %2, %3, dim = 0 : (tensor<97xf64>, tensor<90xf64>, tensor<7xf64>) -> tensor<194xf64>
// CHECK-NEXT:    return %5 : tensor<194xf64>
// CHECK-NEXT:  }
