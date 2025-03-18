// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt | FileCheck %s

module {
  func.func private @foo(%array: memref<85x180x18xf64, 1>, %sum_res: memref<85x180xf64, 1>) {
    %ub = ub.poison : f64
    affine.parallel (%i, %j) = (0, 0) to (85, 180) {
      %rsum = affine.for %k = 0 to 18 iter_args(%running_sum = %ub) -> (f64) {
        %ld = affine.load %array[%i, %j, %k] : memref<85x180x18xf64, 1>
        %add = arith.addf %ld, %running_sum : f64
        affine.yield %add : f64
      }
      affine.store %rsum, %sum_res[%i, %j] : memref<85x180xf64, 1>
    }
    return
  }
}

// CHECK-LABEL:   func.func private @foo_raised(
// CHECK:           %[[UB:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<85x180xf64>
