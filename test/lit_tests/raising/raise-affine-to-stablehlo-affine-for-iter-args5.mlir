// RUN: enzymexlamlir-opt %s "--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=false},enzyme-hlo-opt)" | FileCheck %s
// RUN: enzymexlamlir-opt %s "--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=true},enzyme-hlo-opt)" | FileCheck %s --check-prefix=LOCKSTEP

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

// LOCKSTEP:  func.func private @foo_raised(%arg0: tensor<85x180x18xf64>, %arg1: tensor<85x180xf64>) -> (tensor<85x180x18xf64>, tensor<85x180xf64>) {
// LOCKSTEP-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LOCKSTEP-NEXT{LITERAL}:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<85x180x18xf64>, tensor<f64>) -> tensor<85x180xf64>
// LOCKSTEP-NEXT:    return %arg0, %0 : tensor<85x180x18xf64>, tensor<85x180xf64>
// LOCKSTEP-NEXT:  }
