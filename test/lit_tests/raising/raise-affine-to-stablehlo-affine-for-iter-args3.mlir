// RUN: enzymexlamlir-opt %s "--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=false},enzyme-hlo-opt)" | FileCheck %s
// RUN: enzymexlamlir-opt %s "--pass-pipeline=builtin.module(raise-affine-to-stablehlo{enable_lockstep_for=true},enzyme-hlo-opt)" | FileCheck %s --check-prefix=LOCKSTEP

module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @foo(%array: memref<85x180x18xf64, 1>, %sum_res: memref<85x180xf64, 1>) {
    %cst = arith.constant 5.000000e-01 : f64
    affine.parallel (%i, %j) = (0, 0) to (85, 180) {
      %rsum = affine.for %k = 0 to 18 iter_args(%running_sum = %cst) -> (f64) {
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
// CHECK-SAME:                                  %[[VAL_0:.*]]: tensor<85x180x18xf64>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: tensor<85x180xf64>) -> (tensor<85x180x18xf64>, tensor<85x180xf64>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<18> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<85x180xf64>
// CHECK:           %[[VAL_6:.*]]:2 = stablehlo.while(%[[VAL_7:.*]] = %[[VAL_4]], %[[VAL_8:.*]] = %[[VAL_5]]) : tensor<i64>, tensor<85x180xf64>
// CHECK:             %[[LD_RAW:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_4]], %[[VAL_4]], %[[VAL_7]], sizes = [85, 180, 1] : (tensor<85x180x18xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<85x180x1xf64>
// CHECK:             %[[LD:.*]] = stablehlo.reshape %[[LD_RAW]] : (tensor<85x180x1xf64>) -> tensor<85x180xf64>
// CHECK:             %[[NEW_SUM:.*]] = arith.addf %[[LD]], %[[VAL_8]] : tensor<85x180xf64>
// CHECK:             %[[NEW_IV:.*]] = stablehlo.add %[[VAL_7]], %[[VAL_2]] {enzymexla.bounds = {{.*}}} : tensor<i64>
// CHECK:             stablehlo.return %[[NEW_IV]], %[[NEW_SUM]] : tensor<i64>, tensor<85x180xf64>  
// CHECK:           }
// CHECK:           return %[[VAL_0]], %[[VAL_6]]#1 : tensor<85x180x18xf64>, tensor<85x180xf64>
// CHECK:         }

// LOCKSTEP:  func.func private @foo_raised(%arg0: tensor<85x180x18xf64>, %arg1: tensor<85x180xf64>) -> (tensor<85x180x18xf64>, tensor<85x180xf64>) {
// LOCKSTEP-NEXT:    %cst = stablehlo.constant dense<5.000000e-01> : tensor<85x180x1xf64>
// LOCKSTEP-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LOCKSTEP-NEXT{LITERAL}:    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<85x180x18xf64>, tensor<f64>) -> tensor<85x180xf64>
// LOCKSTEP-NEXT:    %1 = stablehlo.reshape %0 : (tensor<85x180xf64>) -> tensor<85x180x1xf64>
// LOCKSTEP-NEXT:    %2 = stablehlo.add %1, %cst : tensor<85x180x1xf64>
// LOCKSTEP-NEXT:    %3 = stablehlo.reshape %2 : (tensor<85x180x1xf64>) -> tensor<85x180xf64>
// LOCKSTEP-NEXT:    return %arg0, %3 : tensor<85x180x18xf64>, tensor<85x180xf64>
// LOCKSTEP-NEXT:  }
