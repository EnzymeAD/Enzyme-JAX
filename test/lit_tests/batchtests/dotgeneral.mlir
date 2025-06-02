// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

module @reactant_mapslices attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @unbatched_f2(%arg0: tensor<2x3xi64>) -> tensor<2x2xi64> {
    %c = stablehlo.constant dense<0> : tensor<2x2xi64>
    %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xi64>) -> tensor<3x2xi64>
    // CHECK: [[DOT:%.+]] = stablehlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x2x3xi64>, tensor<5x3x2xi64>) -> tensor<5x2x2xi64>
    %9 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xi64>, tensor<3x2xi64>) -> tensor<2x2xi64>
    return %9 : tensor<2x2xi64>
  }
  func.func @main(%arg0: tensor<3x5x2xi64> {tf.aliasing_output = 1 : i32}) -> (tensor<2x5x2xi64>, tensor<3x5x2xi64>) {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x2xi64>) -> tensor<2x5x3xi64>
    %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<2x5x3xi64>) -> tensor<5x2x3xi64>
    %c = stablehlo.constant dense<0> : tensor<2x3xi64>
    %2 = enzyme.batch @unbatched_f2(%1) {batch_shape = array<i64: 5>} : (tensor<5x2x3xi64>) -> tensor<5x2x2xi64>
    %3 = stablehlo.transpose %2, dims = [1, 0, 2] : (tensor<5x2x2xi64>) -> tensor<2x5x2xi64>
    %4 = stablehlo.transpose %3, dims = [2, 1, 0] : (tensor<2x5x2xi64>) -> tensor<2x5x2xi64>
    %5 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<2x5x3xi64>) -> tensor<3x5x2xi64>
    return %4, %5 : tensor<2x5x2xi64>, tensor<3x5x2xi64>
  }
}
