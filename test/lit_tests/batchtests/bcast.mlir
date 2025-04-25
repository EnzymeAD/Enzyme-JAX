// RUN: enzymexlamlir-opt %s --enzyme-batch | FileCheck %s

module @reactant_mapslices attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @unbatched_f2(%arg0: tensor<5x2xi64>) -> tensor<5x2x3xi64> {
    // CHECK: [[BCAST:%.+]] = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<3x5x2xi64>) -> tensor<3x5x2x3xi64>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<5x2xi64>) -> tensor<5x2x3xi64>
    return %1 : tensor<5x2x3xi64>
  }
  func.func @main(%arg0: tensor<3x5x2xi64>) -> (tensor<3x5x2x3xi64>) {
    %2 = enzyme.batch @unbatched_f2(%arg0) {batch_shape = array<i64: 3>} : (tensor<3x5x2xi64>) -> tensor<3x5x2x3xi64>
    return %2 : tensor<3x5x2x3xi64>
  }
}
