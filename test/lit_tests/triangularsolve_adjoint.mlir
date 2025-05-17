// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=triangular_solve_real_adjoint" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module @reactant_fn attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>, %arg2: tensor<32x8xf64>) -> tensor<32x8xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
    %2 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x8xf64>) -> tensor<8x32xf64>
    // CHECK: %3 = "stablehlo.triangular_solve"(%0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = true}> : (tensor<8x8xf64>, tensor<8x32xf64>) -> tensor<8x32xf64>
    %3 = "stablehlo.triangular_solve"(%0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose ADJOINT>, unit_diagonal = true}> : (tensor<8x8xf64>, tensor<8x32xf64>) -> tensor<8x32xf64>
    %5 = stablehlo.transpose %3, dims = [1, 0] : (tensor<8x32xf64>) -> tensor<32x8xf64>
    return %5 : tensor<32x8xf64>
  }
}
