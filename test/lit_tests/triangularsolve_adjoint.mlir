// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=triangular_solve_real_adjoint" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>, %arg2: tensor<32x8xf64>) -> tensor<32x8xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
    %1 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<32x8xf64>) -> tensor<8x32xf64>
    // CHECK: %2 = "stablehlo.triangular_solve"(%0, %1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = true}> : (tensor<8x8xf64>, tensor<8x32xf64>) -> tensor<8x32xf64>
    %2 = "stablehlo.triangular_solve"(%0, %1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose ADJOINT>, unit_diagonal = true}> : (tensor<8x8xf64>, tensor<8x32xf64>) -> tensor<8x32xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<8x32xf64>) -> tensor<32x8xf64>
    return %3 : tensor<32x8xf64>
  }
}
