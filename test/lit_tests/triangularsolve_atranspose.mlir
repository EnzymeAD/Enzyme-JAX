// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=triangular_solve_transpose" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<2x3x8x8xf64>, %arg1: tensor<2x3x32x8xf64>) -> tensor<3x2x32x8xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<2x3x8x8xf64>) -> tensor<3x2x8x8xf64>
    %2 = stablehlo.transpose %arg1, dims = [1, 0, 3, 2] : (tensor<2x3x32x8xf64>) -> tensor<3x2x8x32xf64>
    %3 = "stablehlo.triangular_solve"(%0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = true}> : (tensor<3x2x8x8xf64>, tensor<3x2x8x32xf64>) -> tensor<3x2x8x32xf64>
    %5 = stablehlo.transpose %3, dims = [0, 1, 3, 2] : (tensor<3x2x8x32xf64>) -> tensor<3x2x32x8xf64>
    return %5 : tensor<3x2x32x8xf64>
}

// CHECK: func.func @main1(%arg0: tensor<2x3x8x8xf64>, %arg1: tensor<2x3x32x8xf64>) -> tensor<3x2x32x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg1, dims = [1, 0, 3, 2] : (tensor<2x3x32x8xf64>) -> tensor<3x2x8x32xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %arg0, dims = [1, 0, 2, 3] : (tensor<2x3x8x8xf64>) -> tensor<3x2x8x8xf64>
// CHECK-NEXT:    %2 = "stablehlo.triangular_solve"(%1, %0) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<3x2x8x8xf64>, tensor<3x2x8x32xf64>) -> tensor<3x2x8x32xf64>
// CHECK-NEXT:    %3 = stablehlo.transpose %2, dims = [0, 1, 3, 2] : (tensor<3x2x8x32xf64>) -> tensor<3x2x32x8xf64>
// CHECK-NEXT:    return %3 : tensor<3x2x32x8xf64>
// CHECK-NEXT:  }

module {
  func.func @main2(%arg0: tensor<2x3x8x8xf64>, %arg1: tensor<2x3x32x8xf64>) -> tensor<3x2x32x8xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<2x3x8x8xf64>) -> tensor<3x2x8x8xf64>
    %2 = stablehlo.transpose %arg1, dims = [1, 0, 3, 2] : (tensor<2x3x32x8xf64>) -> tensor<3x2x8x32xf64>
    %3 = "stablehlo.triangular_solve"(%0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<3x2x8x8xf64>, tensor<3x2x8x32xf64>) -> tensor<3x2x8x32xf64>
    %5 = stablehlo.transpose %3, dims = [0, 1, 3, 2] : (tensor<3x2x8x32xf64>) -> tensor<3x2x32x8xf64>
    return %5 : tensor<3x2x32x8xf64>
  }
}

// CHECK: func.func @main2(%arg0: tensor<2x3x8x8xf64>, %arg1: tensor<2x3x32x8xf64>) -> tensor<3x2x32x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg1, dims = [1, 0, 3, 2] : (tensor<2x3x32x8xf64>) -> tensor<3x2x8x32xf64>
// CHECK-NEXT:    %1 = stablehlo.transpose %arg0, dims = [1, 0, 2, 3] : (tensor<2x3x8x8xf64>) -> tensor<3x2x8x8xf64>
// CHECK-NEXT:    %2 = "stablehlo.triangular_solve"(%1, %0) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = true}> : (tensor<3x2x8x8xf64>, tensor<3x2x8x32xf64>) -> tensor<3x2x8x32xf64>
// CHECK-NEXT:    %3 = stablehlo.transpose %2, dims = [0, 1, 3, 2] : (tensor<3x2x8x32xf64>) -> tensor<3x2x32x8xf64>
// CHECK-NEXT:    return %3 : tensor<3x2x32x8xf64>
// CHECK-NEXT:  }

module {
  func.func @main3(%arg0: tensor<2x3x8x8xf64>, %arg1: tensor<2x3x32x8xf64>) -> tensor<3x2x32x8xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<2x3x8x8xf64>) -> tensor<3x2x8x8xf64>
    %2 = stablehlo.transpose %arg1, dims = [1, 0, 3, 2] : (tensor<2x3x32x8xf64>) -> tensor<3x2x8x32xf64>
    %3 = "stablehlo.triangular_solve"(%0, %2) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose ADJOINT>, unit_diagonal = true}> : (tensor<3x2x8x8xf64>, tensor<3x2x8x32xf64>) -> tensor<3x2x8x32xf64>
    %5 = stablehlo.transpose %3, dims = [0, 1, 3, 2] : (tensor<3x2x8x32xf64>) -> tensor<3x2x32x8xf64>
    return %5 : tensor<3x2x32x8xf64>
  }
}

// CHECK: func.func @main3(%arg0: tensor<2x3x8x8xf64>, %arg1: tensor<2x3x32x8xf64>) -> tensor<3x2x32x8xf64> {
// CHECK-NEXT:    %0 = stablehlo.transpose %arg1, dims = [1, 0, 3, 2] : (tensor<2x3x32x8xf64>) -> tensor<3x2x8x32xf64>
// CHECK-NEXT:    %1 = chlo.conj %arg0 : tensor<2x3x8x8xf64> -> tensor<2x3x8x8xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0, 2, 3] : (tensor<2x3x8x8xf64>) -> tensor<3x2x8x8xf64>
// CHECK-NEXT:    %3 = "stablehlo.triangular_solve"(%2, %0) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<3x2x8x8xf64>, tensor<3x2x8x32xf64>) -> tensor<3x2x8x32xf64>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [0, 1, 3, 2] : (tensor<3x2x8x32xf64>) -> tensor<3x2x32x8xf64>
// CHECK-NEXT:    return %4 : tensor<3x2x32x8xf64>
// CHECK-NEXT:  }
