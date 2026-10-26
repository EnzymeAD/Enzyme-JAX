// RUN: enzymexlamlir-opt %s -split-input-file -verify-diagnostics

func.func @mismatched_diagonals(%arg0: tensor<64xf32>, %arg1: tensor<63xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x32xf32>) -> tensor<64x32xf32> {
    // expected-error @+1 {{requires all diagonals to have the same type}}
    %0:2 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<64xf32>, tensor<63xf32>, tensor<64xf32>, tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<i32>)
    return %0#0 : tensor<64x32xf32>
}

// -----

func.func @mismatched_rhs(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // expected-error @+1 {{requires the leading ndim-1 dimensions of `B` to equal the dimensions of the diagonals}}
    %0:2 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>)
    return %0#0 : tensor<64x64xf32>
}

// -----

func.func @mismatched_info(%arg0: tensor<4x64xf32>, %arg1: tensor<4x64xf32>, %arg2: tensor<4x64xf32>, %arg3: tensor<4x64x2xf32>) -> tensor<4x64x2xf32> {
    // expected-error @+1 {{requires `info` to have the batch dimensions of `B`}}
    %0:2 = enzymexla.linalg.tridiagonal_solve %arg0, %arg1, %arg2, %arg3: (tensor<4x64xf32>, tensor<4x64xf32>, tensor<4x64xf32>, tensor<4x64x2xf32>) -> (tensor<4x64x2xf32>, tensor<i32>)
    return %0#0 : tensor<4x64x2xf32>
}
