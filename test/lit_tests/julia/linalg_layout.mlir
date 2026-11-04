// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="reverse-dims=true enzyme-hlo-opt=false" --emit-julia -o /dev/null | FileCheck %s --check-prefix=REV
// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="reverse-dims=false enzyme-hlo-opt=false" --emit-julia -o /dev/null | FileCheck %s --check-prefix=NOREV

module {
  // 1D inner product unboxes to scalar Float64 with `@simd` loop and zero allocations.
  // REV-LABEL: function dot_1d(
  // REV-NOT: undef
  // REV: @simd for
  // NOREV-LABEL: function dot_1d(
  // NOREV-NOT: undef
  // NOREV: @simd for
  func.func @dot_1d(%x: tensor<32xf64>, %y: tensor<32xf64>) -> tensor<f64> {
    %r = stablehlo.dot_general %x, %y, contracting_dims = [0] x [0] : (tensor<32xf64>, tensor<32xf64>) -> tensor<f64>
    return %r : tensor<f64>
  }

  // 2D matmul:
  // Under reverse-dims=true, tensor<4x8> -> Array{Float32, 2}(8, 4), tensor<8x6> -> (6, 8),
  // output -> (6, 4) = rhs * lhs (`LinearAlgebra.mul!(dst, arg1, arg0)`).
  // Under reverse-dims=false, output -> (4, 6) = lhs * rhs (`LinearAlgebra.mul!(dst, arg0, arg1)`).
  // REV-LABEL: function matmul_2d(
  // REV: Array{Float32, 2}(undef, 6, 4)
  // REV: LinearAlgebra.mul!({{.*}}, arg1, arg0)
  // NOREV-LABEL: function matmul_2d(
  // NOREV: Array{Float32, 2}(undef, 4, 6)
  // NOREV: LinearAlgebra.mul!({{.*}}, arg0, arg1)
  func.func @matmul_2d(%a: tensor<4x8xf32>, %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
    %c = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
    return %c : tensor<4x6xf32>
  }

  // Batched 3D matmul lowers to `@inbounds for` with `@view(...)` calling `LinearAlgebra.mul!`.
  // REV-LABEL: function batched_matmul(
  // REV: @inbounds for
  // REV: LinearAlgebra.mul!(@view(
  // NOREV-LABEL: function batched_matmul(
  // NOREV: @inbounds for
  // NOREV: LinearAlgebra.mul!(@view(
  func.func @batched_matmul(%a: tensor<5x4x8xf32>, %b: tensor<5x8x3xf32>) -> tensor<5x4x3xf32> {
    %c = stablehlo.dot_general %a, %b,
      batching_dims = [0] x [0],
      contracting_dims = [2] x [1]
      : (tensor<5x4x8xf32>, tensor<5x8x3xf32>) -> tensor<5x4x3xf32>
    return %c : tensor<5x4x3xf32>
  }

  // General 4D tensor contraction lowers to `dot_general!` nested `@inbounds` loops.
  // REV-LABEL: function tensor_contract_4d(
  // REV: zero(Float32)
  // NOREV-LABEL: function tensor_contract_4d(
  // NOREV: zero(Float32)
  func.func @tensor_contract_4d(%a: tensor<2x3x4x5xf32>, %b: tensor<4x2x6x3xf32>) -> tensor<2x5x6xf32> {
    %c = stablehlo.dot_general %a, %b,
      batching_dims = [0] x [1],
      contracting_dims = [1, 2] x [3, 0]
      : (tensor<2x3x4x5xf32>, tensor<4x2x6x3xf32>) -> tensor<2x5x6xf32>
    return %c : tensor<2x5x6xf32>
  }

  // Cholesky + TriangularSolve lower to `LinearAlgebra.cholesky!` and `ldiv!`/`rdiv!`.
  // REV-LABEL: function cholesky_solve(
  // REV: LinearAlgebra.cholesky!(
  // REV: LinearAlgebra.{{ldiv!|rdiv!}}
  // NOREV-LABEL: function cholesky_solve(
  // NOREV: LinearAlgebra.cholesky!(
  // NOREV: LinearAlgebra.{{ldiv!|rdiv!}}
  func.func @cholesky_solve(%a: tensor<4x4xf64>, %b: tensor<4x2xf64>) -> tensor<4x2xf64> {
    %l = stablehlo.cholesky %a, lower = true : tensor<4x4xf64>
    %x = "stablehlo.triangular_solve"(%l, %b) {
      left_side = true,
      lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    } : (tensor<4x4xf64>, tensor<4x2xf64>) -> tensor<4x2xf64>
    return %x : tensor<4x2xf64>
  }
}
