// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Regression test for #2087: ConcatSliceStack in EnzymeHLOOptPass collapses
// stacked unit slices concatenated across an orthogonal dimension so the
// downstream gather operates directly on the reshaped vector without any
// intermediate slices or concatenates.

module @reactant_quantile {
  func.func @main(%arg0: tensor<4x4xf64>) -> tensor<3xf64> {
    %idx0 = stablehlo.constant dense<[[0], [7], [14]]> : tensor<3x1xi32>
    %idx1 = stablehlo.constant dense<[[1], [8], [15]]> : tensor<3x1xi32>
    %w = stablehlo.constant dense<[0.0, 0.5, 1.0]> : tensor<3xf64>
    %flat = stablehlo.reshape %arg0 : (tensor<4x4xf64>) -> tensor<16x1xf64>

    %s0 = stablehlo.slice %flat [0:1, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s1 = stablehlo.slice %flat [1:2, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s2 = stablehlo.slice %flat [2:3, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s3 = stablehlo.slice %flat [3:4, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %r0 = stablehlo.concatenate %s0, %s1, %s2, %s3, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x4xf64>

    %s4 = stablehlo.slice %flat [4:5, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s5 = stablehlo.slice %flat [5:6, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s6 = stablehlo.slice %flat [6:7, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s7 = stablehlo.slice %flat [7:8, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %r1 = stablehlo.concatenate %s4, %s5, %s6, %s7, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x4xf64>

    %s8 = stablehlo.slice %flat [8:9, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s9 = stablehlo.slice %flat [9:10, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s10 = stablehlo.slice %flat [10:11, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s11 = stablehlo.slice %flat [11:12, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %r2 = stablehlo.concatenate %s8, %s9, %s10, %s11, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x4xf64>

    %s12 = stablehlo.slice %flat [12:13, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s13 = stablehlo.slice %flat [13:14, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s14 = stablehlo.slice %flat [14:15, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %s15 = stablehlo.slice %flat [15:16, 0:1] : (tensor<16x1xf64>) -> tensor<1x1xf64>
    %r3 = stablehlo.concatenate %s12, %s13, %s14, %s15, dim = 1 : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x4xf64>

    %mat = stablehlo.concatenate %r0, %r1, %r2, %r3, dim = 0 : (tensor<1x4xf64>, tensor<1x4xf64>, tensor<1x4xf64>, tensor<1x4xf64>) -> tensor<4x4xf64>
    %t = stablehlo.transpose %mat, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64>
    %vec = stablehlo.reshape %t : (tensor<4x4xf64>) -> tensor<16xf64>

    %g0 = "stablehlo.gather"(%vec, %idx0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<3x1xi32>) -> tensor<3xf64>
    %g1 = "stablehlo.gather"(%vec, %idx1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf64>, tensor<3x1xi32>) -> tensor<3xf64>
    %diff = stablehlo.subtract %g1, %g0 : tensor<3xf64>
    %scaled = stablehlo.multiply %w, %diff : tensor<3xf64>
    %res = stablehlo.add %g0, %scaled : tensor<3xf64>
    return %res : tensor<3xf64>
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-NOT: stablehlo.slice
// CHECK-NOT: stablehlo.concatenate
// CHECK: stablehlo.gather
// CHECK: stablehlo.gather
// CHECK: return
