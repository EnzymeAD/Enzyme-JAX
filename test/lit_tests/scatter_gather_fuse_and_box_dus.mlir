// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=fuse_gather_into_scatter;scatter_point_box_to_dus" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s --check-prefix=FULL
// RUN: stablehlo-translate %s --interpret
// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | stablehlo-translate - --interpret

module {
  // Issue #1892: scatter(in, ind, add(gather(in, ind), upd)) -> scatter-add without gather.
  func.func @fuse_gather_add_into_scatter(
      %arg0: tensor<4x4xf64>,
      %ind: tensor<4x2xi64>,
      %upd: tensor<4xf64>) -> tensor<4x4xf64> {
    %g = "stablehlo.gather"(%arg0, %ind) <{
      dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x4xf64>, tensor<4x2xi64>) -> tensor<4xf64>
    %sum = stablehlo.add %g, %upd : tensor<4xf64>
    %res = "stablehlo.scatter"(%arg0, %ind, %sum) <{
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%old: tensor<f64>, %u: tensor<f64>):
      stablehlo.return %u : tensor<f64>
    }) : (tensor<4x4xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x4xf64>
    return %res : tensor<4x4xf64>
  }
  // CHECK-LABEL: func.func @fuse_gather_add_into_scatter
  // CHECK-NOT:   stablehlo.gather
  // CHECK:       "stablehlo.scatter"(%arg0, %arg1, %arg2)
  // CHECK:       ^bb0(%[[OLD:.+]]: tensor<f64>, %[[U:.+]]: tensor<f64>):
  // CHECK-NEXT:    %[[ADD:.+]] = stablehlo.add %[[OLD]], %[[U]] : tensor<f64>
  // CHECK-NEXT:    stablehlo.return %[[ADD]] : tensor<f64>

  // Multi-op affine update: 2.0 * gather(in, ind) - upd -> fused into scatter body.
  func.func @fuse_gather_affine_into_scatter(
      %arg0: tensor<4x4xf32>,
      %ind: tensor<3x2xi64>,
      %upd: tensor<3xf32>) -> tensor<4x4xf32> {
    %two = stablehlo.constant dense<2.0> : tensor<3xf32>
    %g = "stablehlo.gather"(%arg0, %ind) <{
      dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x4xf32>, tensor<3x2xi64>) -> tensor<3xf32>
    %scaled = stablehlo.multiply %g, %two : tensor<3xf32>
    %diff = stablehlo.subtract %scaled, %upd : tensor<3xf32>
    %res = "stablehlo.scatter"(%arg0, %ind, %diff) <{
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%old: tensor<f32>, %u: tensor<f32>):
      stablehlo.return %u : tensor<f32>
    }) : (tensor<4x4xf32>, tensor<3x2xi64>, tensor<3xf32>) -> tensor<4x4xf32>
    return %res : tensor<4x4xf32>
  }
  // CHECK-LABEL: func.func @fuse_gather_affine_into_scatter
  // CHECK-NOT:   stablehlo.gather
  // CHECK:       "stablehlo.scatter"(%arg0, %arg1, %arg2)
  // CHECK:       ^bb0(%[[OLD2:.+]]: tensor<f32>, %[[U2:.+]]: tensor<f32>):
  // CHECK:         %[[MUL2:.+]] = stablehlo.multiply %[[OLD2]], %{{.+}} : tensor<f32>
  // CHECK-NEXT:    %[[SUB2:.+]] = stablehlo.subtract %[[MUL2]], %[[U2]] : tensor<f32>
  // CHECK-NEXT:    stablehlo.return %[[SUB2]] : tensor<f32>

  // Must NOT fuse when unique_indices = false.
  func.func @no_fuse_when_not_unique(
      %arg0: tensor<4x4xf32>,
      %ind: tensor<2x2xi64>,
      %upd: tensor<2xf32>) -> tensor<4x4xf32> {
    %g = "stablehlo.gather"(%arg0, %ind) <{
      dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 1>
    }> : (tensor<4x4xf32>, tensor<2x2xi64>) -> tensor<2xf32>
    %sum = stablehlo.add %g, %upd : tensor<2xf32>
    %res = "stablehlo.scatter"(%arg0, %ind, %sum) <{
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>,
      unique_indices = false
    }> ({
    ^bb0(%old: tensor<f32>, %u: tensor<f32>):
      stablehlo.return %u : tensor<f32>
    }) : (tensor<4x4xf32>, tensor<2x2xi64>, tensor<2xf32>) -> tensor<4x4xf32>
    return %res : tensor<4x4xf32>
  }
  // CHECK-LABEL: func.func @no_fuse_when_not_unique
  // CHECK:       "stablehlo.gather"
  // CHECK:       "stablehlo.scatter"

  // Issue #1957: ConstantSetindex point scatter over contiguous box -> DUS -> slice + concat.
  func.func @scatter_constant_box_transposed(%arg0: tensor<10x2xf64>) -> tensor<10x2xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
    %c = stablehlo.constant dense<[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]]> : tensor<10x2xi64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<10x2xf64>) -> tensor<2x10xf64>
    %1 = "stablehlo.scatter"(%0, %c, %cst_0) <{
      indices_are_sorted = false,
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      stablehlo.return %cst : tensor<f64>
    }) : (tensor<2x10xf64>, tensor<10x2xi64>, tensor<10xf64>) -> tensor<2x10xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x10xf64>) -> tensor<10x2xf64>
    return %2 : tensor<10x2xf64>
  }
  // CHECK-LABEL: func.func @scatter_constant_box_transposed
  // CHECK-NOT:   stablehlo.scatter
  // CHECK:       stablehlo.dynamic_update_slice
  // FULL-LABEL:  func.func @scatter_constant_box_transposed
  // FULL-NOT:    stablehlo.scatter
  // FULL-NOT:    stablehlo.transpose
  // FULL:        stablehlo.slice %arg0 [0:10, 1:2]

  // Non-constant Setindex over row-major 2x3 box -> reshape + DUS.
  func.func @scatter_nonconst_box(%arg0: tensor<4x5xf32>, %upd: tensor<6xf32>) -> tensor<4x5xf32> {
    %c = stablehlo.constant dense<[[1, 2], [1, 3], [1, 4], [2, 2], [2, 3], [2, 4]]> : tensor<6x2xi64>
    %res = "stablehlo.scatter"(%arg0, %c, %upd) <{
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%old: tensor<f32>, %u: tensor<f32>):
      stablehlo.return %u : tensor<f32>
    }) : (tensor<4x5xf32>, tensor<6x2xi64>, tensor<6xf32>) -> tensor<4x5xf32>
    return %res : tensor<4x5xf32>
  }
  // CHECK-LABEL: func.func @scatter_nonconst_box
  // CHECK-NOT:   stablehlo.scatter
  // CHECK:       %[[RS:.+]] = stablehlo.reshape %arg1 : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK:       stablehlo.dynamic_update_slice %arg0, %[[RS]],

  func.func @main() {
    %i16 = stablehlo.iota dim = 0 : tensor<16xf64>
    %m44 = stablehlo.reshape %i16 : (tensor<16xf64>) -> tensor<4x4xf64>
    %ind4 = stablehlo.constant dense<[[0, 1], [1, 3], [2, 0], [3, 2]]> : tensor<4x2xi64>
    %upd4 = stablehlo.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf64>
    %r1 = func.call @fuse_gather_add_into_scatter(%m44, %ind4, %upd4) : (tensor<4x4xf64>, tensor<4x2xi64>, tensor<4xf64>) -> tensor<4x4xf64>
    %exp1 = stablehlo.constant dense<[[0.0, 11.0, 2.0, 3.0], [4.0, 5.0, 6.0, 27.0], [38.0, 9.0, 10.0, 11.0], [12.0, 13.0, 54.0, 15.0]]> : tensor<4x4xf64>
    check.expect_almost_eq %r1, %exp1 : tensor<4x4xf64>

    %i16f = stablehlo.iota dim = 0 : tensor<16xf32>
    %m44f = stablehlo.reshape %i16f : (tensor<16xf32>) -> tensor<4x4xf32>
    %ind3 = stablehlo.constant dense<[[0, 2], [2, 1], [3, 3]]> : tensor<3x2xi64>
    %upd3 = stablehlo.constant dense<[1.0, 3.0, 5.0]> : tensor<3xf32>
    %r2 = func.call @fuse_gather_affine_into_scatter(%m44f, %ind3, %upd3) : (tensor<4x4xf32>, tensor<3x2xi64>, tensor<3xf32>) -> tensor<4x4xf32>
    %exp2 = stablehlo.constant dense<[[0.0, 1.0, 3.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 15.0, 10.0, 11.0], [12.0, 13.0, 14.0, 25.0]]> : tensor<4x4xf32>
    check.expect_almost_eq %r2, %exp2 : tensor<4x4xf32>

    %i20 = stablehlo.iota dim = 0 : tensor<20xf64>
    %m10x2 = stablehlo.reshape %i20 : (tensor<20xf64>) -> tensor<10x2xf64>
    %r3 = func.call @scatter_constant_box_transposed(%m10x2) : (tensor<10x2xf64>) -> tensor<10x2xf64>
    %z10 = stablehlo.constant dense<0.0> : tensor<10x1xf64>
    %col1 = stablehlo.slice %m10x2 [0:10, 1:2] : (tensor<10x2xf64>) -> tensor<10x1xf64>
    %exp3 = stablehlo.concatenate %z10, %col1, dim = 1 : (tensor<10x1xf64>, tensor<10x1xf64>) -> tensor<10x2xf64>
    check.expect_almost_eq %r3, %exp3 : tensor<10x2xf64>

    %i20f = stablehlo.iota dim = 0 : tensor<20xf32>
    %m45 = stablehlo.reshape %i20f : (tensor<20xf32>) -> tensor<4x5xf32>
    %upd6 = stablehlo.constant dense<[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]> : tensor<6xf32>
    %r4 = func.call @scatter_nonconst_box(%m45, %upd6) : (tensor<4x5xf32>, tensor<6xf32>) -> tensor<4x5xf32>
    %upd23 = stablehlo.reshape %upd6 : (tensor<6xf32>) -> tensor<2x3xf32>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %exp4 = stablehlo.dynamic_update_slice %m45, %upd23, %c1, %c2 : (tensor<4x5xf32>, tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<4x5xf32>
    check.expect_almost_eq %r4, %exp4 : tensor<4x5xf32>

    return
  }
}
