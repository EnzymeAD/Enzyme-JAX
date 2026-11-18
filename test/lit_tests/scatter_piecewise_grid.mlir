// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Reactant.jl#2166: 2D NFFT wrap-around / ifftshift 36x2 Cartesian grid
// ([9, 10, 11, 0, 1, 2] x [9, 10, 11, 0, 1, 2]) scattered into a complex<f64>
// zero splat constant -> separable slice + pad + concatenate, 0 scatters, 0 DUS.
func.func @nfft_2166_wrap_scatter_splat_complex(%upd: tensor<36xcomplex<f64>>) -> tensor<12x12xcomplex<f64>> {
  %zero = stablehlo.constant dense<(0.000000e+00, 0.000000e+00)> : tensor<12x12xcomplex<f64>>
  %idx = stablehlo.constant dense<[
    [9, 9], [10, 9], [11, 9], [0, 9], [1, 9], [2, 9],
    [9, 10], [10, 10], [11, 10], [0, 10], [1, 10], [2, 10],
    [9, 11], [10, 11], [11, 11], [0, 11], [1, 11], [2, 11],
    [9, 0], [10, 0], [11, 0], [0, 0], [1, 0], [2, 0],
    [9, 1], [10, 1], [11, 1], [0, 1], [1, 1], [2, 1],
    [9, 2], [10, 2], [11, 2], [0, 2], [1, 2], [2, 2]
  ]> : tensor<36x2xi32>
  %res = "stablehlo.scatter"(%zero, %idx, %upd) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    unique_indices = true
  }> ({
  ^bb0(%a: tensor<complex<f64>>, %b: tensor<complex<f64>>):
    stablehlo.return %b : tensor<complex<f64>>
  }) : (tensor<12x12xcomplex<f64>>, tensor<36x2xi32>, tensor<36xcomplex<f64>>) -> tensor<12x12xcomplex<f64>>
  return %res : tensor<12x12xcomplex<f64>>
}

// CHECK-LABEL: func.func @nfft_2166_wrap_scatter_splat_complex
// CHECK-NOT: stablehlo.scatter
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: stablehlo.slice
// CHECK: stablehlo.pad
// CHECK: stablehlo.concatenate
// CHECK-NOT: stablehlo.scatter
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: return {{.*}} : tensor<12x12xcomplex<f64>>

// Flattened 36x2 contiguous Cartesian box [2:8, 3:9] into a dynamic buffer -> 1 DUS.
func.func @flattened_contiguous_box_scatter(%buf: tensor<12x12xf32>, %upd: tensor<36xf32>) -> tensor<12x12xf32> {
  %idx = stablehlo.constant dense<[
    [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
    [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
    [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
    [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
    [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8],
    [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8]
  ]> : tensor<36x2xi32>
  %res = "stablehlo.scatter"(%buf, %idx, %upd) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    unique_indices = true
  }> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<12x12xf32>, tensor<36x2xi32>, tensor<36xf32>) -> tensor<12x12xf32>
  return %res : tensor<12x12xf32>
}

// CHECK-LABEL: func.func @flattened_contiguous_box_scatter
// CHECK-NOT: stablehlo.scatter
// CHECK: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.scatter
// CHECK: return {{.*}} : tensor<12x12xf32>

// Flattened 36x2 contiguous Cartesian box [2:8, 3:9] gather -> 1 slice.
func.func @flattened_contiguous_box_gather(%src: tensor<12x12xf32>) -> tensor<36xf32> {
  %idx = stablehlo.constant dense<[
    [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
    [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
    [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
    [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
    [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8],
    [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8]
  ]> : tensor<36x2xi32>
  %res = "stablehlo.gather"(%src, %idx) <{
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    slice_sizes = array<i64: 1, 1>
  }> : (tensor<12x12xf32>, tensor<36x2xi32>) -> tensor<36xf32>
  return %res : tensor<36xf32>
}

// CHECK-LABEL: func.func @flattened_contiguous_box_gather
// CHECK-NOT: stablehlo.gather
// CHECK: stablehlo.slice %arg0 [2:8, 3:9]
// CHECK-NOT: stablehlo.gather
// CHECK: return {{.*}} : tensor<36xf32>

// Wrap-around 36x2 [9, 10, 11, 0, 1, 2]^2 scatter into a dynamic buffer -> 0 scatters.
func.func @wrap_around_general_dest_buffer(%buf: tensor<12x12xf32>, %upd: tensor<36xf32>) -> tensor<12x12xf32> {
  %idx = stablehlo.constant dense<[
    [9, 9], [10, 9], [11, 9], [0, 9], [1, 9], [2, 9],
    [9, 10], [10, 10], [11, 10], [0, 10], [1, 10], [2, 10],
    [9, 11], [10, 11], [11, 11], [0, 11], [1, 11], [2, 11],
    [9, 0], [10, 0], [11, 0], [0, 0], [1, 0], [2, 0],
    [9, 1], [10, 1], [11, 1], [0, 1], [1, 1], [2, 1],
    [9, 2], [10, 2], [11, 2], [0, 2], [1, 2], [2, 2]
  ]> : tensor<36x2xi32>
  %res = "stablehlo.scatter"(%buf, %idx, %upd) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    unique_indices = true
  }> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<12x12xf32>, tensor<36x2xi32>, tensor<36xf32>) -> tensor<12x12xf32>
  return %res : tensor<12x12xf32>
}

// CHECK-LABEL: func.func @wrap_around_general_dest_buffer
// CHECK-NOT: stablehlo.scatter
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// CHECK: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK-NOT: stablehlo.scatter
// CHECK: return {{.*}} : tensor<12x12xf32>

// Strided Cartesian grid ([1, 3, 5, 7] x [0, 3, 6]) into a splat constant -> 1 interior pad.
func.func @strided_splat_constant_scatter_to_pad(%upd: tensor<12xf32>) -> tensor<10x9xf32> {
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<10x9xf32>
  %idx = stablehlo.constant dense<[
    [1, 0], [1, 3], [1, 6],
    [3, 0], [3, 3], [3, 6],
    [5, 0], [5, 3], [5, 6],
    [7, 0], [7, 3], [7, 6]
  ]> : tensor<12x2xi32>
  %res = "stablehlo.scatter"(%zero, %idx, %upd) <{
    indices_are_sorted = true,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    unique_indices = true
  }> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<10x9xf32>, tensor<12x2xi32>, tensor<12xf32>) -> tensor<10x9xf32>
  return %res : tensor<10x9xf32>
}

// CHECK-LABEL: func.func @strided_splat_constant_scatter_to_pad
// CHECK-NOT: stablehlo.scatter
// CHECK-NOT: stablehlo.dynamic_update_slice
// CHECK: stablehlo.pad {{.*}}, low = [1, 0], high = [2, 2], interior = [1, 2] : (tensor<4x3xf32>, tensor<f32>) -> tensor<10x9xf32>
// CHECK: return {{.*}} : tensor<10x9xf32>
