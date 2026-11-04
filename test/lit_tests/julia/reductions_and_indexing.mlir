// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" | FileCheck %s --check-prefix=IR
// RUN: enzymexlamlir-opt %s --shlo-to-julia-pipeline="enzyme-hlo-opt=false" --emit-julia -o /dev/null | FileCheck %s --check-prefix=JL

module {
  // Full tensor reduction to 0-D scalar: zero heap allocations!
  // IR-LABEL: julia.func @full_sum_reduce
  // IR-NOT: julia.alloc
  // JL-LABEL: function full_sum_reduce(
  // JL-NOT: undef
  func.func @full_sum_reduce(%x: tensor<4x8xf32>) -> tensor<f32> {
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %res = stablehlo.reduce(%x init: %init) across dimensions = [0, 1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<f32>
      reducer(%acc: tensor<f32>, %val: tensor<f32>) {
        %s = stablehlo.add %acc, %val : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    return %res : tensor<f32>
  }

  // Partial axis reduction + variadic argmax reduction.
  // IR-LABEL: julia.func @variadic_argmax
  // IR: julia.reduce!
  // JL-LABEL: function variadic_argmax(
  func.func @variadic_argmax(%vals: tensor<3x5xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
    %idxs = stablehlo.iota dim = 1 : tensor<3x5xi32>
    %init_v = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %init_i = stablehlo.constant dense<0> : tensor<i32>
    %max_v, %max_i = "stablehlo.reduce"(%vals, %idxs, %init_v, %init_i) ({
    ^bb0(%av: tensor<f32>, %ai: tensor<i32>, %bv: tensor<f32>, %bi: tensor<i32>):
      %gt = stablehlo.compare GT, %bv, %av, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %nv = stablehlo.select %gt, %bv, %av : tensor<i1>, tensor<f32>
      %ni = stablehlo.select %gt, %bi, %ai : tensor<i1>, tensor<i32>
      stablehlo.return %nv, %ni : tensor<f32>, tensor<i32>
    }) {dimensions = array<i64: 1>} : (tensor<3x5xf32>, tensor<3x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<3xf32>, tensor<3xi32>)
    return %max_v, %max_i : tensor<3xf32>, tensor<3xi32>
  }

  // ReduceWindow 2D max-pooling.
  // IR-LABEL: julia.func @max_pool_2d
  // IR: julia.reduce_window!
  // JL-LABEL: function max_pool_2d(
  func.func @max_pool_2d(%x: tensor<4x4xf32>) -> tensor<2x2xf32> {
    %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %res = "stablehlo.reduce_window"(%x, %init) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %m = stablehlo.maximum %a, %b : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }) {
      window_dimensions = array<i64: 2, 2>,
      window_strides = array<i64: 2, 2>
    } : (tensor<4x4xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %res : tensor<2x2xf32>
  }

  // Gather (embedding row lookup) and Scatter (indexed row addition).
  // IR-LABEL: julia.func @gather_and_scatter
  // IR: julia.gather!
  // IR: julia.scatter!
  // JL-LABEL: function gather_and_scatter(
  func.func @gather_and_scatter(
      %table: tensor<10x4xf32>,
      %indices: tensor<3x1xi32>,
      %updates: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<10x4xf32>) {
    %gathered = "stablehlo.gather"(%table, %indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      slice_sizes = array<i64: 1, 4>
    } : (tensor<10x4xf32>, tensor<3x1xi32>) -> tensor<3x4xf32>

    %scattered = "stablehlo.scatter"(%table, %indices, %updates) ({
    ^bb0(%cur: tensor<f32>, %upd: tensor<f32>):
      %sum = stablehlo.add %cur, %upd : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [1],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1
      >
    } : (tensor<10x4xf32>, tensor<3x1xi32>, tensor<3x4xf32>) -> tensor<10x4xf32>

    return %gathered, %scattered : tensor<3x4xf32>, tensor<10x4xf32>
  }

  // Variadic co-sort, pad, dynamic_slice, dynamic_update_slice, and 1D convolution.
  // IR-LABEL: julia.func @sort_pad_dyn_conv
  // IR: julia.sort!
  // IR: julia.pad!
  // IR: julia.dynamic_update_slice!
  // IR: julia.conv!
  // JL-LABEL: function sort_pad_dyn_conv(
  func.func @sort_pad_dyn_conv(
      %keys: tensor<8xf32>,
      %vals: tensor<8xi32>,
      %idx: tensor<i32>,
      %patch: tensor<3xf32>,
      %inp: tensor<1x10x2xf32>,
      %ker: tensor<3x2x4xf32>) -> (tensor<8xf32>, tensor<8xi32>, tensor<10xf32>, tensor<1x8x4xf32>) {
    %sk, %sv = "stablehlo.sort"(%keys, %vals) ({
    ^bb0(%k0: tensor<f32>, %k1: tensor<f32>, %v0: tensor<i32>, %v1: tensor<i32>):
      %lt = stablehlo.compare LT, %k0, %k1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %lt : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<8xf32>, tensor<8xi32>) -> (tensor<8xf32>, tensor<8xi32>)

    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %padded = stablehlo.pad %sk, %zero, low = [1], high = [1], interior = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<10xf32>
    %ds = stablehlo.dynamic_slice %padded, %idx, sizes = [3] : (tensor<10xf32>, tensor<i32>) -> tensor<3xf32>
    %upd = stablehlo.add %ds, %patch : tensor<3xf32>
    %dus = stablehlo.dynamic_update_slice %padded, %upd, %idx : (tensor<10xf32>, tensor<3xf32>, tensor<i32>) -> tensor<10xf32>

    %conv = stablehlo.convolution(%inp, %ker)
      dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
      window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x10x2xf32>, tensor<3x2x4xf32>) -> tensor<1x8x4xf32>

    return %sk, %sv, %dus, %conv : tensor<8xf32>, tensor<8xi32>, tensor<10xf32>, tensor<1x8x4xf32>
  }
}
