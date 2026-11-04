// RUN: enzymexlamlir-opt --split-input-file --distributed-isolate-gather-scatter %s | FileCheck %s

// Each tensor operand and result of a gather gets a reshard point whose rule
// breaks every dimension.
// CHECK-LABEL: module @isolate_gather
// CHECK: %[[TABLE:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l]) {i=512, j=64, k=512, l=64}, custom>}
// CHECK: %[[IDS:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg1) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([l, m, n]) {i=8, j=1, k=1, l=8, m=1, n=1}, custom>}
// CHECK: %[[GATHER:.+]] = "stablehlo.gather"(%[[TABLE]], %[[IDS]])
// CHECK: %[[OUT:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%[[GATHER]]) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([l, m, n]) {i=8, j=1, k=64, l=8, m=1, n=64}, custom>}
// CHECK: stablehlo.negate %[[OUT]]
module @isolate_gather {
  func.func @main(%table: tensor<512x64xf32>, %ids: tensor<8x1x1xi32>) -> tensor<8x1x64xf32> {
    %g = "stablehlo.gather"(%table, %ids) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
    %n = stablehlo.negate %g : tensor<8x1x64xf32>
    return %n : tensor<8x1x64xf32>
  }
}

// -----

// A scatter's three operands and its result each get a reshard point.
// CHECK-LABEL: module @isolate_scatter
// CHECK: %[[TABLE:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg0)
// CHECK: %[[IDS:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg1)
// CHECK: %[[UPDATES:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg2)
// CHECK: %[[SCATTER:.+]] = "stablehlo.scatter"(%[[TABLE]], %[[IDS]], %[[UPDATES]])
// CHECK: %[[OUT:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%[[SCATTER]])
// CHECK: return %[[OUT]]
module @isolate_scatter {
  func.func @main(%table: tensor<512x64xf32>, %ids: tensor<8x1xi32>, %upd: tensor<8x64xf32>) -> tensor<512x64xf32> {
    %s = "stablehlo.scatter"(%table, %ids, %upd) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false}> ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      %r = stablehlo.add %a, %b : tensor<f32>
      stablehlo.return %r : tensor<f32>
    }) : (tensor<512x64xf32>, tensor<8x1xi32>, tensor<8x64xf32>) -> tensor<512x64xf32>
    return %s : tensor<512x64xf32>
  }
}

// -----

// A scalar index has no dimension to break, so it gets no reshard point.
// CHECK-LABEL: module @negative_scalar_index
// CHECK: %[[TABLE:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg0)
// CHECK-NEXT: "stablehlo.gather"(%[[TABLE]], %arg1)
module @negative_scalar_index {
  func.func @main(%table: tensor<512x64xf32>, %id: tensor<i32>) -> tensor<64xf32> {
    %g = "stablehlo.gather"(%table, %id) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 0>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<512x64xf32>, tensor<i32>) -> tensor<64xf32>
    return %g : tensor<64xf32>
  }
}

// -----

// A gather inside a nested region is left alone.
// CHECK-LABEL: module @negative_nested_gather
// CHECK-NOT: enzymexla_reshard_point
module @negative_nested_gather {
  func.func @main(%pred: tensor<i1>, %table: tensor<512x64xf32>, %ids: tensor<8x1x1xi32>) -> tensor<8x1x64xf32> {
    %r = "stablehlo.if"(%pred) ({
      %g = "stablehlo.gather"(%table, %ids) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
      stablehlo.return %g : tensor<8x1x64xf32>
    }, {
      %z = stablehlo.constant dense<0.0> : tensor<8x1x64xf32>
      stablehlo.return %z : tensor<8x1x64xf32>
    }) : (tensor<i1>) -> tensor<8x1x64xf32>
    return %r : tensor<8x1x64xf32>
  }
}

// -----

// A gather feeding another gather shares one reshard point between them.
// CHECK-LABEL: module @chained_gathers_share_point
// CHECK: %[[FIRST:.+]] = "stablehlo.gather"
// CHECK-NEXT: %[[BETWEEN:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%[[FIRST]])
// CHECK-NEXT: %[[POSITIONS:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%arg2)
// CHECK-NEXT: "stablehlo.gather"(%[[BETWEEN]], %[[POSITIONS]])
module @chained_gathers_share_point {
  func.func @main(%table: tensor<512x64xf32>, %ids: tensor<8x1x1xi32>, %pos: tensor<4x1xi32>) -> tensor<4x1x64xf32> {
    %g = "stablehlo.gather"(%table, %ids) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
    %h = "stablehlo.gather"(%g, %pos) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 64>}> : (tensor<8x1x64xf32>, tensor<4x1xi32>) -> tensor<4x1x64xf32>
    return %h : tensor<4x1x64xf32>
  }
}

// -----

// A reshard point carries the Shardy sharding of the value it sits on.
// CHECK-LABEL: module @point_keeps_sharding
// CHECK: %[[GATHER:.+]] = "stablehlo.gather"
// CHECK-NEXT: %[[OUT:.+]] = stablehlo.custom_call @enzymexla_reshard_point(%[[GATHER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>
// CHECK-NEXT: sdy.reshard %[[OUT]]
module @point_keeps_sharding {
  sdy.mesh @mesh = <["x"=4]>
  func.func @main(%table: tensor<512x64xf32>, %ids: tensor<8x1x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<8x1x64xf32> {
    %g = "stablehlo.gather"(%table, %ids) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<512x64xf32>, tensor<8x1x1xi32>) -> tensor<8x1x64xf32>
    %r = sdy.reshard %g <@mesh, [{}, {}, {"x"}]> : tensor<8x1x64xf32>
    return %r : tensor<8x1x64xf32>
  }
}
