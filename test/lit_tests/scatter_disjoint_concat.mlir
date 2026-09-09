// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// A set-index scatter assigns updates[k] to destination[indices[k]]. The new
// rewrite combines a chain of these assignments when the index sets are
// provably disjoint, so their relative order cannot affect the result:
//
//   first  = scatter(destination, indices_a, updates_a)
//   result = scatter(first,       indices_b, updates_b)
//     =>
//   result = scatter(destination, concat(indices_a, indices_b),
//                                 concat(updates_a, updates_b))
//
// Both scatters must already promise unique indices, and the intermediate
// destination must have no other uses. This runs through the normal optimizer,
// so the checks include constant folding and other existing canonicalizations.

// Three disjoint writes combine even though their batch shapes differ:
//
//   write   index shape   update shape   destination indices
//     a       2x3x1          2x3         [3, 5, 7, 9, 11, 13]
//     b       1x4             4         [30, 33, 36, 39]
//     c       2x2            2x2        [60, 62, 64, 66]
//
// The index-vector dimension is last in a, first in b, and implicit in c.
// Each is a point write into a rank-one destination. Flattening must keep
// every index paired with its original update. Check the complete combined
// index tensor, the matching update order a/b/c, and the set-index region.
// CHECK-LABEL: func.func @disjoint_chain
// CHECK-SAME: (%[[DEST:[^:]+]]: tensor<101xf32>, %[[A:[^:]+]]: tensor<2x3xf32>, %[[B:[^:]+]]: tensor<4xf32>, %[[C:[^:]+]]: tensor<2x2xf32>)
// CHECK-DAG: %[[INDICES:.*]] = stablehlo.constant dense<{{\[}}[3], [5], [7], [9], [11], [13], [30], [33], [36], [39], [60], [62], [64], [66]]> : tensor<14x1xi32>
// CHECK-DAG: %[[FLAT_A:.*]] = stablehlo.reshape %[[A]] : (tensor<2x3xf32>) -> tensor<6xf32>
// CHECK-DAG: %[[FLAT_C:.*]] = stablehlo.reshape %[[C]] : (tensor<2x2xf32>) -> tensor<4xf32>
// CHECK: %[[UPDATES:.*]] = stablehlo.concatenate %[[FLAT_A]], %[[B]], %[[FLAT_C]], dim = 0 : (tensor<6xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<14xf32>
// CHECK-NEXT: %[[RESULT:.*]] = "stablehlo.scatter"(%[[DEST]], %[[INDICES]], %[[UPDATES]])
// CHECK-SAME: indices_are_sorted = false
// CHECK-SAME: scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>
// CHECK-SAME: unique_indices = true
// CHECK-NEXT: ^bb0(%[[OLD:[^:]+]]: tensor<f32>, %[[NEW:[^:]+]]: tensor<f32>):
// CHECK-NEXT: stablehlo.return %[[NEW]] : tensor<f32>
// CHECK-NEXT: }) : (tensor<101xf32>, tensor<14x1xi32>, tensor<14xf32>) -> tensor<101xf32>
// CHECK-NOT: "stablehlo.scatter"
// CHECK: return %[[RESULT]]
func.func @disjoint_chain(%destination: tensor<101xf32>, %updates_a: tensor<2x3xf32>, %updates_b: tensor<4xf32>, %updates_c: tensor<2x2xf32>) -> tensor<101xf32> {
  %i = stablehlo.iota dim = 0 : tensor<6xi32>
  %two = stablehlo.constant dense<2> : tensor<6xi32>
  %three = stablehlo.constant dense<3> : tensor<6xi32>
  %scaled = stablehlo.multiply %i, %two : tensor<6xi32>
  %offset = stablehlo.add %scaled, %three : tensor<6xi32>
  %indices_a = stablehlo.reshape %offset : (tensor<6xi32>) -> tensor<2x3x1xi32>
  %indices_b = stablehlo.constant dense<[[30, 33, 36, 39]]> : tensor<1x4xi32>
  %indices_c = stablehlo.constant dense<[[60, 62], [64, 66]]> : tensor<2x2xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<2x3x1xi32>, tensor<2x3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 0>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<1x4xi32>, tensor<4xf32>) -> tensor<101xf32>
  %third = "stablehlo.scatter"(%second, %indices_c, %updates_c) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<2x2xi32>, tensor<2x2xf32>) -> tensor<101xf32>
  return %third : tensor<101xf32>
}

// a writes [3, 5, 7]; b writes [7, 9, 11]. Both touch destination[7],
// where b must win. A combined unordered scatter would lose that guarantee.
// Keep the two scatters and check that b still consumes the result of a.
// CHECK-LABEL: func.func @overlapping
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0,
// CHECK: "stablehlo.scatter"(%[[FIRST]],
func.func @overlapping(%destination: tensor<101xf32>, %updates_a: tensor<3xf32>, %updates_b: tensor<3xf32>) -> tensor<101xf32> {
  %indices_a = stablehlo.constant dense<[[3], [5], [7]]> : tensor<3x1xi32>
  %indices_b = stablehlo.constant dense<[[7], [9], [11]]> : tensor<3x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}

// The caller supplies both index tensors. Each set promises uniqueness, but
// that says nothing about overlap between sets. Without a disjointness proof,
// retain the chain so any overlapping elements still get their last update.
// CHECK-LABEL: func.func @unknown_indices
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0,
// CHECK: "stablehlo.scatter"(%[[FIRST]],
func.func @unknown_indices(%destination: tensor<101xf32>, %indices_a: tensor<3x1xi32>, %indices_b: tensor<3x1xi32>, %updates_a: tensor<3xf32>, %updates_b: tensor<3xf32>) -> tensor<101xf32> {
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}

// a writes [3, 3, 7]; b writes [20, 22, 24]. Their ranges are disjoint, but a
// writes index 3 twice. The new scatter would promise unique_indices=true,
// so reject the rewrite and preserve a's existing unique_indices=false.
// CHECK-LABEL: func.func @non_unique
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0, {{.*}}unique_indices = false
// CHECK: "stablehlo.scatter"(%[[FIRST]],
func.func @non_unique(%destination: tensor<101xf32>, %updates_a: tensor<3xf32>, %updates_b: tensor<3xf32>) -> tensor<101xf32> {
  %indices_a = stablehlo.constant dense<[[3], [3], [7]]> : tensor<3x1xi32>
  %indices_b = stablehlo.constant dense<[[20], [22], [24]]> : tensor<3x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}

// a ADDS to [3, 5, 7]; b ASSIGNS to [20, 22, 24]. The index sets are disjoint,
// but the combined scatter would use b's assignment region for every update.
// That would incorrectly discard a's addition to the old destination values.
// Keep both scatters, including the first scatter's addition.
// CHECK-LABEL: func.func @different_combiners
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0,
// CHECK: stablehlo.add
// CHECK: "stablehlo.scatter"(%[[FIRST]],
func.func @different_combiners(%destination: tensor<101xf32>, %updates_a: tensor<3xf32>, %updates_b: tensor<3xf32>) -> tensor<101xf32> {
  %indices_a = stablehlo.constant dense<[[3], [5], [7]]> : tensor<3x1xi32>
  %indices_b = stablehlo.constant dense<[[20], [22], [24]]> : tensor<3x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    %sum = stablehlo.add %old, %new : tensor<f32>
    stablehlo.return %sum : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}

// a writes [3, 5, 7]; b writes [20, 22, 24]. These assignments could otherwise
// combine, but the function also returns the destination after only a's writes.
// Preserve both returned states: the first must not include b's updates.
// CHECK-LABEL: func.func @multiple_uses
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0,
// CHECK: %[[SECOND:.*]] = "stablehlo.scatter"(%[[FIRST]],
// CHECK: return %[[FIRST]], %[[SECOND]]
func.func @multiple_uses(%destination: tensor<101xf32>, %updates_a: tensor<3xf32>, %updates_b: tensor<3xf32>) -> (tensor<101xf32>, tensor<101xf32>) {
  %indices_a = stablehlo.constant dense<[[3], [5], [7]]> : tensor<3x1xi32>
  %indices_b = stablehlo.constant dense<[[20], [22], [24]]> : tensor<3x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<101xf32>
  return %first, %second : tensor<101xf32>, tensor<101xf32>
}

// In mathematical integers a would write [252, 254, 256], disjoint from b's
// [0, 2, 4]. But the indices use i8: 2*iota + 126 + 126 wraps to [-4, -2, 0].
// The sets overlap at zero, where b must win. Range inference must account for
// modular arithmetic; check the wrapped indices and retain the write order.
// CHECK-LABEL: func.func @overflow_overlap
// CHECK-DAG: %[[WRAPPED:.*]] = stablehlo.constant dense<{{\[}}[-4], [-2], [0]]> : tensor<3x1xi8>
// CHECK-DAG: %[[SECOND_INDICES:.*]] = stablehlo.constant dense<{{\[}}[0], [2], [4]]> : tensor<3x1xi8>
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0, %[[WRAPPED]], %arg1)
// CHECK: %[[SECOND:.*]] = "stablehlo.scatter"(%[[FIRST]], %[[SECOND_INDICES]], %arg2)
// CHECK: return %[[SECOND]]
func.func @overflow_overlap(%destination: tensor<101xf32>, %updates_a: tensor<3xf32>, %updates_b: tensor<3xf32>) -> tensor<101xf32> {
  %i = stablehlo.iota dim = 0 : tensor<3xi8>
  %two = stablehlo.constant dense<2> : tensor<3xi8>
  %offset = stablehlo.constant dense<126> : tensor<3xi8>
  %scaled = stablehlo.multiply %i, %two : tensor<3xi8>
  %shifted = stablehlo.add %scaled, %offset : tensor<3xi8>
  %wrapped = stablehlo.add %shifted, %offset : tensor<3xi8>
  %indices_a = stablehlo.reshape %wrapped : (tensor<3xi8>) -> tensor<3x1xi8>
  %indices_b = stablehlo.constant dense<[[0], [2], [4]]> : tensor<3x1xi8>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi8>, tensor<3xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi8>, tensor<3xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}

// Converting integer 2 to boolean produces true, then converting back gives 1.
// Treating that conversion as bit truncation would incorrectly infer index 0
// and allow fusion with b's write to index 1. Both actually target index 1.
// Existing canonicalizations eliminate a's overwritten value entirely:
// result = concat(destination[0:1], b, destination[2:101]). Check that result.
// CHECK-LABEL: func.func @boolean_conversion_overlap
// CHECK: %[[BEFORE:.*]] = stablehlo.slice %arg0 [0:1]
// CHECK-NEXT: %[[AFTER:.*]] = stablehlo.slice %arg0 [2:101]
// CHECK-NEXT: %[[RESULT:.*]] = stablehlo.concatenate %[[BEFORE]], %arg2, %[[AFTER]], dim = 0
// CHECK-NEXT: return %[[RESULT]]
func.func @boolean_conversion_overlap(%destination: tensor<101xf32>, %updates_a: tensor<1xf32>, %updates_b: tensor<1xf32>) -> tensor<101xf32> {
  %two = stablehlo.constant dense<2> : tensor<1xi32>
  %boolean = stablehlo.convert %two : (tensor<1xi32>) -> tensor<1xi1>
  %integer = stablehlo.convert %boolean : (tensor<1xi1>) -> tensor<1xi32>
  %indices_a = stablehlo.reshape %integer : (tensor<1xi32>) -> tensor<1x1xi32>
  %indices_b = stablehlo.constant dense<1> : tensor<1x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<1x1xi32>, tensor<1xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<1x1xi32>, tensor<1xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}

// a starts at [3, 7, 11]; b starts at [12, 16, 20]. The start-index ranges are
// disjoint, but each update writes TWO elements. a's last window [11, 12]
// overlaps b's first window [12, 13], so b must win at destination[12].
// This rewrite only handles point updates; keep the window-update chain.
// CHECK-LABEL: func.func @window_updates
// CHECK: %[[FIRST:.*]] = "stablehlo.scatter"(%arg0,
// CHECK: "stablehlo.scatter"(%[[FIRST]],
func.func @window_updates(%destination: tensor<101xf32>, %updates_a: tensor<3x2xf32>, %updates_b: tensor<3x2xf32>) -> tensor<101xf32> {
  %indices_a = stablehlo.constant dense<[[3], [7], [11]]> : tensor<3x1xi32>
  %indices_b = stablehlo.constant dense<[[12], [16], [20]]> : tensor<3x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %updates_a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3x2xf32>) -> tensor<101xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %updates_b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<101xf32>, tensor<3x1xi32>, tensor<3x2xf32>) -> tensor<101xf32>
  return %second : tensor<101xf32>
}
