// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-opt='max_constant_expansion=0' %s | FileCheck %s --check-prefix=SYMBOLIC --implicit-check-not='stablehlo.transpose{{.*}}xi32'

// The original batch order visits x first, then y:
//   indices_a = [[0, 8, 16], [1, 9, 17]]
//   indices_b = [[32, 40, 48], [33, 41, 49]]
//
// The destination stride is 1 along x and 8 along y. Before concatenating,
// visit x fastest by transposing BOTH indices and updates. This preserves
// the association of each value with its destination while allowing update
// producers to use the layout of contiguous destination accesses.
//
// The combined indices must be [0,1,8,9,16,17,32,33,40,41,48,49], with updates
// taken from transpose(a), then transpose(b), in exactly that order.
//
// Also disable constant expansion to model a large index grid. Generate its
// iotas in the new order directly, even though the original grid is shared by
// both scatters. A transpose of the integer grid would invite an extra kernel
// to materialize indices; transposes of the update tensors remain legal.
// SYMBOLIC-LABEL: func.func @contiguous_dimension_fastest
// SYMBOLIC-DAG: %[[STRIDE:.*]] = stablehlo.constant dense<8> : tensor<3x2xi32>
// SYMBOLIC: %[[X:.*]] = stablehlo.iota dim = 1 : tensor<3x2xi32>
// SYMBOLIC: %[[Y:.*]] = stablehlo.iota dim = 0 : tensor<3x2xi32>
// SYMBOLIC: %[[SY:.*]] = stablehlo.multiply %[[Y]], %[[STRIDE]]
// SYMBOLIC: %[[GRID:.*]] = stablehlo.add %[[X]], %[[SY]]
// SYMBOLIC: stablehlo.reshape %[[GRID]] : (tensor<3x2xi32>) -> tensor<6x1xi32>
// SYMBOLIC: "stablehlo.scatter"
// CHECK-LABEL: func.func @contiguous_dimension_fastest
// CHECK-DAG: %[[INDICES:.*]] = stablehlo.constant dense<{{\[}}[0], [1], [8], [9], [16], [17], [32], [33], [40], [41], [48], [49]]> : tensor<12x1xi32>
// CHECK-DAG: %[[AT:.*]] = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-DAG: %[[BT:.*]] = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-DAG: %[[AF:.*]] = stablehlo.reshape %[[AT]] : (tensor<3x2xf32>) -> tensor<6xf32>
// CHECK-DAG: %[[BF:.*]] = stablehlo.reshape %[[BT]] : (tensor<3x2xf32>) -> tensor<6xf32>
// CHECK: %[[UPDATES:.*]] = stablehlo.concatenate %[[AF]], %[[BF]], dim = 0
// CHECK-NEXT: %[[RESULT:.*]] = "stablehlo.scatter"(%arg0, %[[INDICES]], %[[UPDATES]])
// CHECK-SAME: unique_indices = true
// CHECK: return %[[RESULT]]
func.func @contiguous_dimension_fastest(%destination: tensor<80xf32>, %a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<80xf32> {
  %x = stablehlo.iota dim = 0 : tensor<2x3xi32>
  %y = stablehlo.iota dim = 1 : tensor<2x3xi32>
  %stride = stablehlo.constant dense<8> : tensor<2x3xi32>
  %offset = stablehlo.constant dense<32> : tensor<2x3xi32>
  %scaled_y = stablehlo.multiply %y, %stride : tensor<2x3xi32>
  %grid = stablehlo.add %x, %scaled_y : tensor<2x3xi32>
  %shifted = stablehlo.add %grid, %offset : tensor<2x3xi32>
  %indices_a = stablehlo.reshape %grid : (tensor<2x3xi32>) -> tensor<2x3x1xi32>
  %indices_b = stablehlo.reshape %shifted : (tensor<2x3xi32>) -> tensor<2x3x1xi32>
  %first = "stablehlo.scatter"(%destination, %indices_a, %a) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<80xf32>, tensor<2x3x1xi32>, tensor<2x3xf32>) -> tensor<80xf32>
  %second = "stablehlo.scatter"(%first, %indices_b, %b) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<80xf32>, tensor<2x3x1xi32>, tensor<2x3xf32>) -> tensor<80xf32>
  return %second : tensor<80xf32>
}
