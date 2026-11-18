// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=scatter_masked_index_slice" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A lane store into per-block scratch under a loop guard that spans the
// block and element dimensions of the grid (e = 2 * block + element < 7, the
// raising's `e < NE` with two elements per block): the first three blocks are
// full, the last has one element. The scatter is cut to the full blocks,
// with no mask, and the last block keeps its mask (folded over the one
// block, it is a box this pattern cuts next).
func.func @tail(%smem: tensor<4x2x8xf64>, %update: tensor<4x2x3xf64>) -> tensor<4x2x8xf64> {
  %idx = stablehlo.constant dense<[[[[0], [1], [2]], [[0], [1], [2]]]]> : tensor<1x2x3x1xi64>
  %c6 = stablehlo.constant dense<6> : tensor<4x2xi64>
  %c0 = stablehlo.constant dense<0> : tensor<4x2xi64>
  %cm2 = stablehlo.constant dense<-2> : tensor<4xi64>
  %off = stablehlo.constant dense<-1> : tensor<4x2x3x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<4xi64>
  %mi = stablehlo.multiply %cm2, %i : tensor<4xi64>
  %bi = stablehlo.broadcast_in_dim %mi, dims = [0] : (tensor<4xi64>) -> tensor<4x2xi64>
  %j = stablehlo.iota dim = 1 : tensor<4x2xi64>
  %s = stablehlo.subtract %c6, %j : tensor<4x2xi64>
  %e = stablehlo.add %s, %bi : tensor<4x2xi64>
  %m = stablehlo.compare GE, %e, %c0 : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi1>
  %pred = stablehlo.broadcast_in_dim %m, dims = [0, 1] : (tensor<4x2xi1>) -> tensor<4x2x3x1xi1>
  %bidx = stablehlo.broadcast_in_dim %idx, dims = [0, 1, 2, 3] : (tensor<1x2x3x1xi64>) -> tensor<4x2x3x1xi64>
  %sel = stablehlo.select %pred, %bidx, %off : tensor<4x2x3x1xi1>, tensor<4x2x3x1xi64>
  %0 = "stablehlo.scatter"(%smem, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<4x2x8xf64>, tensor<4x2x3x1xi64>, tensor<4x2x3xf64>) -> tensor<4x2x8xf64>
  return %0 : tensor<4x2x8xf64>
}

// CHECK-LABEL: func.func @tail(
// CHECK-SAME: %[[SMEM:[^ ,:)]+]]: tensor<4x2x8xf64>, %[[UPD:[^ ,:)]+]]: tensor<4x2x3xf64>) -> tensor<4x2x8xf64> {
// CHECK: %[[PIN:[^ ,:)]+]] = stablehlo.slice %[[SMEM]] [0:3, 0:2, 0:8] : (tensor<4x2x8xf64>) -> tensor<3x2x8xf64>
// CHECK-NEXT: %[[PUPD:[^ ,:)]+]] = stablehlo.slice %[[UPD]] [0:3, 0:2, 0:3] : (tensor<4x2x3xf64>) -> tensor<3x2x3xf64>
// CHECK-NEXT: %[[PIDX:[^ ,:)]+]] = stablehlo.slice %{{[^ ]+}} [0:3, 0:2, 0:3, 0:1] : (tensor<4x2x3x1xi64>) -> tensor<3x2x3x1xi64>
// CHECK-NEXT: %[[P:[^ ,:)]+]] = "stablehlo.scatter"(%[[PIN]], %[[PIDX]], %[[PUPD]])
// CHECK: }) : (tensor<3x2x8xf64>, tensor<3x2x3x1xi64>, tensor<3x2x3xf64>) -> tensor<3x2x8xf64>
// CHECK-NEXT: %[[TIDX:[^ ,:)]+]] = stablehlo.slice %{{[^ ]+}} [3:4, 0:2, 0:3, 0:1] : (tensor<4x2x3x1xi64>) -> tensor<1x2x3x1xi64>
// CHECK-NEXT: %[[TUPD:[^ ,:)]+]] = stablehlo.slice %[[UPD]] [3:4, 0:2, 0:3] : (tensor<4x2x3xf64>) -> tensor<1x2x3xf64>
// CHECK-NEXT: %[[TIN:[^ ,:)]+]] = stablehlo.slice %[[SMEM]] [3:4, 0:2, 0:8] : (tensor<4x2x8xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT: %[[T:[^ ,:)]+]] = "stablehlo.scatter"(%[[TIN]], %[[TIDX]], %[[TUPD]])
// CHECK: }) : (tensor<1x2x8xf64>, tensor<1x2x3x1xi64>, tensor<1x2x3xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.concatenate %[[P]], %[[T]], dim = 0 : (tensor<3x2x8xf64>, tensor<1x2x8xf64>) -> tensor<4x2x8xf64>
// CHECK-NEXT: return %[[R]] : tensor<4x2x8xf64>
// CHECK-NEXT: }

// The same guard read through a slice of the padded lane mask (the lanes
// 0:3 of a 4-wide block), anded with the constant lane mask.
func.func @sliced(%smem: tensor<4x2x8xf64>, %update: tensor<4x2x3xf64>) -> tensor<4x2x8xf64> {
  %idx = stablehlo.constant dense<[[[[0], [1], [2]], [[0], [1], [2]]]]> : tensor<1x2x3x1xi64>
  %lane = stablehlo.constant dense<[[true, true, true, false], [true, true, true, false]]> : tensor<2x4xi1>
  %c6 = stablehlo.constant dense<6> : tensor<4x2xi64>
  %c0 = stablehlo.constant dense<0> : tensor<4x2xi64>
  %cm2 = stablehlo.constant dense<-2> : tensor<4xi64>
  %off = stablehlo.constant dense<-1> : tensor<4x2x3x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<4xi64>
  %mi = stablehlo.multiply %cm2, %i : tensor<4xi64>
  %bi = stablehlo.broadcast_in_dim %mi, dims = [0] : (tensor<4xi64>) -> tensor<4x2xi64>
  %j = stablehlo.iota dim = 1 : tensor<4x2xi64>
  %s = stablehlo.subtract %c6, %j : tensor<4x2xi64>
  %e = stablehlo.add %s, %bi : tensor<4x2xi64>
  %m = stablehlo.compare GE, %e, %c0 : (tensor<4x2xi64>, tensor<4x2xi64>) -> tensor<4x2xi1>
  %bm = stablehlo.broadcast_in_dim %m, dims = [0, 1] : (tensor<4x2xi1>) -> tensor<4x2x4xi1>
  %bl = stablehlo.broadcast_in_dim %lane, dims = [1, 2] : (tensor<2x4xi1>) -> tensor<4x2x4xi1>
  %a = stablehlo.and %bm, %bl : tensor<4x2x4xi1>
  %sl = stablehlo.slice %a [0:4, 0:2, 0:3] : (tensor<4x2x4xi1>) -> tensor<4x2x3xi1>
  %pred = stablehlo.broadcast_in_dim %sl, dims = [0, 1, 2] : (tensor<4x2x3xi1>) -> tensor<4x2x3x1xi1>
  %bidx = stablehlo.broadcast_in_dim %idx, dims = [0, 1, 2, 3] : (tensor<1x2x3x1xi64>) -> tensor<4x2x3x1xi64>
  %sel = stablehlo.select %pred, %bidx, %off : tensor<4x2x3x1xi1>, tensor<4x2x3x1xi64>
  %0 = "stablehlo.scatter"(%smem, %sel, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [0, 1], scatter_dims_to_operand_dims = [2], index_vector_dim = 3>, unique_indices = false}> ({
  ^bb0(%a0: tensor<f64>, %b0: tensor<f64>):
    stablehlo.return %b0 : tensor<f64>
  }) : (tensor<4x2x8xf64>, tensor<4x2x3x1xi64>, tensor<4x2x3xf64>) -> tensor<4x2x8xf64>
  return %0 : tensor<4x2x8xf64>
}

// CHECK-LABEL: func.func @sliced(
// CHECK: %[[P:[^ ,:)]+]] = "stablehlo.scatter"(%{{[^ ]+}}, %{{[^ ]+}}, %{{[^ ]+}})
// CHECK: }) : (tensor<3x2x8xf64>, tensor<3x2x3x1xi64>, tensor<3x2x3xf64>) -> tensor<3x2x8xf64>
// CHECK: %[[T:[^ ,:)]+]] = "stablehlo.scatter"(%{{[^ ]+}}, %{{[^ ]+}}, %{{[^ ]+}})
// CHECK: }) : (tensor<1x2x8xf64>, tensor<1x2x3x1xi64>, tensor<1x2x3xf64>) -> tensor<1x2x8xf64>
// CHECK-NEXT: %[[R:[^ ,:)]+]] = stablehlo.concatenate %[[P]], %[[T]], dim = 0 : (tensor<3x2x8xf64>, tensor<1x2x8xf64>) -> tensor<4x2x8xf64>
