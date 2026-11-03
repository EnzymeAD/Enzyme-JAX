// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func.func @reshard_cse_with_singleton_broadcast_in_dim
// CHECK-NEXT: %[[B1:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
// CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[B1]] <@mesh, [{}, {"a"}]> : tensor<1x8xf32>
// CHECK-NEXT: %[[RS:.*]] = stablehlo.reshape %[[R0]] : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK-NEXT: %[[B2:.*]] = stablehlo.broadcast_in_dim %[[RS]], dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
// CHECK-NEXT: return %[[R0]], %[[B2]] : tensor<1x8xf32>, tensor<8x1xf32>
func.func @reshard_cse_with_singleton_broadcast_in_dim(%arg0: tensor<8xf32>)
    -> (tensor<1x8xf32>, tensor<8x1xf32>) {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{}, {"a"}]> : tensor<1x8xf32>
  %2 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
  %3 = sdy.reshard %2 <@mesh, [{"a"}, {}]> : tensor<8x1xf32>
  return %1, %3 : tensor<1x8xf32>, tensor<8x1xf32>
}

// CHECK-LABEL: func.func @reshard_cse_with_transpose_and_singleton_reshape_bcast
// CHECK-NEXT: %[[R0:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[R0]], dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[B:.*]] = stablehlo.broadcast_in_dim %[[T]], dims = [1, 2] : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
// CHECK-NEXT: %[[RS:.*]] = stablehlo.reshape %[[B]] : (tensor<1x4x8xf32>) -> tensor<4x1x8xf32>
// CHECK-NEXT: return %[[R0]], %[[RS]] : tensor<8x4xf32>, tensor<4x1x8xf32>
func.func @reshard_cse_with_transpose_and_singleton_reshape_bcast(%arg0: tensor<8x4xf32>)
    -> (tensor<8x4xf32>, tensor<4x1x8xf32>) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
  %1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x4x8xf32>) -> tensor<4x1x8xf32>
  %4 = sdy.reshard %3 <@mesh, [{}, {}, {"a"}]> : tensor<4x1x8xf32>
  return %0, %4 : tensor<8x4xf32>, tensor<4x1x8xf32>
}
