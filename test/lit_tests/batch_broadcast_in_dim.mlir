// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=batch_broadcast_in_dim},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s --check-prefix=TD

module {
  func.func private @scale_and_add(%a: tensor<4xf32>, %b: tensor<f32>) -> tensor<4xf32> {
    %bb = stablehlo.broadcast_in_dim %b, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %prod = stablehlo.multiply %a, %bb : tensor<4xf32>
    %sum = stablehlo.add %prod, %a : tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  // Full unbatching: all batch dimensions are constructed by broadcast_in_dim ->
  // unbatched call followed by post-call broadcast.
  // CHECK-LABEL: func.func @full_unbatch_broadcast(
  // CHECK-NOT: enzyme.batch
  // CHECK: %[[CALL:.*]] = call @scale_and_add(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[OUT:.*]] = stablehlo.broadcast_in_dim %[[CALL]], dims = [1] : (tensor<4xf32>) -> tensor<10x4xf32>
  // CHECK: return %[[OUT]] : tensor<10x4xf32>
  // TD-LABEL: func.func @full_unbatch_broadcast(
  // TD-NOT: enzyme.batch
  // TD: %[[TCALL:.*]] = call @scale_and_add(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // TD: %[[TOUT:.*]] = stablehlo.broadcast_in_dim %[[TCALL]], dims = [1] : (tensor<4xf32>) -> tensor<10x4xf32>
  // TD: return %[[TOUT]] : tensor<10x4xf32>
  func.func @full_unbatch_broadcast(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<10x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<10x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %2 = enzyme.batch @scale_and_add(%0, %1) <{batch_shape = array<i64: 10>}> : (tensor<10x4xf32>, tensor<10xf32>) -> tensor<10x4xf32>
    return %2 : tensor<10x4xf32>
  }

  // Partial unbatching with a splatted constant: batch_shape = [3, 5], where dim 0
  // is broadcast in %arg0 and splatted in %cst, while dim 1 varies in %arg0.
  // CHECK-LABEL: func.func @partial_unbatch_with_splat_const(
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<2.500000e+00> : tensor<5xf32>
  // CHECK: %[[BATCH:.*]] = enzyme.batch @scale_and_add(%arg0, %[[CST]]) <batch_shape = [5]> : (tensor<5x4xf32>, tensor<5xf32>) -> tensor<5x4xf32>
  // CHECK: %[[OUT:.*]] = stablehlo.broadcast_in_dim %[[BATCH]], dims = [1, 2] : (tensor<5x4xf32>) -> tensor<3x5x4xf32>
  // CHECK: return %[[OUT]] : tensor<3x5x4xf32>
  // TD-LABEL: func.func @partial_unbatch_with_splat_const(
  // TD: %[[TCST:.*]] = stablehlo.constant dense<2.500000e+00> : tensor<5xf32>
  // TD: %[[TBATCH:.*]] = enzyme.batch @scale_and_add(%arg0, %[[TCST]]) <batch_shape = [5]> : (tensor<5x4xf32>, tensor<5xf32>) -> tensor<5x4xf32>
  // TD: %[[TOUT:.*]] = stablehlo.broadcast_in_dim %[[TBATCH]], dims = [1, 2] : (tensor<5x4xf32>) -> tensor<3x5x4xf32>
  // TD: return %[[TOUT]] : tensor<3x5x4xf32>
  func.func @partial_unbatch_with_splat_const(%arg0: tensor<5x4xf32>) -> tensor<3x5x4xf32> {
    %cst = stablehlo.constant dense<2.500000e+00> : tensor<3x5xf32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<5x4xf32>) -> tensor<3x5x4xf32>
    %1 = enzyme.batch @scale_and_add(%0, %cst) <{batch_shape = array<i64: 3, 5>}> : (tensor<3x5x4xf32>, tensor<3x5xf32>) -> tensor<3x5x4xf32>
    return %1 : tensor<3x5x4xf32>
  }

  func.func private @two_outputs(%a: tensor<4xf32>, %b: tensor<f32>) -> (tensor<4xf32>, tensor<f32>) {
    %bb = stablehlo.broadcast_in_dim %b, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %sum = stablehlo.add %a, %bb : tensor<4xf32>
    return %sum, %b : tensor<4xf32>, tensor<f32>
  }

  // Singleton input dim [0, 1] on tensor<1x4xf32> + splat constant, multi-result batch.
  // CHECK-LABEL: func.func @singleton_dim_and_multi_result(
  // CHECK-NOT: enzyme.batch
  // CHECK-DAG: %[[CST:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[SQ:.*]] = stablehlo.reshape %arg0 : (tensor<1x4xf32>) -> tensor<4xf32>
  // CHECK: %[[CALL:.*]]:2 = call @two_outputs(%[[SQ]], %[[CST]]) : (tensor<4xf32>, tensor<f32>) -> (tensor<4xf32>, tensor<f32>)
  // CHECK: %[[B0:.*]] = stablehlo.broadcast_in_dim %[[CALL]]#0, dims = [1] : (tensor<4xf32>) -> tensor<6x4xf32>
  // CHECK: %[[B1:.*]] = stablehlo.broadcast_in_dim %[[CALL]]#1, dims = [] : (tensor<f32>) -> tensor<6xf32>
  // CHECK: return %[[B0]], %[[B1]] : tensor<6x4xf32>, tensor<6xf32>
  // TD-LABEL: func.func @singleton_dim_and_multi_result(
  // TD-NOT: enzyme.batch
  // TD: %[[TCALL:.*]]:2 = call @two_outputs(
  func.func @singleton_dim_and_multi_result(%arg0: tensor<1x4xf32>) -> (tensor<6x4xf32>, tensor<6xf32>) {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<6xf32>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<6x4xf32>
    %1:2 = enzyme.batch @two_outputs(%0, %cst) <{batch_shape = array<i64: 6>}> : (tensor<6x4xf32>, tensor<6xf32>) -> (tensor<6x4xf32>, tensor<6xf32>)
    return %1#0, %1#1 : tensor<6x4xf32>, tensor<6xf32>
  }

  func.func private @impure_rng(%state: tensor<2xui64>) -> (tensor<2xui64>, tensor<4xf32>) {
    %s, %r = stablehlo.rng_bit_generator %state, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xf32>)
    return %s, %r : tensor<2xui64>, tensor<4xf32>
  }

  // Must NOT unbatch RNG functions!
  // CHECK-LABEL: func.func @do_not_unbatch_rng(
  // CHECK: enzyme.batch @impure_rng
  // TD-LABEL: func.func @do_not_unbatch_rng(
  // TD: enzyme.batch @impure_rng
  func.func @do_not_unbatch_rng(%state: tensor<2xui64>) -> (tensor<4x2xui64>, tensor<4x4xf32>) {
    %0 = stablehlo.broadcast_in_dim %state, dims = [1] : (tensor<2xui64>) -> tensor<4x2xui64>
    %1:2 = enzyme.batch @impure_rng(%0) <{batch_shape = array<i64: 4>}> : (tensor<4x2xui64>) -> (tensor<4x2xui64>, tensor<4x4xf32>)
    return %1#0, %1#1 : tensor<4x2xui64>, tensor<4x4xf32>
  }
}
