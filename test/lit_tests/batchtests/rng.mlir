// RUN: enzymexlamlir-opt --enzyme-batch %s | FileCheck %s

func.func @main(%seed: tensor<10x2xui64>) -> (tensor<10x2xui64>, tensor<10xf64>) {
    %state, %output = enzyme.batch @rng(%seed) {batch_shape = array<i64: 10>} : (tensor<10x2xui64>) -> (tensor<10x2xui64>, tensor<10xf64>)
    return %state, %output : tensor<10x2xui64>, tensor<10xf64>
}

func.func @rng(%seed: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
    %state, %output = stablehlo.rng_bit_generator %seed, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
    return %state, %output : tensor<2xui64>, tensor<f64>
}

// CHECK:      func.func private @batched_rng(%arg0: tensor<10x2xui64>) -> (tensor<10x2xui64>, tensor<10xf64>) {
// CHECK-NEXT:   %[[SLICED:.*]] = stablehlo.slice %arg0 [0, 0] to [1, 2] step [1, 1] : (tensor<10x2xui64>) -> tensor<1x2xui64>
// CHECK-NEXT:   %[[SEED:.*]] = stablehlo.reshape %[[SLICED]] : (tensor<1x2xui64>) -> tensor<2xui64>
// CHECK-NEXT:   %[[NEW_STATE:.*]], %[[NEW_OUT:.*]] = stablehlo.rng_bit_generator %[[SEED]], algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10xf64>)
// CHECK-NEXT:   %[[BCAST_STATE:.*]] = stablehlo.broadcast_in_dim %[[NEW_STATE]], dims = [1] : (tensor<2xui64>) -> tensor<10x2xui64>
// CHECK-NEXT:   return %[[BCAST_STATE]], %[[NEW_OUT]] : tensor<10x2xui64>, tensor<10xf64>
// CHECK-NEXT: }
