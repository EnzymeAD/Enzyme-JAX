// RUN: enzymexlamlir-opt --enzymexla-cudnn-hlo-opt %s | FileCheck %s

module {
    func.func @dense1(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> (tensor<4x16x16xbf16>) {
        %1 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
        %2 = stablehlo.add %1, %arg2 : tensor<4x16x16xbf16>
        return %2 : tensor<4x16x16xbf16>
    }
}

// CHECK: func.func private @__cudnn_fused_elementwise_dot_0(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16> attributes {no_inline} {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg2 : tensor<4x16x16xbf16>
// CHECK-NEXT:    return %1 : tensor<4x16x16xbf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @dense1(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16> {
// CHECK-NEXT:    %0 = stablehlo.custom_call @__cudnn$fusion(%arg0, %arg1, %arg2) {called_computations = [@__cudnn_fused_elementwise_dot_0]} : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    return %0 : tensor<4x16x16xbf16>
// CHECK-NEXT:  }


module {
    func.func @dense2(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> (tensor<4x16x16xbf16>) {
        %1 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
        %2 = stablehlo.subtract %arg2, %1 : tensor<4x16x16xbf16>
        return %2 : tensor<4x16x16xbf16>
    }
}

// CHECK: func.func private @__cudnn_fused_elementwise_dot_0(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16> attributes {no_inline} {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    %1 = stablehlo.subtract %arg2, %0 : tensor<4x16x16xbf16>
// CHECK-NEXT:    return %1 : tensor<4x16x16xbf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @dense2(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16> {
// CHECK-NEXT:    %0 = stablehlo.custom_call @__cudnn$fusion(%arg0, %arg1, %arg2) {called_computations = [@__cudnn_fused_elementwise_dot_0]} : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    return %0 : tensor<4x16x16xbf16>
// CHECK-NEXT:  }

// A dot_general with no contracting dimensions is a batched outer product.
// cuDNN has no candidate for it ("No candidates could be compiled"), so it must
// be left alone rather than wrapped in a fusion.
module {
    func.func @outer_product(%arg0: tensor<4x16xbf16>, %arg1: tensor<8x4x16xbf16>, %arg2: tensor<4x16x8xbf16>) -> (tensor<4x16x8xbf16>) {
        %1 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [1, 2], contracting_dims = [] x [] : (tensor<4x16xbf16>, tensor<8x4x16xbf16>) -> tensor<4x16x8xbf16>
        %2 = stablehlo.multiply %1, %arg2 : tensor<4x16x8xbf16>
        return %2 : tensor<4x16x8xbf16>
    }
}

// CHECK:      func.func @outer_product
// CHECK-NOT:    stablehlo.custom_call @__cudnn$fusion
// CHECK:        stablehlo.dot_general
// CHECK-NEXT:   stablehlo.multiply

