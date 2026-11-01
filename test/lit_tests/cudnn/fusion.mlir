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

// Two fusions in the same module must get distinct symbols. The counter used to
// be a `static` inside the class template, so the add and multiply
// instantiations both handed out `__cudnn_fused_elementwise_dot_0` and the
// module failed to verify with "redefinition of symbol named ...".
module {
    func.func @dense3(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) {
        %1 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
        %2 = stablehlo.add %1, %arg2 : tensor<4x16x16xbf16>
        %3 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
        %4 = stablehlo.multiply %3, %arg2 : tensor<4x16x16xbf16>
        return %2, %4 : tensor<4x16x16xbf16>, tensor<4x16x16xbf16>
    }
}

// CHECK: func.func private @__cudnn_fused_elementwise_dot_1(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16> attributes {no_inline} {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    %1 = stablehlo.add %0, %arg2 : tensor<4x16x16xbf16>
// CHECK-NEXT:    return %1 : tensor<4x16x16xbf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @__cudnn_fused_elementwise_dot_0(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16> attributes {no_inline} {
// CHECK-NEXT:    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %arg2 : tensor<4x16x16xbf16>
// CHECK-NEXT:    return %1 : tensor<4x16x16xbf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @dense3(%arg0: tensor<4x16x16xbf16>, %arg1: tensor<4x16x16xbf16>, %arg2: tensor<4x16x16xbf16>) -> (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) {
// CHECK-NEXT:    %0 = stablehlo.custom_call @__cudnn$fusion(%arg0, %arg1, %arg2) {called_computations = [@__cudnn_fused_elementwise_dot_1]} : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    %1 = stablehlo.custom_call @__cudnn$fusion(%arg0, %arg1, %arg2) {called_computations = [@__cudnn_fused_elementwise_dot_0]} : (tensor<4x16x16xbf16>, tensor<4x16x16xbf16>, tensor<4x16x16xbf16>) -> tensor<4x16x16xbf16>
// CHECK-NEXT:    return %0, %1 : tensor<4x16x16xbf16>, tensor<4x16x16xbf16>
// CHECK-NEXT:  }
