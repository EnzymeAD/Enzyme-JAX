// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcast_reshape" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s


func.func @r(%1107: tensor<1x4x1x2048xf32>) -> tensor<1x4x1x2048x14xf32> {
    %1108 = stablehlo.reshape %1107 : (tensor<1x4x1x2048xf32>) -> tensor<4x2048xf32>
    %1109 = stablehlo.broadcast_in_dim %1108, dims = [1, 3] : (tensor<4x2048xf32>) -> tensor<1x4x1x2048x14xf32>
  return %1109 : tensor<1x4x1x2048x14xf32>
}

func.func public @r2(%0: tensor<1xf32>) -> (tensor<288xf32>) {
  %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<288xf32>
  return %2 : tensor<288xf32>
}

func.func public @r3(%1056: tensor<1x3072xi32>) -> (tensor<3072x1xi32>) {
  %1057 = "stablehlo.reshape"(%1056) : (tensor<1x3072xi32>) -> tensor<3072xi32>
  %1058 = "stablehlo.broadcast_in_dim"(%1057) {broadcast_dimensions = array<i64: 0>} : (tensor<3072xi32>) -> tensor<3072x1xi32>
  return %1058 : tensor<3072x1xi32>
}

// CHECK:  func.func @r(%arg0: tensor<1x4x1x2048xf32>) -> tensor<1x4x1x2048x14xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x4x1x2048xf32>) -> tensor<1x4x1x2048x14xf32>
// CHECK-NEXT:    return %0 : tensor<1x4x1x2048x14xf32>
// CHECK-NEXT:  }
// CHECK:  func.func public @r2(%arg0: tensor<1xf32>) -> tensor<288xf32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<288xf32>
// CHECK-NEXT:    return %0 : tensor<288xf32>
// CHECK-NEXT:  }
// CHECK:  func.func public @r3(%arg0: tensor<1x3072xi32>) -> tensor<3072x1xi32> {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 0] : (tensor<1x3072xi32>) -> tensor<3072x1xi32>
// CHECK-NEXT:    return %0 : tensor<3072x1xi32>
// CHECK-NEXT:  }
