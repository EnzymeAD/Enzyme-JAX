// RUN: enzymexlamlir-opt --resolve-custom-lowering --allow-unregistered-dialect --enzyme-hlo-opt %s | FileCheck %s

func.func @custom_lowering1(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = stablehlo.sine %arg0 : tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

func.func @custom_lowering2(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c = stablehlo.constant dense<2.000000e+00> : tensor<1x1xf32>
    %0 = stablehlo.cosine %arg0 : tensor<?x?xf32>
    %d1 = "stablehlo.get_dimension_size"(%arg0) {
        dimension = 0 : i64
    } : (tensor<?x?xf32>) -> tensor<i32>
    %d2 = "stablehlo.get_dimension_size"(%arg0) {
        dimension = 1 : i64
    } : (tensor<?x?xf32>) -> tensor<i32>
    %d1_r = stablehlo.reshape %d1 : (tensor<i32>) -> tensor<1xi32>
    %d2_r = stablehlo.reshape %d2 : (tensor<i32>) -> tensor<1xi32>
    %d = stablehlo.concatenate %d1_r, %d2_r, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %1 = "stablehlo.dynamic_broadcast_in_dim"(%c, %d) {
        broadcast_dimensions = array<i64: 0, 1>
    } : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %2 = stablehlo.add %0, %1 : tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
}

enzymexla.lowering.register "mydialect.sin_or_cos" @custom_lowering1 ({"op" = "sin"})
enzymexla.lowering.register "mydialect.sin_or_cos" @custom_lowering2 ({
    "op" = "cos",
    "add" = "true"
})

func.func @main(%arg0: tensor<8x8xf32>) -> tensor<4x4xf32> {
    %0 = "mydialect.sin_or_cos"(%arg0) {
        "enzymexla.lowering.config" = { "op" = "sin" }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>

    %1 = stablehlo.slice %0 [0:4, 0:4] : (tensor<8x8xf32>) -> tensor<4x4xf32>

    %2 = "mydialect.sin_or_cos"(%1) {
        "enzymexla.lowering.config" = { "op" = "cos" }
    } : (tensor<4x4xf32>) -> tensor<4x4xf32>

    %3 = stablehlo.add %2, %1 : tensor<4x4xf32>

    %4 = "mydialect.sin_or_cos"(%3) {
        "enzymexla.lowering.config" = { "add" = "true" }
    } : (tensor<4x4xf32>) -> tensor<4x4xf32>

    return %4 : tensor<4x4xf32>
}

// CHECK: func.func private @custom_lowering2__[[ADD_ID:[0-9]+]](%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:     %0 = stablehlo.cosine %arg0 : tensor<4x4xf32>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst : tensor<4x4xf32>
// CHECK-NEXT:     return %1 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK: func.func private @custom_lowering2__[[COS_ID:[0-9]+]](%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<4x4xf32>
// CHECK-NEXT:     %0 = stablehlo.cosine %arg0 : tensor<4x4xf32>
// CHECK-NEXT:     %1 = stablehlo.add %0, %cst : tensor<4x4xf32>
// CHECK-NEXT:     return %1 : tensor<4x4xf32>
// CHECK-NEXT: }
// CHECK: func.func private @custom_lowering1__[[SIN_ID:[0-9]+]](%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
// CHECK-NEXT:     %0 = stablehlo.sine %arg0 : tensor<8x8xf32>
// CHECK-NEXT:     return %0 : tensor<8x8xf32>
// CHECK-NEXT: }
// CHECK: func.func @main(%arg0: tensor<8x8xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:     %0 = call @custom_lowering1__[[SIN_ID]](%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:4] : (tensor<8x8xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %2 = call @custom_lowering2__[[COS_ID]](%1) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %3 = stablehlo.add %2, %1 : tensor<4x4xf32>
// CHECK-NEXT:     %4 = call @custom_lowering2__[[ADD_ID]](%3) : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     return %4 : tensor<4x4xf32>
// CHECK-NEXT: }
