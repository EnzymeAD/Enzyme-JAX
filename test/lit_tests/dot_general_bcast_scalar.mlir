// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

// Reproduction for issue #1475:
// DotGeneralSimplify handles dot_general(splat_constant, X) by folding it
// into a reduce(X). It does NOT handle the equivalent case where one
// operand is a broadcast_in_dim of a runtime scalar (e.g., a function
// argument). The constant-folding canonicalizers can't help here because
// the scalar value isn't known at compile time.

// Variant A -- broadcast of a CONSTANT scalar: already simplified by the pipeline.
func.func @bcast_rank0_constant(%arg1: tensor<1024x32xf32>) -> tensor<24x32xf32> {
    %s = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %ones = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<24x1024xf32>
    %r = stablehlo.dot_general %ones, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x1024xf32>, tensor<1024x32xf32>) -> tensor<24x32xf32>
    return %r : tensor<24x32xf32>
}
// CHECK-LABEL: func.func @bcast_rank0_constant
// CHECK-NOT: stablehlo.dot_general

// Variant B -- broadcast of a RUNTIME scalar: NOT simplified. This is the gap.
// After issue #1475 is fixed, the dot_general should be replaced by a
// reduce(arg1) multiplied by the scalar (broadcast back to the result shape).
func.func @bcast_rank0_runtime(%s: tensor<f32>, %arg1: tensor<1024x32xf32>) -> tensor<24x32xf32> {
    %ones = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<24x1024xf32>
    %r = stablehlo.dot_general %ones, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x1024xf32>, tensor<1024x32xf32>) -> tensor<24x32xf32>
    return %r : tensor<24x32xf32>
}
// CHECK-LABEL: func.func @bcast_rank0_runtime
// CHECK-NOT: stablehlo.dot_general
// Variant C -- broadcast of a RUNTIME scalar on the RHS operand.
func.func @bcast_rank0_runtime_rhs(%s: tensor<f32>, %arg0: tensor<32x1024xf32>) -> tensor<32x24xf32> {
    %ones = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<1024x24xf32>
    %r = stablehlo.dot_general %arg0, %ones, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1024xf32>, tensor<1024x24xf32>) -> tensor<32x24xf32>
    return %r : tensor<32x24xf32>
}
// CHECK-LABEL: func.func @bcast_rank0_runtime_rhs
// CHECK-NOT: stablehlo.dot_general
