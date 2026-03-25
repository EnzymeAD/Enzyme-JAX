// RUN: enzymexlamlir-opt --stencil-ghost-cell-widening --canonicalize %s | FileCheck %s

// Chain where a reshape (NxM → 1xNxM) shifts the stencil dimension.
// The widening should track dim through the reshape.

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @test(%field: tensor<10x20xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<10x20xf64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %s1 = stablehlo.slice %field [1:10, 0:20] : (tensor<10x20xf64>) -> tensor<9x20xf64>
    %p1 = stablehlo.pad %s1, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<9x20xf64>, tensor<f64>) -> tensor<10x20xf64>
    %rs1 = stablehlo.reshape %p1 : (tensor<10x20xf64>) -> tensor<1x10x20xf64>
    %neg = stablehlo.negate %rs1 : tensor<1x10x20xf64>
    %rs2 = stablehlo.reshape %neg : (tensor<1x10x20xf64>) -> tensor<10x20xf64>
    %s2 = stablehlo.slice %rs2 [1:10, 0:20] : (tensor<10x20xf64>) -> tensor<9x20xf64>
    %p2 = stablehlo.pad %s2, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<9x20xf64>, tensor<f64>) -> tensor<10x20xf64>
    return %p2 : tensor<10x20xf64>
}

// CHECK-LABEL: func.func @test
// Wide pad (K=2 → 10+4=14)
// CHECK: stablehlo.pad {{.*}} low = [2, 0], high = [2, 0]
// CHECK-SAME: tensor<14x20xf64>
// Reshape tracks the widened dim (14 → 1x14x20)
// CHECK: stablehlo.reshape {{.*}} tensor<1x{{[0-9]+}}x20xf64>
// No stencil pads
// CHECK-NOT: low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<9x20
// CHECK: return {{.*}} : tensor<10x20xf64>
