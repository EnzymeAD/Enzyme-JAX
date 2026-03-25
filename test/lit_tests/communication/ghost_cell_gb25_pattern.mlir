// RUN: enzymexlamlir-opt --stencil-ghost-cell-widening --canonicalize %s | FileCheck %s

// Mimics the GB-25 ocean simulation pattern:
// 3 levels of stencil shifts with complementary pads, subtracts, and field accumulation.
// Critical path = 3 (one pad per level on the longest path).

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @ocean_stencil(%field: tensor<100x100xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
                          %coeff: tensor<99x100xf32>) -> (tensor<100x100xf32>, tensor<100x100xf32>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>

    // === Level 1: finite difference on field ===
    %s1 = stablehlo.slice %field [1:100, 0:100] : (tensor<100x100xf32>) -> tensor<99x100xf32>
    %m1 = stablehlo.multiply %coeff, %s1 : tensor<99x100xf32>
    %p1r = stablehlo.pad %m1, %zero, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<99x100xf32>, tensor<f32>) -> tensor<100x100xf32>
    %p1l = stablehlo.pad %m1, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<99x100xf32>, tensor<f32>) -> tensor<100x100xf32>
    %d1 = stablehlo.subtract %p1r, %p1l : tensor<100x100xf32>
    %r1 = stablehlo.add %field, %d1 : tensor<100x100xf32>

    // === Level 2: finite difference on r1 ===
    %s2 = stablehlo.slice %r1 [1:100, 0:100] : (tensor<100x100xf32>) -> tensor<99x100xf32>
    %m2 = stablehlo.multiply %coeff, %s2 : tensor<99x100xf32>
    %p2r = stablehlo.pad %m2, %zero, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<99x100xf32>, tensor<f32>) -> tensor<100x100xf32>
    %p2l = stablehlo.pad %m2, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<99x100xf32>, tensor<f32>) -> tensor<100x100xf32>
    %d2 = stablehlo.subtract %p2r, %p2l : tensor<100x100xf32>
    %r2 = stablehlo.add %r1, %d2 : tensor<100x100xf32>

    // === Level 3: one more finite difference ===
    %s3 = stablehlo.slice %r2 [1:100, 0:100] : (tensor<100x100xf32>) -> tensor<99x100xf32>
    %p3r = stablehlo.pad %s3, %zero, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<99x100xf32>, tensor<f32>) -> tensor<100x100xf32>
    %p3l = stablehlo.pad %s3, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<99x100xf32>, tensor<f32>) -> tensor<100x100xf32>
    %d3 = stablehlo.subtract %p3r, %p3l : tensor<100x100xf32>

    // Two outputs: r2 and d3 (both used outside the stencil)
    return %r2, %d3 : tensor<100x100xf32>, tensor<100x100xf32>
}

// CHECK-LABEL: func.func @ocean_stencil
// One wide pad — critical path is 3 pads deep → K=3, pad by 3 each side (100→106)
// CHECK: stablehlo.pad {{.*}} low = [3, 0], high = [3, 0]
// CHECK-SAME: tensor<106x100xf32>
// Coefficient gets padded to match (local, no communication)
// CHECK: stablehlo.pad %arg1
// No width-1 stencil pads remain
// CHECK-NOT: low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<99x100
// CHECK-NOT: low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<99x100
// Both outputs are correct size
// CHECK: return {{.*}} : tensor<100x100xf32>, tensor<100x100xf32>
