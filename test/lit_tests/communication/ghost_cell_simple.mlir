// RUN: enzymexlamlir-opt --stencil-ghost-cell-widening --canonicalize %s | FileCheck %s

// Two levels of paired shifts (low+high). The pass should:
// 1. Replace ALL 4 stencil pads with slices
// 2. Insert ONE wide pad at the root (width 2)
// 3. No intermediate pads in the output

sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @test(%field: tensor<10x20xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<10x20xf64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %s1 = stablehlo.slice %field [1:10, 0:20] : (tensor<10x20xf64>) -> tensor<9x20xf64>
    %p1r = stablehlo.pad %s1, %zero, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<9x20xf64>, tensor<f64>) -> tensor<10x20xf64>
    %p1l = stablehlo.pad %s1, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<9x20xf64>, tensor<f64>) -> tensor<10x20xf64>
    %d1 = stablehlo.subtract %p1r, %p1l : tensor<10x20xf64>
    %r1 = stablehlo.add %field, %d1 : tensor<10x20xf64>
    %s2 = stablehlo.slice %r1 [1:10, 0:20] : (tensor<10x20xf64>) -> tensor<9x20xf64>
    %p2r = stablehlo.pad %s2, %zero, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<9x20xf64>, tensor<f64>) -> tensor<10x20xf64>
    %p2l = stablehlo.pad %s2, %zero, low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<9x20xf64>, tensor<f64>) -> tensor<10x20xf64>
    %d2 = stablehlo.subtract %p2r, %p2l : tensor<10x20xf64>
    return %d2 : tensor<10x20xf64>
}

// CHECK-LABEL: func.func @test
// One wide pad at the root (width 2 → 10+4=14)
// CHECK: stablehlo.pad {{.*}} low = [2, 0], high = [2, 0]
// CHECK-SAME: tensor<14x20xf64>
// No intermediate stencil pads
// CHECK-NOT: low = [1, 0], high = [0, 0]
// CHECK-NOT: low = [0, 0], high = [1, 0]
// Final result is tensor<10x20>
// CHECK: return {{.*}} : tensor<10x20xf64>
