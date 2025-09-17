// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzymexla-triton-simplify)" %s | FileCheck %s

tt.func @"add_kernel!_lattice_kernel"(%arg0: !tt.ptr<f32>) -> (tensor<8x!tt.ptr<f32>>) {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
    %1 = tt.splat %0 : tensor<!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
    tt.return %1 : tensor<8x!tt.ptr<f32>>
}

// CHECK: tt.func @"add_kernel!_lattice_kernel"(%arg0: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
// CHECK-NEXT:     %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK-NEXT:     tt.return %0 : tensor<8x!tt.ptr<f32>>
// CHECK-NEXT: }
