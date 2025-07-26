// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzymexla-triton-simplify,ttf.func(enzymexla-stablehlo-to-triton-compatible-dialect),enzymexla-triton-simplify)" %s | FileCheck %s

tt.func @"add_kernel!_lattice_kernel"(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %c_0 = stablehlo.constant dense<8> : tensor<8xi64>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<8> : tensor<i32>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<!tt.ptr<f32>>
    %3 = tt.get_program_id x : i32
    %4 = tt.splat %3 : i32 -> tensor<i32>
    %5 = stablehlo.multiply %4, %c_2 : tensor<i32>
    %6 = stablehlo.add %5, %c_1 : tensor<i32>
    %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<i64>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i64>) -> tensor<8xi64>
    %9 = stablehlo.add %8, %c : tensor<8xi64>
    %10 = stablehlo.compare  LT, %9, %c_0 : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
    %11 = tt.splat %1 : tensor<!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
    %12 = tt.addptr %11, %9 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %13 = tt.load %12, %10 : tensor<8x!tt.ptr<f32>>
    %14 = tt.splat %2 : tensor<!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
    %15 = tt.addptr %14, %9 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %16 = tt.load %15, %10 : tensor<8x!tt.ptr<f32>>
    %17 = tt.splat %0 : tensor<!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
    %18 = tt.addptr %17, %9 : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    %19 = stablehlo.add %13, %16 : tensor<8xf32>
    tt.store %18, %19, %10 : tensor<8x!tt.ptr<f32>>
    tt.return
}
