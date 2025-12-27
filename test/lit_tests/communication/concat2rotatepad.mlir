// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(optimize-communication{periodic_concat=0 concat_to_pad_comm=0 concat_to_rotatepad=3 concat_to_dus=1 dus_to_pad_comm=0})" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>
func.func @main1(%15: tensor<20x6144x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x6128x12273xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %21 = stablehlo.slice %15 [8:12, 6:6134, 12271:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<4x6128x1xf32>
    %31 = stablehlo.slice %15 [8:12, 6:6134, 0:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<4x6128x12272xf32>
    %723 = stablehlo.concatenate %21, %31, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x1xf32>, tensor<4x6128x12272xf32>) -> tensor<4x6128x12273xf32>
    return %723 : tensor<4x6128x12273xf32>
}

func.func @main2(%15: tensor<20x6144x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x6128x12273xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
    %21 = stablehlo.slice %15 [8:12, 6:6134, 12271:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<4x6128x1xf32>
    %31 = stablehlo.slice %15 [8:12, 6:6134, 0:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<4x6128x12272xf32>
    %723 = stablehlo.concatenate %31, %21, dim = 2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12272xf32>, tensor<4x6128x1xf32>) -> tensor<4x6128x12273xf32>
    return %723 : tensor<4x6128x12273xf32>
}

// CHECK:  func.func @main1(%arg0: tensor<20x6144x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x6128x12273xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 6:6134, 0:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<4x6128x12272xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12272xf32>, tensor<f32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    %2 = "enzymexla.rotate"(%1) <{amount = 12272 : si32, dimension = 2 : si32}> : (tensor<4x6128x12273xf32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:4, 0:6128, 0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12273xf32>) -> tensor<4x6128x1xf32>
// CHECK-NEXT:    %4 = stablehlo.dynamic_update_slice %1, %3, %c, %c, %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12273xf32>, tensor<4x6128x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    return %4 : tensor<4x6128x12273xf32>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<20x6144x12272xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<4x6128x12273xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"z"}, {"y"}, {"x"}]>}) {
// CHECK-NEXT:    %c = stablehlo.constant dense<12272> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 6:6134, 0:12272] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<20x6144x12272xf32>) -> tensor<4x6128x12272xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 0], high = [0, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12272xf32>, tensor<f32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %0, %cst, low = [0, 0, 1], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12272xf32>, tensor<f32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %2 [0:4, 0:6128, 12272:12273] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12273xf32>) -> tensor<4x6128x1xf32>
// CHECK-NEXT:    %4 = stablehlo.dynamic_update_slice %1, %3, %c_0, %c_0, %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"z"}, {"y"}, {"x"}]>]>} : (tensor<4x6128x12273xf32>, tensor<4x6128x1xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    return %4 : tensor<4x6128x12273xf32>
// CHECK-NEXT:  }
