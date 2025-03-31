// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=131072})" | FileCheck %s

module {
  // CHECK-LABEL: @test_reshape_of_concat_1
  func.func @test_reshape_of_concat_1(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<4xf32> {
    // CHECK: reshape
    // CHECK-NEXT: reshape
    // CHECK-NEXT: concatenate
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x4xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  // CHECK-LABEL: @test_reshape_of_concat_2
  func.func @test_reshape_of_concat_2(%arg0: tensor<1x268x6xf64>, 
                                      %arg1: tensor<1x268x2048xf64>, 
                                      %arg2: tensor<1x268x6xf64>) -> tensor<268x2060xf64> {
    // CHECK: reshape
    // CHECK-NEXT: reshape
    // CHECK-NEXT: reshape
    // CHECK-NEXT: concatenate
    %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 2 {mhlo.sharding = "{devices=[1,2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<1x268x6xf64>, tensor<1x268x2048xf64>, tensor<1x268x6xf64>) -> tensor<1x268x2060xf64>
    %1 = stablehlo.reshape %0 {mhlo.sharding = "{devices=[2,2,2]<=[2,2,2]T(1,0,2) last_tile_dim_replicate}"} : (tensor<1x268x2060xf64>) -> tensor<268x2060xf64>
    return %1 : tensor<268x2060xf64>
  }
}
