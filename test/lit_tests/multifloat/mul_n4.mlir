// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=first" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=4 concat-dimension=last" %s | FileCheck %s --check-prefix=LAST

func.func @test_mul_n4(%arg0: tensor<5xf64>, %arg1: tensor<5xf64>) -> tensor<5xf64> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<5xf64>
  return %0 : tensor<5xf64>
}

// 4-limb representation along the first dim is packed as tensor<4x5xf32>.
// FIRST-LABEL: func.func @test_mul_n4
// FIRST: stablehlo.concatenate {{.*}} dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>
// FIRST: stablehlo.multiply {{.*}} : tensor<1x5xf32>

// LAST-LABEL: func.func @test_mul_n4
// LAST: stablehlo.concatenate {{.*}} : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x4xf32>
// LAST: stablehlo.multiply {{.*}} : tensor<5x1xf32>
