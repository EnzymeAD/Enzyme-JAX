// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=first" %s | FileCheck %s --check-prefix=FIRST
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3 concat-dimension=last" %s | FileCheck %s --check-prefix=LAST

func.func @test_add_n3(%arg0: tensor<5xf64>, %arg1: tensor<5xf64>) -> tensor<5xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<5xf64>
  return %0 : tensor<5xf64>
}

// 3-limb representation along the first dim is packed as tensor<3x5xf32>.
// FIRST-LABEL: func.func @test_add_n3
// FIRST: stablehlo.concatenate {{.*}} dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<3x5xf32>
// FIRST: stablehlo.add %{{.*}} : tensor<1x5xf32>

// LAST-LABEL: func.func @test_add_n3
// LAST: stablehlo.concatenate {{.*}} : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x3xf32>
// LAST: stablehlo.add %{{.*}} : tensor<5x1xf32>
