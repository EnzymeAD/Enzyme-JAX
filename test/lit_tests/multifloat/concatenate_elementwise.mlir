// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @test_combine_add(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<1xf64>, %arg3: tensor<1xf64>) -> tensor<2xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<1xf64>
  %1 = stablehlo.add %arg2, %arg3 : tensor<1xf64>
  %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  return %2 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_combine_add
// CHECK: stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK: stablehlo.concatenate %{{.*}}, %{{.*}}, dim = 1 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// CHECK: stablehlo.add %{{.*}}, %{{.*}} : tensor<1x2xf32>
