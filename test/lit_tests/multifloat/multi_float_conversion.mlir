// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32" %s | FileCheck %s

func.func @test_add(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

// CHECK-LABEL: func.func @test_add
// CHECK-DAG: %[[C1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<f64> to tensor<2xf32>
// CHECK-DAG: %[[C2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<f64> to tensor<2xf32>
// CHECK: stablehlo.slice
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.slice
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.add
// CHECK: stablehlo.subtract
// CHECK: %[[OUT:.*]] = stablehlo.reshape %{{.*}} : (tensor<1xf64>) -> tensor<f64>
// CHECK: return %[[OUT]]
