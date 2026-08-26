// RUN: not enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 expansion-size=3" %s 2>&1 | FileCheck %s

func.func @add(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
  return %0 : tensor<f64>
}

// CHECK: Unsupported expansion size specified: 3
