// RUN: enzymexlamlir-opt --arith-raise %s | FileCheck %s

module {
  // trunc(x) rounds towards zero: select(x >= 0, floor(x), ceil(x)).
  // CHECK-LABEL: @trunc_f32
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  // CHECK: %[[GE:.+]] = stablehlo.compare GE, %arg0, %[[ZERO]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK: %[[FLOOR:.+]] = stablehlo.floor %arg0 : tensor<4xf32>
  // CHECK: %[[CEIL:.+]] = stablehlo.ceil %arg0 : tensor<4xf32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %[[GE]], %[[FLOOR]], %[[CEIL]] : tensor<4xi1>, tensor<4xf32>
  // CHECK: return %[[SEL]] : tensor<4xf32>
  // CHECK-NOT: math.trunc
  func.func @trunc_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.trunc %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: @trunc_f64
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
  // CHECK: %[[GE:.+]] = stablehlo.compare GE, %arg0, %[[ZERO]] : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
  // CHECK: %[[FLOOR:.+]] = stablehlo.floor %arg0 : tensor<4xf64>
  // CHECK: %[[CEIL:.+]] = stablehlo.ceil %arg0 : tensor<4xf64>
  // CHECK: %[[SEL:.+]] = stablehlo.select %[[GE]], %[[FLOOR]], %[[CEIL]] : tensor<4xi1>, tensor<4xf64>
  // CHECK: return %[[SEL]] : tensor<4xf64>
  // CHECK-NOT: math.trunc
  func.func @trunc_f64(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = math.trunc %arg0 : tensor<4xf64>
    return %0 : tensor<4xf64>
  }

  // Scalars are left alone by the raising pass.
  // CHECK-LABEL: @trunc_scalar
  // CHECK: math.trunc %arg0 : f32
  func.func @trunc_scalar(%arg0: f32) -> f32 {
    %0 = math.trunc %arg0 : f32
    return %0 : f32
  }
}
